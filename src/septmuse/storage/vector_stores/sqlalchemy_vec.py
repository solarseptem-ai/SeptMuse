#  Copyright 2026 The sonhhxg0529 Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""SQLAlchemy 通用向量存储 — 跨方言 JSON + numpy 余弦。

任何 SQLAlchemy engine (SQLite/MySQL/PostgreSQL) 均可用。
向量以 JSON list[float] 存储, 检索用 numpy 余弦相似。
upsert 用 DELETE + INSERT 两步模式 (跨方言兼容, 避免 INSERT OR REPLACE 的 SQLite 限制)。
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine
from sqlmodel import Session

from septmuse.core.logging import get_logger
from septmuse.storage.vector_stores.base import VectorEntry, VectorSearchResult, VectorStoreBase

logger = get_logger(__name__)


class SQLAlchemyVectorStore(VectorStoreBase):
    """SQLAlchemy 通用向量存储 (JSON + numpy 余弦, 跨方言)。

    用法:
        from sqlalchemy import create_engine
        engine = create_engine("sqlite:///test.db")
        store = SQLAlchemyVectorStore(engine)
        store.insert_vectors([[1.0, 0.0]], ["m1"], [{"user_id": "alice"}])
        results = store.search_vectors([0.9, 0.1], top_k=5, filters={"user_id": "alice"})
    """

    def __init__(self, engine: Engine) -> None:
        self._engine = engine
        self._create_table()
        logger.info("sqlalchemy_vector_store_ready", dialect=engine.dialect.name)

    def _create_table(self) -> None:
        """建表 — 跨方言 DDL。"""
        with self._engine.connect() as conn:
            conn.execute(
                text(
                    """
                CREATE TABLE IF NOT EXISTS vector_entries (
                    id      VARCHAR(512) PRIMARY KEY,
                    vector  TEXT NOT NULL,
                    payload TEXT DEFAULT '{}'
                )
            """
                )
            )
            conn.commit()

    def insert_vectors(
        self,
        vectors: list[list[float]],
        ids: list[str],
        payloads: list[dict[str, Any]] | None = None,
    ) -> None:
        if len(vectors) != len(ids):
            raise ValueError(f"vectors ({len(vectors)}) and ids ({len(ids)}) length mismatch")
        if payloads is None:
            payloads = [{}] * len(ids)
        elif len(payloads) != len(ids):
            raise ValueError(f"payloads ({len(payloads)}) and ids ({len(ids)}) length mismatch")
        # 维度校验
        if vectors:
            dim = len(vectors[0])
            for i, v in enumerate(vectors):
                if len(v) != dim:
                    raise ValueError(
                        f"vector dimension mismatch: vector[0] has dim {dim}, vector[{i}] has dim {len(v)}"
                    )
        # 跨方言 upsert: DELETE + INSERT 两步 (INSERT OR REPLACE 仅 SQLite 可用)
        with Session(self._engine) as session:
            for vec, vid, payload in zip(vectors, ids, payloads, strict=True):
                session.execute(text("DELETE FROM vector_entries WHERE id = :id").bindparams(id=vid))
                session.execute(
                    text("INSERT INTO vector_entries (id, vector, payload) VALUES (:id, :vec, :payload)").bindparams(
                        id=vid, vec=json.dumps(vec), payload=json.dumps(payload)
                    )
                )
            session.commit()

    def search_vectors(
        self,
        query_vector: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        query = np.array(query_vector, dtype=np.float32)
        query_norm = float(np.linalg.norm(query))
        if query_norm == 0:
            return []

        rows = self._fetch_rows(filters)
        scored: list[tuple[float, str, dict[str, Any]]] = []
        for vid, vec_json, payload_json in rows:
            vec = np.array(json.loads(vec_json), dtype=np.float32)
            if vec.shape != query.shape:
                raise ValueError(
                    f"vector dimension mismatch: query={query.shape} stored={vec.shape} for id={vid}"
                )
            vec_norm = float(np.linalg.norm(vec))
            if vec_norm == 0:
                continue
            score = float(np.dot(query, vec) / (query_norm * vec_norm))
            score = max(0.0, min(1.0, score))
            scored.append((score, vid, json.loads(payload_json) if payload_json else {}))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [VectorSearchResult(id=vid, score=sc, payload=pl) for sc, vid, pl in scored[:top_k]]

    def _fetch_rows(self, filters: dict[str, Any] | None) -> list[tuple[str, str, str]]:
        """取行, payload 过滤推到 SQL 层 (减少 Python 侧加载和反序列化量)。

        SQLite/MySQL: json_extract WHERE 精确过滤, 只加载匹配行。
        其他方言: 全量加载 + Python 侧过滤 (降级路径)。
        """
        dialect = self._engine.dialect.name
        supports_json_extract = dialect in ("sqlite", "mysql")

        where_clauses: list[str] = []
        params: dict[str, Any] = {}

        if filters and supports_json_extract:
            for i, (key, value) in enumerate(filters.items()):
                if value is None:
                    continue
                param_name = f"fv{i}"
                where_clauses.append(f"json_extract(payload, '$.{key}') = :{param_name}")
                params[param_name] = str(value) if not isinstance(value, (int, float, bool)) else value

        sql = "SELECT id, vector, payload FROM vector_entries"
        if where_clauses:
            sql += " WHERE " + " AND ".join(where_clauses)

        with self._engine.connect() as conn:
            result = conn.execute(text(sql).bindparams(**params))
            rows = result.fetchall()

        if not filters or supports_json_extract:
            return [(r[0], r[1], r[2]) for r in rows]

        # 降级: Python 侧 payload 过滤 (非 SQLite/MySQL 方言)
        filtered = []
        for r in rows:
            payload = json.loads(r[2]) if r[2] else {}
            if all(payload.get(k) == v for k, v in filters.items() if v is not None):
                filtered.append((r[0], r[1], r[2]))
        return filtered

    def delete_vector(self, vector_id: str) -> bool:
        with self._engine.connect() as conn:
            result = conn.execute(text("DELETE FROM vector_entries WHERE id = :id").bindparams(id=vector_id))
            conn.commit()
            return result.rowcount > 0

    def get_vector(self, vector_id: str) -> VectorEntry | None:
        with self._engine.connect() as conn:
            result = conn.execute(
                text("SELECT vector, payload FROM vector_entries WHERE id = :id").bindparams(id=vector_id)
            )
            row = result.fetchone()
        if row is None:
            return None
        return VectorEntry(
            id=vector_id,
            vector=json.loads(row[0]),
            payload=json.loads(row[1]) if row[1] else {},
        )

    def list_vectors(
        self,
        filters: dict[str, Any] | None = None,
        limit: int | None = None,
    ) -> list[VectorEntry]:
        rows = self._fetch_rows(filters)
        entries = [
            VectorEntry(
                id=vid,
                vector=json.loads(vec_json),
                payload=json.loads(payload_json) if payload_json else {},
            )
            for vid, vec_json, payload_json in rows
        ]
        if limit is not None:
            entries = entries[:limit]
        return entries

    def update_vector(self, vector_id: str, vector: list[float], payload: dict[str, Any] | None = None) -> bool:
        """原地更新 — DELETE + INSERT 两步（跨方言 upsert）。"""
        with Session(self._engine) as session:
            session.execute(text("DELETE FROM vector_entries WHERE id = :id").bindparams(id=vector_id))
            session.execute(
                text("INSERT INTO vector_entries (id, vector, payload) VALUES (:id, :vec, :payload)").bindparams(
                    id=vector_id, vec=json.dumps(vector), payload=json.dumps(payload or {})
                )
            )
            session.commit()
        return True

    def delete_collection(self) -> None:
        """DROP TABLE。"""
        with self._engine.connect() as conn:
            conn.execute(text("DROP TABLE IF EXISTS vector_entries"))
            conn.commit()

    def get_collection_info(self) -> dict[str, Any]:
        """SELECT COUNT(*)；表不存在时返回 count=0。"""
        if not inspect(self._engine).has_table("vector_entries"):
            return {"name": "vector_entries", "count": 0}
        with self._engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM vector_entries"))
            count = result.scalar() or 0
        return {"name": "vector_entries", "count": count}

    def reset_collection(self) -> None:
        """DROP + CREATE。"""
        self.delete_collection()
        self._create_table()

    def close(self) -> None:
        self._engine.dispose()

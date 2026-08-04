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
"""PgvectorVectorStore — PostgreSQL + pgvector 扩展向量存储。

有 pgvector 扩展时: 用 VECTOR(dim) 列 + <=> 余弦距离算子, 性能远超 numpy。
无 pgvector 时: 降级为 SQLAlchemyVectorStore (JSON + numpy), 日志警告。

需要: pip install pgvector (SQLAlchemy pgvector 支持)
"""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Engine

from septmuse.core.logging import get_logger
from septmuse.storage.vector_stores.base import VectorEntry, VectorSearchResult, VectorStoreBase
from septmuse.storage.vector_stores.sqlalchemy_vec import SQLAlchemyVectorStore

logger = get_logger(__name__)

# 检测 pgvector 是否可用
try:
    from pgvector.sqlalchemy import Vector  # noqa: F401

    PGVECTOR_AVAILABLE = True
except ImportError:
    PGVECTOR_AVAILABLE = False


class PgvectorVectorStore(VectorStoreBase):
    """PostgreSQL + pgvector 向量存储 (有 pgvector 时用扩展, 无则降级)。

    用法:
        from sqlalchemy import create_engine
        engine = create_engine("postgresql://user:pass@host/db")
        store = PgvectorVectorStore(engine)
        store.insert_vectors([[1.0, 0.0]], ["m1"], [{"user_id": "alice"}])
    """

    def __init__(self, engine: Engine, vector_dim: int = 384) -> None:
        self._engine = engine
        self._dim = vector_dim
        self._pgvector_available = PGVECTOR_AVAILABLE and engine.dialect.name == "postgresql"

        if self._pgvector_available:
            self._init_pgvector()
            logger.info("pgvector_store_ready", dim=vector_dim)
        else:
            # 降级为 SQLAlchemyVectorStore (JSON + numpy)
            logger.warning("pgvector_not_available_fallback", dialect=engine.dialect.name)
            self._fallback = SQLAlchemyVectorStore(engine)

    def _init_pgvector(self) -> None:
        """初始化 pgvector 扩展 + 建表。"""
        with self._engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.execute(
                text(
                    f"""
                CREATE TABLE IF NOT EXISTS vector_entries (
                    id      VARCHAR(512) PRIMARY KEY,
                    vector  VECTOR({self._dim}),
                    payload JSONB DEFAULT '{{}}'::jsonb
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
        if not self._pgvector_available:
            return self._fallback.insert_vectors(vectors, ids, payloads)
        if payloads is None:
            payloads = [{}] * len(ids)
        with self._engine.connect() as conn:
            for vec, vid, payload in zip(vectors, ids, payloads, strict=True):
                # 跨方言不可用, 此分支仅 PG 执行; 用 ON CONFLICT 幂等 upsert
                conn.execute(
                    text(
                        "INSERT INTO vector_entries (id, vector, payload) VALUES (:id, :vec::vector, :payload::jsonb) "
                        "ON CONFLICT (id) DO UPDATE SET vector = :vec::vector, payload = :payload::jsonb"
                    ).bindparams(id=vid, vec=json.dumps(vec), payload=json.dumps(payload))
                )
            conn.commit()

    def search_vectors(
        self,
        query_vector: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        if not self._pgvector_available:
            return self._fallback.search_vectors(query_vector, top_k, filters)

        # pgvector 余弦距离: <=> 操作符 (distance 越小越相似, score = 1 - distance)
        query_str = json.dumps(query_vector)
        sql = text(
            """
            SELECT id, vector, payload,
                   (vector <=> :query::vector) AS distance
            FROM vector_entries
            ORDER BY vector <=> :query::vector
            LIMIT :top_k
        """
        ).bindparams(query=query_str, top_k=top_k)

        with self._engine.connect() as conn:
            result = conn.execute(sql)
            rows = result.fetchall()

        results: list[VectorSearchResult] = []
        for row in rows:
            vid, _vec_json, payload_json, distance = row
            score = max(0.0, 1.0 - float(distance))
            payload = json.loads(payload_json) if payload_json else {}
            # payload 过滤 (PG JSONB 字段匹配)
            if filters and not all(payload.get(k) == v for k, v in filters.items()):
                continue
            results.append(VectorSearchResult(id=str(vid), score=score, payload=payload))
        return results

    def delete_vector(self, vector_id: str) -> bool:
        if not self._pgvector_available:
            return self._fallback.delete_vector(vector_id)
        with self._engine.connect() as conn:
            result = conn.execute(text("DELETE FROM vector_entries WHERE id = :id").bindparams(id=vector_id))
            conn.commit()
            return result.rowcount > 0

    def get_vector(self, vector_id: str) -> VectorEntry | None:
        if not self._pgvector_available:
            return self._fallback.get_vector(vector_id)
        with self._engine.connect() as conn:
            result = conn.execute(
                text("SELECT vector, payload FROM vector_entries WHERE id = :id").bindparams(id=vector_id)
            )
            row = result.fetchone()
        if row is None:
            return None
        return VectorEntry(
            id=vector_id,
            vector=json.loads(row[0]) if row[0] else [],
            payload=json.loads(row[1]) if row[1] else {},
        )

    def list_vectors(
        self,
        filters: dict[str, Any] | None = None,
        limit: int | None = None,
    ) -> list[VectorEntry]:
        if not self._pgvector_available:
            return self._fallback.list_vectors(filters, limit)
        with self._engine.connect() as conn:
            result = conn.execute(text("SELECT id, vector, payload FROM vector_entries"))
            rows = result.fetchall()
        entries: list[VectorEntry] = []
        for row in rows:
            payload = json.loads(row[2]) if row[2] else {}
            if filters and not all(payload.get(k) == v for k, v in filters.items()):
                continue
            entries.append(
                VectorEntry(
                    id=str(row[0]),
                    vector=json.loads(row[1]) if row[1] else [],
                    payload=payload,
                )
            )
        if limit is not None:
            entries = entries[:limit]
        return entries

    def close(self) -> None:
        if self._pgvector_available:
            self._engine.dispose()
        else:
            self._fallback.close()

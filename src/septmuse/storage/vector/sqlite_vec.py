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
"""SQLite 向量存储 — 默认零配置实现 (numpy 余弦相似)。

借鉴 mem0 vector_stores/faiss.py 的 numpy 余弦回退模式,
用 SQLite vector_entries 表持久化, JSON list[float] 存向量。

参考模式 (实证):
- numpy 余弦相似: mem0 FaissVectorStore.search 的 fallback 路径
- payload JSON 列: mem0 Qdrant.payload 字段
- user_id 隔离: mem0 search filters 参数
"""

from __future__ import annotations

import json
import sqlite3
import threading
from typing import Any

import numpy as np

from septmuse.core.logging import get_logger
from septmuse.storage.vector.base import VectorEntry, VectorSearchResult, VectorStoreBase

logger = get_logger(__name__)


class SQLiteVectorStore(VectorStoreBase):
    """SQLite 向量存储 (numpy 余弦, 零配置默认)。

    用法:
        conn = sqlite3.connect("mem.db")
        store = SQLiteVectorStore(conn=conn)
        store.insert_vectors([[1.0, 0.0]], ["m1"], [{"user_id": "alice"}])
        results = store.search_vectors([0.9, 0.1], top_k=5, filters={"user_id": "alice"})
    """

    def __init__(self, conn: sqlite3.Connection, lock: threading.Lock | None = None) -> None:
        self.conn = conn
        self._lock = lock or threading.Lock()
        self._create_table()
        logger.info("sqlite_vector_store_ready")

    def _create_table(self) -> None:
        with self._lock:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS vector_entries (
                    id       TEXT PRIMARY KEY,
                    vector   TEXT NOT NULL,
                    payload  TEXT DEFAULT '{}'
                )
                """
            )
            self.conn.commit()

    def insert_vectors(
        self,
        vectors: list[list[float]],
        ids: list[str],
        payloads: list[dict[str, Any]] | None = None,
    ) -> None:
        if len(vectors) != len(ids):
            raise ValueError(f"vectors ({len(vectors)}) and ids ({len(ids)}) length mismatch")
        if payloads is not None and len(payloads) != len(ids):
            raise ValueError(f"payloads ({len(payloads)}) and ids ({len(ids)}) length mismatch")
        if payloads is None:
            payloads = [{}] * len(ids)

        # Validate all vectors in batch have same dimension
        if vectors:
            dim = len(vectors[0])
            for i, v in enumerate(vectors):
                if len(v) != dim:
                    raise ValueError(
                        f"vector dimension mismatch: vector[0] has dim {dim}, vector[{i}] has dim {len(v)}"
                    )

        with self._lock:
            for vec, vid, payload in zip(vectors, ids, payloads, strict=True):
                self.conn.execute(
                    "INSERT OR REPLACE INTO vector_entries (id, vector, payload) VALUES (?, ?, ?)",
                    (vid, json.dumps(vec), json.dumps(payload)),
                )
            self.conn.commit()

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

        with self._lock:
            rows = self._fetch_rows(filters)

        scored: list[tuple[float, str, dict[str, Any]]] = []
        for vid, vec_json, payload_json in rows:
            vec = np.array(json.loads(vec_json), dtype=np.float32)
            if vec.shape != query.shape:
                raise ValueError(f"vector dimension mismatch: query={query.shape} stored={vec.shape} for id={vid}")
            vec_norm = float(np.linalg.norm(vec))
            if vec_norm == 0:
                continue
            score = float(np.dot(query, vec) / (query_norm * vec_norm))
            score = max(0.0, min(1.0, score))
            scored.append((score, vid, json.loads(payload_json) if payload_json else {}))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [VectorSearchResult(id=vid, score=sc, payload=pl) for sc, vid, pl in scored[:top_k]]

    def _fetch_rows(self, filters: dict[str, Any] | None) -> list[tuple[str, str, str]]:
        if not filters:
            return list(self.conn.execute("SELECT id, vector, payload FROM vector_entries"))
        rows = list(self.conn.execute("SELECT id, vector, payload FROM vector_entries"))
        result = []
        for vid, vec_json, payload_json in rows:
            payload = json.loads(payload_json) if payload_json else {}
            if all(payload.get(k) == v for k, v in filters.items()):
                result.append((vid, vec_json, payload_json))
        return result

    def delete_vector(self, vector_id: str) -> bool:
        with self._lock:
            cur = self.conn.execute("DELETE FROM vector_entries WHERE id = ?", (vector_id,))
            self.conn.commit()
            return cur.rowcount > 0

    def get_vector(self, vector_id: str) -> VectorEntry | None:
        with self._lock:
            row = self.conn.execute("SELECT vector, payload FROM vector_entries WHERE id = ?", (vector_id,)).fetchone()
        if row is None:
            return None
        vec_json, payload_json = row
        return VectorEntry(
            id=vector_id,
            vector=json.loads(vec_json),
            payload=json.loads(payload_json) if payload_json else {},
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

    def close(self) -> None:
        self.conn.close()

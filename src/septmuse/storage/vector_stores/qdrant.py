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
"""Qdrant 向量存储 (extras, lazy import)。

upsert/query_points/retrieve/scroll + Filter/FieldCondition 模式。
Qdrant Distance.COSINE 返回的 score 直接是 [0,1] 相似度, 无需转换。
"""

from __future__ import annotations

from typing import Any

from septmuse.core.logging import get_logger
from septmuse.storage.vector_stores.base import VectorEntry, VectorSearchResult, VectorStoreBase

logger = get_logger(__name__)


class QdrantVectorStore(VectorStoreBase):
    """Qdrant 向量存储 (远程服务, 需 Qdrant 实例)。

    用法:
        store = QdrantVectorStore(host="localhost", port=6333)
        store.insert_vectors([[1.0, 0.0]], ["m1"], [{"user_id": "alice"}])
        results = store.search_vectors([0.9, 0.1], top_k=5, filters={"user_id": "alice"})

    Note: qdrant_client 必须 lazy import (extras 未安装时不能破坏 ``import septmuse``)。
    Collection 在首次 insert/search 时按向量维度自动创建 (Distance.COSINE)。
    """

    def __init__(self, host: str, port: int, collection_name: str = "septmuse") -> None:
        from qdrant_client import QdrantClient

        self.collection_name = collection_name
        self.host = host
        self.port = port
        self.client = QdrantClient(host=host, port=port)
        self._collection_ensured = False
        logger.info(
            "qdrant_vector_store_ready",
            collection=collection_name,
            host=host,
            port=port,
        )

    def _ensure_collection(self, dim: int) -> None:
        if self._collection_ensured:
            return
        from qdrant_client.models import Distance, VectorParams

        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)
        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
        self._collection_ensured = True

    def _build_filter(self, filters: dict[str, Any] | None) -> Any:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        if not filters:
            return None
        return Filter(must=[FieldCondition(key=k, match=MatchValue(value=v)) for k, v in filters.items()])

    @staticmethod
    def _extract_vector(raw: Any) -> list[float]:
        if raw is None:
            return []
        if isinstance(raw, list):
            return [float(x) for x in raw]
        if isinstance(raw, dict):
            for key in ("", "dense", "vector"):
                value = raw.get(key)
                if value is not None:
                    return [float(x) for x in value]
        return []

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
        if not vectors:
            return
        self._ensure_collection(len(vectors[0]))
        from qdrant_client.models import PointStruct

        points = [
            PointStruct(id=vid, vector=vec, payload=payload)
            for vid, vec, payload in zip(ids, vectors, payloads, strict=True)
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)

    def search_vectors(
        self,
        query_vector: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        self._ensure_collection(len(query_vector))
        query_filter = self._build_filter(filters)
        hits = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )
        return [VectorSearchResult(id=str(p.id), score=float(p.score), payload=p.payload) for p in hits.points]

    def delete_vector(self, vector_id: str) -> bool:
        existing = self.client.retrieve(
            collection_name=self.collection_name,
            ids=[vector_id],
            with_payload=False,
            with_vectors=False,
        )
        if not existing:
            return False
        from qdrant_client.models import PointIdsList

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=PointIdsList(points=[vector_id]),
        )
        return True

    def get_vector(self, vector_id: str) -> VectorEntry | None:
        result = self.client.retrieve(
            collection_name=self.collection_name,
            ids=[vector_id],
            with_payload=True,
            with_vectors=True,
        )
        if not result:
            return None
        point = result[0]
        return VectorEntry(
            id=str(point.id),
            vector=self._extract_vector(point.vector),
            payload=point.payload,
        )

    def list_vectors(
        self,
        filters: dict[str, Any] | None = None,
        limit: int | None = None,
    ) -> list[VectorEntry]:
        query_filter = self._build_filter(filters)
        result = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=query_filter,
            limit=limit if limit is not None else 100,
            with_payload=True,
            with_vectors=True,
        )
        points, _ = result
        entries: list[VectorEntry] = []
        for point in points:
            entries.append(
                VectorEntry(
                    id=str(point.id),
                    vector=self._extract_vector(point.vector),
                    payload=point.payload,
                )
            )
        return entries

    def close(self) -> None:
        self.client.close()
        logger.info("qdrant_vector_store_closed", collection=self.collection_name)

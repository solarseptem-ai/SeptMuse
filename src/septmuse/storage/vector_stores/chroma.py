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
"""Chroma 向量存储 (extras, lazy import)。

collection.add/query/delete/get 模式。
score 语义对齐 VectorStoreBase: 相似度 [0,1], 越高越相似。
Chroma 返回 L2 距离 (越小越相似), 转换: score = max(0.0, 1.0 - distance)。
"""

from __future__ import annotations

import json
from typing import Any

from septmuse.core.logging import get_logger
from septmuse.storage.vector_stores.base import VectorEntry, VectorSearchResult, VectorStoreBase

logger = get_logger(__name__)


class ChromaVectorStore(VectorStoreBase):
    """Chroma 向量存储 (PersistentClient, 本地持久化)。

    用法:
        store = ChromaVectorStore(persist_path="./chroma")
        store.insert_vectors([[1.0, 0.0]], ["m1"], [{"user_id": "alice"}])
        results = store.search_vectors([0.9, 0.1], top_k=5, filters={"user_id": "alice"})

    Note: chromadb 必须 lazy import (extras 未安装时不能破坏 ``import septmuse``)。
    """

    def __init__(self, persist_path: str, collection_name: str = "septmuse") -> None:
        import chromadb
        from chromadb.config import Settings

        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(
            path=persist_path,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=None,
        )
        logger.info(
            "chroma_vector_store_ready",
            collection=collection_name,
            path=persist_path,
        )

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
        documents = [json.dumps(p, ensure_ascii=False) for p in payloads]
        self.collection.add(
            ids=ids,
            embeddings=vectors,
            metadatas=payloads,
            documents=documents,
        )

    def search_vectors(
        self,
        query_vector: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where=filters if filters else None,
        )
        ids = (results.get("ids") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        search_results: list[VectorSearchResult] = []
        for vid, dist, meta in zip(ids, distances, metadatas, strict=False):
            score = max(0.0, 1.0 - float(dist))
            search_results.append(VectorSearchResult(id=str(vid), score=score, payload=meta))
        return search_results

    def delete_vector(self, vector_id: str) -> bool:
        existing = self.collection.get(ids=[vector_id])
        if not existing.get("ids"):
            return False
        self.collection.delete(ids=[vector_id])
        return True

    def get_vector(self, vector_id: str) -> VectorEntry | None:
        result = self.collection.get(
            ids=[vector_id],
            include=["embeddings", "metadatas"],
        )
        ids = result.get("ids", [])
        if not ids:
            return None
        embeddings = result.get("embeddings") or []
        metadatas = result.get("metadatas") or []
        return VectorEntry(
            id=str(ids[0]),
            vector=list(embeddings[0]) if embeddings and embeddings[0] is not None else [],
            payload=metadatas[0] if metadatas else None,
        )

    def list_vectors(
        self,
        filters: dict[str, Any] | None = None,
        limit: int | None = None,
    ) -> list[VectorEntry]:
        result = self.collection.get(
            where=filters if filters else None,
            limit=limit,
            include=["embeddings", "metadatas"],
        )
        ids = result.get("ids", [])
        embeddings = result.get("embeddings") or []
        metadatas = result.get("metadatas") or []
        entries: list[VectorEntry] = []
        for i, vid in enumerate(ids):
            vec = list(embeddings[i]) if i < len(embeddings) and embeddings[i] is not None else []
            entries.append(
                VectorEntry(
                    id=str(vid),
                    vector=vec,
                    payload=metadatas[i] if i < len(metadatas) else None,
                )
            )
        return entries

    def close(self) -> None:
        logger.info("chroma_vector_store_closed", collection=self.collection_name)

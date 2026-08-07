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
            metadata={"hnsw:space": "cosine"},
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
        # Chroma 元数据只能是 str/int/float/bool, 过滤 None 值, 空 dict 补 _id 兜底
        chroma_metadatas = []
        for i, p in enumerate(payloads):
            clean = {k: v for k, v in p.items() if v is not None}
            if not clean:
                clean = {"_id": ids[i]}
            chroma_metadatas.append(clean)
        documents = [json.dumps(p, ensure_ascii=False) for p in payloads]
        self.collection.upsert(
            ids=ids,
            embeddings=vectors,
            metadatas=chroma_metadatas,
            documents=documents,
        )

    def search_vectors(
        self,
        query_vector: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        # Chroma where 只支持单 key 或 $and/$or, 多 key 需转 $and
        chroma_where = None
        if filters and len(filters) > 0:
            chroma_where = filters if len(filters) == 1 else {"$and": [{k: v} for k, v in filters.items()]}
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where=chroma_where,
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
        embeddings_raw = result.get("embeddings")
        embeddings = list(embeddings_raw) if embeddings_raw is not None else []
        metadatas_raw = result.get("metadatas")
        metadatas = list(metadatas_raw) if metadatas_raw is not None else []
        return VectorEntry(
            id=str(ids[0]),
            vector=list(embeddings[0]) if len(embeddings) > 0 and embeddings[0] is not None else [],
            payload=metadatas[0] if len(metadatas) > 0 else None,
        )

    def list_vectors(
        self,
        filters: dict[str, Any] | None = None,
        limit: int | None = None,
    ) -> list[VectorEntry]:
        # Chroma where 只支持单 key 或 $and/$or
        chroma_where = None
        if filters and len(filters) > 0:
            chroma_where = filters if len(filters) == 1 else {"$and": [{k: v} for k, v in filters.items()]}
        result = self.collection.get(
            where=chroma_where,
            limit=limit,
            include=["embeddings", "metadatas"],
        )
        ids = result.get("ids", [])
        embeddings_raw = result.get("embeddings")
        embeddings = list(embeddings_raw) if embeddings_raw is not None else []
        metadatas_raw = result.get("metadatas")
        metadatas = list(metadatas_raw) if metadatas_raw is not None else []
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

    def update_vector(self, vector_id: str, vector: list[float], payload: dict[str, Any] | None = None) -> bool:
        """Chroma collection.upsert 原地更新。"""
        chroma_meta = {k: v for k, v in (payload or {}).items() if v is not None} or {"_id": vector_id}
        doc = json.dumps(payload or {}, ensure_ascii=False)
        self.collection.upsert(
            ids=[vector_id],
            embeddings=[vector],
            metadatas=[chroma_meta],
            documents=[doc],
        )
        return True

    def delete_collection(self) -> None:
        """删除整个 collection。"""
        self.client.delete_collection(name=self.collection_name)

    def get_collection_info(self) -> dict[str, Any]:
        """collection 元信息。"""
        try:
            count = self.collection.count()
        except Exception:
            count = 0
        return {"name": self.collection_name, "count": count}

    def reset_collection(self) -> None:
        """重置 collection。"""
        self.delete_collection()
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=None,
            metadata={"hnsw:space": "cosine"},
        )

    def close(self) -> None:
        logger.info("chroma_vector_store_closed", collection=self.collection_name)

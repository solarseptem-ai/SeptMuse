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
"""向量存储后端抽象基类 (精简为 5 方法)。

只管向量 CRUD, 不管 memories 表/history 表 (那是 MemoryStore 的职责)。
score 语义: 相似度 [0,1], 越高越相似。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class VectorSearchResult:
    """向量检索结果 (id/score/payload 三字段)。"""

    id: str
    score: float
    payload: dict[str, Any] | None = None


@dataclass
class VectorEntry:
    """向量条目。"""

    id: str
    vector: list[float]
    payload: dict[str, Any] | None = None


class VectorStoreBase(ABC):
    """向量存储后端抽象 (精简为 5 方法)。

    实现方需保证 user_id 隔离 (通过 filters 参数)。
    """

    @abstractmethod
    def insert_vectors(
        self,
        vectors: list[list[float]],
        ids: list[str],
        payloads: list[dict[str, Any]] | None = None,
    ) -> None:
        """批量插入向量。id 与 vector 一一对应。"""
        ...

    @abstractmethod
    def search_vectors(
        self,
        query_vector: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        """向量检索, 返回 top_k 结果 (按 score 降序)。

        filters: payload 字段过滤, 如 {"user_id": "alice"}。
        """
        ...

    @abstractmethod
    def delete_vector(self, vector_id: str) -> bool:
        """删除向量。True=删除成功, False=不存在。"""
        ...

    @abstractmethod
    def get_vector(self, vector_id: str) -> VectorEntry | None:
        """取单条向量。不存在返回 None。"""
        ...

    @abstractmethod
    def list_vectors(
        self,
        filters: dict[str, Any] | None = None,
        limit: int | None = None,
    ) -> list[VectorEntry]:
        """列向量。filters 按 payload 字段过滤。"""
        ...

    @abstractmethod
    def update_vector(
        self,
        vector_id: str,
        vector: list[float],
        payload: dict[str, Any] | None = None,
    ) -> bool:
        """原地更新向量 + payload。True=更新成功，False=不存在。"""
        ...

    @abstractmethod
    def delete_collection(self) -> None:
        """删除整个 collection（所有向量 + payload）。"""
        ...

    def search_batch(
        self,
        queries: list[str],
        vectors_list: list[list[float]],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[list[VectorSearchResult]]:
        """批量搜索。默认循环 search_vectors，子类可 override 做原生批量。"""
        return [self.search_vectors(v, top_k, filters) for v in vectors_list]

    def keyword_search(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult] | None:
        """BM25 关键词搜索。默认返回 None（不支持），Qdrant 后端 override。"""
        return None

    def list_collections(self) -> list[str]:
        """列出所有 collection。默认返回当前 collection 名。"""
        return [getattr(self, "collection_name", "default")]

    def get_collection_info(self) -> dict[str, Any]:
        """collection 元信息。默认返回 name + count。"""
        return {"name": getattr(self, "collection_name", "default"), "count": 0}

    def reset_collection(self) -> None:
        """重置 collection（删除 + 重新创建）。"""
        self.delete_collection()

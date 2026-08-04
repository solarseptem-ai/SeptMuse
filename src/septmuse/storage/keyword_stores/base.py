#  Copyright 2026 The sonhhxg0529 Authors. All Rights Reserved.
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
"""关键词索引后端抽象基类 (借鉴 ReMe keyword_index/base_keyword_index.py, 改同步)。

score 语义: 归一化 BM25 分数 [0,1], 越高越相关 (与 VectorStoreBase 对齐)。
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class KeywordIndexBase(ABC):
    """关键词索引后端抽象。

    实现方需保证:
    - add_docs 幂等 (同 id 覆盖)
    - delete_docs 静默跳过不存在的 id
    - retrieve 返回 score 已归一化
    """

    @abstractmethod
    def add_docs(self, docs: dict[str, str]) -> None:
        """添加或替换文档 (id->text)。已存在的 id 覆盖。"""
        ...

    @abstractmethod
    def retrieve(self, query: str, limit: int = 5) -> dict[str, float]:
        """检索, 返回 {doc_id: score} (按 score 降序, top_k=limit)。"""
        ...

    @abstractmethod
    def delete_docs(self, doc_ids: list[str]) -> None:
        """删除文档。不存在的 id 静默跳过。"""
        ...

    @abstractmethod
    def clear(self) -> None:
        """清空索引。"""
        ...

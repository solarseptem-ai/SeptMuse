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
"""Reranker 抽象基类 — 对纯文本文档打分, 与记忆结构解耦。

设计:
- reranker 只管"用什么打分" (cosine / cross-encoder / LLM / Cohere)
- strategy 管"打什么内容" (整条记忆 / 对话拆分 / 拼接背景)
- 两者正交组合, reranker 操作 list[str], 返回 list[(doc_index, score)]
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseReranker(ABC):
    """重排器抽象基类 — 对纯文本文档打分。

    子类实现 rerank 方法, 返回 (doc_index, score) 降序排列。
    score 统一为相似度 [0,1] (越高越相关)。
    """

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        top_k: int | None = None,
    ) -> list[tuple[int, float]]:
        """对 documents 打分, 返回 [(doc_index, score)] 按相关性降序。

        Args:
            query: 查询文本
            documents: 待打分的纯文本列表
            top_k: 返回前 K 条 (None=全部)

        Returns:
            [(doc_index, score)] 降序, doc_index 指向 documents 列表位置
        """
        ...

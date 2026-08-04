#  Copyright 2026 The sonhhxg0529 Authors. All Rights Reserved.
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Reranker 策略抽象基类 — 解决"文档怎么准备和重建"。

策略解决文档准备和重建问题:
- prepare: 把 HybridResult 列表转为送入 reranker 的纯文本列表 (list[str])
- reconstruct: 把 reranker 返回的分数映射回 HybridResult

一条记忆可能拆为多个 document (如对话拆分), reconstruct 时聚合取最高分。
boost 权重在 reconstruct 中统一应用。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from septmuse.rerankers.boost import DEFAULT_BOOST_WEIGHTS, apply_boost

if TYPE_CHECKING:
    from septmuse.retrieval.hybrid import HybridResult


class BaseRerankerStrategy(ABC):
    """重排策略抽象 — HybridResult <-> list[str] 转换。"""

    @abstractmethod
    def prepare(self, results: list[HybridResult]) -> tuple[list[dict[str, Any]], list[str]]:
        """把 HybridResult 列表转为 (tracker, documents) 供 reranker 打分。

        Returns:
            tracker: 映射条目列表, 每条 {"result_index": int, ...}
            documents: 纯文本列表, 送入 reranker
        """
        ...

    @abstractmethod
    def reconstruct(
        self,
        scored: list[tuple[int, float]],
        tracker: list[dict[str, Any]],
        results: list[HybridResult],
        top_k: int,
        search_filter: dict[str, Any] | None = None,
    ) -> list[HybridResult]:
        """把 reranker 返回的 (doc_index, score) 映射回 HybridResult。

        Args:
            scored: reranker 输出 [(doc_index, score)] 降序
            tracker: prepare 返回的映射条目
            results: 原始 HybridResult 列表
            top_k: 返回数量
            search_filter: boost 权重 (匹配 metadata 字段加权)
        """
        ...

    def _apply_boost_to_result(
        self, result: HybridResult, score: float, search_filter: dict[str, Any] | None
    ) -> float:
        """通用 boost 应用 (子类可调用)。"""
        return apply_boost(result, score, search_filter, DEFAULT_BOOST_WEIGHTS)

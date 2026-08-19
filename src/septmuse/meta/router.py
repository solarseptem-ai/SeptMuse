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
"""元认知路由 — L0 路由决定查哪些命名空间。"""

from __future__ import annotations

from dataclasses import dataclass, field

from septmuse.core.logging import get_logger
from septmuse.embedders.base import Embedder

logger = get_logger(__name__)

# 默认路由阈值
DEFAULT_ROUTE_THRESHOLD = 0.1

# 默认命名空间描述 (对齐架构文档 §3.2 四类内容类型)
DEFAULT_NAMESPACES: dict[str, str] = {
    "working": "current context task state persona human memory blocks",
    "semantic": "facts knowledge preferences likes dislikes user profile triples",
    "episodic": "events timeline history reasoning experience session logs observations",
    "procedural": "rules skills how-to heuristics lessons learned playbook procedures",
}


@dataclass
class RouteResult:
    """路由结果。"""

    namespaces: list[str] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)
    fallback: bool = False  # True = 无匹配, 回退到全部


class MetaRouter:
    """元认知路由器。"""

    def __init__(
        self,
        embedder: Embedder,
        namespaces: dict[str, str] | None = None,
        threshold: float = DEFAULT_ROUTE_THRESHOLD,
    ) -> None:
        self.embedder = embedder
        self.threshold = threshold
        self._namespaces = namespaces if namespaces is not None else dict(DEFAULT_NAMESPACES)
        # 预计算命名空间描述嵌入
        self._ns_embeddings: dict[str, list[float]] = {}
        for name, desc in self._namespaces.items():
            self._ns_embeddings[name] = self.embedder.embed(desc)

    def route(self, query: str) -> RouteResult:
        """路由查询到匹配命名空间。"""
        import numpy as np

        q_emb = self.embedder.embed(query)
        q = np.array(q_emb, dtype=np.float32)
        qnorm = float(np.linalg.norm(q))
        if qnorm > 0:
            q = q / qnorm

        scores: dict[str, float] = {}
        for name, ns_emb in self._ns_embeddings.items():
            emb = np.array(ns_emb, dtype=np.float32)
            norm = float(np.linalg.norm(emb))
            if norm > 0:
                emb = emb / norm
            score = float(np.dot(q, emb))
            scores[name] = score

        # 过滤 + 排序
        matched = [(name, score) for name, score in scores.items() if score >= self.threshold]
        matched.sort(key=lambda x: x[1], reverse=True)

        if not matched:
            # fallback: 返回全部命名空间
            logger.info("meta_route_fallback", query=query[:50], threshold=self.threshold)
            return RouteResult(namespaces=list(self._namespaces.keys()), scores=scores, fallback=True)

        ns_list = [name for name, _ in matched]
        logger.info("meta_route_done", query=query[:50], routed=ns_list, top_score=matched[0][1])
        return RouteResult(namespaces=ns_list, scores=scores, fallback=False)

    def register_namespace(self, name: str, description: str) -> None:
        """注册新命名空间 (动态扩展)。"""
        self._namespaces[name] = description
        self._ns_embeddings[name] = self.embedder.embed(description)

    def list_namespaces(self) -> list[str]:
        """列出所有已注册命名空间。"""
        return list(self._namespaces.keys())

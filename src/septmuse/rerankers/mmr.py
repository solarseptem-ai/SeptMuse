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
"""最大边际相关性 reranker — 去冗余 + 多样性。

贪心迭代选择: mmr = lambda * sim(query, doc) - (1-lambda) * max(sim(doc, selected))
去冗余: 相似度 >0.9 的结果只保留排名靠前的一个。
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from septmuse.rerankers.base import BaseReranker

if TYPE_CHECKING:
    from septmuse.embedders.base import Embedder


class MMRReranker(BaseReranker):
    """MMR 重排器 — 需要 embedder 计算向量相似度。"""

    def __init__(self, embedder: Embedder, lambda_param: float = 0.7, **kwargs) -> None:
        self.embedder = embedder
        self.lambda_param = lambda_param

    def _cosine(self, a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b, strict=True))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        top_k: int | None = None,
    ) -> list[tuple[int, float]]:
        if not documents:
            return []

        tk = top_k or len(documents)
        query_emb = self.embedder.embed(query)
        doc_embs = [self.embedder.embed(doc) for doc in documents]
        query_sims = [self._cosine(query_emb, de) for de in doc_embs]

        selected: list[int] = []
        remaining = list(range(len(documents)))

        while remaining and len(selected) < tk:
            best_idx = -1
            best_score = -float("inf")
            for i in remaining:
                max_sim = max(self._cosine(doc_embs[i], doc_embs[j]) for j in selected) if selected else 0.0
                mmr = self.lambda_param * query_sims[i] - (1 - self.lambda_param) * max_sim
                if mmr > best_score:
                    best_score = mmr
                    best_idx = i
            if best_idx < 0:
                break
            selected.append(best_idx)
            remaining.remove(best_idx)
            # 去冗余
            to_remove = [j for j in remaining if self._cosine(doc_embs[best_idx], doc_embs[j]) > 0.9]
            for j in to_remove:
                remaining.remove(j)

        return [(selected[rank], query_sims[selected[rank]]) for rank in range(len(selected))]

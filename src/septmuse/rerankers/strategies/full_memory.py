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
"""默认策略 — 直接用 r.memory 作为 document (1:1 映射)。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from septmuse.rerankers.strategies.base import BaseRerankerStrategy

if TYPE_CHECKING:
    from septmuse.retrieval.hybrid import HybridResult


class FullMemoryStrategy(BaseRerankerStrategy):
    """默认策略: 一条记忆 = 一个 document, 1:1 映射。"""

    def prepare(self, results: list[HybridResult]) -> tuple[list[dict[str, Any]], list[str]]:
        tracker = [{"result_index": i} for i in range(len(results))]
        documents = [r.memory for r in results]
        return tracker, documents

    def reconstruct(
        self,
        scored: list[tuple[int, float]],
        tracker: list[dict[str, Any]],
        results: list[HybridResult],
        top_k: int,
        search_filter: dict[str, Any] | None = None,
    ) -> list[HybridResult]:
        from septmuse.retrieval.hybrid import HybridResult

        out: list[HybridResult] = []
        for doc_idx, score in scored[:top_k]:
            entry = tracker[doc_idx]
            ri = entry["result_index"]
            r = results[ri]
            final_score = self._apply_boost_to_result(r, score, search_filter)
            out.append(
                HybridResult(
                    id=r.id, memory=r.memory, score=final_score,
                    vector_score=r.vector_score, bm25_score=r.bm25_score,
                    entity_boost=r.entity_boost, metadata=r.metadata, created_at=r.created_at,
                )
            )
        return out

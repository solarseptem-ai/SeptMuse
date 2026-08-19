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
"""对话拆分策略 — 把对话记忆拆为 user+assistant pair 分别打分，取最高分聚合。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from septmuse.rerankers.strategies.base import BaseRerankerStrategy

if TYPE_CHECKING:
    from septmuse.retrieval.hybrid import HybridResult


class SingleTurnStrategy(BaseRerankerStrategy):
    """对话拆分策略: user+assistant pair 粒度打分。"""

    def prepare(self, results: list[HybridResult]) -> tuple[list[dict[str, Any]], list[str]]:
        tracker: list[dict[str, Any]] = []
        documents: list[str] = []
        for i, r in enumerate(results):
            meta = r.metadata or {}
            sources = meta.get("sources", [])
            if not sources:
                tracker.append({"result_index": i, "pair_index": -1})
                documents.append(r.memory)
                continue
            for j in range(0, len(sources), 2):
                user_msg = sources[j].get("content", "") if j < len(sources) else ""
                asst_msg = sources[j + 1].get("content", "") if j + 1 < len(sources) else ""
                if user_msg or asst_msg:
                    doc = f"user: {user_msg}\nassistant: {asst_msg}"
                    tracker.append({"result_index": i, "pair_index": j // 2})
                    documents.append(doc)
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

        best: dict[int, float] = {}
        for doc_idx, score in scored:
            entry = tracker[doc_idx]
            ri = entry["result_index"]
            if ri not in best or score > best[ri]:
                best[ri] = score

        ranked = sorted(best.items(), key=lambda x: x[1], reverse=True)[:top_k]
        out: list[HybridResult] = []
        for ri, score in ranked:
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

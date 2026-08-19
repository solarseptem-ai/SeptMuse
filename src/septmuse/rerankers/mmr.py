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
"""最大边际相关性 reranker — 去冗余 + 多样性。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from septmuse.rerankers.base import BaseReranker

if TYPE_CHECKING:
    from septmuse.embedders.base import Embedder


class MMRReranker(BaseReranker):
    """MMR 重排器 — 需要 embedder 计算向量相似度。"""

    def __init__(self, embedder: Embedder, lambda_param: float = 0.7, **kwargs) -> None:
        self.embedder = embedder
        self.lambda_param = lambda_param

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        """余弦相似度 (numpy 向量化)。"""
        va = np.array(a, dtype=np.float32)
        vb = np.array(b, dtype=np.float32)
        na = float(np.linalg.norm(va))
        nb = float(np.linalg.norm(vb))
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(va, vb) / (na * nb))

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

        query_emb = np.array(self.embedder.embed(query), dtype=np.float32)
        doc_embs = np.array(self.embedder.embed_batch(documents), dtype=np.float32)

        doc_norms = np.linalg.norm(doc_embs, axis=1, keepdims=True)
        doc_norms = np.where(doc_norms == 0, 1.0, doc_norms)
        doc_unit = doc_embs / doc_norms

        q_norm = np.linalg.norm(query_emb)
        q_unit = query_emb / max(float(q_norm), 1e-10)

        query_sims = doc_unit @ q_unit

        sim_matrix = doc_unit @ doc_unit.T
        np.fill_diagonal(sim_matrix, 0.0)

        selected: list[int] = []
        remaining: set[int] = set(range(len(documents)))

        while remaining and len(selected) < tk:
            remaining_list = list(remaining)
            if selected:
                max_sims = np.max(sim_matrix[np.ix_(remaining_list, selected)], axis=1)
            else:
                max_sims = np.zeros(len(remaining_list), dtype=np.float32)

            mmr_scores = self.lambda_param * query_sims[remaining_list] - (1 - self.lambda_param) * max_sims
            best_local = int(np.argmax(mmr_scores))
            best_idx = remaining_list[best_local]

            selected.append(best_idx)
            remaining.discard(best_idx)

            to_remove = {j for j in remaining if float(sim_matrix[best_idx, j]) > 0.9}
            remaining -= to_remove

        return [(selected[rank], float(query_sims[selected[rank]])) for rank in range(len(selected))]

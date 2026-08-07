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
"""BM25 分数归一化与加性融合 (对齐 mem0 scoring.py).

sigmoid 归一化替代简单 max_score 除法:
- 长查询原始 BM25 分偏高 → 提高 midpoint
- 短查询原始 BM25 分偏低 → 降低 midpoint
- sigmoid 输出 [0,1], 与向量余弦分数可比
"""

from __future__ import annotations

import math
from typing import Any


def get_bm25_params(query: str, *, num_terms: int | None = None) -> tuple[float, float]:
    """按查询词数返回 sigmoid 参数 (midpoint, steepness).

    对齐 mem0 get_bm25_params: 词数越多, 原始 BM25 分越高, midpoint 随之提高.

    Returns:
        (midpoint, steepness) — sigmoid 参数
    """
    if num_terms is None:
        # 简单分词: 按空格切分
        num_terms = len(query.split()) if query else 1

    if num_terms <= 3:
        return 5.0, 0.7
    elif num_terms <= 6:
        return 7.0, 0.6
    elif num_terms <= 9:
        return 9.0, 0.5
    elif num_terms <= 15:
        return 10.0, 0.5
    else:
        return 12.0, 0.5


def normalize_bm25(raw_score: float, midpoint: float, steepness: float) -> float:
    """sigmoid 归一化 BM25 原始分到 [0, 1].

    f(x) = 1 / (1 + exp(-steepness * (x - midpoint)))

    raw_score=0 → ~0 (低分压低)
    raw_score=midpoint → 0.5 (中分居中)
    raw_score >> midpoint → ~1 (高分压顶)
    """
    return 1.0 / (1.0 + math.exp(-steepness * (raw_score - midpoint)))


ENTITY_BOOST_WEIGHT = 0.5


def score_and_rank(
    semantic_results: list[dict[str, Any]],
    bm25_scores: dict[str, float],
    entity_boosts: dict[str, float],
    threshold: float,
    top_k: int,
    explain: bool = False,
) -> list[dict[str, Any]]:
    """加性融合: semantic + bm25 + entity_boost, 归一化到 [0,1].

    max_possible 自适应:
        仅语义: 1.0
        语义+BM25: 2.0
        语义+BM25+实体: 2.5
        语义+实体: 1.5
    """
    has_bm25 = bool(bm25_scores)
    has_entity = bool(entity_boosts)

    max_possible = 1.0
    if has_bm25:
        max_possible += 1.0
    if has_entity:
        max_possible += ENTITY_BOOST_WEIGHT

    scored: list[dict[str, Any]] = []

    for result in semantic_results:
        mem_id = result.get("id")
        if mem_id is None:
            continue

        semantic_score = result.get("score") or 0.0
        if semantic_score < threshold:
            continue

        mem_id_str = str(mem_id)
        bm25_score = bm25_scores.get(mem_id_str, 0.0)
        entity_boost = entity_boosts.get(mem_id_str, 0.0)

        raw_combined = semantic_score + bm25_score + entity_boost
        combined = min(raw_combined / max_possible, 1.0)

        scored_result: dict[str, Any] = {
            "id": mem_id_str,
            "score": combined,
            "payload": result.get("payload"),
        }
        if explain:
            scored_result["score_details"] = {
                "semantic_score": semantic_score,
                "bm25_score": bm25_score,
                "entity_boost": entity_boost,
                "raw_score": raw_combined,
                "max_possible_score": max_possible,
                "final_score": combined,
                "threshold": threshold,
            }
        scored.append(scored_result)

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]

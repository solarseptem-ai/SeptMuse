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
"""boost 权重通用函数 — 匹配 search_filter 的结果 score 加权。

匹配逻辑: 检查 result.metadata 中对应字段的值是否与 search_filter 中的值匹配。
匹配则 score *= (1 + weight), clamp [0, 1]。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from septmuse.retrieval.hybrid import HybridResult

# boost 权重默认值: 匹配 user_id 加 0.5, tags 加 0.2, session_id 加 0.3
DEFAULT_BOOST_WEIGHTS: dict[str, float] = {"user_id": 0.5, "tags": 0.2, "session_id": 0.3}


def apply_boost(
    result: HybridResult,
    base_score: float,
    search_filter: dict[str, Any] | None,
    boost_weights: dict[str, float] | None = None,
) -> float:
    """对匹配 search_filter 的结果加权 boost。

    Args:
        result: 检索结果项 (含 metadata)
        base_score: 基础分数
        search_filter: boost 过滤字典 (如 {"user_id": "alice", "tags": ["python"]})
        boost_weights: 字段权重 (None=用 DEFAULT_BOOST_WEIGHTS)

    Returns:
        加权后的分数 [0, 1]
    """
    if not search_filter:
        return base_score

    weights = boost_weights or DEFAULT_BOOST_WEIGHTS
    score = float(base_score)
    meta = result.metadata or {}

    for key, wanted in search_filter.items():
        field_value = meta.get(key)
        if field_value is None:
            continue
        # 支持 list/tuple 字段 (如 tags)
        if isinstance(field_value, (list, tuple, set)):
            wanted_set = wanted if isinstance(wanted, (list, tuple, set)) else [wanted]
            if any(w in field_value for w in wanted_set):
                w = weights.get(key, 0.0)
                if w != 0.0:
                    score = min(1.0, score * (1.0 + w))
        elif field_value == wanted:
            w = weights.get(key, 0.0)
            if w != 0.0:
                score = min(1.0, score * (1.0 + w))

    return score

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
"""用户画像数据模型 — 从 SemanticFact 聚合的结构化画像.

时态适配: SemanticFact 无 valid_at/invalid_at 双时态列 (只有 is_deleted + touch() 更新 updated_at).
画像的"当前有效"= is_deleted=False, "最新"= updated_at 最新.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class UserProfileValue:
    """画像单值 (一个 predicate 的当前/历史值)."""

    value: str
    confidence: float = 1.0
    updated_at: str | None = None
    is_current: bool = True
    source_fact_ids: list[str] = field(default_factory=list)


@dataclass
class UserProfile:
    """用户结构化画像 (从 SemanticFact 聚合).

    attributes/preferences/relationships: dict[predicate -> UserProfileValue] (同 predicate 只留一个, current 优先).
    plans: list[UserProfileValue] (可多值).
    raw_facts: 未分类 predicate 的兜底列表.
    """

    user_id: str
    attributes: dict[str, UserProfileValue] = field(default_factory=dict)
    preferences: dict[str, UserProfileValue] = field(default_factory=dict)
    plans: list[UserProfileValue] = field(default_factory=list)
    relationships: dict[str, UserProfileValue] = field(default_factory=dict)
    raw_facts: list[dict] = field(default_factory=list)
    temporal_summary: dict = field(default_factory=dict)


_ATTR_PREDICATES = {"name", "age", "occupation", "location", "birthday", "email", "phone"}
_PREF_PREDICATES = {"likes", "dislikes", "prefers", "hates", "favorite"}
_PLAN_PREDICATES = {"planning", "intends", "will", "goal"}
_REL_PREDICATES = {"has", "knows", "related_to", "friend", "family"}


def _classify(predicate: str) -> str:
    """predicate -> 分类 (attributes/preferences/plans/relationships/raw)."""
    p = predicate.lower().strip()
    if p in _ATTR_PREDICATES:
        return "attributes"
    if p in _PREF_PREDICATES:
        return "preferences"
    if p in _PLAN_PREDICATES:
        return "plans"
    if p in _REL_PREDICATES:
        return "relationships"
    return "raw"

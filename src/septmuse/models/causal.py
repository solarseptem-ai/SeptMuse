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
"""因果链记忆数据模型。"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum

from sqlmodel import Field, SQLModel


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return f"causal-{uuid.uuid4()}"


class CausalRelation(str, Enum):
    """因果关系类型。"""

    ENABLES = "enables"  # X 使 Y 成为可能
    PREVENTS = "prevents"  # X 阻止 Y 发生
    CAUSES = "causes"  # X 直接导致 Y
    INHIBITS = "inhibits"  # X 抑制 Y (部分阻止)


class CausalEdge(SQLModel, table=True):
    """因果边。"""

    __tablename__ = "septmuse_causal"  # type: ignore[assignment]

    id: str = Field(default_factory=_new_id, primary_key=True)
    cause_event_id: str = Field(index=True, description="原因事件 ID (指向 EpisodicEvent)")
    effect_event_id: str = Field(index=True, description="结果事件 ID (指向 EpisodicEvent)")
    relation: str = Field(
        default=CausalRelation.CAUSES.value,
        index=True,
        description="因果关系: enables|prevents|causes|inhibits",
    )
    confidence: float = Field(default=0.5, description="因果置信度 [0,1]")
    counterfactual_valid: bool = Field(default=False, description="是否已验证反事实")
    user_id: str = Field(index=True, description="用户 ID (多租户隔离)")

    created_at: datetime = Field(default_factory=_utcnow, description="创建时间 UTC")
    is_deleted: bool = Field(default=False, description="软删除")

    def is_positive(self) -> bool:
        """是否正向因果 (enables/causes → 促进)。"""
        return self.relation in (CausalRelation.ENABLES.value, CausalRelation.CAUSES.value)

    def is_negative(self) -> bool:
        """是否负向因果 (prevents/inhibits → 阻止)。"""
        return self.relation in (CausalRelation.PREVENTS.value, CausalRelation.INHIBITS.value)

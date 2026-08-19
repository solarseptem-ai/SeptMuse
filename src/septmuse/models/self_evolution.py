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
"""Agent 自进化数据模型。"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from sqlalchemy import Column, Integer, Text
from sqlalchemy.types import JSON
from sqlmodel import Field, SQLModel

from septmuse.core.logging import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return f"se-{uuid.uuid4().hex[:12]}"


class AgentSelfModel(SQLModel, table=True):
    """Agent 跨会话持久化能力画像。"""

    __tablename__ = "septmuse_agent_self"  # type: ignore[assignment]

    id: str = Field(default_factory=_new_id, primary_key=True)
    agent_id: str = Field(index=True, description="Agent 标识")
    user_id: str = Field(index=True, description="用户 ID")

    strengths: list[str] = Field(default=[], sa_column=Column(JSON), description="擅长领域")
    weaknesses: list[str] = Field(default=[], sa_column=Column(JSON), description="薄弱领域")
    skill_distribution: dict = Field(default={}, sa_column=Column(JSON), description="技能分布")
    top_rules: list[dict] = Field(default=[], sa_column=Column(JSON), description="最佳规则")
    evolution_summary: str = Field(default="", sa_column=Column(Text), description="自然语言自我描述")

    recall_success_rate: float = Field(default=0.0, description="检索成功率 0-1")
    total_recalls: int = Field(default=0, sa_column=Column(Integer), description="总检索次数")
    successful_recalls: int = Field(default=0, sa_column=Column(Integer), description="有效检索次数")

    last_improved_at: datetime = Field(default_factory=_utcnow, description="上次 improve 时间")
    version: int = Field(default=1, sa_column=Column(Integer), description="版本号")

    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)

    def touch(self) -> None:
        self.updated_at = _utcnow()

    def record_recall(self, success: bool) -> None:
        """记录一次检索结果。"""
        self.total_recalls += 1
        if success:
            self.successful_recalls += 1
        if self.total_recalls > 0:
            self.recall_success_rate = self.successful_recalls / self.total_recalls

    def build_evolution_summary(self) -> str:
        """构建自然语言自我描述。"""
        parts = []
        if self.strengths:
            parts.append(f"I am good at: {', '.join(self.strengths[:5])}.")
        if self.weaknesses:
            parts.append(f"I need to improve: {', '.join(self.weaknesses[:5])}.")
        if self.top_rules:
            best = self.top_rules[0]
            rule_text = best.get('rule', '')
            conf = best.get('confidence', 0)
            parts.append(
                f"My most reliable rule is: '{rule_text}' (confidence={conf:.2f})."
            )
        if self.total_recalls > 0:
            rate = self.recall_success_rate
            parts.append(
                f"Recall success rate: {rate:.1%} "
                f"({self.successful_recalls}/{self.total_recalls})."
            )
        if not parts:
            parts.append("I am still learning about myself.")
        self.evolution_summary = " ".join(parts)
        return self.evolution_summary


class RuleSnapshot(SQLModel, table=True):
    """规则版本快照。"""

    __tablename__ = "septmuse_rule_snapshots"  # type: ignore[assignment]

    id: str = Field(default_factory=_new_id, primary_key=True)
    rule_id: str = Field(index=True, description="规则 ID")
    parent_rule_id: str | None = Field(default=None, index=True, description="上一版本 ID")
    rule: str = Field(description="规则内容")
    version: int = Field(default=1, sa_column=Column(Integer), description="版本号")
    change_description: str = Field(default="", description="变更说明")
    refinement_reason: str = Field(default="", description="精炼原因")
    source_event_ids: list[str] = Field(default=[], sa_column=Column(JSON), description="触发事件")
    confidence_at_snapshot: float = Field(default=0.5, description="快照时置信度")
    helpful_count: int = Field(default=0, description="快照时 helpful")
    harmful_count: int = Field(default=0, description="快照时 harmful")

    user_id: str = Field(index=True, description="用户 ID")
    created_at: datetime = Field(default_factory=_utcnow)

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
"""程序记忆数据模型 + 操作 — Playbook 规则退化 (架构文档 §3.2.3)。

数据模型:
- helpful_count / harmful_count: 规则带来正/负面结果次数
- source_tracing: 溯源到具体 episodic session
- deprecated: 规则退化标记 (被证伪则废弃)
- confidence = helpful / (helpful + harmful)

操作:
- record_outcome: helpful/harmful 计数
- 退化: harmful > helpful 且 >=3 次则 deprecated
- should_inject: 废弃规则不注入 context
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from sqlalchemy import Column
from sqlalchemy.types import JSON
from sqlmodel import Field, SQLModel

from septmuse.core.logging import get_logger

if TYPE_CHECKING:
    from septmuse.storage.relational_stores.typed_store import TypedMemoryStore


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return f"rule-{uuid.uuid4()}"


class ProceduralRule(SQLModel, table=True):
    """程序规则 — 带退化的 how-to/skill (架构文档 §3.2.3)。

    - helpful/harmful 计数: 规则被验证正/负面
    - source_tracing: 溯源到 episodic session
    - deprecated: harmful > helpful 时标记废弃, 不再注入 context

    SeptMuse 增量: confidence 自动计算 = helpful/(helpful+harmful)。
    """

    __tablename__ = "septmuse_procedural"  # type: ignore[assignment]

    id: str = Field(default_factory=_new_id, primary_key=True)
    rule: str = Field(description="规则内容 (how-to/skill/heuristic)")
    namespace: str = Field(default="default", index=True, description="命名空间")
    user_id: str = Field(index=True, description="用户 ID")

    # 核心字段
    helpful_count: int = Field(default=0, description="带来正面结果次数")
    harmful_count: int = Field(default=0, description="带来负面结果次数")
    source_tracing: str | None = Field(
        default=None,
        description="溯源到 episodic session",
    )
    deprecated: bool = Field(default=False, description="规则退化标记")

    # SeptMuse 增量
    tags: list[str] = Field(default=[], sa_column=Column(JSON), description="分类标签")

    created_at: datetime = Field(default_factory=_utcnow, description="创建时间 UTC")
    updated_at: datetime = Field(default_factory=_utcnow, description="更新时间 UTC")
    is_deleted: bool = Field(default=False, description="软删除")

    def touch(self) -> None:
        self.updated_at = _utcnow()

    @property
    def confidence(self) -> float:
        """置信度 = helpful / (helpful + harmful)。

        无记录时默认 0.5 (中性)。
        """
        total = self.helpful_count + self.harmful_count
        if total == 0:
            return 0.5
        return self.helpful_count / total

    def record_outcome(self, helpful: bool) -> None:
        """记录一次应用结果。

        Args:
            helpful: True=正面(helpful_count+1), False=负面(harmful_count+1)
        """
        if helpful:
            self.helpful_count += 1
        else:
            self.harmful_count += 1
        # 退化: harmful 超过 helpful 则标记废弃
        if self.harmful_count > self.helpful_count and self.harmful_count >= 3:
            self.deprecated = True
        self.touch()

    def should_inject(self) -> bool:
        """是否应注入 context (废弃规则不注入)。"""
        return not self.deprecated and not self.is_deleted


logger = get_logger(__name__)


class ProceduralMemory:
    """程序记忆操作 (架构文档 §3.2.3)。"""

    def __init__(self, store: TypedMemoryStore) -> None:
        self.store = store

    def add_rule(
        self,
        rule: str,
        *,
        user_id: str,
        namespace: str = "default",
        source_tracing: str | None = None,
        tags: list[str] | None = None,
    ) -> ProceduralRule:
        """添加规则。"""
        return self.store.add_rule(
            rule,
            user_id=user_id,
            namespace=namespace,
            source_tracing=source_tracing,
            tags=tags,
        )

    def record_outcome(self, rule_id: str, helpful: bool) -> ProceduralRule | None:
        """记录规则应用结果 (helpful/harmful 计数 + 自动退化)。"""
        r = self.store.record_rule_outcome(rule_id, helpful)
        if r:
            logger.info(
                "rule_outcome_recorded",
                rule_id=rule_id,
                helpful=helpful,
                helpful_count=r.helpful_count,
                harmful_count=r.harmful_count,
                deprecated=r.deprecated,
            )
        return r

    def get_active_rules(self, *, user_id: str, namespace: str = "default") -> list[ProceduralRule]:
        """获取应注入的规则 (废弃规则不返回)。"""
        return self.store.get_active_rules(user_id=user_id, namespace=namespace)

    def get_all_rules(self, *, user_id: str, include_deprecated: bool = False) -> list[ProceduralRule]:
        return self.store.get_all_rules(user_id=user_id, include_deprecated=include_deprecated)

    def rules_to_prompt(self, *, user_id: str, namespace: str = "default") -> str:
        """编译规则为 prompt 注入文本 (仅 active 规则)。

        格式:
            <procedural_rules>
            - 规则1 (confidence=0.75)
            - 规则2 (confidence=1.0)
            </procedural_rules>
        """
        rules = self.get_active_rules(user_id=user_id, namespace=namespace)
        if not rules:
            return ""
        parts = ["<procedural_rules>"]
        for r in rules:
            parts.append(f"- {r.rule} (confidence={r.confidence:.2f})")
        parts.append("</procedural_rules>")
        return "\n".join(parts)

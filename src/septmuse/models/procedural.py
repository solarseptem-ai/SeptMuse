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
"""程序记忆数据模型 + 操作。"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import Column
from sqlalchemy.types import JSON
from sqlmodel import Field, SQLModel

from septmuse.core.logging import get_logger


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return f"rule-{uuid.uuid4()}"


class ProceduralRule(SQLModel, table=True):
    """程序规则。"""

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
        """置信度 = helpful / (helpful + harmful)。"""
        total = self.helpful_count + self.harmful_count
        if total == 0:
            return 0.5
        return self.helpful_count / total

    def record_outcome(self, helpful: bool) -> None:
        """记录一次应用结果。"""
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

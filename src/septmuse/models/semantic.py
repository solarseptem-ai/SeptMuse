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
"""语义记忆数据模型 — 三元组事实 + 身份子类。

借鉴 LangMem Triple(subject/predicate/object/context) + namespace 多租户。
SeptMuse 增量 (架构文档 §3.2.2): confidence + provenance (创新: 区分事实/推断)。

详见 docs/specs/agent-memory-architecture.md §3.2.2。
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import Column
from sqlalchemy.types import JSON
from sqlmodel import Field, SQLModel


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return f"fact-{uuid.uuid4()}"


class SemanticFact(SQLModel, table=True):
    """语义事实 — 三元组 + 置信度 + 溯源 (架构文档 §3.2.2)。

    对齐 LangMem Triple: subject/predicate/object/context。
    对齐 LangMem namespace: (org_id, user_id) 多租户。
    SeptMuse 创新: confidence + provenance (区分用户陈述/推断/工具结果/观察)。
    """

    __tablename__ = "septmuse_facts"  # type: ignore[assignment]

    id: str = Field(default_factory=_new_id, primary_key=True)
    subject: str = Field(index=True, description="三元组主语")
    predicate: str = Field(index=True, description="三元组谓语")
    object: str = Field(index=True, description="三元组宾语")
    context: str | None = Field(default=None, description="上下文限定")

    # 多租户 (借鉴 LangMem namespace 模板)
    org_id: str = Field(default="default", index=True, description="组织 ID")
    user_id: str = Field(index=True, description="用户 ID (跨 agent 共享键)")

    # SeptMuse 创新: 置信度 + 溯源 (架构文档 §3.2.2)
    confidence: float = Field(default=1.0, description="置信度 [0,1], 区分事实/推断")
    provenance: str = Field(
        default="user",
        description="来源: user(用户陈述) | inferred(推断) | tool(工具结果) | observed(观察)",
    )

    # 标签 (identity 子类用 tags=["identity"], 普通事实无标签)
    tags: list[str] = Field(default=[], sa_column=Column(JSON))

    # 向量共存 (平面B: 同一事实可向量+图+文件)
    embedding: bytes | None = Field(default=None, description="嵌入向量 (JSON bytes)")

    created_at: datetime = Field(default_factory=_utcnow, description="创建时间 UTC")
    updated_at: datetime = Field(default_factory=_utcnow, description="更新时间 UTC")
    is_deleted: bool = Field(default=False, description="软删除标记")

    def touch(self) -> None:
        self.updated_at = _utcnow()

    def as_triple(self) -> dict[str, Any]:
        """返回三元组 dict (对齐 LangMem Triple schema)。"""
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "context": self.context,
        }


def is_identity(fact: SemanticFact) -> bool:
    """判断是否身份记忆子类 (架构文档 §3.2.2: 身份归语义子类, 打 identity 标签)。"""
    return "identity" in (fact.tags or [])

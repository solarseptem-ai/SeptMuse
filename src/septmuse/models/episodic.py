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
"""情节记忆数据模型 + 操作。

数据模型 — 时序事件 + 推理经验 + raw log:
三子类对齐 (架构文档 §3.2.1):
- 时序事件: Zep/Graphiti Episode (content, type, reference_time)
- 推理经验: LangMem Episode (observation, thoughts, action, result)
- raw log: Cass Episodic (原始 session transcript, 高保真防摘要坍缩)

操作 — 三子类统一接口:
- add_temporal_event: Zep Episode (事实 + reference_time)
- add_reasoning_episode: LangMem Episode (observation/thoughts/action/result)
- add_raw_log: Cass Episodic (原始 session log, 高保真)
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any

from sqlmodel import Field, SQLModel

from septmuse.core.logging import get_logger

if TYPE_CHECKING:
    from septmuse.storage.relational_stores.typed_store import TypedMemoryStore


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return f"epi-{uuid.uuid4()}"


class EpisodeType(str, Enum):
    """情节子类 (架构文档 §3.2.1 三子类)。"""

    FACT = "fact"  # 时序事件 (Zep Episode: 事实+时间锚)
    REASONING = "reasoning"  # 推理经验 (LangMem Episode: obs/act/result)
    RAW_LOG = "raw_log"  # 原始日志 (Cass Episodic: 高保真)


class EpisodicEvent(SQLModel, table=True):
    """情节事件 — 统一表, event_type 区分子类 (架构文档 §3.2.1)。

    对齐 Zep Episode: content + reference_time (时间锚点, 时序检索依据)。
    对齐 LangMem Episode: observation/thoughts/action/result (推理经验, 阶段2 reasoning 用)。
    对齐 Cass Episodic: raw transcript (高保真, raw_log 用)。
    """

    __tablename__ = "septmuse_episodic"  # type: ignore[assignment]

    id: str = Field(default_factory=_new_id, primary_key=True)
    event_type: str = Field(
        index=True,
        description="子类型: fact | reasoning | raw_log (EpisodeType)",
    )

    # 通用内容 (所有子类)
    content: str = Field(description="事件内容")
    reference_time: datetime = Field(
        default_factory=_utcnow,
        index=True,
        description="时间锚点 (Zep reference_time, 时序检索依据)",
    )

    # 多租户
    user_id: str = Field(index=True, description="用户 ID")
    agent_id: str | None = Field(default=None, description="agent ID (可选)")
    session_id: str | None = Field(default=None, description="会话 ID (raw_log 关联)")

    # 推理经验专用 (event_type=reasoning, 对齐 LangMem Episode)
    observation: str | None = Field(default=None, description="推理: 情境观察")
    thoughts: str | None = Field(default=None, description="推理: 思考过程")
    action: str | None = Field(default=None, description="推理: 采取行动")
    result: str | None = Field(default=None, description="推理: 结果与原因")

    created_at: datetime = Field(default_factory=_utcnow, description="创建时间 UTC")
    is_deleted: bool = Field(default=False, description="软删除")

    def as_langmem_episode(self) -> dict[str, str | None]:
        """返回 LangMem Episode 格式 (obs/thoughts/action/result)。"""
        return {
            "observation": self.observation,
            "thoughts": self.thoughts,
            "action": self.action,
            "result": self.result,
        }


logger = get_logger(__name__)


class EpisodicMemory:
    """情节记忆操作 (架构文档 §3.2.1, 三子类统一接口)。"""

    def __init__(self, store: TypedMemoryStore) -> None:
        self.store = store

    def add_temporal_event(
        self,
        content: str,
        *,
        user_id: str,
        reference_time: datetime | None = None,
        agent_id: str | None = None,
    ) -> EpisodicEvent:
        """添加时序事件 (对齐 Zep Episode: content + reference_time)。"""
        return self.store.add_episode(
            content,
            user_id=user_id,
            event_type=EpisodeType.FACT,
            reference_time=reference_time,
            agent_id=agent_id,
        )

    def add_reasoning_episode(
        self,
        observation: str,
        thoughts: str,
        action: str,
        result: str,
        *,
        user_id: str,
        agent_id: str | None = None,
    ) -> EpisodicEvent:
        """添加推理经验 (对齐 LangMem Episode: obs/thoughts/action/result)。

        content = 综合叙述; obs/thoughts/action/result 存专用字段。
        """
        content = f"[EPISODE] observation: {observation}\nthoughts: {thoughts}\naction: {action}\nresult: {result}"
        return self.store.add_episode(
            content,
            user_id=user_id,
            event_type=EpisodeType.REASONING,
            observation=observation,
            thoughts=thoughts,
            action=action,
            result=result,
            agent_id=agent_id,
        )

    def add_raw_log(
        self,
        transcript: str,
        *,
        user_id: str,
        session_id: str,
        agent_id: str | None = None,
    ) -> EpisodicEvent:
        """添加原始 session log (对齐 Cass Episodic: 高保真防摘要坍缩)。"""
        return self.store.add_episode(
            transcript,
            user_id=user_id,
            event_type=EpisodeType.RAW_LOG,
            session_id=session_id,
            agent_id=agent_id,
        )

    def get_timeline(
        self,
        *,
        user_id: str,
        event_type: EpisodeType | str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 50,
    ) -> list[EpisodicEvent]:
        """时序查询 (Zep reference_time 模式)。"""
        return self.store.get_episodes(
            user_id=user_id,
            event_type=event_type,
            since=since,
            until=until,
            limit=limit,
        )

    def get_reasoning_episodes(self, *, user_id: str, limit: int = 20) -> list[EpisodicEvent]:
        """获取推理经验 (LangMem Episode 检索)。"""
        return self.store.get_episodes(user_id=user_id, event_type=EpisodeType.REASONING, limit=limit)

    def episode_to_dict(self, event: EpisodicEvent) -> dict[str, Any]:
        """序列化 (含推理字段, 对齐 LangMem Episode 格式)。"""
        return {
            "id": event.id,
            "event_type": event.event_type,
            "content": event.content,
            "reference_time": event.reference_time.isoformat(),
            "user_id": event.user_id,
            "agent_id": event.agent_id,
            "session_id": event.session_id,
            "observation": event.observation,
            "thoughts": event.thoughts,
            "action": event.action,
            "result": event.result,
        }

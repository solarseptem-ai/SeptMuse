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
"""V2 情节记忆子组件 — EpisodicEvent CRUD + 时序查询 + 三子类。

继承 LongTermMemory ABC, 委托 TypedMemoryStore。
数据模型共享 models/episodic.py 的 EpisodicEvent, 不 import models/ 的操作类。

详见 docs/specs/2026-08-04-v2-memory-architecture.md §2.3 + §4。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlmodel import Session

from septmuse.core.logging import get_logger
from septmuse.memory.base import LongTermMemory
from septmuse.models.episodic import EpisodeType, EpisodicEvent
from septmuse.storage.relational_stores.typed_store import TypedMemoryStore

logger = get_logger(__name__)


class EpisodicMemory(LongTermMemory):
    """V2 情节记忆 — 三子类统一接口 (时序事件 + 推理经验 + raw log)。

    构造参数 (与 V1 models/episodic.py 一致):
        em = EpisodicMemory(store=typed_store)

    与 V1 区别: 继承 LongTermMemory ABC, 实现 invalidate/get_history/get_all。
    """

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
        """添加时序事件 (Zep Episode: content + reference_time)。"""
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
        """添加推理经验 (LangMem Episode: obs/thoughts/action/result)。"""
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
        """添加原始 session log (Cass Episodic: 高保真)。"""
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
        """序列化 (含推理字段)。"""
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

    # ── LongTermMemory ABC 实现 ──

    def invalidate(self, memory_id: str, *, invalid_at: str | None = None) -> bool:
        """标记事件不再为真 (软删除 is_deleted=True)。"""
        with Session(self.store.engine) as session:
            event = session.get(EpisodicEvent, memory_id)
            if not event or event.is_deleted:
                return False
            event.is_deleted = True
            session.add(event)
            session.commit()
            return True

    def get_history(self, memory_id: str) -> list[dict[str, Any]]:
        """获取记忆变更历史 (审计用, 暂返回基本信息)。"""
        return [{"id": memory_id, "event": "no_history_available"}]

    def get_all(self, *, user_id: str, limit: int = 100) -> list[dict[str, Any]]:
        """列出用户全部情节事件 (分页)。"""
        events = self.store.get_episodes(user_id=user_id, limit=limit)
        return [self.episode_to_dict(e) for e in events]

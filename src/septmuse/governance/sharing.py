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
"""跨 agent 记忆共享 — user_id/agent_id 并列模式。

模式说明:
- user_id/agent_id/run_id 三 ID 并列写入 metadata + filters
- 单传 user_id (不传 agent_id) → 该用户记忆跨 agent 共享
- 隔离靠 vector_store metadata filter
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from septmuse.core.logging import get_logger
from septmuse.storage.base import MemoryStore

logger = get_logger(__name__)


@dataclass
class MemoryScope:
    """记忆共享作用域 (user_id/agent_id 并列模式)。

    user_id 是主共享键 (跨 agent); agent_id 限定到特定 agent。
    单传 user_id → 跨 agent 共享; 同时传 user_id + agent_id → 限定该 agent。
    """

    user_id: str
    agent_id: str | None = None

    def is_shared(self) -> bool:
        """是否跨 agent 共享 (无 agent_id 限定)。"""
        return self.agent_id is None

    def to_filter(self) -> dict[str, str]:
        """转为 metadata filter。"""
        f: dict[str, str] = {"user_id": self.user_id}
        if self.agent_id is not None:
            f["agent_id"] = self.agent_id
        return f


class SharedMemoryAccessor:
    """跨 agent 共享查询器。

    用法:
        accessor = SharedMemoryAccessor(store)
        agents = accessor.list_agents("alice")
        users = accessor.list_users("bot1")
        is_shared = accessor.is_cross_agent("alice")
    """

    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def list_agents(self, user_id: str) -> list[str]:
        """列出该用户的所有 agent (去重)。"""
        return self.store.list_agents(user_id)

    def list_users(self, agent_id: str) -> list[str]:
        """列出该 agent 的所有用户。"""
        return self.store.list_users(agent_id)

    def is_cross_agent(self, user_id: str) -> bool:
        """检查该用户记忆是否跨 agent 共享。"""
        agents = self.list_agents(user_id)
        if len(agents) >= 2:
            return True
        shared = self.store.get_shared_memories(user_id, limit=100)
        return any(m.get("agent_id") is None for m in shared)

    def get_shared_memories(self, user_id: str, limit: int = 100) -> list[dict[str, Any]]:
        """获取跨 agent 共享的记忆。"""
        return self.store.get_shared_memories(user_id, limit)

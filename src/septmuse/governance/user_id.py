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
"""user_id 跨 agent 共享 — mem0 三 ID 并列模式 (架构文档 §5.5)。

借鉴 (源码实证 mem0/main.py):
- _build_filters_and_metadata: user_id/agent_id/run_id 三 ID 并列写入 metadata + filters
- 单传 user_id (不传 agent_id) → 该用户记忆跨 agent 共享 (任何 agent 用同 user_id 可读)
- 隔离靠 vector_store metadata filter, 非 SQLiteManager
- _build_session_scope: "agent_id=a1&user_id=u1" 字典序拼接 (session 上下文)

SeptMuse 简化:
- MemoryScope: 封装 user_id + agent_id 的共享作用域
- SharedMemoryAccessor: 跨 agent 共享查询 (list_agents / list_users / is_cross_agent)

详见 docs/specs/agent-memory-architecture.md §5.5 共享。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from septmuse.core.logging import get_logger
from septmuse.storage.base import MemoryStore

logger = get_logger(__name__)


@dataclass
class MemoryScope:
    """记忆共享作用域 (对齐 mem0 user_id/agent_id 并列模式)。

    user_id 是主共享键 (跨 agent); agent_id 限定到特定 agent。
    单传 user_id → 跨 agent 共享; 同时传 user_id + agent_id → 限定该 agent。
    """

    user_id: str
    agent_id: str | None = None

    def is_shared(self) -> bool:
        """是否跨 agent 共享 (无 agent_id 限定)。"""
        return self.agent_id is None

    def to_filter(self) -> dict[str, str]:
        """转为 metadata filter (对齐 mem0 effective_query_filters)。"""
        f: dict[str, str] = {"user_id": self.user_id}
        if self.agent_id is not None:
            f["agent_id"] = self.agent_id
        return f


class SharedMemoryAccessor:
    """跨 agent 共享查询器 (对齐 mem0 user_id 共享模式)。

    用法:
        accessor = SharedMemoryAccessor(store)
        agents = accessor.list_agents("alice")  # alice 被哪些 agent 共享
        users = accessor.list_users("bot1")     # bot1 有哪些用户的记忆
        is_shared = accessor.is_cross_agent("alice")  # alice 是否跨 agent
    """

    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def list_agents(self, user_id: str) -> list[str]:
        """列出该用户的所有 agent (跨 agent 共享, 对齐 mem0 metadata filter)。

        返回有该 user_id 记忆的所有 agent_id (去重)。
        agent_id 为 NULL 的记忆表示跨 agent 共享。
        """
        return self.store.list_agents(user_id)

    def list_users(self, agent_id: str) -> list[str]:
        """列出该 agent 的所有用户 (对齐 mem0 agent-scoped 查询)。"""
        return self.store.list_users(agent_id)

    def is_cross_agent(self, user_id: str) -> bool:
        """检查该用户记忆是否跨 agent 共享 (对齐 mem0 单传 user_id 模式)。

        True = 多个 agent 为同一用户存了记忆 (或有无 agent_id 的共享记忆)。
        """
        agents = self.list_agents(user_id)
        if len(agents) >= 2:
            return True
        # 检查是否有无 agent_id 的共享记忆 (list_agents 排除 NULL, 需额外检查)
        shared = self.store.get_shared_memories(user_id, limit=100)
        return any(m.get("agent_id") is None for m in shared)

    def get_shared_memories(self, user_id: str, limit: int = 100) -> list[dict[str, Any]]:
        """获取跨 agent 共享的记忆 (对齐 mem0 单传 user_id 查询)。

        返回该 user_id 的所有记忆 (不限 agent_id)。
        """
        return self.store.get_shared_memories(user_id, limit)

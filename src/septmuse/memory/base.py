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
"""记忆 ABC 分层 — 区分短期记忆 vs 长期记忆的类型标记。

设计决策 (brainstorming 确认):
- ABC 只做类型标记 + 各层特有方法, 不强制统一 add/search
- 各子类 add 方法保持原名 (add_fact / add_raw_log / add_rule / core_memory_append)
- ABC 价值: isinstance 判断短期 vs 长期 + 各层特有方法约束

详见 docs/specs/2026-08-04-v2-memory-architecture.md §2。
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class MemoryABC(ABC):  # noqa: B024
    """记忆根抽象 — 类型标记, 不强制统一 add/search 方法。

    各子类 add 方法保持原名 (add_fact / add_raw_log / add_rule / core_memory_append),
    因为参数签名不同 (add_fact(subject,predicate,object) vs add_raw_log(transcript))。
    ABC 价值: 类型标记 (isinstance 判断短期 vs 长期) + 各层特有方法约束。
    """


class ShortTermMemory(MemoryABC):
    """短期记忆 — context window 内, 零检索即可见。

    特征:
    - 编译注入 system prompt (compile_to_prompt)
    - 容量有限 (token/char limit)
    - 超限驱逐到长期记忆
    - 跨会话持久化但每次会话加载到 context

    各子类 add 方法保持原名 (core_memory_append / core_memory_replace),
    不强制统一为 add()。
    """

    @abstractmethod
    def compile_to_prompt(self) -> str:
        """编译为可注入 system prompt 的文本。"""
        ...

    @abstractmethod
    def get_limit(self) -> int:
        """获取容量上限 (字符数或 token 数)。"""
        ...

    @abstractmethod
    def evict_overflow(self) -> list[dict]:
        """驱逐超限内容到长期记忆, 返回被驱逐的内容列表。"""
        ...


class LongTermMemory(MemoryABC):
    """长期记忆 — 跨会话持久, 需检索召回。

    特征:
    - 持久化到 DB (SQLite/PG/MySQL)
    - 需要向量/BM25/图检索才能召回
    - 双时态 (valid_at / invalid_at)
    - 软删除 + 历史保留

    各子类 add 方法保持原名 (add_fact / add_raw_log / add_rule),
    不强制统一为 add()。
    """

    @abstractmethod
    def invalidate(self, memory_id: str, *, invalid_at: str | None = None) -> bool:
        """标记事实不再为真 (双时态: 设 invalid_at, 不删除)。"""
        ...

    @abstractmethod
    def get_history(self, memory_id: str) -> list[dict]:
        """获取记忆变更历史 (审计用)。"""
        ...

    @abstractmethod
    def get_all(self, *, user_id: str, limit: int = 100) -> list[dict]:
        """列出用户全部记忆 (分页)。"""
        ...

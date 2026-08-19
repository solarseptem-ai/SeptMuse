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
"""记忆 ABC 分层 — 区分短期记忆 vs 长期记忆的类型标记。"""

from __future__ import annotations

from abc import ABC, abstractmethod


class MemoryABC(ABC):  # noqa: B024
    """记忆根抽象 — 类型标记, 不强制统一 add/search 方法。"""


class ShortTermMemory(MemoryABC):
    """短期记忆 — context window 内, 零检索即可见。"""

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
    """长期记忆 — 跨会话持久, 需检索召回。"""

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

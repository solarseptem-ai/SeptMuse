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
"""工作记忆独立后端 ABC — Block CRUD 接口。

设计决策 (brainstorming 确认):
- WorkingMemory 走独立 WorkingMemoryStore (非 typed_store), 彻底分库
- 默认 SQLite + 内存缓存 (与长时记忆同 .db 文件)
- 可选 Redis (SEPTMUSE_WORKING_MEMORY_BACKEND=redis)

详见 docs/specs/2026-08-04-v2-memory-architecture.md §5.1 + §6。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from septmuse.models.block import Block


class WorkingMemoryStore(ABC):
    """工作记忆存储后端 ABC — Block CRUD 接口。

    子类:
    - SQLiteStore: SQLite + 内存缓存 (默认, 零配置)
    - RedisStore: Redis (可选, 需 SEPTMUSE_REDIS_URL)
    """

    @abstractmethod
    def get_blocks(self, agent_id: str) -> list[Block]:
        """加载 agent 的全部 block。"""
        ...

    @abstractmethod
    def save_block(self, block: Block) -> Block:
        """保存 block (INSERT or UPDATE, 按 id upsert)。"""
        ...

    @abstractmethod
    def update_block_value(self, agent_id: str, label: str, value: str) -> Block | None:
        """更新 block value。"""
        ...

    @abstractmethod
    def delete_block(self, agent_id: str, label: str) -> bool:
        """删除 block。"""
        ...

    @abstractmethod
    def ensure_default_blocks(self, agent_id: str) -> list[Block]:
        """确保 agent 有默认 block (human + persona), 无则创建。"""
        ...

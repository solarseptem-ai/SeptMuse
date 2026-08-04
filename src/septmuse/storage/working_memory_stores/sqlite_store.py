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
"""SQLite 工作记忆后端 — Block CRUD, 共享 engine (同 .db 文件)。

零配置默认: 与长时记忆共享 SQLite engine, 独立类实现。
"""

from __future__ import annotations

from sqlmodel import Session, select

from septmuse.core.logging import get_logger
from septmuse.models.block import Block, default_blocks
from septmuse.storage.working_memory_stores.base import WorkingMemoryStore

logger = get_logger(__name__)


class SQLiteWorkingMemoryStore(WorkingMemoryStore):
    """SQLite 工作记忆后端 (共享 engine, 独立实现)。

    用法:
        store = SQLiteWorkingMemoryStore(engine=mem.store.engine)
        blocks = store.ensure_default_blocks("agent-1")
    """

    def __init__(self, engine) -> None:
        self.engine = engine

    def get_blocks(self, agent_id: str) -> list[Block]:
        """加载 agent 的全部 block。"""
        with Session(self.engine) as session:
            stmt = select(Block).where(Block.agent_id == agent_id)
            return list(session.exec(stmt).all())

    def save_block(self, block: Block) -> Block:
        """保存 block (INSERT or UPDATE, 按 id upsert)。"""
        with Session(self.engine) as session:
            existing = session.get(Block, block.id)
            if existing:
                existing.label = block.label
                existing.value = block.value
                existing.limit = block.limit
                existing.read_only = block.read_only
                existing.tags = block.tags
                existing.touch()
                session.add(existing)
            else:
                session.add(block)
            session.commit()
            session.refresh(existing or block)
            return existing or block

    def update_block_value(self, agent_id: str, label: str, value: str) -> Block | None:
        """更新 block value。"""
        with Session(self.engine) as session:
            stmt = select(Block).where(Block.agent_id == agent_id, Block.label == label)
            block = session.exec(stmt).first()
            if not block:
                return None
            block.value = value
            block.touch()
            session.add(block)
            session.commit()
            session.refresh(block)
            return block

    def delete_block(self, agent_id: str, label: str) -> bool:
        """删除 block。"""
        with Session(self.engine) as session:
            stmt = select(Block).where(Block.agent_id == agent_id, Block.label == label)
            block = session.exec(stmt).first()
            if not block:
                return False
            session.delete(block)
            session.commit()
            return True

    def ensure_default_blocks(self, agent_id: str) -> list[Block]:
        """确保 agent 有默认 block (human + persona), 无则创建。"""
        blocks = self.get_blocks(agent_id)
        if blocks:
            return blocks
        for block in default_blocks(agent_id):
            self.save_block(block)
        logger.info("default_blocks_created", agent_id=agent_id)
        return self.get_blocks(agent_id)

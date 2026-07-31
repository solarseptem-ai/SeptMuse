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
"""工作记忆 Block 数据模型 + 操作。

数据模型 (借鉴 letta/schemas/block.py):
- value: 块内容 (LLM context 内可见)
- limit: 字符上限 (治理: 防 context 溢出, 借鉴 Hermes char_limit)
- label: 段标签 (如 "human" / "persona")
- read_only: agent 是否只读
- tags: 关联标签
- sanitize_value_null_bytes: 移除 null 字节防 PG 错误 (对齐 letta field_validator)

操作 (借鉴 letta/schemas/memory.py BasicBlockMemory):
- get_block(label): 遍历找 label, 找不到 raise KeyError
- update_block_value(label, value): 遍历找 label 改 value, 找不到 raise ValueError
- set_block(block): 同 label 替换或 append
- core_memory_append(label, content): current + "\\n" + content
- core_memory_replace(label, old, new): old not in current → raise ValueError, replace

XML 编译格式 (架构文档 §3.1.1):
    <memory>
      <block label="human">...</block>
      <block label="persona">...</block>
    </memory>
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any
from xml.sax.saxutils import escape

from pydantic import field_validator
from sqlalchemy import Column
from sqlalchemy.types import JSON
from sqlmodel import Field, SQLModel

from septmuse.core.logging import get_logger

# 默认块字符上限 (借鉴 letta CORE_MEMORY_BLOCK_CHAR_LIMIT + Hermes memory_char_limit)
DEFAULT_BLOCK_CHAR_LIMIT = 2000


def _utcnow() -> datetime:
    """UTC 当前时间 (带 tzinfo)。"""
    return datetime.now(timezone.utc)


def _new_block_id() -> str:
    """生成块唯一 ID。"""
    return f"block-{uuid.uuid4()}"


class Block(SQLModel, table=True):
    """工作记忆块 — context window 内的保留区, agent 可自编辑。

    对齐 letta Block: value/limit/label/read_only/tags/description。
    SeptMuse 增量: agent_id (跨 agent 共享键), created_at/updated_at (时序)。
    """

    __tablename__ = "septmuse_blocks"  # type: ignore[assignment]  # SQLModel mypy stub 已知误报

    id: str = Field(default_factory=_new_block_id, primary_key=True)
    agent_id: str = Field(index=True, description="归属 agent (跨 agent 共享键)")
    label: str = Field(index=True, description="段标签, 如 human/persona/task")
    value: str = Field(default="", description="块内容, LLM context 内可见")
    limit: int = Field(default=DEFAULT_BLOCK_CHAR_LIMIT, description="字符上限")
    read_only: bool = Field(default=False, description="agent 是否只读")
    description: str | None = Field(default=None, description="块描述")

    # tags 用 JSON 列存储 list[str] (对齐 letta tags)
    tags: list[str] = Field(default=[], sa_column=Column(JSON))

    created_at: datetime = Field(default_factory=_utcnow, description="创建时间 UTC")
    updated_at: datetime = Field(default_factory=_utcnow, description="更新时间 UTC")

    @field_validator("value", mode="before")
    @classmethod
    def sanitize_value_null_bytes(cls, v: Any) -> Any:
        """移除 null 字节, 防止 PostgreSQL 编码错误 (对齐 letta)。"""
        if isinstance(v, str):
            return v.replace("\x00", "")
        return v

    def touch(self) -> None:
        """更新 updated_at 时间戳。"""
        self.updated_at = _utcnow()


def default_blocks(agent_id: str) -> list[Block]:
    """生成默认块集: human + persona (对齐 letta DEFAULT_BLOCKS)。

    直接构造 Block(label=...) 而非子类, 避免 SQLModel 单表继承复杂度。
    """
    return [
        Block(agent_id=agent_id, label="human", value=""),
        Block(agent_id=agent_id, label="persona", value=""),
    ]


logger = get_logger(__name__)


class WorkingMemory:
    """工作记忆 — 持有 Block 列表, 提供 agent 自编辑工具 + XML 编译。

    对齐 letta BasicBlockMemory, 简化为直接持有 blocks (无 agent_state 间接层)。
    """

    def __init__(
        self,
        agent_id: str,
        blocks: list[Block] | None = None,
        store: Any | None = None,
    ) -> None:
        """初始化工作记忆。

        Args:
            agent_id: 归属 agent ID (跨 agent 共享键)
            blocks: 初始块列表; None 时用 default_blocks(agent_id)
            store: TypedMemoryStore | None; 非空时操作后自动持久化
        """
        self.agent_id = agent_id
        self.store = store
        self.blocks: list[Block] = blocks if blocks is not None else default_blocks(agent_id)
        logger.debug("working_memory_init", agent_id=agent_id, block_count=len(self.blocks))

    def list_block_labels(self) -> list[str]:
        """返回所有块标签 (对齐 letta list_block_labels)。"""
        return [b.label for b in self.blocks]

    def get_block(self, label: str) -> Block:
        """按 label 取块 (对齐 letta get_block)。

        Raises:
            KeyError: label 不存在时
        """
        keys: list[str] = []
        for block in self.blocks:
            if block.label == label:
                return block
            keys.append(block.label)
        raise KeyError(f"Block field {label} does not exist (available = {', '.join(keys)})")

    def get_blocks(self) -> list[Block]:
        """返回全部块 (对齐 letta get_blocks)。"""
        return self.blocks

    def set_block(self, block: Block) -> None:
        """设置块: 同 label 替换, 否则 append (对齐 letta set_block)。"""
        for i, b in enumerate(self.blocks):
            if b.label == block.label:
                self.blocks[i] = block
                if self.store is not None:
                    self.store.save_block(block)
                return
        self.blocks.append(block)
        if self.store is not None:
            self.store.save_block(block)

    def update_block_value(self, label: str, value: str) -> None:
        """更新块 value (对齐 letta update_block_value)。

        Raises:
            ValueError: value 非 str 或 label 不存在
        """
        if not isinstance(value, str):
            raise ValueError("Provided value must be a string")
        for block in self.blocks:
            if block.label == label:
                block.value = value
                block.touch()
                if self.store is not None:
                    self.store.save_block(block)
                return
        raise ValueError(f"Block with label {label} does not exist")

    def core_memory_append(self, label: str, content: str) -> None:
        """追加到块内容 (对齐 letta core_memory_append)。

        new_value = current_value + "\\n" + content
        超限记日志 (驱逐逻辑见 eviction.py)。
        """
        current_value = str(self.get_block(label).value)
        new_value = current_value + "\n" + str(content)
        self._check_limit(label, new_value)
        self.update_block_value(label=label, value=new_value)
        logger.info("core_memory_append", label=label, added_len=len(str(content)))

    def core_memory_replace(self, label: str, old_content: str, new_content: str) -> None:
        """替换块内容 (对齐 letta core_memory_replace)。

        old_content 必须在 current_value 中 (精确匹配), 否则 raise ValueError。
        new_content 为空串时表示删除该片段。

        Raises:
            ValueError: old_content 未找到
        """
        current_value = str(self.get_block(label).value)
        if old_content not in current_value:
            raise ValueError(f"Old content '{old_content}' not found in memory block '{label}'")
        new_value = current_value.replace(str(old_content), str(new_content))
        self._check_limit(label, new_value)
        self.update_block_value(label=label, value=new_value)
        logger.info(
            "core_memory_replace",
            label=label,
            old_len=len(str(old_content)),
            new_len=len(str(new_content)),
        )

    def compile_to_xml(self) -> str:
        """编译为 XML 注入 system prompt (架构文档 §3.1.1)。

        格式:
            <memory>
              <block label="human">值</block>
              <block label="persona">值</block>
            </memory>
        """
        parts: list[str] = ["<memory>"]
        for block in self.blocks:
            escaped_value = escape(block.value)
            parts.append(f'  <block label="{escape(block.label)}">{escaped_value}</block>')
        parts.append("</memory>")
        return "\n".join(parts)

    def _check_limit(self, label: str, new_value: str) -> None:
        """检查块是否超限, 超限记 warning (驱逐逻辑见 eviction.py, 阶段3)。"""
        block = self.get_block(label)
        if len(new_value) > block.limit:
            logger.warning(
                "block_limit_exceeded",
                label=label,
                length=len(new_value),
                limit=block.limit,
                hint="eviction to long-term pending (eviction.py, 阶段3)",
            )

    def to_dict(self) -> dict:
        """序列化为 dict (用于 API 响应/持久化)。"""
        return {
            "agent_id": self.agent_id,
            "blocks": [
                {
                    "id": b.id,
                    "label": b.label,
                    "value": b.value,
                    "limit": b.limit,
                    "read_only": b.read_only,
                    "tags": b.tags,
                }
                for b in self.blocks
            ],
            "compiled": self.compile_to_xml(),
        }

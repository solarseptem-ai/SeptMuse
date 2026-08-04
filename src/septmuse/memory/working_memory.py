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
"""V2 工作记忆子组件 — Block CRUD + compile_to_prompt + 超限驱逐。

继承 ShortTermMemory ABC, 委托 WorkingMemoryStore (独立后端)。
数据模型共享 models/block.py 的 Block, 不 import models/ 的操作类。

详见 docs/specs/2026-08-04-v2-memory-architecture.md §2.3 + §4。
"""

from __future__ import annotations

from typing import Any
from xml.sax.saxutils import escape

from septmuse.core.logging import get_logger
from septmuse.memory.base import ShortTermMemory
from septmuse.models.block import Block
from septmuse.storage.working_memory_stores.base import WorkingMemoryStore

logger = get_logger(__name__)


class WorkingMemory(ShortTermMemory):
    """V2 工作记忆 — Block CRUD + XML 编译 + 超限驱逐。

    构造参数 (store 在前, 对齐 V2 组合模式):
        wm = WorkingMemory(store=wm_store, agent_id="default")

    与 V1 models/block.py 的 WorkingMemory 区别:
    - 继承 ShortTermMemory ABC
    - 委托 WorkingMemoryStore (独立后端, 非 typed_store)
    - 实现 ABC 方法: compile_to_prompt / get_limit / evict_overflow
    """

    def __init__(self, *, store: WorkingMemoryStore, agent_id: str) -> None:
        self.store = store
        self.agent_id = agent_id
        self.blocks: list[Block] = store.ensure_default_blocks(agent_id)
        logger.debug("v2_working_memory_init", agent_id=agent_id, block_count=len(self.blocks))

    def list_block_labels(self) -> list[str]:
        """返回所有块标签。"""
        return [b.label for b in self.blocks]

    def get_block(self, label: str) -> Block:
        """按 label 取块。"""
        for block in self.blocks:
            if block.label == label:
                return block
        raise KeyError(f"Block field {label} does not exist (available = {', '.join(self.list_block_labels())})")

    def get_blocks(self) -> list[Block]:
        """返回全部块。"""
        return self.blocks

    def set_block(self, block: Block) -> None:
        """设置块: 同 label 替换, 否则 append。"""
        for i, b in enumerate(self.blocks):
            if b.label == block.label:
                self.blocks[i] = self.store.save_block(block)
                return
        self.blocks.append(self.store.save_block(block))

    def update_block_value(self, label: str, value: str) -> None:
        """更新块 value。"""
        if not isinstance(value, str):
            raise ValueError("Provided value must be a string")
        for i, block in enumerate(self.blocks):
            if block.label == label:
                block.value = value
                block.touch()
                self.blocks[i] = self.store.save_block(block)
                return
        raise ValueError(f"Block with label {label} does not exist")

    def core_memory_append(self, label: str, content: str) -> None:
        """追加到块内容。"""
        current_value = str(self.get_block(label).value)
        new_value = current_value + "\n" + str(content)
        self._check_limit(label, new_value)
        self.update_block_value(label=label, value=new_value)
        logger.info("core_memory_append", label=label, added_len=len(str(content)))

    def core_memory_replace(self, label: str, old_content: str, new_content: str) -> None:
        """替换块内容片段。"""
        current_value = str(self.get_block(label).value)
        if old_content not in current_value:
            raise ValueError(f"Old content '{old_content}' not found in memory block '{label}'")
        new_value = current_value.replace(str(old_content), str(new_content))
        self._check_limit(label, new_value)
        self.update_block_value(label=label, value=new_value)

    def compile_to_xml(self) -> str:
        """编译为 XML 注入 system prompt。"""
        parts: list[str] = ["<memory>"]
        for block in self.blocks:
            escaped_value = escape(block.value)
            parts.append(f'  <block label="{escape(block.label)}">{escaped_value}</block>')
        parts.append("</memory>")
        return "\n".join(parts)

    def _check_limit(self, label: str, new_value: str) -> None:
        """检查块是否超限, 超限记 warning。"""
        block = self.get_block(label)
        if len(new_value) > block.limit:
            logger.warning(
                "block_limit_exceeded",
                label=label,
                length=len(new_value),
                limit=block.limit,
            )

    # ── ShortTermMemory ABC 实现 ──

    def compile_to_prompt(self) -> str:
        """编译为可注入 system prompt 的文本 (复用 compile_to_xml)。"""
        return self.compile_to_xml()

    def get_limit(self) -> int:
        """获取容量上限 (所有 block limit 之和)。"""
        return sum(b.limit for b in self.blocks)

    def evict_overflow(self) -> list[dict]:
        """驱逐超限内容到长期记忆, 返回被驱逐的内容列表。

        简化版: 记 warning, 暂不实际驱逐 (后续实现驱逐到 episodic)。
        """
        evicted: list[dict] = []
        for block in self.blocks:
            if len(block.value) > block.limit:
                overflow = block.value[block.limit :]
                evicted.append({"label": block.label, "overflow": overflow, "limit": block.limit})
                logger.info("block_evicted", label=block.label, overflow_len=len(overflow))
        return evicted

    def to_dict(self) -> dict[str, Any]:
        """序列化为 dict。"""
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

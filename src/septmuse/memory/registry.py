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
"""类型×形态 注册表 (架构文档 §7.2 自研, 借鉴 cognee search_core_registry)。

cognee 用 search_core_registry (SearchType→Retriever 路由表) + registered_community_retrievers
(可扩展插件表)。SeptMuse 适配为 MemoryType→Handler 路由表。

注册内容类型 (平面A) × 存储形态 (平面B) 的组合格子,
每个格子注册一个 handler 函数。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from septmuse.core.logging import get_logger

logger = get_logger(__name__)

# 内容类型 (平面A)
MEMORY_TYPES = ["working", "semantic", "episodic", "procedural"]

# 存储形态 (平面B)
STORAGE_FORMS = ["block", "vector", "graph", "file", "activation", "parametric"]


@dataclass
class RegistryEntry:
    """注册表条目 (类型×形态 格子)。"""

    memory_type: str
    storage_form: str
    handler: Callable[..., Any]
    description: str = ""


class MemoryRegistry:
    """类型×形态 注册表 (借鉴 cognee search_core_registry)。

    用法:
        registry = MemoryRegistry()
        registry.register("semantic", "vector", semantic_vector_handler)
        handler = registry.get("semantic", "vector")  # → handler
        entries = registry.list_by_type("semantic")  # → all forms for semantic
    """

    def __init__(self) -> None:
        self._entries: dict[str, RegistryEntry] = {}

    def register(self, memory_type: str, storage_form: str, handler: Callable[..., Any], description: str = "") -> None:
        """注册类型×形态格子的 handler。"""
        key = self._key(memory_type, storage_form)
        self._entries[key] = RegistryEntry(
            memory_type=memory_type,
            storage_form=storage_form,
            handler=handler,
            description=description,
        )
        logger.info("registry_registered", type=memory_type, form=storage_form)

    def get(self, memory_type: str, storage_form: str) -> Callable[..., Any] | None:
        """获取 handler (None=未注册)。"""
        key = self._key(memory_type, storage_form)
        entry = self._entries.get(key)
        return entry.handler if entry else None

    def list_by_type(self, memory_type: str) -> list[RegistryEntry]:
        """列出某类型的所有形态 handler。"""
        return [e for e in self._entries.values() if e.memory_type == memory_type]

    def list_by_form(self, storage_form: str) -> list[RegistryEntry]:
        """列出某形态的所有类型 handler。"""
        return [e for e in self._entries.values() if e.storage_form == storage_form]

    def list_all(self) -> list[RegistryEntry]:
        """列出全部注册条目。"""
        return list(self._entries.values())

    def is_registered(self, memory_type: str, storage_form: str) -> bool:
        """检查是否已注册。"""
        return self._key(memory_type, storage_form) in self._entries

    @staticmethod
    def _key(memory_type: str, storage_form: str) -> str:
        return f"{memory_type}:{storage_form}"


# 全局默认注册表 (单例)
_default_registry: MemoryRegistry | None = None


def get_default_registry() -> MemoryRegistry:
    """获取全局默认注册表 (惰性初始化)。"""
    global _default_registry
    if _default_registry is None:
        _default_registry = MemoryRegistry()
    return _default_registry

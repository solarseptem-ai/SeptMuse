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
"""记忆访问权限 — 4 层权限检查 (sync + async)。

层1: memory 存在 + state=active
层2: 无 app_id → 用户自己访问, 放行
层3: app_id 非空即 active
层4: app 白名单 (默认全部可访问, P2.2 未来扩展)
"""

from __future__ import annotations

from enum import Enum

from septmuse.core.logging import get_logger
from septmuse.storage.async_base import AsyncMemoryStore
from septmuse.storage.base import MemoryStore

logger = get_logger(__name__)


class MemoryState(str, Enum):
    """记忆状态枚举。"""

    ACTIVE = "active"
    PAUSED = "paused"
    ARCHIVED = "archived"
    DELETED = "deleted"


# ── sync ──────────────────────────────────────────────────────────────────


def check_memory_access_permissions(
    store: MemoryStore,
    memory_id: str,
    app_id: str | None = None,
) -> tuple[bool, str]:
    """4 层权限检查 (sync 版)。

    Args:
        store: 记忆存储后端
        memory_id: 目标记忆 ID
        app_id: 访问方应用 ID; None 表示用户自己访问

    Returns:
        (allowed, reason): True=放行, False=拒绝 + 原因
    """
    # 层1: memory 存在 + state=active
    mem = store.get(memory_id)
    if not mem:
        return False, "memory not found"
    state = mem.get("state", MemoryState.ACTIVE)
    if state is not None and state != MemoryState.ACTIVE:
        return False, f"memory state is {state} (not active)"

    # 层2: 无 app_id (None) → 用户自己访问, 放行
    if app_id is None:
        return True, "self access"

    # 层3: app_id 非空即 active
    if not app_id.strip():
        return False, "app_id is empty"

    # 层4: app 白名单 (默认全部可访问)
    return True, f"app {app_id} access granted"


# ── async ─────────────────────────────────────────────────────────────────


async def async_check_memory_access_permissions(
    store: AsyncMemoryStore,
    memory_id: str,
    app_id: str | None = None,
) -> tuple[bool, str]:
    """4 层权限检查 (async 版)。

    Args:
        store: 异步记忆存储后端
        memory_id: 目标记忆 ID
        app_id: 访问方应用 ID; None 表示用户自己访问

    Returns:
        (allowed, reason): True=放行, False=拒绝 + 原因
    """
    # 层1: memory 存在 + state=active
    mem = await store.get(memory_id)
    if not mem:
        return False, "memory not found"
    state = mem.get("state", MemoryState.ACTIVE)
    if state is not None and state != MemoryState.ACTIVE:
        return False, f"memory state is {state} (not active)"

    # 层2: 无 app_id (None) → 用户自己访问, 放行
    if app_id is None:
        return True, "self access"

    # 层3: app_id 非空即 active
    if not app_id.strip():
        return False, "app_id is empty"

    # 层4: app 白名单 (默认全部可访问)
    return True, f"app {app_id} access granted"

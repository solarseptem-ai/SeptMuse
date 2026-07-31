"""异步权限检查 — 4 层权限检查（async 版，与 sync 版并存）。

层1: memory 存在 + state=active
层2: 无 app_id → 用户自己访问, 放行
层3: app_id 非空即 active
层4: app 白名单（默认全部可访问）
"""

from __future__ import annotations

from septmuse.governance.permissions import MemoryState
from septmuse.storage.async_base import AsyncMemoryStore


async def async_check_memory_access_permissions(
    store: AsyncMemoryStore,
    memory_id: str,
    app_id: str | None = None,
) -> tuple[bool, str]:
    """4 层权限检查（async 版）。

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

    # 层4: app 白名单（默认全部可访问）
    return True, f"app {app_id} access granted"

"""async 权限检查函数测试。"""

from septmuse.governance.access import async_check_memory_access_permissions
from septmuse.storage.async_base import AsyncMemoryStore


class _FakeStore(AsyncMemoryStore):
    """测试用假 store。"""

    def __init__(self, mem=None):
        self._mem = mem

    async def add(self, content, embedding, *, user_id, **kwargs):
        pass

    async def search(self, query_embedding, *, user_id, **kwargs):
        return []

    async def get_all(self, *, user_id, **kwargs):
        return []

    async def get(self, memory_id):
        return self._mem

    async def delete(self, memory_id):
        pass

    async def update(self, memory_id, content, embedding, *, metadata=None):
        return True

    async def get_history(self, memory_id):
        return []

    async def close(self):
        pass


async def test_active_memory_allowed():
    """active 记忆 + 无 app_id → 放行。"""
    store = _FakeStore(mem={"id": "m1", "memory": "hello"})
    allowed, reason = await async_check_memory_access_permissions(store, "m1")
    assert allowed is True
    assert "self access" in reason


async def test_not_found_rejected():
    """不存在的记忆 → 拒绝。"""
    store = _FakeStore(mem=None)
    allowed, reason = await async_check_memory_access_permissions(store, "nonexistent")
    assert allowed is False
    assert "not found" in reason


async def test_with_app_id_allowed():
    """active 记忆 + 有 app_id → 放行。"""
    store = _FakeStore(mem={"id": "m1", "memory": "hello"})
    allowed, reason = await async_check_memory_access_permissions(store, "m1", app_id="my-app")
    assert allowed is True
    assert "granted" in reason


async def test_empty_app_id_rejected():
    """active 记忆 + 空 app_id → 拒绝。"""
    store = _FakeStore(mem={"id": "m1", "memory": "hello"})
    allowed, reason = await async_check_memory_access_permissions(store, "m1", app_id="")
    assert allowed is False
    assert "empty" in reason


async def test_none_app_id_allowed():
    """active 记忆 + None app_id → 放行（用户自己访问）。"""
    store = _FakeStore(mem={"id": "m1", "memory": "hello"})
    allowed, _ = await async_check_memory_access_permissions(store, "m1", app_id=None)
    assert allowed is True

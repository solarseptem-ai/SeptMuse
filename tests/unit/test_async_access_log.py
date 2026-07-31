#  Copyright 2026 The sonhhxg0529 Authors. All Rights Reserved.
"""async 访问日志函数测试。"""

from septmuse.governance.async_access_log import async_record_access
from septmuse.storage.async_base import AsyncMemoryStore


class _FakeStoreWithLog(AsyncMemoryStore):
    """支持访问日志的假 store。"""

    async def add(self, content, embedding, *, user_id, **kwargs):
        pass

    async def search(self, query_embedding, *, user_id, **kwargs):
        return []

    async def get_all(self, *, user_id, **kwargs):
        return []

    async def get(self, memory_id):
        return None

    async def delete(self, memory_id):
        pass

    async def update(self, memory_id, content, embedding, *, metadata=None):
        return True

    async def get_history(self, memory_id):
        return []

    async def close(self):
        pass

    async def _record_access_log(self, memory_id, app_id, access_type, metadata=None):
        return "log-123"


class _FakeStoreNoLog(AsyncMemoryStore):
    """不支持访问日志的假 store。"""

    async def add(self, content, embedding, *, user_id, **kwargs):
        pass

    async def search(self, query_embedding, *, user_id, **kwargs):
        return []

    async def get_all(self, *, user_id, **kwargs):
        return []

    async def get(self, memory_id):
        return None

    async def delete(self, memory_id):
        pass

    async def update(self, memory_id, content, embedding, *, metadata=None):
        return True

    async def get_history(self, memory_id):
        return []

    async def close(self):
        pass


class _FakeStoreRaisesLog(AsyncMemoryStore):
    """_record_access_log 抛异常的假 store。"""

    async def add(self, content, embedding, *, user_id, **kwargs):
        pass

    async def search(self, query_embedding, *, user_id, **kwargs):
        return []

    async def get_all(self, *, user_id, **kwargs):
        return []

    async def get(self, memory_id):
        return None

    async def delete(self, memory_id):
        pass

    async def update(self, memory_id, content, embedding, *, metadata=None):
        return True

    async def get_history(self, memory_id):
        return []

    async def close(self):
        pass

    async def _record_access_log(self, memory_id, app_id, access_type, metadata=None):
        raise RuntimeError("DB locked")


async def test_returns_log_id():
    """支持日志的 store 返回 log_id。"""
    store = _FakeStoreWithLog()
    log_id = await async_record_access(store, "m1", "app1", "get")
    assert log_id == "log-123"


async def test_unsupported_store_returns_none():
    """不支持日志的 store 返回 None。"""
    store = _FakeStoreNoLog()
    log_id = await async_record_access(store, "m1", "app1", "get")
    assert log_id is None


async def test_swallows_exceptions():
    """_record_access_log 抛异常时不传播，返回 None。"""
    store = _FakeStoreRaisesLog()
    log_id = await async_record_access(store, "m1", "app1", "get")
    assert log_id is None

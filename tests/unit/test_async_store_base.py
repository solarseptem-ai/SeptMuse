"""AsyncMemoryStore ABC 测试。"""
import inspect

import pytest

from septmuse.storage.async_base import AsyncMemoryStore


def test_all_methods_are_async():
    """所有公开方法都是 async def。"""
    for name in ["add", "search", "get_all", "get", "delete", "update", "get_history", "close"]:
        method = getattr(AsyncMemoryStore, name)
        assert inspect.iscoroutinefunction(method), f"{name} 不是 async def"


def test_default_methods_are_async():
    """默认实现方法也是 async。"""
    for name in ["keyword_search", "hybrid_search", "get_access_logs", "get_temporal_valid", "get_temporal_interval"]:
        method = getattr(AsyncMemoryStore, name)
        assert inspect.iscoroutinefunction(method), f"{name} 不是 async def"


def test_cannot_instantiate_abc():
    """不能直接实例化 ABC。"""
    with pytest.raises(TypeError):
        AsyncMemoryStore()

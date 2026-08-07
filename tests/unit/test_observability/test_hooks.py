"""track_operation 装饰器测试。"""

import asyncio

from prometheus_client import REGISTRY

from septmuse.observability.collector import MetricsCollector
from septmuse.observability.hooks import time_block, track_operation


def test_track_operation_sync():
    """同步函数装饰器记录延迟。"""

    @track_operation("add")
    def add(a, b):
        return a + b

    MetricsCollector.get().configure(enabled=True)
    result = add(1, 2)
    assert result == 3

    samples = list(REGISTRY.collect())
    op_metric = [s for s in samples if s.name == "septmuse_memory_operation_duration_seconds"]
    assert len(op_metric) > 0
    found = any(s.labels.get("operation") == "add" for s in op_metric[0].samples)
    assert found


def test_track_operation_async():
    """异步函数装饰器记录延迟。"""

    @track_operation("search")
    async def search(q):
        return f"result:{q}"

    MetricsCollector.get().configure(enabled=True)
    result = asyncio.run(search("hello"))
    assert result == "result:hello"

    samples = list(REGISTRY.collect())
    op_metric = [s for s in samples if s.name == "septmuse_memory_operation_duration_seconds"]
    assert len(op_metric) > 0
    found = any(s.labels.get("operation") == "search" for s in op_metric[0].samples)
    assert found


def test_time_block():
    """time_block 上下文管理器记录耗时。"""
    MetricsCollector.get().configure(enabled=True)
    with time_block("hybrid_search_components_seconds", {"component": "vector"}):
        pass

    samples = list(REGISTRY.collect())
    hybrid_metric = [s for s in samples if s.name == "septmuse_hybrid_search_components_seconds"]
    assert len(hybrid_metric) > 0
    found = any(s.labels.get("component") == "vector" for s in hybrid_metric[0].samples)
    assert found


def test_noop_when_disabled():
    """未启用时装饰器不报错。"""

    @track_operation("add")
    def add(a, b):
        return a + b

    result = add(1, 2)
    assert result == 3

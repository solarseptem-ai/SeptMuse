"""埋点辅助 — track_operation 装饰器 + time_block 上下文管理器。

track_operation: 装饰 Memory facade 方法，自动记录 memory_operation_duration_seconds。
time_block: 用于 HybridRetriever 内部函数的 component 级计时。
"""

from __future__ import annotations

import asyncio
import functools
import time
from collections.abc import Iterator
from contextlib import contextmanager

from septmuse.observability.collector import MetricsCollector


def track_operation(operation: str):
    """装饰器 — 记录 memory_operation_duration_seconds（同步+异步兼容）。

    Args:
        operation: 操作名 (add/search/update/delete/get/invalidate)
    """

    def decorator(func):
        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                collector = MetricsCollector.get()
                start = time.perf_counter()
                try:
                    return await func(*args, **kwargs)
                finally:
                    collector.observe(
                        "memory_operation_duration_seconds",
                        time.perf_counter() - start,
                        labels={"operation": operation},
                    )

            return async_wrapper

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            collector = MetricsCollector.get()
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                collector.observe(
                    "memory_operation_duration_seconds",
                    time.perf_counter() - start,
                    labels={"operation": operation},
                )

        return sync_wrapper

    return decorator


@contextmanager
def time_block(metric_name: str, labels: dict[str, str] | None = None) -> Iterator[None]:
    """上下文管理器 — 记录指定 Histogram 指标的耗时。

    用于 HybridRetriever 内部函数的 component 级计时：

        with time_block("hybrid_search_components_seconds", {"component": "vector"}):
            results = self.store.search(...)
    """
    collector = MetricsCollector.get()
    start = time.perf_counter()
    try:
        yield
    finally:
        collector.observe(metric_name, time.perf_counter() - start, labels=labels)

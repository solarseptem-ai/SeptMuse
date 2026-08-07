"""可观测性测试 conftest — 每个测试前重置 MetricsCollector 单例。"""

import pytest

from septmuse.observability.collector import MetricsCollector


@pytest.fixture(autouse=True)
def _reset_metrics():
    """每个测试前重置 metrics 单例 + 注销所有自定义指标。"""
    MetricsCollector.reset()
    yield
    MetricsCollector.reset()

"""MetricsCollector 单例测试 — 单例、no-op、inc/dec/observe/set_gauge。"""

from prometheus_client import REGISTRY

from septmuse.observability.collector import MetricsCollector


def test_singleton():
    """get() 返回同一实例。"""
    a = MetricsCollector.get()
    b = MetricsCollector.get()
    assert a is b


def test_noop_when_disabled():
    """未 configure 时所有方法不报错。"""
    c = MetricsCollector.get()
    assert not c.enabled
    c.inc("api_requests_total", {"endpoint": "/", "method": "GET", "status": "200"})
    c.dec("api_requests_in_flight", {"endpoint": "/"})
    c.observe("embed_duration_seconds", 0.001, {"backend": "hash"})
    c.set_gauge("septmuse_uptime_seconds", 42.0)


def test_inc_after_configure():
    """configure(enabled=True) 后 inc 记录到 prometheus_client。"""
    c = MetricsCollector.get()
    c.configure(enabled=True)
    assert c.enabled
    c.inc("api_requests_total", {"endpoint": "/", "method": "GET", "status": "200"})
    c.inc("api_requests_total", {"endpoint": "/", "method": "GET", "status": "200"})

    samples = list(REGISTRY.collect())
    # prometheus_client 会自动去掉 Counter 名的 _total 后缀
    api_metric = [s for s in samples if s.name == "septmuse_api_requests"]
    assert len(api_metric) > 0
    total = sum(s.value for s in api_metric[0].samples)
    assert total >= 2.0


def test_reset_clears_singleton():
    """reset() 后 get() 返回新实例。"""
    a = MetricsCollector.get()
    a.configure(enabled=True)
    MetricsCollector.reset()
    b = MetricsCollector.get()
    assert a is not b
    assert not b.enabled

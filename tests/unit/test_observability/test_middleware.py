"""PrometheusMiddleware 测试 — 请求计数、状态码、延迟、in_flight。"""

from fastapi import FastAPI
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

from septmuse.observability.collector import MetricsCollector
from septmuse.observability.middleware import PrometheusMiddleware


def _make_app() -> FastAPI:
    app = FastAPI()

    @app.get("/test")
    async def test_endpoint():
        return {"ok": True}

    MetricsCollector.get().configure(enabled=True)
    app.add_middleware(PrometheusMiddleware)
    return app


def test_request_counted():
    """GET /test 后 api_requests_total 增加。"""
    app = _make_app()
    client = TestClient(app)
    client.get("/test")

    samples = list(REGISTRY.collect())
    # prometheus_client 会自动去掉 Counter 名的 _total 后缀
    req_metric = [s for s in samples if s.name == "septmuse_api_requests"]
    assert len(req_metric) > 0
    found = any(
        s.labels.get("endpoint") == "/test" and s.labels.get("status") == "200"
        for s in req_metric[0].samples
    )
    assert found


def test_duration_recorded():
    """GET /test 后 api_request_duration_seconds 有 bucket。"""
    app = _make_app()
    client = TestClient(app)
    client.get("/test")

    samples = list(REGISTRY.collect())
    dur_metric = [s for s in samples if s.name == "septmuse_api_request_duration_seconds"]
    assert len(dur_metric) > 0
    found = any(s.labels.get("endpoint") == "/test" for s in dur_metric[0].samples)
    assert found

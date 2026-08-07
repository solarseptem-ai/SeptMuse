"""PrometheusMiddleware — 自动记录 REST API RED 指标。

RED = Rate / Errors / Duration
- Rate: api_requests_total (Counter, endpoint+method+status)
- Duration: api_request_duration_seconds (Histogram, endpoint+method)
- In-flight: api_requests_in_flight (Gauge, endpoint)
"""

from __future__ import annotations

import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from septmuse.observability.collector import MetricsCollector


class PrometheusMiddleware(BaseHTTPMiddleware):
    """自动记录所有 HTTP 请求的 RED 指标。"""

    async def dispatch(self, request: Request, call_next) -> Response:
        collector = MetricsCollector.get()
        endpoint = request.url.path
        method = request.method

        collector.inc("api_requests_in_flight", labels={"endpoint": endpoint})

        start = time.perf_counter()
        try:
            response = await call_next(request)
            return response
        finally:
            elapsed = time.perf_counter() - start
            collector.dec("api_requests_in_flight", labels={"endpoint": endpoint})
            status = str(getattr(response, "status_code", 500))
            collector.inc(
                "api_requests_total",
                labels={"endpoint": endpoint, "method": method, "status": status},
            )
            collector.observe(
                "api_request_duration_seconds",
                elapsed,
                labels={"endpoint": endpoint, "method": method},
            )

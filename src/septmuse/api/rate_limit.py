"""Rate limiting middleware — in-memory sliding window, zero dependencies.

Per-worker in-memory rate limiting. Multi-worker deployments get limit × workers
effective ceiling (acceptable for P0 abuse prevention; Redis-backed limiting is P2).

Env:
    SEPTMUSE_RATE_LIMIT: requests per minute per client (default: 120, 0=disabled)
"""

from __future__ import annotations

import os
import time
from collections import defaultdict

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from septmuse.api.auth import EXEMPT_PATHS
from septmuse.core.logging import get_logger

logger = get_logger(__name__)

_DEFAULT_RPM = 120


def _get_rpm() -> int:
    val = os.getenv("SEPTMUSE_RATE_LIMIT", "")
    if not val:
        return _DEFAULT_RPM
    try:
        n = int(val)
    except ValueError:
        return _DEFAULT_RPM
    return max(0, n)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Sliding-window rate limiter — N requests per 60s per client identity."""

    def __init__(self, app, requests_per_minute: int = 0) -> None:
        super().__init__(app)
        self.limit = requests_per_minute or _get_rpm()
        self._window = 60.0
        self._hits: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next) -> JSONResponse | object:
        if not self.limit or request.url.path in EXEMPT_PATHS:
            return await call_next(request)

        key = (
            request.headers.get("X-API-Key")
            or request.headers.get("Authorization", "")
            or (request.client.host if request.client else "unknown")
        )

        now = time.monotonic()
        hits = [t for t in self._hits[key] if now - t < self._window]
        if len(hits) >= self.limit:
            logger.warning("rate_limited", client=key, limit=self.limit)
            return JSONResponse(
                status_code=429,
                content={"detail": f"Rate limit exceeded: {self.limit} requests per minute"},
                headers={"Retry-After": str(int(self._window))},
            )
        hits.append(now)
        self._hits[key] = hits
        return await call_next(request)


def setup_rate_limit(app) -> None:
    """Mount rate limiting middleware if SEPTMUSE_RATE_LIMIT != 0."""
    rpm = _get_rpm()
    if rpm == 0:
        logger.info("rate_limit_disabled")
        return
    app.add_middleware(RateLimitMiddleware, requests_per_minute=rpm)
    logger.info("rate_limit_enabled", rpm=rpm)

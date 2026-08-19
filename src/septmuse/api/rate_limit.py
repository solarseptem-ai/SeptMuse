#  Copyright 2026 The sonhhxg0529 Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""速率限制中间件 — 内存滑动窗口。"""

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
    """滑动窗口速率限制器。"""

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
    """挂载速率限制中间件。"""
    rpm = _get_rpm()
    if rpm == 0:
        logger.info("rate_limit_disabled")
        return
    app.add_middleware(RateLimitMiddleware, requests_per_minute=rpm)
    logger.info("rate_limit_enabled", rpm=rpm)

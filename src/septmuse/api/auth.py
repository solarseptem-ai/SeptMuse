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
"""API key 认证中间件 (架构文档 §11.3 安全)。

模式:
- 开发模式 (默认): SEPTMUSE_API_KEY 环境变量未设 → 不拦截, 日志警告一次
- 生产模式: SEPTMUSE_API_KEY 设了 → 所有 API 请求需 Authorization: Bearer <key> 或 X-API-Key: <key>

豁免路径: /docs /redoc /openapi.json /health /favicon.ico (Swagger UI 可访问)

用法:
    export SEPTMUSE_API_KEY=sk-septmuse-xxx
    septmuse serve --with-rest    # 自动启用认证
"""

from __future__ import annotations

import os

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from septmuse.core.logging import get_logger

logger = get_logger(__name__)

# 豁免路径 (Swagger UI + 健康检查)
EXEMPT_PATHS = frozenset(
    {
        "/docs",
        "/redoc",
        "/openapi.json",
        "/health",
        "/favicon.ico",
    }
)


def get_configured_api_key() -> str | None:
    """获取已配置的 API key (环境变量 SEPTMUSE_API_KEY)。未设返回 None (开发模式)。"""
    return os.getenv("SEPTMUSE_API_KEY") or None


def _extract_token(request: Request) -> str | None:
    """从 Authorization: Bearer <token> 或 X-API-Key 提取 key。"""
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:].strip()
    return request.headers.get("X-API-Key")


class ApiKeyMiddleware(BaseHTTPMiddleware):
    """API key 认证中间件。

    开发模式 (SEPTMUSE_API_KEY 未设): 不拦截, 仅启动时警告一次。
    生产模式 (SEPTMUSE_API_KEY 已设): 未认证请求返回 401。
    """

    def __init__(self, app, api_key: str | None = None) -> None:
        super().__init__(app)
        self.api_key = api_key or get_configured_api_key()
        self.dev_mode = self.api_key is None
        if self.dev_mode:
            logger.warning(
                "api_auth_dev_mode",
                reason="SEPTMUSE_API_KEY not set — all endpoints unauthenticated",
            )
        else:
            logger.info("api_auth_enabled", mode="bearer|x-api-key")

    async def dispatch(self, request: Request, call_next) -> Response:
        # 开发模式: 不拦截
        if self.dev_mode:
            return await call_next(request)

        # 豁免路径: Swagger UI + 健康检查
        if request.url.path in EXEMPT_PATHS:
            return await call_next(request)

        # 验证 token
        token = _extract_token(request)
        if not token or token != self.api_key:
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing API key. Set Authorization: Bearer <key> or X-API-Key header."},
            )

        return await call_next(request)


def setup_auth(app, api_key: str | None = None) -> None:
    """挂载 API key 认证中间件到 FastAPI app。

    Args:
        app: FastAPI 实例
        api_key: 显式指定 API key; None 时从 SEPTMUSE_API_KEY 环境变量读
    """
    app.add_middleware(ApiKeyMiddleware, api_key=api_key)

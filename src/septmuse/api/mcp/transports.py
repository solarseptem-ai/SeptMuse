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
"""MCP transports — Streamable HTTP + SSE 挂载到 FastAPI。

- SseServerTransport("/mcp/messages/") + @router.get SSE endpoint
- StreamableHTTPServerTransport(mcp_session_id=None, is_json_response_enabled=True) stateless
- capture_send 拦截 ASGI 响应避免 FastAPI double-response
- @router.api_route streamable_http endpoint
- contextvars 从 URL 路径参数 set user_id/client_name

stdio transport 见 server.run_stdio (FastMCP.run 默认)。
"""

from __future__ import annotations

import anyio
from fastapi import APIRouter, FastAPI, Request
from mcp.server.sse import SseServerTransport
from mcp.server.streamable_http import StreamableHTTPServerTransport
from starlette.responses import Response

from septmuse.api.mcp.context import client_name_var, user_id_var
from septmuse.core.logging import get_logger

logger = get_logger(__name__)


def _mount_sse(app: FastAPI, mcp_server) -> APIRouter:
    """挂载 SSE transport。"""
    sse = SseServerTransport("/mcp/sse/messages/")
    router = APIRouter(prefix="/mcp")

    @router.get("/sse/{client_name}/{user_id}")
    async def handle_sse(request: Request) -> None:
        """SSE 连接, 从路径参数取 user_id/client_name set contextvar。"""
        uid = request.path_params.get("user_id", "")
        client = request.path_params.get("client_name", "")
        user_token = user_id_var.set(uid)
        client_token = client_name_var.set(client)
        try:
            async with sse.connect_sse(request.scope, request.receive, request._send) as (  # type: ignore[attr-defined]
                read_stream,
                write_stream,
            ):
                await mcp_server._mcp_server.run(  # type: ignore[attr-defined]
                    read_stream,
                    write_stream,
                    mcp_server._mcp_server.create_initialization_options(),  # type: ignore[attr-defined]
                )
        finally:
            user_id_var.reset(user_token)
            client_name_var.reset(client_token)

    @router.post("/sse/messages/")
    async def handle_post_message(request: Request) -> dict:
        body = await request.body()

        async def receive():
            return {"type": "http.request", "body": body, "more_body": False}

        async def send(message):
            return None

        await sse.handle_post_message(request.scope, receive, send)
        return {"status": "ok"}

    app.include_router(router)
    logger.info("mcp_sse_mounted")
    return router


def _mount_http(app: FastAPI, mcp_server) -> APIRouter:
    """挂载 Streamable HTTP transport。

    使用 MCP spec 2025-03-26+ 的 Streamable HTTP, stateless 模式。
    """
    router = APIRouter(prefix="/mcp")

    @router.api_route("/http/{client_name}/{user_id}", methods=["POST", "GET", "DELETE"])
    async def handle_streamable_http(request: Request) -> Response:
        uid = request.path_params.get("user_id", "")
        client = request.path_params.get("client_name", "")
        user_token = user_id_var.set(uid)
        client_token = client_name_var.set(client)

        response_started = False
        response_status = 200
        response_headers: list[tuple[bytes, bytes]] = []
        response_body = bytearray()

        async def capture_send(message):
            nonlocal response_started, response_status
            if message["type"] == "http.response.start":
                response_started = True
                response_status = message["status"]
                response_headers.extend(message.get("headers", []))
            elif message["type"] == "http.response.body":
                response_body.extend(message.get("body", b""))

        try:
            transport = StreamableHTTPServerTransport(
                mcp_session_id=None,
                is_json_response_enabled=True,
            )
            async with anyio.create_task_group() as tg:

                async def run_server(*, task_status=anyio.TASK_STATUS_IGNORED):
                    async with transport.connect() as (read_stream, write_stream):
                        task_status.started()
                        await mcp_server._mcp_server.run(  # type: ignore[attr-defined]
                            read_stream,
                            write_stream,
                            mcp_server._mcp_server.create_initialization_options(),  # type: ignore[attr-defined]
                            stateless=True,
                        )

                await tg.start(run_server)
                await transport.handle_request(request.scope, request.receive, capture_send)
                await transport.terminate()
                tg.cancel_scope.cancel()
        finally:
            user_id_var.reset(user_token)
            client_name_var.reset(client_token)

        if not response_started:
            return Response(status_code=500, content=b"Transport did not produce a response")

        return Response(
            content=bytes(response_body),
            status_code=response_status,
            headers={k.decode(): v.decode() for k, v in response_headers},
        )

    app.include_router(router)
    logger.info("mcp_http_mounted")
    return router


def mount_sse(app: FastAPI, mcp_server) -> APIRouter:
    """挂载 SSE transport (对外入口)。"""
    return _mount_sse(app, mcp_server)


def mount_http(app: FastAPI, mcp_server) -> APIRouter:
    """挂载 Streamable HTTP transport (对外入口)。"""
    return _mount_http(app, mcp_server)

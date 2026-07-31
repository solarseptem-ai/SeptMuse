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
"""MCP server 实例 + lazy Memory + setup + run_stdio。

源码参考 mem0/openmemory/api/app/mcp_server.py:
- mcp = FastMCP("...")  实例化
- get_memory_client_safe()  lazy + 失败不崩
- setup_mcp_server(app: FastAPI)  挂载 router

SeptMuse 增量 (架构文档 §13.3):
- run_stdio()  stdio transport (mem0 openmemory 无, 仅 http/sse)
  使 `septmuse mcp` 可在 Claude Code 本地零服务运行
"""

from __future__ import annotations

import os

from fastapi import FastAPI
from mcp.server.fastmcp import FastMCP

from septmuse.core.logging import get_logger

logger = get_logger(__name__)

# MCP server 实例 (对齐 mem0: mcp = FastMCP("mem0-mcp-server"))
mcp = FastMCP("septmuse-mcp-server")

# lazy Memory 单例 (对齐 mem0 get_memory_client_safe — 失败不崩)
_mem_instance = None


def get_memory_safe():
    """获取 Memory 实例, 失败返回 None (不崩 server)。

    源码参考 mem0 mcp_server.py get_memory_client_safe。

    SeptMuse 增量: 默认用 HashEmbedder (零模型加载, 离线可用), 与 CLI _make_memory 一致。
    若 SEPTMUSE_EMBEDDER 环境变量非 "hash", 则用 Memory() 默认 (onnx/hash)。
    """
    global _mem_instance
    if _mem_instance is not None:
        return _mem_instance
    try:
        from septmuse.embedders.hash import HashEmbedder
        from septmuse.experimental import ExperimentalMemory

        embedder_env = os.getenv("SEPTMUSE_EMBEDDER", "hash").lower()
        _mem_instance = ExperimentalMemory(embedder=HashEmbedder()) if embedder_env == "hash" else ExperimentalMemory()
        logger.info("mcp_memory_ready", embedder=embedder_env)
    except Exception as e:
        logger.warning("mcp_memory_unavailable", error=str(e))
        return None
    return _mem_instance


def setup_mcp_server(app: FastAPI) -> None:
    """挂载 MCP server 到 FastAPI (http/sse transport)。

    源码参考 mem0 mcp_server.py setup_mcp_server(app): app.include_router(mcp_router)

    注意: 必须先 import tools 模块触发 @mcp.tool 装饰器注册, 否则 SSE/HTTP 模式下
    tools/list 返回空 (stdio 模式在 run_stdio 里 import, 此处补上 http/sse 路径)。
    """
    from septmuse.api.auth import setup_auth
    from septmuse.api.mcp import tools  # noqa: F401  注册 @mcp.tool 工具
    from septmuse.api.mcp.transports import mount_http, mount_sse

    setup_auth(app)
    mcp._mcp_server.name = "septmuse-mcp-server"  # type: ignore[attr-defined]
    mount_http(app, mcp)
    mount_sse(app, mcp)
    logger.info("mcp_server_mounted", transport="http+sse")


def run_stdio() -> None:
    """启动 stdio MCP server (Claude Code / Cursor 等本地编辑器用)。

    FastMCP.run() 默认 transport='stdio'。
    启动前从环境变量初始化默认 user_id (stdio 无 URL 路径)。
    """
    from septmuse.api.mcp import tools  # noqa: F401  注册 @mcp.tool 工具
    from septmuse.api.mcp.context import user_id_var

    # stdio 模式: 从环境变量初始化默认 user_id (可被工具显式参数覆盖)
    env_uid = os.getenv("SEPTMUSE_USER_ID", "")
    if env_uid:
        user_id_var.set(env_uid)
        logger.info("mcp_stdio_user_from_env", user_id=env_uid)

    logger.info("mcp_stdio_starting")
    mcp.run()


if __name__ == "__main__":
    # python -m septmuse.api.mcp.server 启动时, server.py 作为 __main__ 执行。
    # 后续 run_stdio → import tools → from septmuse.api.mcp.server import mcp 会触发
    # server.py 第二次执行 (作为 septmuse.api.mcp.server 模块), 导致 mcp 实例分裂
    # (装饰器注册到第二个实例, run() 用第一个实例, tools/list 返回空)。
    # 修复: 把 __main__ 注册到 sys.modules, 后续 import 复用同一模块同一 mcp 实例。
    import sys

    sys.modules.setdefault("septmuse.api.mcp.server", sys.modules[__name__])
    run_stdio()

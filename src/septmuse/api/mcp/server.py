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
"""MCP server 实例 + lazy Memory + setup + run_stdio。"""

from __future__ import annotations

import os

from fastapi import FastAPI
from mcp.server.fastmcp import FastMCP

from septmuse.core.logging import get_logger

logger = get_logger(__name__)

# MCP server 实例
mcp = FastMCP("septmuse-mcp-server")

# lazy Memory 单例 (失败不崩)
_mem_instance = None


def get_memory_safe():
    """获取 Memory 实例, 失败返回 None。"""
    global _mem_instance
    if _mem_instance is not None:
        return _mem_instance
    try:
        from septmuse.embedders.hash import HashEmbedder
        from septmuse.memory.main import Memory

        embedder_env = os.getenv("SEPTMUSE_EMBEDDER", "hash").lower()
        _mem_instance = Memory(embedder=HashEmbedder()) if embedder_env == "hash" else Memory()
        logger.info("mcp_memory_ready", embedder=embedder_env)
    except Exception as e:
        logger.warning("mcp_memory_unavailable", error=str(e))
        return None
    return _mem_instance


def setup_mcp_server(app: FastAPI) -> None:
    """挂载 MCP server 到 FastAPI。"""
    from septmuse.api.auth import setup_auth
    from septmuse.api.mcp import tools  # noqa: F401  注册 @mcp.tool 工具
    from septmuse.api.mcp.transports import mount_http, mount_sse

    setup_auth(app)
    mcp._mcp_server.name = "septmuse-mcp-server"  # type: ignore[attr-defined]
    mount_http(app, mcp)
    mount_sse(app, mcp)
    logger.info("mcp_server_mounted", transport="http+sse")

    @app.on_event("startup")
    def _start_evolution_scheduler() -> None:
        mem = get_memory_safe()
        if mem is not None:
            from septmuse.evolution.scheduler import EvolutionScheduler

            scheduler = EvolutionScheduler(mem)
            scheduler.start()
            app.state.evolution_scheduler = scheduler
            logger.info("evolution_scheduler_started_via_mcp_setup")

    @app.on_event("shutdown")
    def _stop_evolution_scheduler() -> None:
        scheduler = getattr(app.state, "evolution_scheduler", None)
        if scheduler is not None:
            scheduler.stop()
            logger.info("evolution_scheduler_stopped_via_mcp_setup")


def run_stdio() -> None:
    """启动 stdio MCP server。"""
    from septmuse.api.mcp import tools  # noqa: F401  注册 @mcp.tool 工具
    from septmuse.api.mcp.context import user_id_var

    # stdio 模式: 从环境变量初始化默认 user_id (可被工具显式参数覆盖)
    env_uid = os.getenv("SEPTMUSE_USER_ID", "")
    if env_uid:
        user_id_var.set(env_uid)
        logger.info("mcp_stdio_user_from_env", user_id=env_uid)

    # 启动后台演化调度器
    scheduler = None
    mem = get_memory_safe()
    if mem is not None:
        from septmuse.evolution.scheduler import EvolutionScheduler

        scheduler = EvolutionScheduler(mem)
        scheduler.start()
        logger.info("mcp_stdio_evolution_started")

    logger.info("mcp_stdio_starting")
    try:
        mcp.run()
    finally:
        if scheduler is not None:
            scheduler.stop(timeout=10)
            logger.info("mcp_stdio_evolution_stopped")


if __name__ == "__main__":
    # 防止 __main__ 与模块导入产生两份 mcp 实例
    import sys

    sys.modules.setdefault("septmuse.api.mcp.server", sys.modules[__name__])
    run_stdio()

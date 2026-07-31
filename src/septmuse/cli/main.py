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
"""septmuse 命令行入口。

子命令:
- ``init``   : 初始化本地记忆库
- ``add``    : 添加记忆
- ``search`` : 检索记忆
- ``dump``   : 导出记忆
- ``serve``  : 启动 HTTP/SSE MCP server
- ``mcp``    : 启动 stdio MCP server
- ``version``: 显示版本号

用法:
    septmuse init --user alice
    septmuse add "我喜欢 Python" --user alice
    septmuse search "喜欢什么" --user alice
    septmuse dump --user alice
    septmuse serve --port 8000 --with-rest
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _resolve_db_path(db_path: str | None) -> str:
    """解析 SQLite 路径: 参数 > 环境变量 > 默认 ~/.septmuse/septmuse.db。"""
    if db_path:
        return db_path
    env = os.getenv("SEPTMUSE_DB_PATH")
    if env:
        return env
    return str(Path.home() / ".septmuse" / "septmuse.db")


def _make_memory(db_path: str | None):
    """创建 Memory 实例 (默认 HashEmbedder, 零模型加载)。"""
    from septmuse.configs.defaults import MemoryConfig
    from septmuse.embedders.hash import HashEmbedder
    from septmuse.experimental import ExperimentalMemory

    path = _resolve_db_path(db_path)
    return ExperimentalMemory(config=MemoryConfig(db_path=path), embedder=HashEmbedder())


def _build_parser() -> argparse.ArgumentParser:
    """构建 argparse 解析器。"""
    parser = argparse.ArgumentParser(
        prog="septmuse",
        description="SeptMuse — agent 记忆系统",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # init
    p_init = sub.add_parser("init", help="初始化本地记忆库")
    p_init.add_argument("--user", default=os.getenv("SEPTMUSE_USER_ID", "default"), help="用户 ID")
    p_init.add_argument("--db-path", default=None, help="SQLite 路径 (默认 ~/.septmuse/septmuse.db)")
    p_init.set_defaults(func=_cmd_init)

    # add (Task 3 实现)
    p_add = sub.add_parser("add", help="添加记忆")
    p_add.add_argument("content", help="记忆内容")
    p_add.add_argument(
        "--user", "--user-id", dest="user", default=os.getenv("SEPTMUSE_USER_ID", "default"), help="用户 ID"
    )
    p_add.add_argument(
        "--type",
        choices=["verbatim", "semantic", "episodic", "procedural"],
        default="verbatim",
        help="记忆类型",
    )
    p_add.add_argument("--agent", default=None, help="agent ID")
    p_add.add_argument("--session", default=None, help="会话 ID (对齐 mem0 run_id)")
    p_add.add_argument("--valid-at", default=None, help="事实开始为真的时间 (ISO 8601)")
    p_add.add_argument("--db-path", default=None, help="SQLite 路径")
    p_add.set_defaults(func=_cmd_add)

    # invalidate (bitemporal)
    p_inv = sub.add_parser("invalidate", help="手动标记事实不再为真")
    p_inv.add_argument("memory_id", help="记忆 ID")
    p_inv.add_argument("--invalid-at", default=None, help="失效时间 (ISO 8601)")
    p_inv.add_argument("--db-path", default=None, help="SQLite 路径")
    p_inv.set_defaults(func=_cmd_invalidate)

    # search (Task 4 实现)
    p_search = sub.add_parser("search", help="检索记忆")
    p_search.add_argument("query", help="查询文本")
    p_search.add_argument(
        "--user", "--user-id", dest="user", default=os.getenv("SEPTMUSE_USER_ID", "default"), help="用户 ID"
    )
    p_search.add_argument("--top-k", type=int, default=5, help="返回数")
    p_search.add_argument("--threshold", type=float, default=0.1, help="相似阈值")
    p_search.add_argument("--session-id", default=None, help="会话 ID（仅搜该会话的记忆）")
    p_search.add_argument("--db-path", default=None, help="SQLite 路径")
    p_search.set_defaults(func=_cmd_search)

    # dump (Task 5 实现)
    p_dump = sub.add_parser("dump", help="导出记忆")
    p_dump.add_argument("--user", default=os.getenv("SEPTMUSE_USER_ID", "default"), help="用户 ID")
    p_dump.add_argument("--session-id", default=None, help="会话 ID（仅导出该会话的记忆）")
    p_dump.add_argument("--format", choices=["json", "markdown"], default="json", help="输出格式")
    p_dump.add_argument("--output", default=None, help="输出文件路径 (默认 stdout)")
    p_dump.add_argument("--db-path", default=None, help="SQLite 路径")
    p_dump.set_defaults(func=_cmd_dump)

    # update
    p_update = sub.add_parser("update", help="更新记忆")
    p_update.add_argument("memory_id", help="记忆 ID")
    p_update.add_argument("content", help="新内容")
    p_update.add_argument("--user", default=os.getenv("SEPTMUSE_USER_ID", "default"), help="用户 ID")
    p_update.add_argument("--db-path", default=None, help="SQLite 路径")
    p_update.set_defaults(func=_cmd_update)

    # history
    p_history = sub.add_parser("history", help="查看记忆变更历史")
    p_history.add_argument("memory_id", help="记忆 ID")
    p_history.add_argument("--db-path", default=None, help="SQLite 路径")
    p_history.set_defaults(func=_cmd_history)

    # block
    p_block = sub.add_parser("block", help="工作记忆 Block 操作")
    block_sub = p_block.add_subparsers(dest="block_cmd", required=True)
    p_block_set = block_sub.add_parser("set", help="设置 block value")
    p_block_set.add_argument("agent_id", help="agent ID")
    p_block_set.add_argument("label", help="block 标签")
    p_block_set.add_argument("value", help="新内容")
    p_block_set.add_argument("--db-path", default=None, help="SQLite 路径")
    p_block_set.set_defaults(func=_cmd_block_set)
    p_block_list = block_sub.add_parser("list", help="列出 block")
    p_block_list.add_argument("agent_id", help="agent ID")
    p_block_list.add_argument("--db-path", default=None, help="SQLite 路径")
    p_block_list.set_defaults(func=_cmd_block_list)

    # serve (Task 6 实现)
    p_serve = sub.add_parser("serve", help="启动 HTTP/SSE MCP server")
    p_serve.add_argument("--host", default="127.0.0.1", help="绑定地址")
    p_serve.add_argument("--port", type=int, default=8000, help="端口")
    p_serve.add_argument("--with-rest", action="store_true", help="额外挂载 REST API")
    p_serve.add_argument("--db-path", default=None, help="SQLite 路径")
    p_serve.set_defaults(func=_cmd_serve)

    # mcp (已有, 保留)
    p_mcp = sub.add_parser("mcp", help="启动 stdio MCP server")
    p_mcp.set_defaults(func=_cmd_mcp)

    # version
    p_ver = sub.add_parser("version", help="显示版本号")
    p_ver.set_defaults(func=_cmd_version)

    # backends — 列出所有能力后端及可用性
    p_backends = sub.add_parser("backends", help="列出所有能力后端及可用性")
    p_backends.set_defaults(func=_cmd_backends)

    # config — 配置自省
    p_config = sub.add_parser("config", help="配置自省")
    config_sub = p_config.add_subparsers(dest="config_action", required=True)
    p_config_show = config_sub.add_parser("show", help="显示当前生效配置")
    p_config_show.set_defaults(func=_cmd_config_show)

    # migrate — 运行数据库迁移
    p_migrate = sub.add_parser("migrate", help="运行数据库迁移")
    p_migrate.add_argument("--db-path", default=None, help="SQLite 路径 (默认 ~/.septmuse/septmuse.db)")
    p_migrate.set_defaults(func=_cmd_migrate)

    return parser


def _cmd_init(args: argparse.Namespace) -> int:
    """初始化本地记忆库。"""
    db_path = _resolve_db_path(args.db_path)
    db = Path(db_path)
    db.parent.mkdir(parents=True, exist_ok=True)

    m = _make_memory(args.db_path)
    m.add(f"septmuse initialized for user {args.user}", user_id=args.user)
    print(f"initialized: {db}")
    return 0


def _cmd_add(args: argparse.Namespace) -> int:
    """添加记忆 (verbatim/semantic/episodic/procedural)。"""
    m = _make_memory(args.db_path)

    if args.type == "verbatim":
        result = m.add(
            args.content,
            user_id=args.user,
            agent_id=args.agent,
            session_id=args.session,
            infer=False,
            valid_at=args.valid_at,
        )
        mid = result["results"][0]["id"] if result.get("results") else None
        print(json.dumps({"memory_id": mid, "type": "verbatim"}, ensure_ascii=False))
    elif args.type == "semantic":
        parts = args.content.split(None, 2)
        subject = parts[0] if len(parts) > 0 else args.content
        predicate = parts[1] if len(parts) > 1 else "is"
        obj = parts[2] if len(parts) > 2 else ""
        result = m.add_fact(subject, predicate, obj, user_id=args.user)
        print(json.dumps({"id": result["id"], "type": "semantic"}, ensure_ascii=False))
    elif args.type == "episodic":
        result = m.add_episode(args.content, user_id=args.user)
        print(json.dumps({"id": result["id"], "type": "episodic"}, ensure_ascii=False))
    elif args.type == "procedural":
        result = m.add_rule(args.content, user_id=args.user)
        print(json.dumps({"id": result["id"], "type": "procedural"}, ensure_ascii=False))
    return 0


def _cmd_invalidate(args: argparse.Namespace) -> int:
    """手动标记事实不再为真 (设置 invalid_at + expired_at, 不删除记忆)。"""
    m = _make_memory(args.db_path)
    result = m.invalidate(args.memory_id, invalid_at=args.invalid_at)
    print(json.dumps(result, ensure_ascii=False))
    return 0


def _cmd_search(args: argparse.Namespace) -> int:
    """检索记忆, JSON 数组输出。"""
    m = _make_memory(args.db_path)
    results = m.search(
        args.query,
        user_id=args.user,
        session_id=args.session_id,
        top_k=args.top_k,
        threshold=args.threshold,
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


def _cmd_dump(args: argparse.Namespace) -> int:
    """导出记忆 (json/markdown, stdout 或文件)。"""
    m = _make_memory(args.db_path)
    data = m.get_all(user_id=args.user, session_id=args.session_id)

    if args.format == "markdown":
        lines: list[str] = []
        for item in data.get("results", []):
            mid = item.get("id", "?")
            memory = item.get("memory", item.get("rule", item.get("content", str(item))))
            lines.append(f"- **{mid}**: {memory}")
        output = "\n".join(lines) if lines else "(无记忆)"
    else:
        output = json.dumps(data, ensure_ascii=False, indent=2)

    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
    else:
        print(output)
    return 0


def _cmd_update(args: argparse.Namespace) -> int:
    """更新记忆。"""
    m = _make_memory(args.db_path)
    result = m.update(args.memory_id, args.content)
    print(json.dumps(result, ensure_ascii=False))
    return 0


def _cmd_history(args: argparse.Namespace) -> int:
    """查看记忆变更历史。"""
    m = _make_memory(args.db_path)
    history = m.get_history(args.memory_id)
    print(json.dumps(history, ensure_ascii=False, default=str, indent=2))
    return 0


def _cmd_block_set(args: argparse.Namespace) -> int:
    """设置 block value。"""
    m = _make_memory(args.db_path)
    result = m.update_block(args.agent_id, args.label, args.value)
    print(json.dumps(result, ensure_ascii=False))
    return 0


def _cmd_block_list(args: argparse.Namespace) -> int:
    """列出 block。"""
    m = _make_memory(args.db_path)
    blocks = m.get_blocks(args.agent_id)
    print(json.dumps(blocks, ensure_ascii=False, indent=2, default=str))
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    """启动 HTTP/SSE MCP server (uvicorn + FastAPI)。"""
    try:
        import uvicorn
    except ImportError:
        print(
            "septmuse serve: uvicorn 未安装。请运行: pip install uvicorn",
            file=sys.stderr,
        )
        return 1

    from fastapi import FastAPI

    from septmuse.api.mcp.server import setup_mcp_server

    app = FastAPI(title="SeptMuse MCP Server")
    setup_mcp_server(app)

    if args.with_rest:
        from septmuse.api.rest import register_routes

        m = _make_memory(args.db_path)
        register_routes(app, m)

    print(f"septmuse serve: http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def _cmd_mcp(args: argparse.Namespace) -> int:
    """启动 stdio MCP server。"""
    try:
        from septmuse.api.mcp.server import run_stdio
    except ImportError:
        print("septmuse mcp: MCP server 尚未实现。", file=sys.stderr)
        return 1
    run_stdio()
    return 0


def _cmd_version(args: argparse.Namespace) -> int:
    """显示版本号。"""
    from septmuse import __version__

    print(__version__)
    return 0


def _cmd_backends(args: argparse.Namespace) -> int:
    """列出所有能力后端及可用性。"""
    from septmuse.services.providers import ALL_PROVIDERS

    for capability, provider in ALL_PROVIDERS.items():
        parts = []
        for backend in provider.list_backends():
            mark = "+" if provider.is_available(backend) else "-"
            parts.append(f"{backend}[{mark}]")
        print(f"{capability:20s} {' '.join(parts)}")
    return 0


def _cmd_config_show(args: argparse.Namespace) -> int:
    """显示当前生效配置。"""
    from septmuse.configs import default_config

    config = default_config()
    capabilities = [
        ("embedder", config.embedder.backend),
        ("vector_store", config.vector_store.backend),
        ("llm", config.llm.backend if config.llm else "null"),
        ("reranker", config.reranker.backend),
        ("entity_extractor", config.entity_extractor.backend),
        ("keyword_index", config.keyword_index.backend),
        ("graph_store", config.graph_store.backend),
        ("search_recipe", config.search_recipe),
        ("infer", str(config.infer)),
    ]
    for name, value in capabilities:
        print(f"{name:20s} value={value}")
    return 0


def _cmd_migrate(args: argparse.Namespace) -> int:
    """运行数据库迁移。"""
    import sqlite3

    from septmuse.storage.migrations import MIGRATIONS
    from septmuse.storage.migrations.runner import MigrationRunner

    db_path = _resolve_db_path(args.db_path)
    conn = sqlite3.connect(db_path)
    try:
        runner = MigrationRunner(conn, "sqlite")
        applied = runner.run()
        if applied:
            print(f"已应用 {len(applied)} 个迁移:")
            for v in applied:
                desc = next((m.description for m in MIGRATIONS if m.version == v), "")
                print(f"  {v} - {desc}")
        else:
            print("所有迁移已应用，无需操作")
        print(f"schema_version: {len(MIGRATIONS)} migrations total")
    finally:
        conn.close()
    return 0


def main() -> int:
    """septmuse CLI 主入口。"""
    parser = _build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

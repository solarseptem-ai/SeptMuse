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
"""septmuse 命令行入口。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


def _resolve_db_path(db_path: str | None) -> str:
    """解析 SQLite 路径: 参数 > 环境变量 > 默认 ~/.septmuse/septmuse.db。"""
    if db_path:
        return db_path
    env = os.getenv("SEPTMUSE_DB_PATH")
    if env:
        return env
    return str(Path.home() / ".septmuse" / "septmuse.db")


_ops_cache: dict[str, Any] = {}


def _get_ops(db_path: str | None) -> Any:
    """获取 MemoryOperations 实例 (按 db_path 缓存, 避免重复创建连接)。"""
    from septmuse.api.operations import MemoryOperations
    path = _resolve_db_path(db_path)
    if path not in _ops_cache:
        ops = MemoryOperations()
        ops._db_path = path
        _ops_cache[path] = ops
    return _ops_cache[path]


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

    # add
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
    p_add.add_argument("--session", default=None, help="会话 ID")
    p_add.add_argument("--valid-at", default=None, help="事实开始为真的时间 (ISO 8601)")
    p_add.add_argument("--db-path", default=None, help="SQLite 路径")
    p_add.set_defaults(func=_cmd_add)

    # invalidate (bitemporal)
    p_inv = sub.add_parser("invalidate", help="手动标记事实不再为真")
    p_inv.add_argument("memory_id", help="记忆 ID")
    p_inv.add_argument("--invalid-at", default=None, help="失效时间 (ISO 8601)")
    p_inv.add_argument("--db-path", default=None, help="SQLite 路径")
    p_inv.set_defaults(func=_cmd_invalidate)

    # search
    p_search = sub.add_parser("search", help="检索记忆")
    p_search.add_argument("query", help="查询文本")
    p_search.add_argument(
        "--user", "--user-id", dest="user", default=os.getenv("SEPTMUSE_USER_ID", "default"), help="用户 ID"
    )
    p_search.add_argument("--top-k", type=int, default=5, help="返回数")
    p_search.add_argument("--threshold", type=float, default=0.1, help="相似阈值")
    p_search.add_argument("--session-id", default=None, help="会话 ID（仅搜该会话的记忆）")
    p_search.add_argument("--filters", default=None, help='字段过滤 JSON (如 \'{"agent_id":"a1"}\')')
    p_search.add_argument(
        "--reranker", choices=["noop", "mmr", "cross_encoder", "llm"], default=None, help="重排器"
    )
    p_search.add_argument("--explain", action="store_true", help="返回 score_details")
    p_search.add_argument("--db-path", default=None, help="SQLite 路径")
    p_search.set_defaults(func=_cmd_search)

    # dump
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

    # get
    p_get = sub.add_parser("get", help="获取单条记忆")
    p_get.add_argument("memory_id", help="记忆 ID")
    p_get.add_argument("--db-path", default=None, help="SQLite 路径")
    p_get.set_defaults(func=_cmd_get)

    # delete
    p_del = sub.add_parser("delete", help="删除记忆")
    p_del.add_argument("memory_id", help="记忆 ID")
    p_del.add_argument("--db-path", default=None, help="SQLite 路径")
    p_del.set_defaults(func=_cmd_delete)

    # history
    p_history = sub.add_parser("history", help="查看记忆变更历史")
    p_history.add_argument("memory_id", help="记忆 ID")
    p_history.add_argument("--db-path", default=None, help="SQLite 路径")
    p_history.set_defaults(func=_cmd_history)

    # remember — V2 写入编排
    p_remember = sub.add_parser("remember", help="V2 记忆写入编排")
    p_remember.add_argument("text", help="待记忆文本")
    p_remember.add_argument("--user", default=os.getenv("SEPTMUSE_USER_ID", "default"), help="用户 ID")
    p_remember.add_argument("--agent", default=None, help="agent ID")
    p_remember.add_argument("--session", default=None, help="会话 ID")
    p_remember.add_argument("--db-path", default=None, help="SQLite 路径")
    p_remember.set_defaults(func=_cmd_remember)

    # recall — V2 召回编排
    p_recall = sub.add_parser("recall", help="V2 记忆召回编排")
    p_recall.add_argument("query", help="召回查询")
    p_recall.add_argument("--user", default=os.getenv("SEPTMUSE_USER_ID", "default"), help="用户 ID")
    p_recall.add_argument("--session", default=None, help="会话 ID")
    p_recall.add_argument("--top-k", type=int, default=5, help="返回记忆条数")
    p_recall.add_argument("--auto-compress", action="store_true", help="先压缩上下文")
    p_recall.add_argument("--recipe", default=None, help="检索配方名称")
    p_recall.add_argument("--db-path", default=None, help="SQLite 路径")
    p_recall.set_defaults(func=_cmd_recall)

    # improve — V2 演化编排
    p_improve = sub.add_parser("improve", help="V2 记忆演化编排")
    p_improve.add_argument("--user", default=os.getenv("SEPTMUSE_USER_ID", "default"), help="用户 ID")
    p_improve.add_argument("--limit", type=int, default=20, help="处理记忆数量上限")
    p_improve.add_argument("--db-path", default=None, help="SQLite 路径")
    p_improve.set_defaults(func=_cmd_improve)

    # forget — V2 遗忘编排
    p_forget = sub.add_parser("forget", help="V2 记忆遗忘编排")
    p_forget.add_argument("memory_id", help="要遗忘的记忆 ID")
    p_forget.add_argument("--user", default=os.getenv("SEPTMUSE_USER_ID", "default"), help="用户 ID")
    p_forget.add_argument("--db-path", default=None, help="SQLite 路径")
    p_forget.set_defaults(func=_cmd_forget)

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

    # serve
    p_serve = sub.add_parser("serve", help="启动 HTTP/SSE MCP server")
    p_serve.add_argument("--host", default="127.0.0.1", help="绑定地址")
    p_serve.add_argument("--port", type=int, default=8000, help="端口")
    p_serve.add_argument("--with-rest", action="store_true", help="额外挂载 REST API")
    p_serve.add_argument("--workers", type=int, default=1, help="worker 进程数 (多 worker 需配合 PostgreSQL)")
    p_serve.add_argument("--db-path", default=None, help="SQLite 路径")
    p_serve.set_defaults(func=_cmd_serve)

    # mcp
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

    # cognify — 构建知识图谱
    p_cognify = sub.add_parser("cognify", help="构建知识图谱 (存记忆→抽三元组→存实体/关系→建链接)")
    p_cognify.add_argument("text", help="输入文本")
    p_cognify.add_argument("--user", default=os.getenv("SEPTMUSE_USER_ID", "default"), help="用户 ID")
    p_cognify.add_argument("--db-path", default=None, help="SQLite 路径")
    p_cognify.set_defaults(func=_cmd_cognify)

    # search-graph — BFS 图遍历检索
    p_sg = sub.add_parser("search-graph", help="BFS 图遍历检索")
    p_sg.add_argument("memory_id", help="种子记忆 ID")
    p_sg.add_argument("--max-depth", type=int, default=3, help="BFS 最大深度")
    p_sg.add_argument("--user", default=os.getenv("SEPTMUSE_USER_ID", "default"), help="用户 ID")
    p_sg.add_argument("--db-path", default=None, help="SQLite 路径")
    p_sg.set_defaults(func=_cmd_search_graph)

    # search-interval — 时态区间查询
    p_si = sub.add_parser("search-interval", help="时间区间查询 ([start, end) 内为真的相关记忆)")
    p_si.add_argument("--start", required=True, help="起始时间 (ISO 8601)")
    p_si.add_argument("--end", required=True, help="结束时间 (ISO 8601)")
    p_si.add_argument("query", help="查询文本")
    p_si.add_argument("--user", default=os.getenv("SEPTMUSE_USER_ID", "default"), help="用户 ID")
    p_si.add_argument("--db-path", default=None, help="SQLite 路径")
    p_si.set_defaults(func=_cmd_search_interval)

    # reflect — 会话蒸馏
    p_reflect = sub.add_parser("reflect", help="会话蒸馏 (提取教训→procedural rules)")
    p_reflect.add_argument("--user", default=os.getenv("SEPTMUSE_USER_ID", "default"), help="用户 ID")
    p_reflect.add_argument("--limit", type=int, default=50, help="回顾记忆数")
    p_reflect.add_argument("--db-path", default=None, help="SQLite 路径")
    p_reflect.set_defaults(func=_cmd_reflect)

    # entities — 搜索实体
    p_entities = sub.add_parser("entities", help="搜索实体")
    p_entities.add_argument("query", help="查询文本")
    p_entities.add_argument("--user-id", dest="user_id", default=os.getenv("SEPTMUSE_USER_ID", "default"), help="用户 ID")
    p_entities.add_argument("--top-k", type=int, default=5, help="返回数")
    p_entities.add_argument("--db-path", default=None, help="SQLite 路径")
    p_entities.set_defaults(func=_cmd_entities)

    # entity-list — 列出实体
    p_el = sub.add_parser("entity-list", help="列出用户全部实体")
    p_el.add_argument("--user-id", dest="user_id", default=os.getenv("SEPTMUSE_USER_ID", "default"), help="用户 ID")
    p_el.add_argument("--entity-type", dest="entity_type", default=None, help="实体类型过滤")
    p_el.add_argument("--limit", type=int, default=100, help="返回数")
    p_el.add_argument("--db-path", default=None, help="SQLite 路径")
    p_el.set_defaults(func=_cmd_entity_list)

    # access-logs — 访问日志
    p_al = sub.add_parser("access-logs", help="查询记忆访问日志")
    p_al.add_argument("memory_id", help="记忆 ID")
    p_al.add_argument("--limit", type=int, default=100, help="返回数")
    p_al.add_argument("--db-path", default=None, help="SQLite 路径")
    p_al.set_defaults(func=_cmd_access_logs)

    # backup — SQLite 备份
    p_backup = sub.add_parser("backup", help="备份 SQLite 数据库 (VACUUM INTO)")
    p_backup.add_argument("--output", default=None, help="输出文件路径 (默认 <db_path>.backup-<timestamp>.db)")
    p_backup.add_argument("--db-path", default=None, help="SQLite 路径")
    p_backup.set_defaults(func=_cmd_backup)

    return parser


def _cmd_init(args: argparse.Namespace) -> int:
    """初始化本地记忆库。"""
    db_path = _resolve_db_path(args.db_path)
    db = Path(db_path)
    db.parent.mkdir(parents=True, exist_ok=True)

    ops = _get_ops(args.db_path)
    ops.add(f"septmuse initialized for user {args.user}", user_id=args.user)
    print(f"initialized: {db}")
    return 0


def _cmd_add(args: argparse.Namespace) -> int:
    """添加记忆 (verbatim/semantic/episodic/procedural)。"""
    ops = _get_ops(args.db_path)

    if args.type == "verbatim":
        result = ops.add(
            args.content,
            user_id=args.user,
            agent_id=args.agent,
            session_id=args.session,
            infer=False,
            valid_at=args.valid_at,
        )
        mid = result["results"][0]["id"] if result.get("results") else None
        print(json.dumps({"memory_id": mid, "type": "verbatim"}, ensure_ascii=False))
    else:
        result = ops.add_typed(args.content, args.type, user_id=args.user)
        if "id" in result:
            print(json.dumps({"id": result["id"], "type": args.type}, ensure_ascii=False))
        else:
            print(json.dumps(result, ensure_ascii=False))
    return 0


def _cmd_invalidate(args: argparse.Namespace) -> int:
    """手动标记事实不再为真 (设置 invalid_at + expired_at, 不删除记忆)。"""
    ops = _get_ops(args.db_path)
    result = ops.invalidate(args.memory_id, invalid_at=args.invalid_at)
    print(json.dumps(result, ensure_ascii=False))
    return 0


def _cmd_search(args: argparse.Namespace) -> int:
    """检索记忆, JSON 数组输出。"""
    ops = _get_ops(args.db_path)
    filters = json.loads(args.filters) if args.filters else None
    reranker = args.reranker or None
    results = ops.search(
        args.query,
        user_id=args.user,
        session_id=args.session_id,
        top_k=args.top_k,
        threshold=args.threshold,
        filters=filters,
        reranker=reranker,
        explain=args.explain,
    )
    print(json.dumps(results, ensure_ascii=False, indent=2, default=str))
    return 0


def _cmd_dump(args: argparse.Namespace) -> int:
    """导出记忆 (json/markdown, stdout 或文件)。"""
    ops = _get_ops(args.db_path)
    data = ops.get_all(user_id=args.user, session_id=args.session_id)

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
    ops = _get_ops(args.db_path)
    result = ops.update(args.memory_id, args.content)
    print(json.dumps(result, ensure_ascii=False))
    return 0


def _cmd_history(args: argparse.Namespace) -> int:
    """查看记忆变更历史。"""
    ops = _get_ops(args.db_path)
    history = ops.get_history(args.memory_id)
    print(json.dumps(history, ensure_ascii=False, default=str, indent=2))
    return 0


def _cmd_block_set(args: argparse.Namespace) -> int:
    """设置 block value。"""
    ops = _get_ops(args.db_path)
    result = ops.update_block(args.agent_id, args.label, args.value)
    print(json.dumps(result, ensure_ascii=False))
    return 0


def _cmd_block_list(args: argparse.Namespace) -> int:
    """列出 block。"""
    ops = _get_ops(args.db_path)
    blocks = ops.get_blocks(args.agent_id)
    print(json.dumps(blocks, ensure_ascii=False, indent=2, default=str))
    return 0


def _cmd_backup(args: argparse.Namespace) -> int:
    """备份 SQLite 数据库 (VACUUM INTO 一致性快照)。"""
    db_path = _resolve_db_path(args.db_path)

    if os.getenv("SEPTMUSE_DB_URL") and not db_path.endswith(".db"):
        print("backup 命令仅支持 SQLite。PostgreSQL 请用 pg_dump，MySQL 请用 mysqldump。", file=sys.stderr)
        return 1

    if args.output:
        output = args.output
    else:
        from datetime import datetime

        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        output = f"{db_path}.backup-{ts}.db"

    from sqlalchemy import create_engine, text

    engine = create_engine(f"sqlite:///{db_path}")
    try:
        with engine.connect() as conn:
            conn.execute(text(f"VACUUM INTO '{output}'"))
        print(f"backup: {output}")
    finally:
        engine.dispose()
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

    workers = max(1, args.workers)

    if workers > 1:
        if not os.getenv("SEPTMUSE_DB_URL"):
            print(
                "WARNING: SQLite + 多 worker 可能导致写冲突。生产环境请用 PostgreSQL (SEPTMUSE_DB_URL)。",
                file=sys.stderr,
            )
        if not args.with_rest:
            print("WARNING: 多 worker 仅支持 --with-rest 模式，回退到 1 worker。", file=sys.stderr)
            workers = 1

    if workers > 1:
        print(f"septmuse serve: http://{args.host}:{args.port} ({workers} workers)")
        uvicorn.run("septmuse.api.rest:app", host=args.host, port=args.port, workers=workers)
        return 0

    app = FastAPI(title="SeptMuse MCP Server")
    setup_mcp_server(app)

    from septmuse.api.rate_limit import setup_rate_limit

    setup_rate_limit(app)

    if args.with_rest:
        from septmuse.api.rest import register_routes

        ops = _get_ops(args.db_path)
        register_routes(app, ops.memory)

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
    from sqlalchemy import create_engine

    from septmuse.storage.migrations import MIGRATIONS
    from septmuse.storage.migrations.runner import MigrationRunner

    db_path = _resolve_db_path(args.db_path)
    engine = create_engine(f"sqlite:///{db_path}")
    try:
        runner = MigrationRunner(engine)
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
        engine.dispose()
    return 0


def _cmd_cognify(args: argparse.Namespace) -> int:
    """构建知识图谱 (存记忆→抽三元组→存实体/关系→建链接)。"""
    ops = _get_ops(args.db_path)
    result = ops.cognify(args.text, user_id=args.user)
    print(json.dumps(result, ensure_ascii=False, default=str, indent=2))
    return 0


def _cmd_search_graph(args: argparse.Namespace) -> int:
    """BFS 图遍历检索。"""
    ops = _get_ops(args.db_path)
    results = ops.search_graph(args.memory_id, max_depth=args.max_depth)
    print(json.dumps(results, ensure_ascii=False, default=str, indent=2))
    return 0


def _cmd_search_interval(args: argparse.Namespace) -> int:
    """时间区间查询 ([start, end) 内为真的相关记忆)。"""
    ops = _get_ops(args.db_path)
    results = ops.search_interval(args.start, args.end, args.query, user_id=args.user)
    print(json.dumps(results, ensure_ascii=False, default=str, indent=2))
    return 0


def _cmd_reflect(args: argparse.Namespace) -> int:
    """会话蒸馏 (提取教训→procedural rules)。"""
    ops = _get_ops(args.db_path)
    result = ops.reflect(user_id=args.user, limit=args.limit)
    print(json.dumps(result, ensure_ascii=False, default=str, indent=2))
    return 0


def _cmd_entities(args: argparse.Namespace) -> int:
    """搜索实体。"""
    ops = _get_ops(args.db_path)
    results = ops.search_entities(args.query, user_id=args.user_id, top_k=args.top_k)
    print(json.dumps(results, ensure_ascii=False, default=str, indent=2))
    return 0


def _cmd_entity_list(args: argparse.Namespace) -> int:
    """列出用户全部实体。"""
    ops = _get_ops(args.db_path)
    results = ops.list_entities(user_id=args.user_id, entity_type=args.entity_type, limit=args.limit)
    print(json.dumps(results, ensure_ascii=False, default=str, indent=2))
    return 0


def _cmd_access_logs(args: argparse.Namespace) -> int:
    """查询记忆访问日志。"""
    ops = _get_ops(args.db_path)
    results = ops.get_access_logs(args.memory_id, limit=args.limit)
    print(json.dumps(results, ensure_ascii=False, default=str, indent=2))
    return 0


def _cmd_get(args: argparse.Namespace) -> int:
    """获取单条记忆。"""
    ops = _get_ops(args.db_path)
    result = ops.get(args.memory_id)
    if result is None:
        print(json.dumps({"error": "not found"}, ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False, default=str, indent=2))
    return 0


def _cmd_delete(args: argparse.Namespace) -> int:
    """删除记忆。"""
    ops = _get_ops(args.db_path)
    result = ops.delete(args.memory_id)
    print(json.dumps(result, ensure_ascii=False))
    return 0


def _cmd_remember(args: argparse.Namespace) -> int:
    """V2 记忆写入编排。"""
    ops = _get_ops(args.db_path)
    result = ops.remember(
        args.text,
        user_id=args.user,
        agent_id=args.agent,
        session_id=args.session,
    )
    print(json.dumps(result, ensure_ascii=False, default=str, indent=2))
    return 0


def _cmd_recall(args: argparse.Namespace) -> int:
    """V2 记忆召回编排。"""
    ops = _get_ops(args.db_path)
    result = ops.recall(
        args.query,
        user_id=args.user,
        session_id=args.session,
        top_k=args.top_k,
        auto_compress=args.auto_compress,
        recipe=args.recipe,
    )
    print(json.dumps(result, ensure_ascii=False, default=str, indent=2))
    return 0


def _cmd_improve(args: argparse.Namespace) -> int:
    """V2 记忆演化编排。"""
    ops = _get_ops(args.db_path)
    result = ops.improve(user_id=args.user, limit=args.limit)
    print(json.dumps(result, ensure_ascii=False, default=str, indent=2))
    return 0


def _cmd_forget(args: argparse.Namespace) -> int:
    """V2 记忆遗忘编排。"""
    ops = _get_ops(args.db_path)
    result = ops.forget(args.memory_id, user_id=args.user)
    print(json.dumps(result, ensure_ascii=False, default=str, indent=2))
    return 0


def main() -> int:
    """septmuse CLI 主入口。"""
    parser = _build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

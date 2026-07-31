# SeptMuse CLI 5 命令实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现 `septmuse init/add/search/dump/serve` 5 个 CLI 命令，让 `pip install septmuse` 后命令行开箱即用。

**Architecture:** 重写 `cli/main.py` 为 argparse 子命令 dispatch（零新依赖）。`serve` 延迟 import uvicorn + 复用 `setup_mcp_server`。`serve --with-rest` 需重构 `api/rest/__init__.py` 抽取 `register_routes(app, memory)`。所有命令默认用 `HashEmbedder`（零模型加载）。

**Tech Stack:** Python 3.10+, argparse (stdlib), FastAPI, uvicorn (optional extra)

## Global Constraints

- ruff line-length 120，lint 规则 `E F I W UP B SIM RUF`（忽略 E501 RUF002 RUF003）
- PYTHONPATH=src 运行 pytest（包未 pip install -e .）
- CLI 默认用 `HashEmbedder`，不触发 sentence-transformers 模型加载
- 项目非 git repo，无 commit 步骤，用 "验证" 替代
- uvicorn 不进核心依赖，延迟 import + 友好报错
- 测试用 `tmp_path` + `monkeypatch` 设置 `sys.argv`，直接调 `main()` + `capsys` 验证
- Memory facade 签名（已确认）：
  - `add(messages, *, user_id, agent_id=None, metadata=None, infer=None) → {"results": [{"id","memory","event":"ADD"}], "relations": []}`
  - `search(query, *, user_id, top_k=None, threshold=None) → list[{"id","memory","score","metadata","created_at"}]`
  - `get_all(*, user_id) → {"results": [...]}`
  - `add_fact(subject, predicate, object, *, user_id, ...) → {"id","triple","event":"ADD"}`
  - `add_episode(content, *, user_id, ...) → {"id","event_type","reference_time"}`
  - `add_rule(rule, *, user_id, namespace="default", ...) → {"id","rule","event":"ADD"}`

---

### Task 1: 重构 api/rest 抽取 register_routes

**Files:**
- Modify: `src/septmuse/api/rest/__init__.py`
- Test: `tests/unit/test_rbac_rest_openai.py`（已有，验证不退化）

**Interfaces:**
- Produces: `register_routes(app: FastAPI, memory: Memory) -> None` — 把 `create_app` 内的路由注册逻辑抽取为独立函数，供 CLI `serve --with-rest` 调用

**Why:** `serve --with-rest` 需要把 REST 路由挂到同一个 FastAPI app（同时挂 MCP + REST）。当前 `create_app` 把路由定义耦合在工厂内部，无法复用。

- [ ] **Step 1: 读当前 create_app 确认要抽取的行范围**

Run: `Get-Content src/septmuse/api/rest/__init__.py | Select-Object -First 200`
确认：line 113-193 是 8 个路由定义，line 107 是 `app.state.memory = memory`。

- [ ] **Step 2: 重构 — 抽取 register_routes**

把 `create_app` 内的 `app.state.memory = memory` + 全部 `@app.post/@app.get/@app.delete` 路由移到新函数 `register_routes(app, memory)`。`create_app` 改为创建 app 后调 `register_routes`。

修改 `src/septmuse/api/rest/__init__.py`，把 `create_app` 函数替换为：

```python
def register_routes(app: FastAPI, memory: Memory) -> None:
    """注册 REST 路由到已有 FastAPI app。

    供 CLI serve --with-rest 使用 (同一 app 同时挂 MCP + REST)。
    """
    app.state.memory = memory

    @app.post("/memories", status_code=201)
    async def add_memory(req: AddMemoryRequest) -> dict[str, Any]:
        """添加记忆 (架构文档 §11.2)。"""
        m: Memory = app.state.memory
        if req.memory_type == "semantic":
            parts = req.content.split(None, 2)
            subject = parts[0] if len(parts) > 0 else req.content
            predicate = parts[1] if len(parts) > 1 else "is"
            obj = parts[2] if len(parts) > 2 else ""
            return m.add_fact(subject, predicate, obj, user_id=req.user_id)
        elif req.memory_type == "episodic":
            return m.add_episode(req.content, user_id=req.user_id)
        elif req.memory_type == "procedural":
            return m.add_rule(req.content, user_id=req.user_id)
        else:
            return m.add(req.content, user_id=req.user_id, agent_id=req.agent_id, infer=req.infer)

    @app.get("/memories")
    async def list_memories(
        user_id: str = Query(..., description="用户 ID"),
    ) -> dict[str, Any]:
        """列出记忆 (对齐 mem0 get_all)。"""
        return app.state.memory.get_all(user_id=user_id)

    @app.get("/memories/{memory_id}")
    async def get_memory(memory_id: str) -> dict[str, Any]:
        """取单条记忆。"""
        result = app.state.memory.get(memory_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"memory {memory_id} not found")
        return result

    @app.delete("/memories/{memory_id}")
    async def delete_memory(memory_id: str) -> dict[str, str]:
        """删除记忆 (软删除)。"""
        return app.state.memory.delete(memory_id)

    @app.post("/memories/search")
    async def search_memories(req: SearchRequest) -> list[dict[str, Any]]:
        """统一检索 (元认知路由)。"""
        return app.state.memory.search(req.query, user_id=req.user_id, top_k=req.top_k, threshold=req.threshold)

    @app.post("/memories/search/causal")
    async def causal_search(req: CausalRequest) -> dict[str, Any]:
        """反事实因果查询 (架构文档 §6.1)。"""
        return app.state.memory.counterfactual(
            req.cause_event_id, req.effect_event_id, user_id=req.user_id
        )

    @app.get("/memories/meta/coverage")
    async def coverage_report(
        user_id: str = Query(..., description="用户 ID"),
    ) -> dict[str, Any]:
        """元认知覆盖报告 (架构文档 §6.3 L1)。"""
        return app.state.memory.coverage_report(user_id=user_id)

    @app.post("/memories/rehearse")
    async def rehearse(req: RehearseRequest) -> dict[str, Any]:
        """主动复述 (架构文档 §6.2)。"""
        if req.memory_id:
            return app.state.memory.rehearse(req.memory_id, user_id=req.user_id)
        candidates = app.state.memory.find_rehearse_candidates(user_id=req.user_id)
        for c in candidates:
            app.state.memory.rehearse(c["memory_id"], user_id=req.user_id)
        return {"rehearsed": len(candidates)}

    @app.post("/memories/capture")
    async def capture(req: CaptureRequest) -> dict[str, Any]:
        """PostToolUse 捕获 (架构文档 §5.1)。"""
        return app.state.memory.capture(req.text, user_id=req.user_id, agent_id=req.agent_id)

    @app.get("/agents/{user_id}/memories")
    async def get_shared_memories(user_id: str) -> dict[str, Any]:
        """跨 agent 共享读 (架构文档 §5.5)。"""
        agents = app.state.memory.list_agents(user_id)
        return {"user_id": user_id, "agents": agents, "is_cross_agent": app.state.memory.is_cross_agent(user_id)}

    @app.get("/health")
    async def health() -> dict[str, str]:
        """健康检查。"""
        return {"status": "ok"}


def create_app(memory: Memory | None = None) -> FastAPI:
    """创建 FastAPI app (可注入 Memory 实例便于测试)。

    用法:
        app = create_app()
        # uvicorn septmuse.api.rest:app

    测试:
        app = create_app(my_test_memory)
        # TestClient(app).post("/memories", json={...})
    """
    if memory is None:
        memory = Memory(
            config=MemoryConfig(db_path=":memory:"),
            embedder=HashEmbedder(),
        )

    app = FastAPI(
        title="SeptMuse Memory API",
        description="Agent 记忆系统 REST API (架构文档 §11.2)",
        version="0.1.0",
    )
    register_routes(app, memory)
    return app
```

注意：保留文件末尾的 `app = create_app()` 默认实例。

- [ ] **Step 3: 运行 ruff 验证语法**

Run: `$env:PYTHONPATH="src"; ruff check src/septmuse/api/rest/__init__.py`
Expected: `All checks passed!`

- [ ] **Step 4: 运行现有 REST 测试确认不退化**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_rbac_rest_openai.py -q -k "not test_mount_routes" 2>&1 | Select-Object -Last 5`
Expected: 已有测试不退化（可能因 FastAPI 版本有 12 个 TypeError，那是既有问题，不算本次引入）

- [ ] **Step 5: 运行全回归确认不退化**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/ -q --ignore=tests/unit/test_rbac_rest_openai.py -k "not test_mount_routes" 2>&1 | Select-Object -Last 3`
Expected: `485 passed, 9 skipped, 1 deselected`

---

### Task 2: CLI 骨架 + init + version 命令

**Files:**
- Modify: `src/septmuse/cli/main.py`（完全重写）
- Test: `tests/unit/test_cli.py`（新建）

**Interfaces:**
- Produces: `main() -> int`（argparse dispatch 入口）、`_build_parser() -> argparse.ArgumentParser`、`_resolve_db_path(db_path: str | None) -> str`、`_make_memory(db_path: str | None) -> Memory`、`_cmd_init(args) -> int`、`_cmd_version(args) -> int`

- [ ] **Step 1: 写 init + version 的失败测试**

创建 `tests/unit/test_cli.py`：

```python
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
"""CLI 命令测试 (init/add/search/dump/serve)。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


def _run_cli(argv: list[str], monkeypatch, capsys) -> int:
    """辅助: 设置 sys.argv 并调 main()。"""
    monkeypatch.setattr(sys, "argv", ["septmuse"] + argv)
    from septmuse.cli.main import main

    rc = main()
    out = capsys.readouterr()
    return rc, out.out, out.err


class TestVersion:
    def test_version(self, monkeypatch, capsys):
        rc, out, _ = _run_cli(["version"], monkeypatch, capsys)
        assert rc == 0
        assert "0.1.0" in out


class TestInit:
    def test_init_creates_db(self, tmp_path, monkeypatch, capsys):
        db = tmp_path / "test.db"
        rc, out, _ = _run_cli(["init", "--user", "alice", "--db-path", str(db)], monkeypatch, capsys)
        assert rc == 0
        assert db.exists()
        assert "initialized" in out.lower()

    def test_init_default_db(self, tmp_path, monkeypatch, capsys):
        # 用 SEPTMUSE_DB_PATH 覆盖默认路径避免污染 ~/.septmuse/
        db = tmp_path / "default.db"
        monkeypatch.setenv("SEPTMUSE_DB_PATH", str(db))
        rc, out, _ = _run_cli(["init", "--user", "bob"], monkeypatch, capsys)
        assert rc == 0
        assert db.exists()

    def test_init_creates_parent_dir(self, tmp_path, monkeypatch, capsys):
        db = tmp_path / "nested" / "deep" / "test.db"
        rc, out, _ = _run_cli(["init", "--user", "alice", "--db-path", str(db)], monkeypatch, capsys)
        assert rc == 0
        assert db.exists()
```

- [ ] **Step 2: 运行测试验证失败**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_cli.py -v 2>&1 | Select-Object -Last 10`
Expected: FAIL（当前 main() 不识别 init/version 子命令的 argparse 格式）

- [ ] **Step 3: 重写 cli/main.py — 骨架 + init + version**

完整重写 `src/septmuse/cli/main.py`：

```python
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
    from septmuse import Memory, MemoryConfig
    from septmuse.providers.embedders.hash import HashEmbedder

    path = _resolve_db_path(db_path)
    return Memory(config=MemoryConfig(db_path=path), embedder=HashEmbedder())


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
    p_add.add_argument("--user", default=os.getenv("SEPTMUSE_USER_ID", "default"), help="用户 ID")
    p_add.add_argument(
        "--type",
        choices=["verbatim", "semantic", "episodic", "procedural"],
        default="verbatim",
        help="记忆类型",
    )
    p_add.add_argument("--agent", default=None, help="agent ID")
    p_add.add_argument("--db-path", default=None, help="SQLite 路径")
    p_add.set_defaults(func=_cmd_add)

    # search (Task 4 实现)
    p_search = sub.add_parser("search", help="检索记忆")
    p_search.add_argument("query", help="查询文本")
    p_search.add_argument("--user", default=os.getenv("SEPTMUSE_USER_ID", "default"), help="用户 ID")
    p_search.add_argument("--top-k", type=int, default=5, help="返回数")
    p_search.add_argument("--threshold", type=float, default=0.1, help="相似阈值")
    p_search.add_argument("--db-path", default=None, help="SQLite 路径")
    p_search.set_defaults(func=_cmd_search)

    # dump (Task 5 实现)
    p_dump = sub.add_parser("dump", help="导出记忆")
    p_dump.add_argument("--user", default=os.getenv("SEPTMUSE_USER_ID", "default"), help="用户 ID")
    p_dump.add_argument("--format", choices=["json", "markdown"], default="json", help="输出格式")
    p_dump.add_argument("--output", default=None, help="输出文件路径 (默认 stdout)")
    p_dump.add_argument("--db-path", default=None, help="SQLite 路径")
    p_dump.set_defaults(func=_cmd_dump)

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
    """添加记忆 (Task 3 实现)。"""
    raise NotImplementedError("Task 3")


def _cmd_search(args: argparse.Namespace) -> int:
    """检索记忆 (Task 4 实现)。"""
    raise NotImplementedError("Task 4")


def _cmd_dump(args: argparse.Namespace) -> int:
    """导出记忆 (Task 5 实现)。"""
    raise NotImplementedError("Task 5")


def _cmd_serve(args: argparse.Namespace) -> int:
    """启动 HTTP/SSE MCP server (Task 6 实现)。"""
    raise NotImplementedError("Task 6")


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


def main() -> int:
    """septmuse CLI 主入口。"""
    parser = _build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
```

注意：`_cmd_add`/`_cmd_search`/`_cmd_dump`/`_cmd_serve` 是占位 `raise NotImplementedError`，后续 task 填充。argparse 骨架一次建好避免后续反复修改。

- [ ] **Step 4: 运行 init + version 测试验证通过**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_cli.py::TestVersion tests/unit/test_cli.py::TestInit -v 2>&1 | Select-Object -Last 10`
Expected: 4 passed

- [ ] **Step 5: ruff 验证**

Run: `$env:PYTHONPATH="src"; ruff check src/septmuse/cli/main.py`
Expected: `All checks passed!`

---

### Task 3: add 命令

**Files:**
- Modify: `src/septmuse/cli/main.py`（替换 `_cmd_add`）
- Test: `tests/unit/test_cli.py`（追加 TestAdd）

**Interfaces:**
- Consumes: `Memory.add(messages, *, user_id, agent_id=None, infer=None) → {"results": [{"id","memory","event":"ADD"}], "relations": []}`
- Consumes: `Memory.add_fact(subject, predicate, object, *, user_id) → {"id","triple","event":"ADD"}`
- Consumes: `Memory.add_episode(content, *, user_id) → {"id","event_type","reference_time"}`
- Consumes: `Memory.add_rule(rule, *, user_id) → {"id","rule","event":"ADD"}`

- [ ] **Step 1: 写 add 的失败测试**

在 `tests/unit/test_cli.py` 追加：

```python
class TestAdd:
    def _setup_db(self, tmp_path, monkeypatch):
        """辅助: 先 init 一个 db。"""
        db = tmp_path / "test.db"
        _run_cli(["init", "--user", "alice", "--db-path", str(db)], monkeypatch, capsys if False else None)
        return str(db)

    def test_add_verbatim(self, tmp_path, monkeypatch, capsys):
        db = tmp_path / "test.db"
        _run_cli(["init", "--user", "alice", "--db-path", str(db)], monkeypatch, capsys)
        rc, out, _ = _run_cli(
            ["add", "我喜欢 Python", "--user", "alice", "--db-path", str(db)],
            monkeypatch,
            capsys,
        )
        assert rc == 0
        data = json.loads(out)
        assert "memory_id" in data or "id" in data

    def test_add_semantic(self, tmp_path, monkeypatch, capsys):
        db = tmp_path / "test.db"
        _run_cli(["init", "--user", "alice", "--db-path", str(db)], monkeypatch, capsys)
        rc, out, _ = _run_cli(
            ["add", "alice likes python", "--user", "alice", "--type", "semantic", "--db-path", str(db)],
            monkeypatch,
            capsys,
        )
        assert rc == 0
        data = json.loads(out)
        assert "id" in data

    def test_add_episodic(self, tmp_path, monkeypatch, capsys):
        db = tmp_path / "test.db"
        _run_cli(["init", "--user", "alice", "--db-path", str(db)], monkeypatch, capsys)
        rc, out, _ = _run_cli(
            ["add", "用户登录", "--user", "alice", "--type", "episodic", "--db-path", str(db)],
            monkeypatch,
            capsys,
        )
        assert rc == 0
        data = json.loads(out)
        assert "id" in data

    def test_add_procedural(self, tmp_path, monkeypatch, capsys):
        db = tmp_path / "test.db"
        _run_cli(["init", "--user", "alice", "--db-path", str(db)], monkeypatch, capsys)
        rc, out, _ = _run_cli(
            ["add", "先检查权限再执行", "--user", "alice", "--type", "procedural", "--db-path", str(db)],
            monkeypatch,
            capsys,
        )
        assert rc == 0
        data = json.loads(out)
        assert "id" in data
```

注意：`_setup_db` 方法有 `capsys if False else None` 的问题。改为直接在每条测试里调 `_run_cli(["init", ...])` 初始化（如上 test_add_verbatim 所示）。删除 `_setup_db` 方法。

修正后的 TestAdd（无 `_setup_db`）：

```python
class TestAdd:
    def test_add_verbatim(self, tmp_path, monkeypatch, capsys):
        db = tmp_path / "test.db"
        _run_cli(["init", "--user", "alice", "--db-path", str(db)], monkeypatch, capsys)
        rc, out, _ = _run_cli(
            ["add", "我喜欢 Python", "--user", "alice", "--db-path", str(db)],
            monkeypatch,
            capsys,
        )
        assert rc == 0
        data = json.loads(out)
        assert "memory_id" in data or "id" in data

    def test_add_semantic(self, tmp_path, monkeypatch, capsys):
        db = tmp_path / "test.db"
        _run_cli(["init", "--user", "alice", "--db-path", str(db)], monkeypatch, capsys)
        rc, out, _ = _run_cli(
            ["add", "alice likes python", "--user", "alice", "--type", "semantic", "--db-path", str(db)],
            monkeypatch,
            capsys,
        )
        assert rc == 0
        data = json.loads(out)
        assert "id" in data

    def test_add_episodic(self, tmp_path, monkeypatch, capsys):
        db = tmp_path / "test.db"
        _run_cli(["init", "--user", "alice", "--db-path", str(db)], monkeypatch, capsys)
        rc, out, _ = _run_cli(
            ["add", "用户登录", "--user", "alice", "--type", "episodic", "--db-path", str(db)],
            monkeypatch,
            capsys,
        )
        assert rc == 0
        data = json.loads(out)
        assert "id" in data

    def test_add_procedural(self, tmp_path, monkeypatch, capsys):
        db = tmp_path / "test.db"
        _run_cli(["init", "--user", "alice", "--db-path", str(db)], monkeypatch, capsys)
        rc, out, _ = _run_cli(
            ["add", "先检查权限再执行", "--user", "alice", "--type", "procedural", "--db-path", str(db)],
            monkeypatch,
            capsys,
        )
        assert rc == 0
        data = json.loads(out)
        assert "id" in data
```

- [ ] **Step 2: 运行测试验证失败**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_cli.py::TestAdd -v 2>&1 | Select-Object -Last 10`
Expected: FAIL（`NotImplementedError: Task 3`）

- [ ] **Step 3: 实现 _cmd_add**

替换 `src/septmuse/cli/main.py` 中的 `_cmd_add`：

```python
def _cmd_add(args: argparse.Namespace) -> int:
    """添加记忆 (verbatim/semantic/episodic/procedural)。"""
    m = _make_memory(args.db_path)

    if args.type == "verbatim":
        result = m.add(args.content, user_id=args.user, agent_id=args.agent, infer=False)
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
```

需要在文件顶部加 `import json`。

- [ ] **Step 4: 运行测试验证通过**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_cli.py::TestAdd -v 2>&1 | Select-Object -Last 10`
Expected: 4 passed

- [ ] **Step 5: ruff 验证**

Run: `$env:PYTHONPATH="src"; ruff check src/septmuse/cli/main.py`
Expected: `All checks passed!`

---

### Task 4: search 命令

**Files:**
- Modify: `src/septmuse/cli/main.py`（替换 `_cmd_search`）
- Test: `tests/unit/test_cli.py`（追加 TestSearch）

**Interfaces:**
- Consumes: `Memory.search(query, *, user_id, top_k=None, threshold=None) → list[{"id","memory","score","metadata","created_at"}]`

- [ ] **Step 1: 写 search 的失败测试**

在 `tests/unit/test_cli.py` 追加：

```python
class TestSearch:
    def test_search_returns_json_array(self, tmp_path, monkeypatch, capsys):
        db = tmp_path / "test.db"
        _run_cli(["init", "--user", "alice", "--db-path", str(db)], monkeypatch, capsys)
        _run_cli(["add", "我喜欢 Python", "--user", "alice", "--db-path", str(db)], monkeypatch, capsys)
        rc, out, _ = _run_cli(
            ["search", "喜欢什么", "--user", "alice", "--db-path", str(db)],
            monkeypatch,
            capsys,
        )
        assert rc == 0
        results = json.loads(out)
        assert isinstance(results, list)
        assert len(results) > 0

    def test_search_no_results(self, tmp_path, monkeypatch, capsys):
        db = tmp_path / "test.db"
        _run_cli(["init", "--user", "alice", "--db-path", str(db)], monkeypatch, capsys)
        rc, out, _ = _run_cli(
            ["search", "完全不相关的内容xyz", "--user", "alice", "--db-path", str(db), "--threshold", "0.99"],
            monkeypatch,
            capsys,
        )
        assert rc == 0
        results = json.loads(out)
        assert isinstance(results, list)
        assert len(results) == 0

    def test_search_top_k(self, tmp_path, monkeypatch, capsys):
        db = tmp_path / "test.db"
        _run_cli(["init", "--user", "alice", "--db-path", str(db)], monkeypatch, capsys)
        _run_cli(["add", "测试1", "--user", "alice", "--db-path", str(db)], monkeypatch, capsys)
        _run_cli(["add", "测试2", "--user", "alice", "--db-path", str(db)], monkeypatch, capsys)
        rc, out, _ = _run_cli(
            ["search", "测试", "--user", "alice", "--db-path", str(db), "--top-k", "1"],
            monkeypatch,
            capsys,
        )
        assert rc == 0
        results = json.loads(out)
        assert len(results) <= 1
```

- [ ] **Step 2: 运行测试验证失败**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_cli.py::TestSearch -v 2>&1 | Select-Object -Last 10`
Expected: FAIL（`NotImplementedError: Task 4`）

- [ ] **Step 3: 实现 _cmd_search**

替换 `_cmd_search`：

```python
def _cmd_search(args: argparse.Namespace) -> int:
    """检索记忆, JSON 数组输出。"""
    m = _make_memory(args.db_path)
    results = m.search(args.query, user_id=args.user, top_k=args.top_k, threshold=args.threshold)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0
```

- [ ] **Step 4: 运行测试验证通过**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_cli.py::TestSearch -v 2>&1 | Select-Object -Last 10`
Expected: 3 passed

- [ ] **Step 5: ruff 验证**

Run: `$env:PYTHONPATH="src"; ruff check src/septmuse/cli/main.py`
Expected: `All checks passed!`

---

### Task 5: dump 命令

**Files:**
- Modify: `src/septmuse/cli/main.py`（替换 `_cmd_dump`）
- Test: `tests/unit/test_cli.py`（追加 TestDump）

**Interfaces:**
- Consumes: `Memory.get_all(*, user_id) → {"results": [...]}`

- [ ] **Step 1: 写 dump 的失败测试**

在 `tests/unit/test_cli.py` 追加：

```python
class TestDump:
    def test_dump_json_stdout(self, tmp_path, monkeypatch, capsys):
        db = tmp_path / "test.db"
        _run_cli(["init", "--user", "alice", "--db-path", str(db)], monkeypatch, capsys)
        _run_cli(["add", "测试记忆", "--user", "alice", "--db-path", str(db)], monkeypatch, capsys)
        rc, out, _ = _run_cli(
            ["dump", "--user", "alice", "--db-path", str(db)],
            monkeypatch,
            capsys,
        )
        assert rc == 0
        data = json.loads(out)
        assert "results" in data
        assert len(data["results"]) >= 1

    def test_dump_markdown(self, tmp_path, monkeypatch, capsys):
        db = tmp_path / "test.db"
        _run_cli(["init", "--user", "alice", "--db-path", str(db)], monkeypatch, capsys)
        _run_cli(["add", "测试记忆", "--user", "alice", "--db-path", str(db)], monkeypatch, capsys)
        rc, out, _ = _run_cli(
            ["dump", "--user", "alice", "--format", "markdown", "--db-path", str(db)],
            monkeypatch,
            capsys,
        )
        assert rc == 0
        assert "- **" in out

    def test_dump_to_file(self, tmp_path, monkeypatch, capsys):
        db = tmp_path / "test.db"
        out_file = tmp_path / "dump.json"
        _run_cli(["init", "--user", "alice", "--db-path", str(db)], monkeypatch, capsys)
        _run_cli(["add", "测试记忆", "--user", "alice", "--db-path", str(db)], monkeypatch, capsys)
        rc, _, _ = _run_cli(
            ["dump", "--user", "alice", "--db-path", str(db), "--output", str(out_file)],
            monkeypatch,
            capsys,
        )
        assert rc == 0
        assert out_file.exists()
        data = json.loads(out_file.read_text(encoding="utf-8"))
        assert "results" in data
```

- [ ] **Step 2: 运行测试验证失败**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_cli.py::TestDump -v 2>&1 | Select-Object -Last 10`
Expected: FAIL（`NotImplementedError: Task 5`）

- [ ] **Step 3: 实现 _cmd_dump**

替换 `_cmd_dump`：

```python
def _cmd_dump(args: argparse.Namespace) -> int:
    """导出记忆 (json/markdown, stdout 或文件)。"""
    m = _make_memory(args.db_path)
    data = m.get_all(user_id=args.user)

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
```

- [ ] **Step 4: 运行测试验证通过**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_cli.py::TestDump -v 2>&1 | Select-Object -Last 10`
Expected: 3 passed

- [ ] **Step 5: ruff 验证**

Run: `$env:PYTHONPATH="src"; ruff check src/septmuse/cli/main.py`
Expected: `All checks passed!`

---

### Task 6: serve 命令

**Files:**
- Modify: `src/septmuse/cli/main.py`（替换 `_cmd_serve`）
- Test: `tests/unit/test_cli.py`（追加 TestServe）

**Interfaces:**
- Consumes: `setup_mcp_server(app: FastAPI) -> None`（来自 `api/mcp/server.py`）
- Consumes: `register_routes(app: FastAPI, memory: Memory) -> None`（来自 Task 1，`api/rest/__init__.py`）
- Consumes: `uvicorn.run(app, host=..., port=...)`（延迟 import）

- [ ] **Step 1: 写 serve 的失败测试**

在 `tests/unit/test_cli.py` 追加：

```python
class TestServe:
    def test_serve_no_uvicorn(self, tmp_path, monkeypatch, capsys):
        """uvicorn 未安装时报友好错误。"""
        # 模拟 uvicorn 不可用
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "uvicorn":
                raise ImportError("No module named 'uvicorn'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        db = str(tmp_path / "test.db")
        rc, _, err = _run_cli(
            ["serve", "--db-path", db],
            monkeypatch,
            capsys,
        )
        assert rc == 1
        assert "uvicorn" in err.lower()

    def test_serve_argparse(self, tmp_path, monkeypatch, capsys):
        """serve 参数解析正确 (不真正启动 server)。"""
        from septmuse.cli.main import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["serve", "--host", "0.0.0.0", "--port", "9999", "--with-rest"])
        assert args.host == "0.0.0.0"
        assert args.port == 9999
        assert args.with_rest is True

    def test_serve_default_args(self, tmp_path, monkeypatch, capsys):
        """serve 默认参数。"""
        from septmuse.cli.main import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["serve"])
        assert args.host == "127.0.0.1"
        assert args.port == 8000
        assert args.with_rest is False
```

- [ ] **Step 2: 运行测试验证失败**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_cli.py::TestServe -v 2>&1 | Select-Object -Last 10`
Expected: FAIL（`NotImplementedError: Task 6`，test_serve_argparse 和 test_serve_default_args 可能通过因为只测解析）

- [ ] **Step 3: 实现 _cmd_serve**

替换 `_cmd_serve`：

```python
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
```

- [ ] **Step 4: 运行测试验证通过**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_cli.py::TestServe -v 2>&1 | Select-Object -Last 10`
Expected: 3 passed

注意：`test_serve_no_uvicorn` mock 了 `builtins.__import__`，需要确认 mock 正确拦截 uvicorn import 但不影响其他 import。如果 mock 有问题，调整 mock 策略为 `monkeypatch.setitem(sys.modules, "uvicorn", None)`。

如果 `test_serve_no_uvicorn` 因 mock 复杂度过高不稳定，可改为：

```python
    def test_serve_no_uvicorn(self, tmp_path, monkeypatch, capsys):
        """uvicorn 未安装时报友好错误。"""
        import sys as _sys
        monkeypatch.setitem(_sys.modules, "uvicorn", None)
        db = str(tmp_path / "test.db")
        rc, _, err = _run_cli(
            ["serve", "--db-path", db],
            monkeypatch,
            capsys,
        )
        assert rc == 1
        assert "uvicorn" in err.lower()
```

（`sys.modules["uvicorn"] = None` 会让 `import uvicorn` 抛 `ImportError`，这是标准库测试技巧。）

- [ ] **Step 5: ruff 验证**

Run: `$env:PYTHONPATH="src"; ruff check src/septmuse/cli/main.py`
Expected: `All checks passed!`

---

### Task 7: pyproject.toml server extras + 全回归

**Files:**
- Modify: `pyproject.toml`
- Test: 全回归

- [ ] **Step 1: 新增 server extras**

在 `pyproject.toml` 的 `[project.optional-dependencies]` 部分，在 `parametric` 行之后、`dev` 行之前新增：

```toml
# HTTP server (CLI serve 命令需要)
server = ["uvicorn>=0.29"]
```

然后更新 `all` 行（在 `parametric` 后加 `server`）：

```toml
all = ["septmuse[openai,anthropic,ollama,postgres,graph,activation,parametric,server,dev]"]
```

- [ ] **Step 2: ruff 全量验证**

Run: `$env:PYTHONPATH="src"; ruff check src/ tests/`
Expected: `All checks passed!`

- [ ] **Step 3: CLI 测试全量验证**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_cli.py -v 2>&1 | Select-Object -Last 20`
Expected: 全部 passed（TestVersion 1 + TestInit 3 + TestAdd 4 + TestSearch 3 + TestDump 3 + TestServe 3 = 17 passed）

- [ ] **Step 4: 全回归验证**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/ -q --ignore=tests/unit/test_rbac_rest_openai.py -k "not test_mount_routes" 2>&1 | Select-Object -Last 5`
Expected: `502 passed, 9 skipped, 1 deselected`（485 + 17 新增 = 502）

- [ ] **Step 5: 手动冒烟测试**

Run: `$env:PYTHONPATH="src"; python -m septmuse.cli.main --help`
Expected: 列出 init/add/search/dump/serve/mcp/version 7 命令

Run: `$env:PYTHONPATH="src"; python -m septmuse.cli.main version`
Expected: `0.1.0`

---

## Self-Review

**1. Spec coverage:**
- §4.2 init 命令 → Task 2 ✓
- §4.2 add 命令 → Task 3 ✓
- §4.2 search 命令 → Task 4 ✓
- §4.2 dump 命令 → Task 5 ✓
- §4.2 serve 命令 → Task 6 ✓
- §4.3 环境变量 → Task 2 argparse `default=os.getenv(...)` ✓
- §4.4 register_routes 重构 → Task 1 ✓
- §4.5 HashEmbedder → Task 2 `_make_memory` ✓
- §8 server extras → Task 7 ✓

**2. Placeholder scan:** 无 TBD/TODO，所有代码步骤有完整代码。`_cmd_add`/`_cmd_search`/`_cmd_dump`/`_cmd_serve` 在 Task 2 是 `raise NotImplementedError` 占位，但后续 Task 3-6 分别填充——这是 TDD 的标准做法，不是占位符。

**3. Type consistency:** `register_routes(app, memory)` 在 Task 1 定义，Task 6 `_cmd_serve` 中 `from septmuse.api.rest import register_routes` 调用——签名一致。`_make_memory(db_path)` 在 Task 2 定义，Task 3-6 都调用——签名一致。`_resolve_db_path(db_path)` 在 Task 2 定义，`_make_memory` 内部调用——签名一致。

无问题。

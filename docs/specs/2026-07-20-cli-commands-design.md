# SeptMuse CLI 5 命令设计规格

> 日期: 2026-07-20
> 对应架构文档: `agent-memory-architecture.md` §11.2 API 草案 + `package-structure.md` §13 MCP 服务
> 状态: 待实现

---

## 1. 背景

当前 `src/septmuse/cli/main.py` 在自己的 `_print_help()` 里列出了 6 个命令（mcp / serve / init / add / search / dump），但注释写"阶段1 MVP 逐步实现中, 当前仅 mcp/serve/version 可用"。实际只有 `mcp` 和 `version` 可用，`serve` 打印"尚未实现"返回 1，`init`/`add`/`search`/`dump` 完全没有函数体。

这导致 SeptMuse 无法脱离 Python 代码直接使用——零配置可用性的"最后一公里"断了。用户 `pip install septmuse` 后只能写 Python 代码调 `Memory()`，不能在终端直接 `septmuse add "..."` / `septmuse search "..."`。

## 2. 目标

实现 5 个 CLI 命令，让 `septmuse` 命令行开箱即用：

| 命令 | 功能 |
|------|------|
| `init` | 初始化本地记忆库（创建目录 + SQLite 文件 + 建表） |
| `add` | 命令行添加记忆（4 种 content_type） |
| `search` | 命令行检索记忆，JSON 输出 |
| `dump` | 导出记忆到 stdout 或文件（JSON / Markdown） |
| `serve` | 启动 HTTP/SSE MCP server（uvicorn + FastAPI） |

## 3. 非目标

- 不实现记忆更新/编辑能力（Memory facade 的 update/replace 方法 + REST PUT 端点）——后续单独做
- 不实现 e2e 端到端测试——后续单独做
- 不新增 click/typer 依赖——继续用 argparse（stdlib），对齐 mem0 CLI 风格
- 不把 uvicorn 加入核心依赖——延迟 import + 友好报错，保持核心依赖最小化

## 4. 设计

### 4.1 参数解析: argparse

重写 `cli/main.py`，从纯 `sys.argv` 手动 dispatch 改为 argparse 子命令。理由:

- 零新依赖（stdlib），与现有风格一致
- mem0 CLI 源码也用 argparse（实证）
- argparse 足以处理 5 个子命令的参数解析

顶层 parser 结构:

```python
parser = argparse.ArgumentParser(
    prog="septmuse",
    description="SeptMuse — agent 记忆系统",
)
sub = parser.add_subparsers(dest="cmd", required=True)

# 每个子命令一个 add_parser
p_init = sub.add_parser("init", ...)
p_init.add_argument("--user", default=os.getenv("SEPTMUSE_USER_ID", "default"), ...)
...
```

### 4.2 命令接口

#### `septmuse init`

```
septmuse init [--user USER_ID] [--db-path PATH]
```

- `--user`: 用户 ID，默认读 `SEPTMUSE_USER_ID` 环境变量，再默认 `"default"`
- `--db-path`: SQLite 路径，默认读 `SEPTMUSE_DB_PATH`，再默认 `~/.septmuse/septmuse.db`
- 行为:
  1. 创建 `~/.septmuse/` 目录（`Path.home() / ".septmuse"`，`mkdir(parents=True, exist_ok=True)`）
  2. 实例化 `Memory(config=MemoryConfig(db_path=db_path), embedder=HashEmbedder())` 触发建表
  3. 写入一条 welcome 记忆验证可用: `m.add(f"septmuse initialized for user {user_id}", user_id=user_id)`
  4. 打印路径和确认信息到 stdout
- 退出码: 成功 0，失败 1

#### `septmuse add`

```
septmuse add "记忆内容" [--user USER_ID] [--type TYPE] [--agent AGENT_ID] [--db-path PATH]
```

- 位置参数: 记忆内容文本
- `--user`: 用户 ID，默认读 `SEPTMUSE_USER_ID`，再默认 `"default"`
- `--type`: `verbatim`(默认) / `semantic` / `episodic` / `procedural`
- `--agent`: agent ID，默认 None
- `--db-path`: SQLite 路径，默认读 `SEPTMUSE_DB_PATH`，再默认 `~/.septmuse/septmuse.db`
- 行为: 按 type 调用不同 facade 方法
  - `verbatim` → `Memory.add(content, user_id=..., agent_id=..., infer=False)`
  - `semantic` → `Memory.add_fact(subject, predicate, obj, user_id=...)`（content 按 `split(None, 2)` 拆三元）
  - `episodic` → `Memory.add_episode(content, user_id=...)`
  - `procedural` → `Memory.add_rule(content, user_id=...)`
- 输出: `memory_id` 到 stdout（JSON 格式 `{"memory_id": "...", "type": "..."}`）
- 退出码: 成功 0，失败 1

#### `septmuse search`

```
septmuse search "查询" [--user USER_ID] [--top-k N] [--threshold T] [--db-path PATH]
```

- 位置参数: 查询文本
- `--user`: 用户 ID，默认读 `SEPTMUSE_USER_ID`，再默认 `"default"`
- `--top-k`: 返回数，默认 5（对齐 MemoryConfig）
- `--threshold`: 相似阈值，默认 0.1（对齐 MemoryConfig）
- `--db-path`: SQLite 路径，默认读 `SEPTMUSE_DB_PATH`，再默认 `~/.septmuse/septmuse.db`
- 行为: 调 `Memory.search(query, user_id=..., top_k=..., threshold=...)`
- 输出: JSON 数组到 stdout（`json.dumps(results, ensure_ascii=False, indent=2)`）
- 退出码: 成功 0，无结果也 0（空数组），失败 1

#### `septmuse dump`

```
septmuse dump [--user USER_ID] [--format json|markdown] [--output PATH] [--db-path PATH]
```

- `--user`: 用户 ID，默认读 `SEPTMUSE_USER_ID`，再默认 `"default"`
- `--format`: `json`(默认) / `markdown`
- `--output`: 输出文件路径，默认 None → stdout
- `--db-path`: SQLite 路径，默认读 `SEPTMUSE_DB_PATH`，再默认 `~/.septmuse/septmuse.db`
- 行为: 调 `Memory.get_all(user_id=...)`
- 输出格式:
  - `json`: `json.dumps(data, ensure_ascii=False, indent=2)`
  - `markdown`: 每条记忆一行 `- **{memory_id}**: {content}`（带 metadata 子项）
- 退出码: 成功 0，失败 1

#### `septmuse serve`

```
septmuse serve [--host HOST] [--port PORT] [--with-rest]
```

- `--host`: 绑定地址，默认 `127.0.0.1`
- `--port`: 端口，默认 `8000`
- `--with-rest`: 额外挂载 REST API 路由（`api/rest/__init__.py` 的端点）
- 行为:
  1. 延迟 `import uvicorn`，未装报 `pip install uvicorn` 退出 1
  2. 创建 FastAPI app
  3. 调 `setup_mcp_server(app)` 挂载 MCP http/sse transport
  4. `--with-rest` 时额外挂载 REST 路由（从 `api/rest` import `create_app` 并合并路由，或在同一 app 上注册路由）
  5. `uvicorn.run(app, host=..., port=...)`
- 退出码: 启动失败 1

### 4.3 通用选项

所有命令共享的环境变量（对齐 `configs/defaults.py`）:

| 环境变量 | 用途 | 默认值 |
|---------|------|--------|
| `SEPTMUSE_DB_PATH` | SQLite 路径 | `~/.septmuse/septmuse.db` |
| `SEPTMUSE_USER_ID` | 默认用户 ID | `default` |
| `SEPTMUSE_EMBEDDER` | 嵌入模型 | `all-MiniLM-L6-v2` |
| `SEPTMUSE_LLM` | LLM provider | None（verbatim 模式） |

全局选项:
- `--help` / `-h`: argparse 内置，顶层 + 每子命令
- `--version` / `-V`: 打印 `septmuse.__version__`

### 4.4 serve 的 REST 挂载策略

`--with-rest` 需要把 REST 路由挂到同一个 FastAPI app。两种方式:

**方式 1（推荐）**: 在 `api/rest/__init__.py` 暴露一个 `register_routes(app, memory)` 函数，把 `create_app` 里的路由注册逻辑抽取出来，`serve --with-rest` 调用它。

**方式 2**: `create_app()` 返回独立 app，用 `FastAPI.mount` 合并。但 FastAPI 不推荐 mount 另一个 FastAPI app（路由前缀问题）。

选方式 1，需重构 `api/rest/__init__.py` 把 `create_app` 内的路由注册抽取为 `register_routes(app, memory)`。

### 4.5 embedder 选择

CLI 命令默认用 `HashEmbedder`（零模型加载，零 API key），不触发 sentence-transformers。理由:

- CLI 是零配置使用的入口，不能依赖模型下载
- `HashEmbedder` 已实现（`providers/embedders/hash.py`，47 行），纯 numpy 哈希嵌入
- 检索质量足够 CLI 验证用
- 用户需要高质量检索时，写 Python 代码配 `SentenceTransformersEmbedder` 或 `OpenAIEmbedder`

## 5. 数据流

### 5.1 add 数据流

```
septmuse add "我喜欢 Python" --user alice --type verbatim
  → argparse 解析
  → Memory(config=MemoryConfig(db_path=env_or_default), embedder=HashEmbedder())
  → m.add("我喜欢 Python", user_id="alice", infer=False)
  → SQLite store.add() 写入 + 向量化
  → stdout: {"memory_id": "...", "type": "verbatim"}
```

### 5.2 search 数据流

```
septmuse search "喜欢什么" --user alice --top-k 5
  → Memory(config=..., embedder=HashEmbedder())
  → m.search("喜欢什么", user_id="alice", top_k=5, threshold=0.1)
  → SQLite store.search() 向量余弦检索
  → stdout: [{...}, {...}] (JSON)
```

### 5.3 serve 数据流

```
septmuse serve --port 8000 --with-rest
  → 延迟 import uvicorn
  → app = FastAPI(...)
  → setup_mcp_server(app)  # 挂载 /mcp/http/{client}/{user} + /mcp/sse/{client}/{user}
  → register_routes(app, Memory())  # --with-rest 时挂载 /memories/*
  → uvicorn.run(app, host="127.0.0.1", port=8000)
```

## 6. 错误处理

| 错误场景 | 处理 |
|---------|------|
| 未知子命令 | argparse 自动报错 + 退出码 2 |
| 缺少必填参数（如 `--user`） | argparse 自动报错 + 退出码 2 |
| SQLite 路径不可写 | 捕获 `OSError`，打印中文错误 + 退出码 1 |
| `serve` 无 uvicorn | 打印 `pip install uvicorn` 提示 + 退出码 1 |
| Memory 操作异常 | 捕获 `Exception`，打印 `error: {e}` 到 stderr + 退出码 1 |

## 7. 测试策略

### 7.1 单元测试 `tests/unit/test_cli.py`

用 `capsys` + 临时 `--db-path` 直接调 `main()` 函数:

```python
def test_init_creates_db(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["septmuse", "init", "--user", "alice", "--db-path", str(tmp_path/"test.db")])
    assert main() == 0
    assert (tmp_path / "test.db").exists()

def test_add_verbatim(tmp_path, monkeypatch):
    db = str(tmp_path / "test.db")
    monkeypatch.setattr(sys, "argv", ["septmuse", "add", "我喜欢 Python", "--user", "alice", "--db-path", db])
    assert main() == 0

def test_search_returns_json(tmp_path, monkeypatch, capsys):
    # 先 add 再 search
    db = str(tmp_path / "test.db")
    monkeypatch.setattr(sys, "argv", ["septmuse", "add", "我喜欢 Python", "--user", "alice", "--db-path", db])
    main()
    monkeypatch.setattr(sys, "argv", ["septmuse", "search", "喜欢什么", "--user", "alice", "--db-path", db])
    assert main() == 0
    out = capsys.readouterr().out
    results = json.loads(out)
    assert isinstance(results, list)

def test_dump_json(tmp_path, monkeypatch, capsys):
    # add 后 dump
    ...
    assert main() == 0
    out = capsys.readouterr().out
    data = json.loads(out)
    assert "results" in data or "memories" in data

def test_dump_markdown(tmp_path, monkeypatch, capsys):
    ...
    out = capsys.readouterr().out
    assert "- **" in out  # markdown 列表格式

def test_dump_to_file(tmp_path, monkeypatch):
    out_file = tmp_path / "dump.json"
    monkeypatch.setattr(sys, "argv", ["septmuse", "dump", "--user", "alice", "--db-path", db, "--output", str(out_file)])
    assert main() == 0
    assert out_file.exists()

def test_serve_no_uvicorn(monkeypatch):
    # mock uvicorn import 失败
    monkeypatch.setattr(sys, "argv", ["septmuse", "serve"])
    # 模拟 uvicorn 未安装
    ...

def test_version(capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["septmuse", "--version"])
    assert main() == 0
    out = capsys.readouterr().out
    assert "0.1.0" in out
```

### 7.2 覆盖矩阵

| 命令 | 正常路径 | 错误路径 |
|------|---------|---------|
| init | 创建 db + welcome 记忆 | 路径不可写 |
| add | 4 种 type 各一 | 缺 --user |
| search | 有结果 + 无结果 | 缺 --user |
| dump | json + markdown + file | 缺 --user |
| serve | argparse 解析 | 无 uvicorn |
| version | 输出版本 | - |

## 8. 依赖变更

### pyproject.toml

新增 `server` optional-dependencies:

```toml
[project.optional-dependencies]
# ... 现有 extras ...
server = ["uvicorn>=0.29"]
all = ["septmuse[openai,anthropic,ollama,postgres,graph,activation,parametric,server,dev]"]
```

uvicorn 不进核心依赖。`septmuse serve` 延迟 import + 友好报错。

## 9. 验收标准

- [ ] `septmuse init --user alice --db-path /tmp/test.db` 创建文件且退出码 0
- [ ] `septmuse add "测试" --user alice --db-path /tmp/test.db` 输出 memory_id
- [ ] `septmuse search "测试" --user alice --db-path /tmp/test.db` 输出 JSON 数组
- [ ] `septmuse dump --user alice --db-path /tmp/test.db` 输出 JSON
- [ ] `septmuse dump --user alice --format markdown --db-path /tmp/test.db` 输出 markdown
- [ ] `septmuse serve --port 9999` 无 uvicorn 时报错 + 提示安装
- [ ] `septmuse --version` 输出 0.1.0
- [ ] `septmuse --help` 列出 5 命令
- [ ] `ruff check src/ tests/` 全绿
- [ ] `pytest tests/unit/test_cli.py` 全绿
- [ ] `pytest tests/ -q` 全回归不退化（485 passed 不降）

## 10. 文件清单

| 文件 | 操作 | 行数估算 |
|------|------|---------|
| `src/septmuse/cli/main.py` | 重写 | ~220 |
| `src/septmuse/api/rest/__init__.py` | 重构: 抽取 `register_routes(app, memory)` | +30 |
| `tests/unit/test_cli.py` | 新建 | ~180 |
| `pyproject.toml` | 新增 `server` extras | +3 |

## 11. 不做

- 不实现 Memory facade 的 update/replace 方法（#2+#4 缺口）
- 不实现 REST PUT 端点（#4 缺口）
- 不实现 e2e 端到端测试（#3 缺口）
- 不新增 click/typer 依赖
- 不把 uvicorn 加入核心依赖
- 不触发 sentence-transformers 加载（CLI 默认用 HashEmbedder）

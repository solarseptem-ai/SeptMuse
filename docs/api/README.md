# SeptMuse 接口文档

SeptMuse 提供四种 API 入口，覆盖不同集成场景：

| 入口 | 文件 | 适用场景 | 传输方式 |
|------|------|----------|----------|
| **Python API** | [python-api.md](python-api.md) | Python 应用直接集成 | 进程内调用 |
| **REST API** | [rest-api.md](rest-api.md) | HTTP 微服务集成 | FastAPI (:8000) |
| **MCP Tools** | [mcp-tools.md](mcp-tools.md) | LLM Agent 工具调用 | stdio / SSE / Streamable HTTP |
| **CLI** | [cli.md](cli.md) | 命令行操作 / 脚本 | argparse 子命令 |

## 快速开始

### Python（零配置）

```python
from septmuse import Memory

m = Memory()                                    # SQLite + HashEmbedder, 离线可用
m.add("我喜欢 Python", user_id="alice")
results = m.search("alice 喜欢什么", user_id="alice")
# [{"id": "...", "memory": "我喜欢 Python", "score": 0.85, ...}]
```

### REST

```bash
# 启动 server
septmuse serve --with-rest --port 8000

# 添加记忆
curl -X POST http://localhost:8000/memories \
  -H "Content-Type: application/json" \
  -d '{"content": "我喜欢 Python", "user_id": "alice"}'

# 检索
curl -X POST http://localhost:8000/memories/search \
  -H "Content-Type: application/json" \
  -d '{"query": "喜欢什么", "user_id": "alice"}'
```

### MCP

```bash
# stdio (Claude Desktop / Cursor 等 MCP 客户端)
septmuse mcp

# 或挂载到已有 FastAPI
septmuse serve --port 8000   # SSE + Streamable HTTP
```

MCP 客户端配置示例（`claude_desktop_config.json`）：

```json
{
  "mcpServers": {
    "septmuse": {
      "command": "septmuse",
      "args": ["mcp"]
    }
  }
}
```

### CLI

```bash
septmuse init --user alice
septmuse add "我喜欢 Python" --user alice
septmuse search "喜欢什么" --user alice
septmuse dump --user alice --format json
```

## 架构概览

```
┌─────────────────────────────────────────────────┐
│                  Memory facade                    │  Python API
│            (orchestration/memory.py)              │
├──────────┬──────────────┬───────────────────────┤
│   CLI    │   REST API   │      MCP Tools        │
│ argparse │   FastAPI    │     FastMCP @tool      │
├──────────┴──────────────┴───────────────────────┤
│              Storage Layer                        │
│  SQLite (default) / pgvector / Chroma / Qdrant  │
├──────────────────────────────────────────────────┤
│              Embedder + Reranker                  │
│  Hash(default) / ONNX / ST  +  Noop/MMR/Cross    │
└──────────────────────────────────────────────────┘
```

## 认证与权限

| 层级 | 机制 | 状态码 |
|------|------|--------|
| 认证 | `SEPTMUSE_API_KEY` 环境变量 | 401 = 未认证（API key 缺失/错误） |
| 授权 | `check_memory_access_permissions` (state 检查) | 403 = 存在但非 active（deleted/archived/paused） |
| 不存在 | `store.get` 过滤 `is_deleted=0` | 404 = 从未存在 |

- 未设 `SEPTMUSE_API_KEY` = 开发模式（无认证，启动时警告一次）
- 已设 `SEPTMUSE_API_KEY` = 生产模式（所有请求需 `Authorization: Bearer <key>`）

## Score 约定

所有 score 统一为**相似度 [0, 1]**，越高越相似：

- 向量 cosine similarity
- BM25 归一化到 [0, 1]
- RRF 融合（k=60）
- 图检索 graph_score = 1/2^depth（深度衰减）

## 环境变量

| 变量 | 默认值 | 作用 |
|------|--------|------|
| `SEPTMUSE_DB_PATH` | `~/.septmuse/septmuse.db` | SQLite 路径；`:memory:` 内存库 |
| `SEPTMUSE_EMBEDDER` | `hash` | `hash`/`onnx`/`onnx-zh`/`auto`/`st` |
| `SEPTMUSE_API_KEY` | 未设 | 未设=开发模式；已设=生产模式（Bearer 认证） |
| `SEPTMUSE_USER_ID` | `default`（CLI） | CLI/MCP 默认 user_id |
| `SEPTMUSE_VECTOR_BACKEND` | `sqlite` | `sqlite`/`pgvector`/`chroma`/`qdrant` |
| `SEPTMUSE_KEYWORD_BACKEND` | `sqlite_bm25` | `sqlite_bm25`/`rank_bm25`/`none` |
| `SEPTMUSE_GRAPH_BACKEND` | `sqlite` | `sqlite`/`age`/`neo4j` |
| `SEPTMUSE_LLM` | 未设 | `openai`/`ollama`/`anthropic`/`dashscope` |
| `SEPTMUSE_LLM_MODEL` | 未设 | 覆盖 provider 默认模型 |
| `SEPTMUSE_INFER` | `false` | `true` 启用 LLM 抽取事实 |
| `SEPTMUSE_LANG` | 未设 | `zh`/`en`（仅 `auto` embedder 生效） |
| `SEPTMUSE_MODEL_CACHE` | `~/.septmuse/models/` | ONNX 模型缓存目录 |
| `SEPTMUSE_ENTITY_EXTRACTOR` | `regex` | `regex`/`spacy`/`none` |
| `SEPTMUSE_RERANKER` | `noop` | `noop`/`mmr`/`cross_encoder`/`llm` |
| `SEPTMUSE_TEST_PG_DSN` | 未设 | 测试用 Postgres DSN |
| `SEPTMUSE_TEST_NEO4J_URI` | 未设 | 测试用 Neo4j URI |

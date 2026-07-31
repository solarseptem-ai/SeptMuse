# CLI 命令参考

> 源码：`src/septmuse/cli/main.py`

`septmuse` CLI 基于 argparse，提供 12 个子命令。默认使用 SQLite + HashEmbedder（零配置、离线可用）。

## 全局选项

所有命令支持 `--db-path` 参数指定 SQLite 路径（默认 `~/.septmuse/septmuse.db` 或 `SEPTMUSE_DB_PATH` 环境变量）。

---

## init

初始化本地记忆库。

```bash
septmuse init [--user USER] [--db-path PATH]
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--user` | `SEPTMUSE_USER_ID` 或 `default` | 用户 ID |
| `--db-path` | `~/.septmuse/septmuse.db` | SQLite 路径 |

**示例：**

```bash
septmuse init --user alice
# initialized: ~/.septmuse/septmuse.db
```

---

## add

添加记忆，支持四种类型。

```bash
septmuse add CONTENT [--user USER] [--type TYPE] [--agent AGENT_ID] [--valid-at TIME] [--db-path PATH]
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `content` | （必填） | 记忆内容 |
| `--user` / `--user-id` | `SEPTMUSE_USER_ID` 或 `default` | 用户 ID |
| `--type` | `verbatim` | `verbatim`/`semantic`/`episodic`/`procedural` |
| `--agent` | `None` | agent ID |
| `--valid-at` | `None` | 事实开始为真的时间（ISO 8601） |
| `--db-path` | 默认 | SQLite 路径 |

**类型说明：**

| 类型 | 存储方式 | 示例 |
|------|----------|------|
| `verbatim` | 原文存储（默认，不 LLM 抽取） | `septmuse add "我喜欢 Python" --user alice` |
| `semantic` | 三元组（content 按空格拆分为 subject/predicate/object） | `septmuse add "Alice 喜欢 Python" --type semantic --user alice` |
| `episodic` | 情节事件（时序） | `septmuse add "部署到生产环境" --type episodic --user alice` |
| `procedural` | 程序规则 | `septmuse add "部署前必须跑测试" --type procedural --user alice` |

**输出（JSON）：**

```json
{"memory_id": "mem_abc123", "type": "verbatim"}
```

---

## invalidate

手动标记事实不再为真（双时态）。设置 `invalid_at` + `expired_at`，不删除记忆。

```bash
septmuse invalidate MEMORY_ID [--invalid-at TIME] [--db-path PATH]
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `memory_id` | （必填） | 记忆 ID |
| `--invalid-at` | 当前时间 | 失效时间（ISO 8601） |

---

## search

检索记忆，JSON 数组输出。

```bash
septmuse search QUERY [--user USER] [--top-k N] [--threshold F] [--reranker R] [--db-path PATH]
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `query` | （必填） | 查询文本 |
| `--user` / `--user-id` | `default` | 用户 ID |
| `--top-k` | `5` | 返回数 |
| `--threshold` | `0.1` | 相似度阈值（0-1） |
| `--reranker` | `None`（用配置默认） | `noop`/`mmr`/`cross_encoder`/`llm` |

**示例：**

```bash
septmuse search "喜欢什么" --user alice --top-k 3
```

**输出（JSON 数组）：**

```json
[
  {
    "id": "mem_abc123",
    "memory": "我喜欢 Python",
    "score": 0.85,
    "vector_score": 0.90,
    "bm25_score": 0.80,
    "metadata": null,
    "created_at": "2026-07-27T10:00:00Z"
  }
]
```

---

## dump

导出全部记忆。

```bash
septmuse dump [--user USER] [--format FORMAT] [--output FILE] [--db-path PATH]
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--user` | `default` | 用户 ID |
| `--format` | `json` | `json`/`markdown` |
| `--output` | stdout | 输出文件路径 |

**示例：**

```bash
septmuse dump --user alice --format markdown --output memories.md
```

---

## update

更新记忆内容。

```bash
septmuse update MEMORY_ID CONTENT [--user USER] [--db-path PATH]
```

| 参数 | 说明 |
|------|------|
| `memory_id` | 记忆 ID |
| `content` | 新内容 |

---

## history

查看记忆变更历史（ADD/UPDATE/DELETE 记录）。

```bash
septmuse history MEMORY_ID [--db-path PATH]
```

---

## block

工作记忆 Block 操作（对齐 Letta Block）。

### block set

```bash
septmuse block set AGENT_ID LABEL VALUE [--db-path PATH]
```

### block list

```bash
septmuse block list AGENT_ID [--db-path PATH]
```

**示例：**

```bash
septmuse block set agent-001 persona "You are a helpful assistant"
septmuse block list agent-001
```

---

## entities

搜索实体。

```bash
septmuse entities QUERY [--user-id USER] [--top-k N] [--db-path PATH]
```

**输出格式：**

```
  Python (TOPIC) -> ['mem_abc123', 'mem_def456']
```

---

## entity-list

列出用户全部实体。

```bash
septmuse entity-list [--user-id USER] [--entity-type TYPE] [--limit N] [--db-path PATH]
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--entity-type` | `None` | 实体类型过滤（PROPER/QUOTED/TOPIC/IDENTIFIER） |
| `--limit` | `100` | 返回数 |

---

## serve

启动 HTTP/SSE MCP server（uvicorn + FastAPI）。

```bash
septmuse serve [--host HOST] [--port PORT] [--with-rest] [--db-path PATH]
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--host` | `127.0.0.1` | 绑定地址 |
| `--port` | `8000` | 端口 |
| `--with-rest` | `False` | 额外挂载 REST API |

**示例：**

```bash
septmuse serve --with-rest --port 8000
# septmuse serve: http://127.0.0.1:8000
```

---

## mcp

启动 stdio MCP server（供 Claude Desktop / Cursor 等 MCP 客户端调用）。

```bash
septmuse mcp
```

无需额外参数。MCP 客户端通过 stdio 协议连接。

---

## version

显示版本号。

```bash
septmuse version
```

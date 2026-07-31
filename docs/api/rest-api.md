# REST API 端点参考

> 源码：`src/septmuse/api/rest/__init__.py`

SeptMuse REST API 基于 FastAPI，由 `create_app()` 创建。默认挂在 `:8000`。

## 启动

```bash
# 方式1: CLI 启动 (MCP + REST 同一 app)
septmuse serve --with-rest --port 8000

# 方式2: Python 代码
from septmuse.api.rest import create_app
app = create_app()
# uvicorn septmuse.api.rest:app
```

## 认证

| 模式 | 条件 | 行为 |
|------|------|------|
| 开发模式 | `SEPTMUSE_API_KEY` 未设 | 无认证，启动时警告一次 |
| 生产模式 | `SEPTMUSE_API_KEY` 已设 | 请求需 `Authorization: Bearer <key>`，否则 401 |

## 权限（401 vs 403 vs 404）

| 状态码 | 含义 | 触发条件 |
|--------|------|----------|
| 401 | 未认证 | API key 缺失/错误 |
| 403 | 授权失败 | 记忆存在但 state != active（deleted/archived/paused） |
| 404 | 不存在 | 记忆从未存在（`get_history` 也为空） |

GET/DELETE 端点都做权限检查（`check_memory_access_permissions`）。

## 请求/响应格式

所有端点使用 JSON。`Content-Type: application/json`。

---

## 记忆 CRUD

### POST /memories

添加记忆。

**请求体：**

```json
{
  "content": "我喜欢 Python",
  "user_id": "alice",
  "agent_id": "agent-001",
  "memory_type": "verbatim",
  "infer": false,
  "valid_at": "2024-01-01T00:00:00"
}
```

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `content` | `str` | — | 记忆内容 |
| `user_id` | `str` | — | 用户 ID |
| `agent_id` | `str` | `None` | agent ID |
| `memory_type` | `str` | `"verbatim"` | `verbatim`/`semantic`/`episodic`/`procedural` |
| `infer` | `bool` | `false` | `true`=LLM 抽取事实 |
| `valid_at` | `str` | `None` | 事实开始为真的时间 |

`memory_type` 为 `semantic` 时，`content` 按空格拆分为 subject/predicate/object。
`memory_type` 为 `episodic` 时，走 `add_episode`。
`memory_type` 为 `procedural` 时，走 `add_rule`。

**响应（201）：**

```json
{
  "results": [{"id": "mem_xxx", "memory": "我喜欢 Python", "event": "ADD"}],
  "relations": []
}
```

---

### GET /memories

列出记忆。

| 参数 | 类型 | 说明 |
|------|------|------|
| `user_id` | `str` | 用户 ID（必填） |
| `app_id` | `str` | 应用 ID（可选，审计用） |

**响应：**

```json
{"results": [{"id": "mem_xxx", "memory": "...", "metadata": null, "created_at": "..."}]}
```

---

### GET /memories/{memory_id}

取单条记忆。权限层校验存在性 + `state=active`。

**响应：**

```json
{"id": "mem_xxx", "memory": "我喜欢 Python", "metadata": null, "created_at": "..."}
```

**错误：** 403（存在但非 active）/ 404（不存在）

---

### PUT /memories/{memory_id}

更新记忆内容。

**请求体：**

```json
{"text": "我喜欢 Rust", "metadata": {"updated_by": "alice"}}
```

**响应：**

```json
{"id": "mem_xxx", "memory": "我喜欢 Rust", "event": "UPDATE"}
```

**错误：** 404（不存在）

---

### DELETE /memories/{memory_id}

删除记忆（软删除）。权限层校验 + 记录访问日志。

| 参数 | 类型 | 说明 |
|------|------|------|
| `app_id` | `str` | 应用 ID（可选，审计用） |

**响应：**

```json
{"status": "deleted", "memory_id": "mem_xxx"}
```

---

### POST /memories/{memory_id}/invalidate

手动标记事实不再为真（双时态）。

**请求体：**

```json
{"invalid_at": "2026-07-27T00:00:00"}
```

**响应：**

```json
{"id": "mem_xxx", "invalid_at": "...", "expired_at": "...", "event": "INVALIDATE"}
```

---

### GET /memories/{memory_id}/history

获取记忆变更历史（ADD/UPDATE/DELETE 记录）。

**响应：**

```json
[{"id": "log_xxx", "memory_id": "mem_xxx", "action": "ADD", "timestamp": "..."}]
```

---

### GET /memories/{memory_id}/access-logs

获取记忆访问日志（审计用）。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `limit` | `int` | `100` | 返回日志数上限（>=1） |

**响应：**

```json
{"logs": [{"memory_id": "mem_xxx", "app_id": "...", "action": "get", "timestamp": "..."}]}
```

---

## 检索

### POST /memories/search

统一检索（元认知路由）。

**请求体：**

```json
{
  "query": "喜欢什么",
  "user_id": "alice",
  "top_k": 5,
  "threshold": 0.1,
  "reranker": "mmr",
  "explain": false
}
```

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `query` | `str` | — | 查询文本 |
| `user_id` | `str` | — | 用户 ID |
| `top_k` | `int` | `5` | 返回数 |
| `threshold` | `float` | `0.1` | 相似度阈值 |
| `reranker` | `str \| None` | `None` | `noop`/`mmr`/`cross_encoder`/`llm` |
| `explain` | `bool` | `false` | `true`=返回 score_details |

**响应：**

```json
{
  "results": [
    {
      "id": "mem_xxx",
      "memory": "我喜欢 Python",
      "score": 0.85,
      "vector_score": 0.90,
      "bm25_score": 0.80,
      "entity_boost": 0.05
    }
  ]
}
```

---

### POST /memories/search/causal

反事实因果查询。

**请求体：**

```json
{"cause_event_id": "evt_xxx", "effect_event_id": "evt_yyy", "user_id": "alice"}
```

**响应：**

```json
{"would_still_occur": false, "confidence": 0.8, "reasoning": "..."}
```

---

### POST /memories/rehearse

主动复述（强化低强度高价值记忆）。

**请求体：**

```json
{"memory_id": "mem_xxx", "user_id": "alice"}
```

`memory_id` 为空时批量复述所有候选。

---

### POST /memories/capture

PostToolUse 捕获。

**请求体：**

```json
{"text": "...", "user_id": "alice", "agent_id": "agent-001"}
```

**响应：**

```json
{"captured": true, "memory_id": "mem_xxx", "deduped": false, "redacted": false}
```

---

## 元认知

### GET /memories/meta/coverage

元认知覆盖报告（L1）。

| 参数 | 说明 |
|------|------|
| `user_id` | 用户 ID（必填） |

**响应：**

```json
{
  "overall_score": 0.75,
  "weak_areas": ["math"],
  "strong_areas": ["programming"],
  "namespaces": [{"namespace": "programming", "count": 42, "coverage_score": 0.9}],
  "summary": "..."
}
```

---

## 工作记忆 Block

### GET /memories/working/blocks/{agent_id}

列出 agent 的全部 block。

---

### PUT /memories/working/blocks/{agent_id}/{label}

更新 block value。

**请求体：** `{"value": "new content"}`

---

### POST /memories/working/blocks/{agent_id}/{label}/append

追加 block 内容（对齐 Letta core_memory_append）。

**请求体：** `{"content": " appended text"}`

---

### POST /memories/working/blocks/{agent_id}/{label}/replace

替换 block 内容片段（对齐 Letta core_memory_replace）。

**请求体：** `{"old_content": "old", "new_content": "new"}`

---

## 跨 Agent 共享

### GET /agents/{user_id}/memories

跨 agent 共享读。

**响应：**

```json
{"user_id": "alice", "agents": ["agent-001", "agent-002"], "is_cross_agent": true}
```

---

## 实体

### GET /entities

搜索实体。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `query` | `str` | — | 搜索查询（必填） |
| `user_id` | `str` | `default` | 用户 ID |
| `top_k` | `int` | `5` | 返回数（>=1） |

**响应：**

```json
{"results": [{"id": "ent_xxx", "entity_text": "Python", "entity_type": "TOPIC", "linked_memory_ids": ["mem_xxx"], "score": 0.9}]}
```

---

### GET /entities/list

列出实体。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `user_id` | `str` | `default` | 用户 ID |
| `entity_type` | `str \| None` | `None` | 实体类型过滤 |
| `limit` | `int` | `100` | 返回数（>=1） |

---

## 健康检查

### GET /health

```json
{"status": "ok"}
```

---

## 端点总览

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/memories` | 添加记忆 |
| GET | `/memories` | 列出记忆 |
| GET | `/memories/{memory_id}` | 取单条 |
| PUT | `/memories/{memory_id}` | 更新记忆 |
| DELETE | `/memories/{memory_id}` | 删除记忆 |
| POST | `/memories/{memory_id}/invalidate` | 标记失效 |
| GET | `/memories/{memory_id}/history` | 变更历史 |
| GET | `/memories/{memory_id}/access-logs` | 访问日志 |
| POST | `/memories/search` | 检索 |
| POST | `/memories/search/causal` | 因果查询 |
| POST | `/memories/rehearse` | 主动复述 |
| POST | `/memories/capture` | PostToolUse 捕获 |
| GET | `/memories/meta/coverage` | 覆盖报告 |
| GET | `/memories/working/blocks/{agent_id}` | 列出 block |
| PUT | `/memories/working/blocks/{agent_id}/{label}` | 更新 block |
| POST | `/memories/working/blocks/{agent_id}/{label}/append` | 追加 block |
| POST | `/memories/working/blocks/{agent_id}/{label}/replace` | 替换 block |
| GET | `/agents/{user_id}/memories` | 跨 agent 共享 |
| GET | `/entities` | 搜索实体 |
| GET | `/entities/list` | 列出实体 |
| GET | `/health` | 健康检查 |

共 21 个端点。Swagger 文档：`http://localhost:8000/docs`

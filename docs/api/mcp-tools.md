# MCP 工具参考

> 源码：`src/septmuse/api/mcp/tools.py`

SeptMuse MCP server 通过 FastMCP `@mcp.tool` 注册 18 个工具。支持三种 transport：stdio / SSE / Streamable HTTP。

## transport 配置

| 启动方式 | 命令 | transport |
|----------|------|-----------|
| stdio | `septmuse mcp` | stdio（供 Claude Desktop / Cursor） |
| HTTP | `septmuse serve --port 8000` | SSE + Streamable HTTP |

## user_id 解析

所有带 `user_id` 参数的工具遵循统一解析逻辑：

1. 显式参数 `user_id` 优先
2. 缺省回退 `contextvar`（`user_id_var`，兼容 stdio/http）
3. 均为空时返回 `"Error: user_id not provided"`

## 错误处理

MCP 工具不抛异常，错误时返回字符串 `"Error: ..."`（便于 LLM 客户端处理）。

---

## 基础 5 工具（对齐 mem0）

### add_memories

添加记忆。用户告知任何偏好/事实时调用。

```python
add_memories(content: str, user_id: str = "", infer: bool = False, valid_at: str | None = None) -> str
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `content` | `str` | — | 记忆内容 |
| `user_id` | `str` | `""` | 用户 ID（空则用 contextvar） |
| `infer` | `bool` | `False` | `True`=LLM 抽取事实；`False`=原文存 |
| `valid_at` | `str \| None` | `None` | 事实开始为真的时间（ISO 8601） |

**返回：** JSON 字符串 `{"results": [{"id","memory","event":"ADD"}], "relations": []}`

---

### search_memory

搜索记忆。每次用户提问时调用，召回相关记忆。

```python
search_memory(query: str, user_id: str = "", top_k: int = 5, app_id: str = "", reranker: str | None = None) -> str
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `query` | `str` | — | 查询文本 |
| `user_id` | `str` | `""` | 用户 ID |
| `top_k` | `int` | `5` | 返回数 |
| `app_id` | `str` | `""` | 应用 ID（多租户权限隔离 + 审计日志） |
| `reranker` | `str \| None` | `None` | `noop`/`mmr`/`cross_encoder`/`llm` |

**返回：** JSON 字符串 `{"results": [{"id","memory","score",...}]}`

搜索自动记录访问日志（`record_access`，审计用）。

---

### list_memories

列出用户全部记忆。

```python
list_memories(user_id: str = "") -> str
```

**返回：** JSON 字符串 `{"results": [{"id","memory","metadata","created_at",...}]}`

---

### delete_memories

按 ID 删除指定记忆（软删除）。

```python
delete_memories(memory_ids: list[str], user_id: str = "") -> str
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `memory_ids` | `list[str]` | 记忆 ID 列表 |
| `user_id` | `str` | 用户 ID |

**返回：** `"Successfully deleted N/M memories"`

---

### delete_all_memories

删除用户全部记忆。

```python
delete_all_memories(user_id: str = "") -> str
```

**返回：** `"Successfully deleted all N memories"`

---

## Block 工具（对齐 Letta）

### update_memory

更新已有记忆的内容或 metadata。

```python
update_memory(memory_id: str, content: str = "", user_id: str = "", metadata: str = "") -> str
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `memory_id` | `str` | 记忆 ID |
| `content` | `str` | 新内容（空=不改） |
| `metadata` | `str` | JSON 字符串（空=不改） |

---

### update_block

更新工作记忆 Block 的值（对齐 Letta `update_block_value`）。

```python
update_block(agent_id: str, label: str, value: str, user_id: str = "") -> str
```

---

### core_memory_append

追加内容到工作记忆 Block（对齐 Letta `core_memory_append`）。

```python
core_memory_append(agent_id: str, label: str, content: str, user_id: str = "") -> str
```

---

### core_memory_replace

替换工作记忆 Block 中的内容片段（对齐 Letta `core_memory_replace`）。

```python
core_memory_replace(agent_id: str, label: str, old_content: str, new_content: str, user_id: str = "") -> str
```

---

### get_blocks

列出 agent 的工作记忆 Block 列表。

```python
get_blocks(agent_id: str, user_id: str = "") -> str
```

---

### get_memory_history

查看记忆的变更历史（ADD/UPDATE/DELETE 记录）。

```python
get_memory_history(memory_id: str, user_id: str = "") -> str
```

---

## 实体工具（Task 8）

### search_entities

搜索实体，返回实体文本 + 类型 + linked_memory_ids。

```python
search_entities(query: str, user_id: str = "", top_k: int = 5) -> str
```

**返回：** JSON `[{"id","entity_text","entity_type","linked_memory_ids","score"}]`

---

### list_entities

列出用户全部实体（可选 entity_type 过滤）。

```python
list_entities(user_id: str = "", entity_type: str = "", limit: int = 100) -> str
```

**返回：** JSON `[{"id","entity_text","entity_type","linked_memory_ids","created_at"}]`

---

## 时态工具

### invalidate_memory

手动标记事实不再为真（设置 `invalid_at` + `expired_at`，不删除记忆）。

```python
invalidate_memory(memory_id: str, invalid_at: str | None = None) -> str
```

**返回：** JSON `{"id","invalid_at","expired_at","event":"INVALIDATE"}` 或 `{"id","event":"NOT_FOUND"}`

---

## 扩展工具（架构文档 §13.4）

### remember_episode

记录成功交互的推理经验（观察/思考/行动/结果）。借鉴 LangMem Episode。

```python
remember_episode(observation: str, thoughts: str, action: str, outcome: str, user_id: str = "") -> str
```

> 参数名用 `outcome` 而非 `result`，因 FastMCP 把返回值包装为 pydantic model 的 `result` 字段，参数名 `result` 会冲突。

---

### causal_query

反事实因果查询：若某事件未发生，结果是否仍成立（架构文档 §6.1）。

```python
causal_query(cause_event_id: str, hypothesized_effect: str, user_id: str = "") -> str
```

**返回：** JSON `{"would_still_occur": bool, "confidence": float, "reasoning": str}`

---

### rehearse

触发主动复述强化低强度高价值记忆（架构文档 §6.2）。

```python
rehearse(user_id: str = "", memory_id: str = "") -> str
```

`memory_id` 为空时批量复述所有候选记忆。

---

### coverage_report

生成元认知覆盖报告：agent 记住了什么/记不住什么（架构文档 §6.3）。

```python
coverage_report(user_id: str = "") -> str
```

**返回：** JSON `{"overall_score": float, "weak_areas": [...], "strong_areas": [...], "namespaces": [...], "summary": str}`

---

## 工具总数

共 18 个工具（`mcp_tools_registered, count=18`）：

| 分类 | 工具 |
|------|------|
| 基础 CRUD（对齐 mem0） | `add_memories`, `search_memory`, `list_memories`, `delete_memories`, `delete_all_memories` |
| Block（对齐 Letta） | `update_memory`, `update_block`, `core_memory_append`, `core_memory_replace`, `get_blocks`, `get_memory_history` |
| 实体 | `search_entities`, `list_entities` |
| 时态 | `invalidate_memory` |
| 架构扩展 | `remember_episode`, `causal_query`, `rehearse`, `coverage_report` |

## 注意事项

- **禁止** `from __future__ import annotations`（FastMCP `func_metadata` 把返回注解当字符串解析会炸）
- 工具签名必须用具体类型（`list[str]` 而非 `list`）
- `user_id` 默认从 `contextvars` 读，错误时返回字符串 `"Error: ..."`（非异常）

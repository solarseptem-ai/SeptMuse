# mem0 风格 filters dict + session_id 全入口暴露设计

> 日期: 2026-07-31
> 状态: 设计已批准

## 目标

与 mem0 API 对齐：filters dict 全量操作符（eq/ne/gt/gte/lt/lte/in/nin/contains/icontains/wildcard + AND/OR/NOT）+ session_id 在 REST/CLI/MCP 全入口暴露。

## 设计

### 1. FiltersParser（新建 `storage/filters.py`）

解析 mem0 风格 filters dict → SQL WHERE 子句 + 参数列表。

**直接字段**（memories 表列）：`user_id` / `agent_id` / `session_id` / `run_id`（映射到 session_id）/ `state`

**metadata 字段**：通过 `json_extract(metadata, '$.key')` 查询

**操作符映射**：

| 操作符 | SQL | 适用 |
|--------|-----|------|
| 省略（直接值） | `= ?` | 全部 |
| `eq` | `= ?` | 全部 |
| `ne` | `<> ?` | 全部 |
| `gt` | `> ?` | 数值 |
| `gte` | `>= ?` | 数值 |
| `lt` | `< ?` | 数值 |
| `lte` | `<= ?` | 数值 |
| `in` | `IN (?, ?, ...)` | 全部 |
| `nin` | `NOT IN (?, ?, ...)` | 全部 |
| `contains` | `LIKE '%' || ? || '%'` | 文本 |
| `icontains` | `LOWER(col) LIKE LOWER('%' || ? || '%')` | 文本 |
| `"*"`（通配符） | `IS NOT NULL` | 存在性检查 |

**逻辑运算**：

| 运算 | SQL 构造 |
|------|----------|
| `AND` | `(clause1 AND clause2 AND ...)` |
| `OR` | `(clause1 OR clause2 OR ...)` |
| `NOT` | `NOT (clause1)` |

**直接参数优先**：如果 `session_id=` 和 `filters={"session_id": "x"}` 同时存在，直接参数覆盖 filters 中的同名 key（filters 中的被移除）。

### 2. Store 层改造

`SQLiteMemoryStore` + `AsyncSQLiteMemoryStore` 的 `search` / `get_all` / `get` / `delete` 加 `filters: dict[str, Any] | None = None` 参数。

WHERE 子句构建逻辑：
1. 基础条件：`user_id=? AND is_deleted=0 AND (state='active' OR state IS NULL)`
2. 直接参数追加：`session_id = ?` / `agent_id = ?`
3. filters 解析：`FiltersParser().parse(filters)` → 追加 WHERE 子句

PG 后端用 `metadata->>'key'` 替代 `json_extract(metadata, '$.key')`。

### 3. Facade 层

`Memory.search()` / `Memory.get_all()` / `AsyncMemory.search()` / `AsyncMemory.get_all()` 加 `filters: dict[str, Any] | None = None` 参数，透传 store。

### 4. REST API

- `GET /memories` 加 `session_id: str | None = None` query 参数
- `POST /memories/search` 的 `SearchRequest` 加 `filters: dict[str, Any] | None = None` 字段

### 5. CLI

- `septmuse search` 加 `--session-id` / `--agent-id`
- `septmuse dump` 加 `--session-id` / `--agent-id`

### 6. MCP

`search_memory` 工具加 `filters: dict | None` 参数。

### 7. 测试

`tests/unit/test_filters.py`：FiltersParser 全量操作符 + 逻辑运算 + 直接参数优先。

### 文件清单

| 文件 | 操作 |
|------|------|
| `storage/filters.py` | 新建 — FiltersParser |
| `storage/sqlite/store.py` | 修改 — search/get_all/get/delete 加 filters |
| `storage/async_sqlite/store.py` | 修改 — 同上 |
| `memory/main.py` | 修改 — search/get_all 加 filters |
| `memory/async_main.py` | 修改 — search/get_all 加 filters |
| `api/rest/__init__.py` | 修改 — GET /memories + SearchRequest |
| `cli/main.py` | 修改 — search/dump 加 --session-id |
| `api/mcp/tools.py` | 修改 — search_memory 加 filters |
| `tests/unit/test_filters.py` | 新建 — FiltersParser 测试 |

## 验证标准

1. ruff check 全绿
2. 现有 1076 passed + 36 skipped + 23 failed 不退化
3. 新增 ~15 个 FiltersParser 测试
4. REST `GET /memories?session_id=xxx` 可用
5. CLI `septmuse search --session-id xxx` 可用

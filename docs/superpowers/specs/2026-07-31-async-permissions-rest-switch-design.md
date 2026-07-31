# async 权限检查 + REST API 核心端点切换设计

> 日期: 2026-07-31
> 状态: 设计已批准

## 目标

创建 async 版权限检查函数 + 访问日志函数，解锁 REST API 核心端点切换到 AsyncMemory。

## 背景

当前 REST API 21+ 端点全用 `ExperimentalMemory`（sync）。`check_memory_access_permissions` 和 `record_access` 直接调 sync `store.get()` / `store._record_access_log()`。`AsyncSQLiteMemoryStore` 的同名方法是 async，不兼容。

## 设计

### 1. 新建 async 权限检查函数

**文件**: `governance/async_permissions.py`

```python
async def async_check_memory_access_permissions(
    store: AsyncMemoryStore, memory_id: str, app_id: str | None = None
) -> tuple[bool, str]:
```

逻辑与 sync 版 `check_memory_access_permissions` 完全一致（4 层检查），只是 `await store.get(memory_id)`。

### 2. 新建 async 访问日志函数

**文件**: `governance/async_access_log.py`

```python
async def async_record_access(
    store: AsyncMemoryStore, memory_id: str, app_id: str | None,
    access_type: str, metadata: dict[str, Any] | None = None,
) -> str | None:
```

逻辑与 sync 版 `record_access` 一致：`hasattr(store, "_record_access_log")` 检查 + `await store._record_access_log(...)`。吞错（日志失败不阻塞业务）。

### 3. AsyncSQLiteMemoryStore 补齐访问日志

**文件**: `storage/async_sqlite/store.py`（修改）

- `_create_tables` 加 `memory_access_logs` 表（DDL 与 sync 版一致）
- 加 `async _record_access_log(memory_id, app_id, access_type, metadata) -> str | None`
- 加 `async get_access_logs(memory_id, limit=100) -> list[dict]`（覆盖 ABC 默认）

`memory_access_logs` DDL:
```sql
CREATE TABLE IF NOT EXISTS memory_access_logs (
    id           TEXT PRIMARY KEY,
    memory_id    TEXT NOT NULL,
    app_id       TEXT,
    access_type  TEXT NOT NULL,
    metadata     TEXT,
    accessed_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_access_logs_memory ON memory_access_logs(memory_id);
```

### 4. AsyncMemory 加 invalidate

**文件**: `memory/async_main.py`（修改）

```python
async def invalidate(self, memory_id: str, *, invalid_at: str | None = None) -> dict[str, Any]:
    """异步标记事实不再为真。"""
    return await self.store.invalidate(memory_id, invalid_at=invalid_at)
```

### 5. REST API 核心端点切换

**文件**: `api/rest/__init__.py`（修改）

#### create_app 改造

```python
def create_app(memory=None) -> FastAPI:
    if memory is None:
        config = MemoryConfig()  # 从环境变量读 db_path
        async_memory = AsyncMemory(config=config, embedder=HashEmbedder())
        memory = ExperimentalMemory(config=config, embedder=HashEmbedder())
    elif isinstance(memory, MemoryConfig):
        async_memory = AsyncMemory(config=memory, embedder=HashEmbedder())
        memory = ExperimentalMemory(config=memory, embedder=HashEmbedder())
    ...
    app.state.async_memory = async_memory
    app.state.memory = memory
    register_routes(app, memory, async_memory)
```

**行为变更**: 默认 `MemoryConfig()` 不再硬编码 `db_path=":memory:"`。REST 测试已设 `SEPTMUSE_DB_PATH` 环境变量（文件路径），双 memory 共享同一 DB 文件。

#### 核心端点改造（9 个）

| 端点 | 改造 |
|------|------|
| POST /memories | `await app.state.async_memory.add(...)` |
| GET /memories | `await app.state.async_memory.get_all(...)` → `{"results": results}` + `async_record_access` |
| POST /memories/search | `await app.state.async_memory.search(...)` → `{"results": results}` |
| GET /memories/{id} | `async_check_permissions` + `await async_memory.get(...)` + `async_record_access` |
| PUT /memories/{id} | `await async_memory.update(...)` → `{"event": "UPDATE", ...}` 格式适配 |
| DELETE /memories/{id} | `async_check_permissions` + `await async_memory.delete(...)` + `async_record_access` |
| GET /memories/{id}/history | `await async_memory.get_history(...)` |
| GET /memories/{id}/access-logs | `await async_memory.store.get_access_logs(...)` |
| POST /memories/{id}/invalidate | `await async_memory.invalidate(...)` |

#### 实验端点（12 个）: 不变

保持用 `app.state.memory`（ExperimentalMemory，sync）。

### 6. 测试

**新建**:
- `tests/unit/test_async_permissions.py` — 5 测试: active/deleted/paused/no_app_id/empty_app_id
- `tests/unit/test_async_access_log.py` — 3 测试: returns_log_id/swallows_exceptions/unsupported_store

## 验证标准

1. ruff check 全绿
2. 37 个 REST 测试保持通过
3. 23 个预存在失败不超过基线
4. 1050+ passed（新增 8 个 async 测试）

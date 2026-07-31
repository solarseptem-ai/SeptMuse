# 轻量级数据迁移机制设计

> 日期: 2026-07-31
> 状态: 设计已批准

## 目标

为 SeptMuse 创建轻量级 schema 迁移机制：版本追踪 + 有序迁移 + init 自动执行 + CLI 手动触发。零外部依赖，符合"零配置"哲学。

## 背景

当前迁移靠 `SQLiteMemoryStore.__init__` 中硬编码调用 4 个 `_migrate_add_*` 方法（运行时 `ALTER TABLE`），无版本追踪、无顺序保证、无状态验证。`AsyncSQLiteMemoryStore` 全列硬编码（旧 DB 需迁移）。`PGVectorStore` 用 PG 特有语法。`alembic/` 空壳。

## 设计

### 1. `schema_version` 表

每个 DB 自动创建，追踪已应用的迁移版本：

```sql
CREATE TABLE IF NOT EXISTS schema_version (
    version     TEXT PRIMARY KEY,   -- "001", "002", ...
    description TEXT,
    applied_at  TEXT NOT NULL
);
```

由 MigrationRunner 自动创建（不依赖 store 的 `_create_tables`）。

### 2. 迁移模块（5 个）

每个迁移一个 Python 模块，包含 `VERSION`、`DESCRIPTION`、`upgrade(ctx)` 函数。

| 模块 | 版本 | 内容 |
|------|------|------|
| `m001_initial_schema.py` | "001" | CREATE memories（基础 9 列）+ history + idx_memories_user |
| `m002_state_columns.py` | "002" | ALTER memories ADD state / deleted_at / app_id |
| `m003_session_id.py` | "003" | ALTER memories ADD session_id |
| `m004_temporal.py` | "004" | ALTER memories ADD valid_at / invalid_at / expired_at |
| `m005_access_logs.py` | "005" | CREATE memory_access_logs + idx_access_logs_memory |

**不纳入迁移的表**（自管理，已用 `CREATE TABLE IF NOT EXISTS` 幂等创建）：
- `septmuse_entities`（EntityStore 自建）
- `entity_relations`（CognifyPipeline 自建）
- `docs`（SQLiteBM25Index 自建）
- `memory_links`（SQLiteGraphStore 自建）
- `vector_entries`（SQLiteVectorStore 自建）

### 3. MigrationContext（backend 抽象）

统一 SQLite/PG 的迁移操作接口：

```python
class MigrationContext:
    def __init__(self, conn, backend="sqlite"):
        self.conn = conn
        self.backend = backend

    def has_column(self, table: str, column: str) -> bool:
        """检查列是否存在（SQLite 用 PRAGMA，PG 用 information_schema）。"""
        ...

    def has_table(self, table: str) -> bool:
        """检查表是否存在。"""
        ...

    def execute(self, sql: str):
        """执行 DDL。PG 自动加 IF NOT EXISTS。"""
        ...
```

### 4. MigrationRunner（sync + async）

```python
class MigrationRunner:
    """同步迁移执行器。"""
    def __init__(self, conn, backend="sqlite"):
        ...

    def run(self) -> list[str]:
        """检查 schema_version，执行未应用的迁移，返回新应用的版本列表。"""
        self._ensure_schema_version_table()
        applied = self._get_applied_versions()
        newly = []
        for m in MIGRATIONS:
            if m.version in applied:
                continue
            ctx = MigrationContext(self.conn, self.backend)
            m.upgrade(ctx)
            self._record(m)
            newly.append(m.version)
        return newly


class AsyncMigrationRunner:
    """异步迁移执行器（aiosqlite）。"""
    async def run(self) -> list[str]:
        """同上，但 await conn.execute()。"""
        ...
```

### 5. 迁移注册表

```python
# src/septmuse/storage/migrations/__init__.py
from septmuse.storage.migrations.m001_initial_schema import Migration as M001
from septmuse.storage.migrations.m002_state_columns import Migration as M002
...

MIGRATIONS = [M001, M002, M003, M004, M005]
```

每个迁移模块导出一个 `Migration` dataclass/NamedTuple：
```python
Migration = namedtuple("Migration", ["version", "description", "upgrade"])
```

### 6. Store 集成

**SQLiteMemoryStore（sync）**：
- `__init__` 中删除 4 个 `_migrate_add_*` 调用 + `_create_access_logs_table` 调用
- 保留 `_create_tables()`（CREATE IF NOT EXISTS memories + history）
- 新增：`MigrationRunner(self.conn, "sqlite").run()`

**AsyncSQLiteMemoryStore（async）**：
- `_ensure_conn` 中保留 `_create_tables()`（CREATE IF NOT EXISTS 全表）
- 新增：`await AsyncMigrationRunner(self._conn, "sqlite").run()`
- 新 DB：迁移快速跑完（幂等），旧 DB：补齐缺失列

**PGVectorStore（sync）**：
- `_create_tables` 中保留现有 DDL
- 新增：`MigrationRunner(cur, "postgres").run()`

### 7. CLI 命令

`septmuse migrate` — 手动触发迁移，打印已应用版本：

```
$ septmuse migrate
已应用迁移:
  001 - initial schema (memories + history)
  002 - state columns (state/deleted_at/app_id)
  ...
schema_version: 5 migrations applied
```

### 8. 测试

| 测试文件 | 测试内容 |
|----------|----------|
| `tests/unit/test_migration_runner.py` | 空库全量迁移 / 已迁移跳过 / 幂等 / schema_version 表正确 |
| `tests/unit/test_migration_context.py` | has_column / has_table / PG IF NOT EXISTS 适配 |
| `tests/unit/test_migrations.py` | 5 个迁移模块各自的 DDL 正确性 |

### 9. 文件结构

```
src/septmuse/storage/migrations/
  __init__.py              — MIGRATIONS 注册表
  context.py               — MigrationContext
  runner.py                — MigrationRunner + AsyncMigrationRunner
  m001_initial_schema.py   — CREATE memories + history
  m002_state_columns.py    — ALTER ADD state/deleted_at/app_id
  m003_session_id.py       — ALTER ADD session_id
  m004_temporal.py         — ALTER ADD valid_at/invalid_at/expired_at
  m005_access_logs.py      — CREATE memory_access_logs
```

## 验证标准

1. ruff check 全绿
2. 新增 ~10 个测试全通过
3. 现有 1058 passed 不退化（23 failed 基线不变）
4. `septmuse migrate` CLI 命令可用
5. 新建 DB 和旧 DB 都能正确迁移

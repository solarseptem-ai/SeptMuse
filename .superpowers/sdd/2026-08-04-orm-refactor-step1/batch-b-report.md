# Batch B Report — SQLiteMemoryStore → ORMMemoryStore 迁移

## Status: DONE_WITH_CONCERNS

## 测试结果汇总

| 文件 | passed | skipped | failed | 总计 |
|------|--------|---------|--------|------|
| `tests/unit/test_permissions.py` | 8 | 1 | 0 | 9 |
| `tests/unit/test_memory_state.py` | 9 | 0 | 0 | 9 |
| `tests/unit/test_temporal.py` | 25 | 0 | 0 | 25 |
| `tests/unit/test_update.py` | 41 | 0 | 0 | 41 |
| `tests/unit/test_database_service.py` | 9 | 0 | 0 | 9 |
| **合计** | **92** | **1** | **0** | **93** |

Ruff check: All checks passed (line-length 120, no unused imports, isort 正确).

## 改动文件

1. **`tests/unit/test_permissions.py`**
   - 构造: `SQLiteMemoryStore(db_path=...)` → `ORMemoryStore(create_engine(...))`
   - 删除 fixture 中的 ALTER TABLE 迁移块 (state 列已由 SQLModel 创建)
   - `store.conn.execute("UPDATE ...")` → `with store.engine.connect() as conn: conn.execute(text(...)); conn.commit()`
   - **skip** `test_check_none_state_treated_as_active` (见 Concerns)

2. **`tests/unit/test_memory_state.py`**
   - 构造: `SQLiteMemoryStore(db_path=...)` → `ORMemoryStore(create_engine(...))`
   - `store.conn.execute("SELECT ...")` → `with store.engine.connect() as conn: conn.execute(text(...)).fetchone()`
   - `test_old_data_migration_sets_active`: 原 DROP TABLE + CREATE 旧 schema + `_migrate_add_state_columns()` → 改为验证 ORM 插入的记忆 state 默认 'active' (迁移概念在 ORMMemoryStore 不存在)
   - `test_columns_not_duplicated_on_re_migration`: `_migrate_add_state_columns()` 两次 → `_create_tables()` 两次 + `inspect(engine).get_columns()`

3. **`tests/unit/test_temporal.py`**
   - 构造: `SQLiteMemoryStore(db_path=...)` → `ORMemoryStore(create_engine(...))` (13 处直接构造)
   - `store.conn.execute("PRAGMA ...")` → `inspect(engine).get_columns("memories")`
   - `_migrate_add_temporal_columns()` → `_create_tables()` (幂等性测试)
   - `store.conn.execute("SELECT ...")` → `engine.connect() + text()`
   - 非直接构造的测试 (ExperimentalMemory/CLI/REST) 不受影响

4. **`tests/unit/test_update.py`**
   - 构造: `SQLiteMemoryStore(db_path=...)` → `ORMemoryStore(create_engine(...))` (5 处)
   - `test_update_history_recorded`: `with store._lock:` + `store.conn.execute(...)` → `with engine.connect() as conn: conn.execute(text(...)).fetchall()` (移除 lock)

5. **`tests/unit/test_database_service.py`**
   - `test_engine_query_works`: `SQLiteMemoryStore(db_path=...)` 建表 → `ORMemoryStore(create_engine(...))` 建表
   - `test_session_maker`: 同上
   - 注释引用更新 (SQLiteMemoryStore → ORMMemoryStore)

## 迁移模式应用

| 模式 | 应用位置 |
|------|----------|
| Construction 替换 | 全部 5 文件 |
| `store.conn.execute` → `engine.connect() + text()` | permissions, memory_state, temporal, update |
| `PRAGMA table_info` → `inspect(engine).get_columns()` | memory_state, temporal |
| `store._lock` 移除 | update (test_update_history_recorded) |
| `DROP TABLE/CREATE TABLE` 移除 | memory_state (test_old_data_migration_sets_active) |
| `ALTER TABLE` 移除 | permissions (fixture), memory_state |

## Concerns

### 1. `test_check_none_state_treated_as_active` (test_permissions.py) — SKIPPED

**原因**: `MemoryTable.state` 定义为 `state: str = Field(default="active")`，SQLModel 生成 NOT NULL 约束。原测试通过 `UPDATE memories SET state = NULL` 模拟旧数据 NULL state，在 ORMMemoryStore 下触发 `IntegrityError: NOT NULL constraint failed: memories.state`。

**影响**: `check_memory_access_permissions` 中的 `state or "active"` 防御逻辑仍在源码中保留，但 DB 层已结构上保证 state 不为 NULL，该路径无法通过真实 DB 触达。测试标记为 `@pytest.mark.skip` 并保留向后兼容意图说明。

### 2. `test_old_data_migration_sets_active` (test_memory_state.py) — 改写

**原因**: 原测试 DROP TABLE + CREATE 旧 schema (无 state 列) + INSERT 旧行 + 调用 `_migrate_add_state_columns()` 验证迁移后 state='active'。ORMMemoryStore 无此迁移方法 — `SQLModel.metadata.create_all()` 一次性创建完整 schema (含 state 列)。

**处理**: 改写为验证 ORM 插入的记忆 state 默认 'active'，保留测试名和"旧数据迁移"语义文档，但测试逻辑从"ALTER TABLE 迁移"改为"SQLModel 默认值验证"。

### 3. `_migrate_add_*` 系列方法不再被测试覆盖

**原因**: `test_memory_state.py` 和 `test_temporal.py` 原本测试 `_migrate_add_state_columns()` 和 `_migrate_add_temporal_columns()` 的幂等性。ORMMemoryStore 用 `_create_tables()` (SQLModel.metadata.create_all) 替代，已改写为测试 `_create_tables()` 幂等性。

**影响**: SQLiteMemoryStore 删除 (Step 3) 后，`_migrate_add_*` 方法将无测试覆盖。但 ORMMemoryStore 不需要这些方法 (SQLModel 建表即完整 schema)，所以删除是安全的。

### 4. `state` 列 NOT NULL 差异

ORMMemoryStore 的 `state` 列 NOT NULL (SQLModel `Field(default="active")` + `str` 类型)；SQLiteMemoryStore 的 `state` 列允许 NULL (`ALTER TABLE ADD COLUMN state TEXT DEFAULT 'active'`)。这是 schema 强化的正向变更，但需确认生产环境无依赖 state=NULL 的逻辑。

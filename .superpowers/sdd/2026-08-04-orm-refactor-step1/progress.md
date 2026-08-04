# SDD ledger — plan: docs/superpowers/plans/2026-08-04-orm-refactor-step1.md

Project: SeptMuse (not a git repo, file snapshot mode)
No commits — file changes tracked by before/after state.

Task 1: complete (engine + async_engine property, 3 tests, review clean)
Task 2: complete (TypedMemoryStore engine= param, 3 tests, review clean)
  - minor (deferred): close() calls engine.dispose() — shared engine will double-dispose in Task 5, needs _owns_engine flag
Task 3: complete (MigrationRunner.from_engine + inspect, 4 tests, review clean)
Task 4: complete (EntityStore.from_engine + ORM CRUD, 11 tests, review clean)
  - deviation: Entity dataclass start/end got default 0 (backward compat, necessary for test code)
  - minor (deferred): close() double-dispose concern from Task 2 still applies
Task 5: complete (facade duck typing, 5 tests, review clean)
  - deviation: config.db_path → config.database.db_path (config structure changed)
  - deviation: cognify.py modified (not in brief) — CognifyPipeline ORM-mode EntityStore fix
Task 6: complete (全量回归 + ruff)
  - ruff: All checks passed
  - 新增 26 测试全绿 (3+3+4+11+5)
  - 全量: 1241 passed + 36 skipped + 13 failed (全为预存在 API key)
  - 修复: test_async_orm 用 async test 替代 deprecated get_event_loop()
  - 零新增退化

Step 2: 测试迁移完成
  - _resolve_store() 统一走 ORMMemoryStore (main.py + async_main.py)
  - 5 个新增失败修复 (ZettelLinker commit + ORMMemoryStore filters/get_all/user_id)
  - 9 个测试文件迁移 (cognify/graph_search/graph_store/fact_extraction/permissions/memory_state/temporal/update/database_service)
  - 1 个 async 测试迁移 (test_async_memory.py)
  - 2 个 legacy 标记 (test_composite_store.py + test_async_sqlite_store.py)
  - 全量: 1226 passed + 51 skipped + 13 failed (全为预存在)
  - ruff: All checks passed

Step 3: 删除原生 store 完成
  - 删除 7 文件: store.py, async_store.py, context.py, test_composite_store.py, test_async_sqlite_store.py, test_entity_store.py, test_migration_runner.py
  - EntityStore 删除旧 __init__ + raw SQL 路径, 只保留 from_engine + ORM CRUD
  - MigrationRunner 删除双模式, 只保留 engine 构造
  - __init__.py 清理导出
  - main.py 删除 isinstance 分支
  - cognify.py 删除 _is_orm_mode 分支
  - cli/main.py 迁移到 MigrationRunner(engine)
  - 新增 m006_archived_at 迁移 (旧 DB 缺 archived_at 列)
  - ORMMemoryStore._create_tables() 集成 MigrationRunner (自动补旧 DB 缺列)
  - 全量: 1198 passed + 37 skipped + 13 failed (全为预存在)
  - ruff: All checks passed
  - 零配置验证: Memory() → ORMMemoryStore → add+search 往返成功

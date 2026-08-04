# Batch A Migration Report — Step 1: SQLiteMemoryStore → ORMMemoryStore

**Date:** 2026-08-04  
**Task:** Migrate 4 test files from `SQLiteMemoryStore` to `ORMMemoryStore` so SQLiteMemoryStore can be deleted in Step 3.

---

## Status: DONE

All 4 test files migrated successfully. All tests pass, ruff check passes, no remaining `SQLiteMemoryStore` references in the migrated files.

---

## Test Summary

| File | Tests Passed | Tests Skipped | Total |
|------|-------------|---------------|-------|
| `tests/unit/test_cognify.py` | 16 | 0 | 16 |
| `tests/unit/test_graph_search.py` | 15 | 0 | 15 |
| `tests/unit/test_graph_store.py` | 27 | 4 | 31 |
| `tests/unit/test_fact_extraction.py` | 19 | 0 | 19 |
| **Combined** | **77** | **4** | **81** |

The 4 skipped tests are `TestAGEGraphStoreIntegration` tests requiring `SEPTMUSE_TEST_PG_DSN` (Postgres + Apache AGE) — not related to this migration.

---

## Files Changed

### 1. `tests/unit/test_cognify.py`
- **5 SQLiteMemoryStore constructions** replaced with `ORMMemoryStore` (in `TestCognifyPipelineDirect` class):
  - `test_pipeline_with_mock_llm`
  - `test_pipeline_without_entity_store`
  - `test_pipeline_without_graph_store`
  - `test_pipeline_relations_idempotent`
  - `test_pipeline_empty_text`
- **4 SQLiteGraphStore constructions** updated: `SQLiteGraphStore(store.conn, store._lock)` → `SQLiteGraphStore(raw_conn, threading.Lock())` via `store.engine.raw_connection()`
- **4 EntityStore constructions** updated: `EntityStore(store.conn, store._lock, embedder)` → `EntityStore.from_engine(store.engine, embedder)`
- Added module-level imports: `import threading`, `from sqlmodel import create_engine`, `from septmuse.storage.relational_stores.orm_store import ORMMemoryStore`
- Removed all local `from septmuse.storage.relational_stores.store import SQLiteMemoryStore` imports

### 2. `tests/unit/test_graph_search.py`
- **1 SQLiteMemoryStore construction** replaced in `store_and_graph` fixture
- **1 SQLiteGraphStore construction** updated: `SQLiteGraphStore(store.conn, store._lock)` → `SQLiteGraphStore(raw_conn, threading.Lock())`
- Updated module-level imports: replaced `SQLiteMemoryStore` with `ORMemoryStore`, added `threading` and `create_engine`

### 3. `tests/unit/test_graph_store.py`
- **2 SQLiteMemoryStore constructions** replaced (`graph_store` and `store_with_data` fixtures, both using `:memory:`)
- **1 SQLiteGraphStore construction** updated in `graph_store` fixture
- Type annotations updated: `Iterator[SQLiteMemoryStore]` → `Iterator[ORMMemoryStore]`, `store_with_data: SQLiteMemoryStore` → `store_with_data: ORMMemoryStore` (6 occurrences)
- Comment/docstring updated to reference `ORMMemoryStore`
- Updated module-level imports: replaced `SQLiteMemoryStore` with `ORMemoryStore`, added `threading` and `create_engine`

### 4. `tests/unit/test_fact_extraction.py`
- **1 SQLiteMemoryStore construction** replaced in `test_extract_and_store_returns_linked_memory_ids` (verbatim_store)
- Local import updated: `from septmuse.storage.relational_stores.store import SQLiteMemoryStore` → `from sqlmodel import create_engine` + `from septmuse.storage.relational_stores.orm_store import ORMMemoryStore`

---

## Migration Patterns Applied

1. **Construction**: `SQLiteMemoryStore(db_path=tmp)` → `ORMMemoryStore(create_engine(f"sqlite:///{tmp}"))`
2. **`:memory:` SQLite**: `SQLiteMemoryStore(db_path=":memory:")` → `ORMMemoryStore(create_engine("sqlite://"))` (SQLAlchemy uses StaticPool by default for in-memory SQLite, so all connections share the same DB)
3. **SQLiteGraphStore**: `SQLiteGraphStore(store.conn, store._lock)` → `SQLiteGraphStore(store.engine.raw_connection(), threading.Lock())`
4. **EntityStore**: `EntityStore(store.conn, store._lock, embedder)` → `EntityStore.from_engine(store.engine, embedder)`
5. **No raw SQL access** via `store.conn` was needed in these 4 test files (no PRAGMA or direct SQL queries to migrate)

---

## Verification Evidence

```
$env:PYTHONPATH="src"; python -m pytest tests/unit/test_cognify.py tests/unit/test_graph_search.py tests/unit/test_graph_store.py tests/unit/test_fact_extraction.py -v --tb=short
→ 77 passed, 4 skipped in 23.21s

ruff check --no-cache <4 files>
→ All checks passed!

ruff format --check <4 files>
→ 4 files already formatted

Select-String -Pattern "SQLiteMemoryStore" <4 files>
→ (no output — zero remaining references)
```

---

## Concerns

None. All migrations follow the provided patterns exactly. The `:memory:` SQLite case works correctly because SQLAlchemy uses `StaticPool` by default for in-memory SQLite databases, ensuring `engine.raw_connection()` shares the same in-memory database as `Session(engine)`. This was verified by the existing `test_orm_memory_store.py` which uses the same `create_engine("sqlite://")` pattern.

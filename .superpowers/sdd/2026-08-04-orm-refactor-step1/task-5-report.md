# Task 5: Facade Duck Typing — Report

## What was implemented

### 1. Memory facade duck typing (`src/septmuse/memory/main.py`)
Replaced `isinstance(self.store, SQLiteMemoryStore)` checks with duck typing using `getattr(self.store, "engine", None)`:
- **graph_store**: ORMMemoryStore path gets `raw_connection()` from engine + creates `SQLiteGraphStore`; SQLiteMemoryStore path unchanged
- **typed_store**: ORMMemoryStore path uses `TypedMemoryStore(engine=store_engine)` (shared engine); SQLiteMemoryStore path uses `TypedMemoryStore(db_path=...)` (self-created engine)
- **entity_store**: ORMMemoryStore path uses `EntityStore.from_engine(store_engine, embedder)`; SQLiteMemoryStore path unchanged

### 2. TypedMemoryStore double-dispose fix (`src/septmuse/storage/relational_stores/typed_store.py`)
- Added `self._owns_engine = False` when engine is shared (passed in), `self._owns_engine = True` when self-created
- `close()` now checks `getattr(self, "_owns_engine", True)` before calling `engine.dispose()` — shared engines are not disposed by TypedMemoryStore

### 3. CognifyPipeline ORM-mode fix (`src/septmuse/extraction/cognify.py`)
- `CognifyPipeline.__init__` now checks `entity_store._is_orm_mode()` — when ORM mode (from `EntityStore.from_engine()`), gets a raw connection from `entity_store._engine` and creates a new `threading.Lock()` instead of using `entity_store._conn` (which is None in ORM mode)
- This was needed because the ORM-mode EntityStore has `_conn=None` and `_lock=None`, causing `'NoneType' object does not support the context manager protocol` in `_init_relations_table()` and `_store_relation()`

## TDD Evidence

### RED (before implementation)
```
FAILED tests/unit/test_facade_orm_path.py::test_facade_orm_path_entity_store_not_none   (entity_store is None)
FAILED tests/unit/test_facade_orm_path.py::test_facade_orm_path_typed_store_shares_engine (typed_store doesn't share engine)
FAILED tests/unit/test_facade_orm_path.py::test_facade_orm_path_graph_store_not_none     (graph_store is None)
2 passed (add_search_roundtrip + cognify_works — cognify passed via try/except catching the NoneType error)
```

### GREEN (after implementation)
```
tests/unit/test_facade_orm_path.py::test_facade_orm_path_entity_store_not_none PASSED [ 20%]
tests/unit/test_facade_orm_path.py::test_facade_orm_path_typed_store_shares_engine PASSED [ 40%]
tests/unit/test_facade_orm_path.py::test_facade_orm_path_graph_store_not_none PASSED [ 60%]
tests/unit/test_facade_orm_path.py::test_facade_orm_path_add_search_roundtrip PASSED [ 80%]
tests/unit/test_facade_orm_path.py::test_facade_orm_path_cognify_works PASSED [100%]
============================== 5 passed in 2.75s ==============================
```

### Lint
```
ruff check --no-cache: All checks passed!
ruff format --check: 4 files already formatted
```

### Regression (existing facade tests)
```
6 failed (pre-existing API key failures — unchanged), 109 passed, 3 skipped
```
The 6 failures are all in `TestResolveEmbedderOpenAI` and `TestResolveLLMBaseUrl` — pre-existing API key/provider configuration issues, not caused by this change.

## Files Changed

1. **`tests/unit/test_facade_orm_path.py`** (NEW) — 5 tests for ORMMemoryStore facade path
   - Note: Adapted `config.db_path = ...` from brief to `config.database.db_path = ...` because `MemoryConfig.db_path` is now a read-only property delegating to `DatabaseConfig.db_path` (config was refactored to nested structure after brief was written)

2. **`src/septmuse/memory/main.py`** (MODIFIED) — duck typing in `Memory.__init__`
   - Replaced 2 `isinstance(self.store, SQLiteMemoryStore)` checks with `getattr(self.store, "engine", None)` duck typing
   - 3 code paths now: ORM engine (shared) → SQLiteMemoryStore → fallback

3. **`src/septmuse/storage/relational_stores/typed_store.py`** (MODIFIED) — `_owns_engine` flag
   - `__init__`: sets `self._owns_engine = False` (shared) or `True` (self-created)
   - `close()`: only disposes if `_owns_engine` is True

4. **`src/septmuse/extraction/cognify.py`** (MODIFIED) — ORM-mode EntityStore support
   - `__init__`: when `entity_store._is_orm_mode()`, gets `raw_connection()` from engine + creates `threading.Lock()`

## Self-Review Findings

### Adaptation from brief
The brief's test code used `config.db_path = str(tmp_path / "facade.db")`, but `MemoryConfig.db_path` is a read-only property (delegating to `config.database.db_path`). Adapted to `config.database.db_path = ...` to work with the current nested config structure. The intent is preserved — the config points to the tmp db file.

### Extra fix: cognify.py
The brief only listed `main.py` and `typed_store.py` as files to modify. However, the cognify test (`test_facade_orm_path_cognify_works`) required fixing `CognifyPipeline.__init__` to handle ORM-mode EntityStore (where `_conn` and `_lock` are None). This was discovered during TDD — the test exposed that `CognifyPipeline` assumed `entity_store._conn` was always non-None. The fix is minimal: check `_is_orm_mode()` and get a raw connection from the engine when in ORM mode. All 46 existing cognify/entity_store/typed_store tests still pass.

### Double-dispose fix verification
The `_owns_engine` flag prevents `TypedMemoryStore.close()` from disposing a shared engine. The engine's owner (ORMMemoryStore) is responsible for disposal. This was identified as a known issue from Task 2 review.

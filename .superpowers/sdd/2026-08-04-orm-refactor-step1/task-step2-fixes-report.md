# Task Step 2 — ORMMemoryStore 4 Test Failures Fix Report

**Date:** 2026-08-04
**Status:** DONE
**Test Summary:** 4/4 previously failing tests now pass; 0 regressions introduced.

---

## Root Cause Analysis & Fixes

### Failure 1: `test_evolution.py::TestZettelLinker::test_link_dedup`
**Error:** `AssertionError: assert 1 == 0` — second `link_on_add` call created a duplicate link instead of finding existing one.

**Root Cause:** `SQLiteGraphStore.add_edge()` executes `INSERT OR IGNORE` but never calls `self.conn.commit()`. With ORMMemoryStore, the graph_store's `raw_conn` (from `engine.raw_connection()`) and ORM Sessions share the same underlying SQLite connection (StaticPool for `sqlite://`). When `store.search()` creates a new `Session` between the two `link_on_add` calls, SQLAlchemy resets the connection's transaction state, rolling back the uncommitted edge INSERT. `get_neighbors()` then returns empty, so the dedup check fails.

**Fix:** Added `self.conn.commit()` after the INSERT in `SQLiteGraphStore.add_edge()`. This is consistent with `delete_edge()` which already commits.

**File:** `src/septmuse/storage/graph_stores/sqlite.py` — `add_edge()` method.

---

### Failures 2 & 3: `test_retrieval.py::TestHybridRetriever::test_filters_session_id` and `test_filters_no_match`
**Error:** session_id filter passed via `filters={"session_id": "s1"}` was silently dropped — all memories returned regardless of session_id.

**Root Cause:** Two bugs in `ORMMemoryStore`:

1. **`get_all()` completely ignored the `filters` parameter** — no filter logic at all (SQLiteMemoryStore applies filters via `FiltersParser`).
2. **`search()` unconditionally stripped `session_id` and `run_id` from `filters`** — even when the `session_id` parameter was `None`. SQLiteMemoryStore only strips these keys when the `session_id` parameter is explicitly provided (to avoid duplication).

The HybridRetriever passes `filters={"session_id": "s1"}` (not the `session_id` parameter), so ORMMemoryStore dropped it entirely.

**Fix:**
- `get_all()`: Added filter support matching SQLiteMemoryStore's pattern — only strip `session_id`/`run_id` from filters when `session_id` parameter is not None.
- `search()`: Changed from unconditional strip to conditional strip (only when `session_id` parameter is not None), matching SQLiteMemoryStore's behavior.

**File:** `src/septmuse/storage/relational_stores/orm_store.py` — `get_all()` and `search()` methods.

---

### Failure 4: `test_sharing_meta.py::TestSharedMemoryAccessor::test_get_shared_memories`
**Error:** `KeyError: 'user_id'` — result dict from `get_shared_memories` doesn't include `user_id`.

**Root Cause:** `ORMMemoryStore.get_shared_memories()` returns dict with keys `id`, `memory`, `metadata`, `created_at`, `agent_id` — but NOT `user_id`. SQLiteMemoryStore's version includes `user_id` in both the SELECT and the result dict.

**Fix:** Added `"user_id": mem.user_id` to the result dict in `ORMMemoryStore.get_shared_memories()`.

**File:** `src/septmuse/storage/relational_stores/orm_store.py` — `get_shared_memories()` method.

---

## Additional Consistency Fix

Applied the same `get_all()` and `search()` filter fixes to `AsyncORMMemoryStore` in `src/septmuse/storage/relational_stores/async_orm_store.py` — identical bug pattern (unconditional session_id strip, get_all ignores filters). No `get_shared_memories` method exists in the async store.

---

## Verification

### Failing tests (all pass now):
```
tests/unit/test_evolution.py::TestZettelLinker::test_link_dedup PASSED
tests/unit/test_retrieval.py::TestHybridRetriever::test_filters_session_id PASSED
tests/unit/test_retrieval.py::TestHybridRetriever::test_filters_no_match PASSED
tests/unit/test_sharing_meta.py::TestSharedMemoryAccessor::test_get_shared_memories PASSED
4 passed in 0.47s
```

### Regression check (affected areas):
```
tests/unit/test_evolution.py + test_retrieval.py + test_sharing_meta.py +
test_graph_store.py + test_graph_search.py + test_orm_memory_store.py +
test_facade_orm_path.py + test_async_orm_memory_store.py
164 passed, 4 skipped in 14.66s
```

### Full unit suite:
```
1209 passed, 36 skipped, 13 failed in 69.16s
```
The 13 failures are all pre-existing LLM/embedder provider tests (`test_llm_providers.py`, `test_memory.py::TestResolveEmbedderOpenAI`, `test_memory.py::TestResolveLLMBaseUrl`) — they require API keys or packages not installed. Unrelated to these changes.

### Lint:
```
ruff check --no-cache src/septmuse/storage/graph_stores/sqlite.py \
  src/septmuse/storage/relational_stores/orm_store.py \
  src/septmuse/storage/relational_stores/async_orm_store.py
All checks passed!
```

---

## Files Changed

| File | Change |
|------|--------|
| `src/septmuse/storage/graph_stores/sqlite.py` | Added `self.conn.commit()` in `add_edge()` |
| `src/septmuse/storage/relational_stores/orm_store.py` | Fixed `get_all()` filter support, `search()` conditional session_id strip, `get_shared_memories()` user_id |
| `src/septmuse/storage/relational_stores/async_orm_store.py` | Fixed `get_all()` filter support, `search()` conditional session_id strip |

## Concerns

None. All fixes address actual bugs where ORMMemoryStore diverged from SQLiteMemoryStore's contract. The `add_edge` commit fix is also a latent bug in the SQLiteMemoryStore path (it happened to work because SQLiteMemoryStore's `conn` isn't shared with ORM Sessions).

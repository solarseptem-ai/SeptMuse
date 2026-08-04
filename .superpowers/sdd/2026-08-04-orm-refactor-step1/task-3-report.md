# Task 3 Report: MigrationRunner.from_engine + inspect

## What I Implemented

Refactored `src/septmuse/storage/migrations/runner.py` to support a dual-mode `MigrationRunner`:

1. **New ORM path** — `MigrationRunner.from_engine(engine)` classmethod + `__init__(..., engine=...)` keyword-only param. When an engine is provided, all SQL operations route through SQLAlchemy `text()` + `engine.begin()`/`engine.connect()`, and schema introspection uses `sqlalchemy.inspect()` (`get_columns`, `has_table`) for cross-dialect support.
2. **Legacy path preserved** — `__init__(conn, backend)` still works with raw `sqlite3` connections, falling back to `MigrationContext` for `has_column`/`has_table`. Attribute renamed `self.conn` → `self._conn` (private); verified no callers access `.conn` directly.
3. **`_commit()`** is a no-op on the engine path (`engine.begin()` auto-commits).
4. New imports: `from sqlalchemy import inspect, text`.

The entire file was replaced per the brief (84 → 134 lines).

## TDD Evidence

### RED (before implementation)

Command:
```
$env:PYTHONPATH="src"; python -m pytest tests/unit/test_migration_runner_orm.py -v
```

Output (4 failed):
```
tests/unit/test_migration_runner_orm.py::test_migration_runner_from_engine_sqlite FAILED [ 25%]
tests/unit/test_migration_runner_orm.py::test_migration_runner_from_engine_has_table FAILED [ 50%]
tests/unit/test_migration_runner_orm.py::test_migration_runner_from_engine_runs_migrations FAILED [ 75%]
tests/unit/test_migration_runner_orm.py::test_migration_runner_old_constructor_still_works FAILED [100%]

E       AttributeError: type object 'MigrationRunner' has no attribute 'from_engine'
```

### GREEN (after implementation)

Command:
```
$env:PYTHONPATH="src"; python -m pytest tests/unit/test_migration_runner_orm.py -v
```

Output:
```
tests/unit/test_migration_runner_orm.py::test_migration_runner_from_engine_sqlite PASSED [ 25%]
tests/unit/test_migration_runner_orm.py::test_migration_runner_from_engine_has_table PASSED [ 50%]
tests/unit/test_migration_runner_orm.py::test_migration_runner_from_engine_runs_migrations PASSED [ 75%]
tests/unit/test_migration_runner_orm.py::test_migration_runner_old_constructor_still_works PASSED [100%]

============================== 4 passed in 0.68s ==============================
```

### Lint

Command:
```
ruff check --no-cache src/septmuse/storage/migrations/runner.py tests/unit/test_migration_runner_orm.py
```

Output:
```
All checks passed!
```

Note: Initial lint raised `UP037` (redundant quotes on `"MigrationRunner"` return annotation, since `from __future__ import annotations` is present). Auto-fixed with `ruff check --fix` (explicitly allowed by the brief). Re-verified clean.

### Regression Check

Existing migration tests (no test code modified):
```
$env:PYTHONPATH="src"; python -m pytest tests/unit/test_migration_runner.py tests/unit/test_migrations.py -v
=> 18 passed in 0.44s
```

Broader storage regression:
```
$env:PYTHONPATH="src"; python -m pytest tests/unit/test_async_sqlite_store.py tests/unit/test_composite_store.py tests/unit/test_relational_store_factory.py -q
=> 17 passed in 5.03s
```

Combined:
```
$env:PYTHONPATH="src"; python -m pytest tests/unit/test_migration_runner_orm.py tests/unit/test_migration_runner.py tests/unit/test_migrations.py -q
=> 22 passed in 0.71s
```

## Files Changed

| File | Action | Lines |
|------|--------|-------|
| `src/septmuse/storage/migrations/runner.py` | Modified (full replacement) | 84 → 134 |
| `tests/unit/test_migration_runner_orm.py` | Created (new) | 76 |

## Self-Review Findings

1. **Backward compatibility verified** — All 3 callers (`cli/main.py:459`, `relational_stores/async_store.py:65`, `relational_stores/store.py:80`) use positional `MigrationRunner(conn, "sqlite")` and never access `.conn` attribute. The rename to `self._conn` is safe.
2. **`from_engine` uses `inspect(engine)`** — Replaces the old PRAGMA (SQLite) / information_schema (PG) approach with a unified cross-dialect API. `get_columns` returns dicts with `"name"` key; `has_table` returns bool directly.
3. **Transaction handling** — Engine path uses `engine.begin()` (auto-commit DDL/DML) and `engine.connect()` (read-only for `_get_applied_versions`). Legacy path still uses manual `conn.commit()` via `_commit()`.
4. **One auto-fix applied** — `UP037` removed quotes from `"MigrationRunner"` → `MigrationRunner` on line 37. This is a no-op semantic change (behavior identical with `from __future__ import annotations`), required to pass lint.
5. **Broad `except Exception` in `_get_applied_versions`** — Preserved from original code; returns empty set on any error (defensive for first-run / missing table). Not a regression.
6. **No commit** — Per brief instructions (file-snapshot mode, no git).

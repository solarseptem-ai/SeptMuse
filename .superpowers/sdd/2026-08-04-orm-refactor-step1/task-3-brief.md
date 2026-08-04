### Task 3: MigrationRunner.from_engine + inspect

**Files:**
- Modify: `src/septmuse/storage/migrations/runner.py`（MigrationRunner 类）
- Test: `tests/unit/test_migration_runner_orm.py`

**Interfaces:**
- Consumes: `Engine` from Task 1
- Produces: `MigrationRunner.from_engine(engine)` classmethod

**Global Constraints:**
- PYTHONPATH=src 运行 pytest（包未 pip install -e .）
- ruff line-length 120，select E/F/I/W/UP/B/SIM/RUF，ignore E501/RUF001-003
- 禁止 `ruff format <file>`（Windows 清空 bug），用 `ruff check --fix` + `ruff check --no-cache`
- 现有测试固定不动，仅新增测试
- `pytest_asyncio_mode = "auto"`，async 测试无需 @pytest.mark.asyncio
- 代码注释用中文
- 不用 git（文件快照模式），无 commit 步骤

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_migration_runner_orm.py
"""MigrationRunner.from_engine + SQLAlchemy inspect 路径测试。"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from sqlmodel import create_engine

from septmuse.storage.migrations.runner import MigrationRunner


def test_migration_runner_from_engine_sqlite(tmp_path):
    """from_engine 在 SQLite 上正确检测列。"""
    engine = create_engine(f"sqlite:///{tmp_path / 'mig.db'}")
    runner = MigrationRunner.from_engine(engine)
    assert runner.backend == "sqlite"

    # 先建一张表
    with engine.connect() as conn:
        conn.exec_driver_sql("CREATE TABLE test_table (id TEXT, name TEXT)")
        conn.commit()

    # 检测列存在
    assert runner._has_column("test_table", "id") is True
    assert runner._has_column("test_table", "name") is True
    assert runner._has_column("test_table", "nonexistent") is False


def test_migration_runner_from_engine_has_table(tmp_path):
    """from_engine 检测表存在。"""
    engine = create_engine(f"sqlite:///{tmp_path / 'mig2.db'}")
    runner = MigrationRunner.from_engine(engine)

    with engine.connect() as conn:
        conn.exec_driver_sql("CREATE TABLE foo (id TEXT)")
        conn.commit()

    assert runner._has_table("foo") is True
    assert runner._has_table("nonexistent_table") is False


def test_migration_runner_from_engine_runs_migrations(tmp_path):
    """from_engine 完整执行迁移。"""
    engine = create_engine(f"sqlite:///{tmp_path / 'mig3.db'}")
    runner = MigrationRunner.from_engine(engine)
    newly = runner.run()
    # 首次运行应有迁移被应用
    assert isinstance(newly, list)
    assert len(newly) > 0

    # 二次运行不应重复应用
    runner2 = MigrationRunner.from_engine(engine)
    newly2 = runner2.run()
    assert len(newly2) == 0


def test_migration_runner_old_constructor_still_works(tmp_path):
    """旧 __init__(conn, backend) 向后兼容。"""
    import sqlite3

    conn = sqlite3.connect(str(tmp_path / "old.db"))
    runner = MigrationRunner(conn, "sqlite")
    assert runner.backend == "sqlite"
    assert runner._conn is conn
    newly = runner.run()
    assert isinstance(newly, list)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_migration_runner_orm.py -v`
Expected: FAIL with `AttributeError: type object 'MigrationRunner' has no attribute 'from_engine'`

- [ ] **Step 3: Write minimal implementation**

完全替换 `src/septmuse/storage/migrations/runner.py` 的内容为：

```python
"""迁移执行器 — 检查 schema_version 表，执行未应用的迁移。

sync 版，支持 SQLite + PG。async store 在 to_thread 中调用。
双模式: __init__(conn, backend) 旧路径 / from_engine(engine) ORM 路径。
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import inspect, text

from septmuse.core.logging import get_logger

logger = get_logger(__name__)


class MigrationRunner:
    """同步迁移执行器（SQLite + PG + MySQL via SQLAlchemy inspect）。"""

    def __init__(
        self,
        conn: Any | None = None,
        backend: str = "sqlite",
        *,
        engine: Any | None = None,
    ) -> None:
        self._engine = engine
        if engine is not None:
            self._conn = None
            self.backend = engine.dialect.name
        else:
            self._conn = conn
            self.backend = backend

    @classmethod
    def from_engine(cls, engine: Any) -> "MigrationRunner":
        """从 SQLAlchemy Engine 构造（跨方言 inspect 路径）。"""
        return cls(engine=engine)

    def _utcnow_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _ensure_schema_version_table(self) -> None:
        """创建 schema_version 追踪表（幂等）。"""
        if self._engine is not None:
            with self._engine.begin() as conn:
                conn.execute(
                    text(
                        "CREATE TABLE IF NOT EXISTS schema_version "
                        "(version TEXT PRIMARY KEY, description TEXT, applied_at TEXT NOT NULL)"
                    )
                )
        else:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS schema_version "
                "(version TEXT PRIMARY KEY, description TEXT, applied_at TEXT NOT NULL)"
            )

    def _get_applied_versions(self) -> set[str]:
        """获取已应用的迁移版本集合。"""
        try:
            if self._engine is not None:
                with self._engine.connect() as conn:
                    result = conn.execute(text("SELECT version FROM schema_version"))
                    return {r[0] for r in result.fetchall()}
            else:
                cur = self._conn.execute("SELECT version FROM schema_version")
                return {r[0] for r in cur.fetchall()}
        except Exception:
            return set()

    def _commit(self) -> None:
        """提交事务。"""
        if self._engine is not None:
            return  # engine.begin() 自动提交
        if hasattr(self._conn, "commit"):
            self._conn.commit()

    def _has_column(self, table: str, column: str) -> bool:
        """检查列是否存在（engine 路径用 SQLAlchemy inspect）。"""
        if self._engine is not None:
            cols = [c["name"] for c in inspect(self._engine).get_columns(table)]
            return column in cols
        from septmuse.storage.migrations.context import MigrationContext

        ctx = MigrationContext(self._conn, self.backend)
        return ctx.has_column(table, column)

    def _has_table(self, table: str) -> bool:
        """检查表是否存在。"""
        if self._engine is not None:
            return inspect(self._engine).has_table(table)
        from septmuse.storage.migrations.context import MigrationContext

        ctx = MigrationContext(self._conn, self.backend)
        return ctx.has_table(table)

    def run(self) -> list[str]:
        """检查 schema_version，执行未应用的迁移，返回新应用的版本列表。"""
        from septmuse.storage.migrations import MIGRATIONS

        self._ensure_schema_version_table()
        self._commit()
        applied = self._get_applied_versions()
        newly: list[str] = []
        for m in MIGRATIONS:
            if m.version in applied:
                continue
            for step in m.steps(self.backend):
                if step.check_column and self._has_column(*step.check_column):
                    continue
                if step.check_table and self._has_table(step.check_table):
                    continue
                if self._engine is not None:
                    with self._engine.begin() as conn:
                        conn.execute(text(step.sql))
                else:
                    self._conn.execute(step.sql)
            if self._engine is not None:
                with self._engine.begin() as conn:
                    conn.execute(
                        text("INSERT INTO schema_version (version, description, applied_at) VALUES (:v, :d, :t)"),
                        {"v": m.version, "d": m.description, "t": self._utcnow_iso()},
                    )
            else:
                self._conn.execute(
                    "INSERT INTO schema_version (version, description, applied_at) VALUES (?, ?, ?)",
                    (m.version, m.description, self._utcnow_iso()),
                )
            newly.append(m.version)
            logger.info("migration_applied", version=m.version, description=m.description)
        self._commit()
        return newly
```

- [ ] **Step 4: Run test to verify it passes**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_migration_runner_orm.py -v`
Expected: 4 passed

- [ ] **Step 5: Lint**

Run: `ruff check --no-cache src/septmuse/storage/migrations/runner.py tests/unit/test_migration_runner_orm.py`
Expected: All checks passed

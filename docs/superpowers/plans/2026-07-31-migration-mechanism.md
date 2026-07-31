# 轻量级数据迁移机制 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 创建轻量级 schema 迁移机制：版本追踪 + 有序迁移 + init 自动执行 + CLI 手动触发

**Architecture:** schema_version 表追踪已应用迁移；5 个迁移模块（m001-m005）产出 MigrationStep 列表；MigrationRunner 检查版本并执行未应用的迁移；Store init 时自动调用，CLI 命令手动触发。零外部依赖。

**Tech Stack:** sqlite3、aiosqlite、psycopg、pytest-asyncio

## 全局约束

- **PYTHONPATH=src** 运行所有测试（PowerShell: `$env:PYTHONPATH="src"`）
- **ruff line-length=120**，只用 `ruff check --no-cache`（禁用 ruff format）
- **不是 git 仓库**，无 commit 步骤
- **代码注释用中文**，不暴露任何开源库参考来源
- **现有测试固定不动**，仅新增测试
- **pytest 基线**：1058 passed + 36 skipped + 23 failed（不退化）
- **pytest_asyncio_mode = "auto"**
- 工作目录：E:\sonhhxg0529\vibe_coding_project\solarseptem-ai\solarseptem-ai-platform\SeptMuse

## 文件结构

**新建：**
- `src/septmuse/storage/migrations/__init__.py` — MIGRATIONS 注册表 + Migration namedtuple
- `src/septmuse/storage/migrations/context.py` — MigrationContext（has_column/has_table/execute）
- `src/septmuse/storage/migrations/runner.py` — MigrationRunner（sync，支持 SQLite + PG）
- `src/septmuse/storage/migrations/m001_initial_schema.py` — CREATE memories + history
- `src/septmuse/storage/migrations/m002_state_columns.py` — ALTER ADD state/deleted_at/app_id
- `src/septmuse/storage/migrations/m003_session_id.py` — ALTER ADD session_id
- `src/septmuse/storage/migrations/m004_temporal.py` — ALTER ADD valid_at/invalid_at/expired_at
- `src/septmuse/storage/migrations/m005_access_logs.py` — CREATE memory_access_logs
- `tests/unit/test_migration_runner.py` — Runner + Context 测试
- `tests/unit/test_migrations.py` — 5 个迁移模块 DDL 正确性测试

**修改：**
- `src/septmuse/storage/sqlite/store.py` — 删除 4 个 `_migrate_add_*` + `_create_access_logs_table` 调用，替换为 `MigrationRunner(self.conn, "sqlite").run()`
- `src/septmuse/storage/async_sqlite/store.py` — `_init_dual_write` 中加 `MigrationRunner(sync_conn, "sqlite").run()`
- `src/septmuse/cli/main.py` — 加 `migrate` 子命令

---

## Task 1: MigrationStep + MigrationContext + Migration 注册表

**Files:**
- Create: `src/septmuse/storage/migrations/__init__.py`
- Create: `src/septmuse/storage/migrations/context.py`
- Test: `tests/unit/test_migration_runner.py`（Context 部分）

**Interfaces:**
- Produces: `MigrationStep(sql, check_column, check_table)` dataclass，`Migration` namedtuple，`MigrationContext(conn, backend)`

- [ ] **Step 1: 写 MigrationStep + Migration + MigrationContext 失败测试**

```python
# tests/unit/test_migration_runner.py
"""迁移机制测试 — MigrationContext + MigrationRunner + 迁移模块。"""
import sqlite3

import pytest

from septmuse.storage.migrations.context import MigrationContext
from septmuse.storage.migrations.runner import MigrationRunner


@pytest.fixture
def conn():
    """临时 SQLite 连接。"""
    c = sqlite3.connect(":memory:")
    yield c
    c.close()


def test_migration_context_has_column_false(conn):
    """has_column 在列不存在时返回 False。"""
    conn.execute("CREATE TABLE test (id TEXT)")
    ctx = MigrationContext(conn, "sqlite")
    assert ctx.has_column("test", "name") is False


def test_migration_context_has_column_true(conn):
    """has_column 在列存在时返回 True。"""
    conn.execute("CREATE TABLE test (id TEXT, name TEXT)")
    ctx = MigrationContext(conn, "sqlite")
    assert ctx.has_column("test", "name") is True


def test_migration_context_has_table_false(conn):
    """has_table 在表不存在时返回 False。"""
    ctx = MigrationContext(conn, "sqlite")
    assert ctx.has_table("nonexistent") is False


def test_migration_context_has_table_true(conn):
    """has_table 在表存在时返回 True。"""
    conn.execute("CREATE TABLE test (id TEXT)")
    ctx = MigrationContext(conn, "sqlite")
    assert ctx.has_table("test") is True


def test_migration_context_execute(conn):
    """execute 执行 DDL。"""
    ctx = MigrationContext(conn, "sqlite")
    ctx.execute("CREATE TABLE test (id TEXT)")
    assert ctx.has_table("test") is True
```

- [ ] **Step 2: 运行确认失败**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_migration_runner.py -v`
Expected: FAIL — `No module named 'septmuse.storage.migrations'`

- [ ] **Step 3: 写 migrations 包 + MigrationStep + Migration**

```python
# src/septmuse/storage/migrations/__init__.py
"""轻量级迁移注册表 — 有序迁移模块列表。

每个迁移模块导出 VERSION, DESCRIPTION, steps(backend) -> list[MigrationStep]。
MigrationRunner 按 MIGRATIONS 列表顺序执行未应用的迁移。
"""
from __future__ import annotations

from collections import namedtuple
from dataclasses import dataclass, field


@dataclass
class MigrationStep:
    """单个迁移步骤。

    check_column: (table, column) — 仅在列不存在时执行（SQLite ALTER TABLE 幂等性）。
    check_table: table name — 仅在表不存在时执行。
    两者都 None：无条件执行。
    """
    sql: str
    check_column: tuple[str, str] | None = None
    check_table: str | None = None


Migration = namedtuple("Migration", ["version", "description", "steps"])
```

- [ ] **Step 4: 写 MigrationContext**

```python
# src/septmuse/storage/migrations/context.py
"""迁移上下文 — 统一 SQLite/PG 的 DDL 操作接口。

为 MigrationRunner 提供 has_column/has_table/execute 方法，
屏蔽 SQLite (PRAGMA) 和 PG (information_schema) 的差异。
"""
from __future__ import annotations

import sqlite3
from typing import Any


class MigrationContext:
    """同步迁移上下文（SQLite + PG）。"""

    def __init__(self, conn: Any, backend: str = "sqlite") -> None:
        self.conn = conn
        self.backend = backend

    def has_column(self, table: str, column: str) -> bool:
        """检查列是否存在。"""
        if self.backend == "sqlite":
            cols = [r[1] for r in self.conn.execute(f"PRAGMA table_info({table})").fetchall()]
        else:
            cur = self.conn.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_name = %s",
                (table,),
            )
            cols = [r[0] for r in cur.fetchall()]
        return column in cols

    def has_table(self, table: str) -> bool:
        """检查表是否存在。"""
        if self.backend == "sqlite":
            cur = self.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
            )
            r = cur.fetchone()
            return r is not None
        else:
            cur = self.conn.execute(f"SELECT to_regclass('public.{table}')")
            r = cur.fetchone()
            return r is not None and r[0] is not None

    def execute(self, sql: str) -> None:
        """执行 DDL 语句。"""
        self.conn.execute(sql)
```

- [ ] **Step 5: 写 MigrationRunner 占位（后续 Task 3 补全）**

```python
# src/septmuse/storage/migrations/runner.py
"""迁移执行器 — 检查 schema_version 表，执行未应用的迁移。"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from septmuse.core.logging import get_logger
from septmuse.storage.migrations import Migration, MigrationStep

logger = get_logger(__name__)


class MigrationRunner:
    """同步迁移执行器（SQLite + PG）。"""

    def __init__(self, conn: Any, backend: str = "sqlite") -> None:
        self.conn = conn
        self.backend = backend

    def _utcnow_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _ensure_schema_version_table(self) -> None:
        """创建 schema_version 追踪表（幂等）。"""
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_version (
                version     TEXT PRIMARY KEY,
                description TEXT,
                applied_at  TEXT NOT NULL
            )
            """
        )

    def _get_applied_versions(self) -> set[str]:
        """获取已应用的迁移版本集合。"""
        try:
            cur = self.conn.execute("SELECT version FROM schema_version")
            return {r[0] for r in cur.fetchall()}
        except Exception:
            return set()

    def _commit(self) -> None:
        """提交事务（PG cursor 无 commit 方法时跳过）。"""
        if hasattr(self.conn, "commit"):
            self.conn.commit()

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
                self.conn.execute(step.sql)
            # 记录迁移
            self.conn.execute(
                "INSERT INTO schema_version (version, description, applied_at) VALUES (?, ?, ?)",
                (m.version, m.description, self._utcnow_iso()),
            )
            newly.append(m.version)
            logger.info("migration_applied", version=m.version, description=m.description)
        self._commit()
        return newly

    def _has_column(self, table: str, column: str) -> bool:
        from septmuse.storage.migrations.context import MigrationContext
        ctx = MigrationContext(self.conn, self.backend)
        return ctx.has_column(table, column)

    def _has_table(self, table: str) -> bool:
        from septmuse.storage.migrations.context import MigrationContext
        ctx = MigrationContext(self.conn, self.backend)
        return ctx.has_table(table)
```

- [ ] **Step 6: 在 __init__.py 加 MIGRATIONS 列表占位**

在 `src/septmuse/storage/migrations/__init__.py` 末尾加：

```python
# 迁移注册表（Task 2 填充实际迁移模块）
# 每个迁移模块导出 VERSION, DESCRIPTION, steps(backend) -> list[MigrationStep]
MIGRATIONS: list[Migration] = []
```

- [ ] **Step 7: 运行测试确认通过**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_migration_runner.py -v`
Expected: PASS（5 个 Context 测试全通过）

- [ ] **Step 8: ruff 检查**

Run: `ruff check --no-cache src/septmuse/storage/migrations/ tests/unit/test_migration_runner.py`
Expected: All checks passed!

---

## Task 2: 5 个迁移模块 + 注册表填充

**Files:**
- Create: `src/septmuse/storage/migrations/m001_initial_schema.py`
- Create: `src/septmuse/storage/migrations/m002_state_columns.py`
- Create: `src/septmuse/storage/migrations/m003_session_id.py`
- Create: `src/septmuse/storage/migrations/m004_temporal.py`
- Create: `src/septmuse/storage/migrations/m005_access_logs.py`
- Modify: `src/septmuse/storage/migrations/__init__.py`（填充 MIGRATIONS）
- Test: `tests/unit/test_migrations.py`

**Interfaces:**
- Consumes: `MigrationStep`（Task 1），`Migration`（Task 1）
- Produces: 5 个迁移模块 + `MIGRATIONS` 注册表

- [ ] **Step 1: 写迁移模块 DDL 测试**

```python
# tests/unit/test_migrations.py
"""5 个迁移模块 DDL 正确性测试。"""
import sqlite3

import pytest

from septmuse.storage.migrations import MIGRATIONS


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    yield c
    c.close()


def test_migrations_list_has_5():
    """注册表有 5 个迁移。"""
    assert len(MIGRATIONS) == 5


def test_migrations_versions_sequential():
    """版本号从 001 到 005 有序。"""
    versions = [m.version for m in MIGRATIONS]
    assert versions == ["001", "002", "003", "004", "005"]


def test_m001_creates_memories_and_history(conn):
    """m001 创建 memories + history 表。"""
    m001 = MIGRATIONS[0]
    assert m001.version == "001"
    for step in m001.steps("sqlite"):
        conn.execute(step.sql)
    conn.commit()
    # 检查表存在
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()]
    assert "memories" in tables
    assert "history" in tables


def test_m002_adds_state_columns(conn):
    """m002 添加 state/deleted_at/app_id 列。"""
    conn.execute("CREATE TABLE memories (id TEXT, user_id TEXT, content TEXT, embedding TEXT, metadata TEXT, created_at TEXT, updated_at TEXT, is_deleted INTEGER)")
    m002 = MIGRATIONS[1]
    assert m002.version == "002"
    for step in m002.steps("sqlite"):
        conn.execute(step.sql)
    conn.commit()
    cols = [r[1] for r in conn.execute("PRAGMA table_info(memories)").fetchall()]
    assert "state" in cols
    assert "deleted_at" in cols
    assert "app_id" in cols


def test_m003_adds_session_id(conn):
    """m003 添加 session_id 列。"""
    conn.execute("CREATE TABLE memories (id TEXT)")
    m003 = MIGRATIONS[2]
    assert m003.version == "003"
    for step in m003.steps("sqlite"):
        conn.execute(step.sql)
    conn.commit()
    cols = [r[1] for r in conn.execute("PRAGMA table_info(memories)").fetchall()]
    assert "session_id" in cols


def test_m004_adds_temporal_columns(conn):
    """m004 添加 valid_at/invalid_at/expired_at 列。"""
    conn.execute("CREATE TABLE memories (id TEXT)")
    m004 = MIGRATIONS[3]
    assert m004.version == "004"
    for step in m004.steps("sqlite"):
        conn.execute(step.sql)
    conn.commit()
    cols = [r[1] for r in conn.execute("PRAGMA table_info(memories)").fetchall()]
    assert "valid_at" in cols
    assert "invalid_at" in cols
    assert "expired_at" in cols


def test_m005_creates_access_logs(conn):
    """m005 创建 memory_access_logs 表。"""
    m005 = MIGRATIONS[4]
    assert m005.version == "005"
    for step in m005.steps("sqlite"):
        conn.execute(step.sql)
    conn.commit()
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()]
    assert "memory_access_logs" in tables
```

- [ ] **Step 2: 运行确认失败**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_migrations.py -v`
Expected: FAIL — `MIGRATIONS` 为空

- [ ] **Step 3: 写 m001_initial_schema.py**

```python
# src/septmuse/storage/migrations/m001_initial_schema.py
"""m001: 初始 schema — memories + history 表。"""
from __future__ import annotations

from septmuse.storage.migrations import MigrationStep

VERSION = "001"
DESCRIPTION = "initial schema (memories + history)"


def steps(backend: str = "sqlite") -> list[MigrationStep]:
    """创建 memories + history 基础表。"""
    return [
        MigrationStep(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id          TEXT PRIMARY KEY,
                user_id     TEXT NOT NULL,
                agent_id    TEXT,
                content     TEXT NOT NULL,
                embedding   TEXT NOT NULL,
                metadata    TEXT,
                created_at  TEXT,
                updated_at  TEXT,
                is_deleted  INTEGER DEFAULT 0
            )
            """
        ),
        MigrationStep("CREATE INDEX IF NOT EXISTS idx_memories_user ON memories(user_id)"),
        MigrationStep(
            """
            CREATE TABLE IF NOT EXISTS history (
                id          TEXT PRIMARY KEY,
                memory_id   TEXT,
                old_memory  TEXT,
                new_memory  TEXT,
                event       TEXT,
                created_at  TEXT,
                is_deleted  INTEGER
            )
            """
        ),
    ]
```

- [ ] **Step 4: 写 m002_state_columns.py**

```python
# src/septmuse/storage/migrations/m002_state_columns.py
"""m002: 添加 state/deleted_at/app_id 列。"""
from __future__ import annotations

from septmuse.storage.migrations import MigrationStep

VERSION = "002"
DESCRIPTION = "add state/deleted_at/app_id columns"


def steps(backend: str = "sqlite") -> list[MigrationStep]:
    """ALTER TABLE 添加状态机列。"""
    if backend == "postgres":
        return [
            MigrationStep("ALTER TABLE memories ADD COLUMN IF NOT EXISTS state TEXT DEFAULT 'active'"),
            MigrationStep("ALTER TABLE memories ADD COLUMN IF NOT EXISTS deleted_at TEXT"),
            MigrationStep("ALTER TABLE memories ADD COLUMN IF NOT EXISTS app_id TEXT"),
        ]
    return [
        MigrationStep(
            "ALTER TABLE memories ADD COLUMN state TEXT DEFAULT 'active'",
            check_column=("memories", "state"),
        ),
        MigrationStep(
            "ALTER TABLE memories ADD COLUMN deleted_at TEXT",
            check_column=("memories", "deleted_at"),
        ),
        MigrationStep(
            "ALTER TABLE memories ADD COLUMN app_id TEXT",
            check_column=("memories", "app_id"),
        ),
    ]
```

- [ ] **Step 5: 写 m003_session_id.py**

```python
# src/septmuse/storage/migrations/m003_session_id.py
"""m003: 添加 session_id 列。"""
from __future__ import annotations

from septmuse.storage.migrations import MigrationStep

VERSION = "003"
DESCRIPTION = "add session_id column"


def steps(backend: str = "sqlite") -> list[MigrationStep]:
    """ALTER TABLE 添加会话 ID 列。"""
    if backend == "postgres":
        return [
            MigrationStep("ALTER TABLE memories ADD COLUMN IF NOT EXISTS session_id TEXT"),
        ]
    return [
        MigrationStep(
            "ALTER TABLE memories ADD COLUMN session_id TEXT",
            check_column=("memories", "session_id"),
        ),
    ]
```

- [ ] **Step 6: 写 m004_temporal.py**

```python
# src/septmuse/storage/migrations/m004_temporal.py
"""m004: 添加 valid_at/invalid_at/expired_at 时态列。"""
from __future__ import annotations

from septmuse.storage.migrations import MigrationStep

VERSION = "004"
DESCRIPTION = "add temporal columns (valid_at/invalid_at/expired_at)"


def steps(backend: str = "sqlite") -> list[MigrationStep]:
    """ALTER TABLE 添加双时态列。"""
    if backend == "postgres":
        return [
            MigrationStep("ALTER TABLE memories ADD COLUMN IF NOT EXISTS valid_at TEXT"),
            MigrationStep("ALTER TABLE memories ADD COLUMN IF NOT EXISTS invalid_at TEXT"),
            MigrationStep("ALTER TABLE memories ADD COLUMN IF NOT EXISTS expired_at TEXT"),
        ]
    return [
        MigrationStep(
            "ALTER TABLE memories ADD COLUMN valid_at TEXT",
            check_column=("memories", "valid_at"),
        ),
        MigrationStep(
            "ALTER TABLE memories ADD COLUMN invalid_at TEXT",
            check_column=("memories", "invalid_at"),
        ),
        MigrationStep(
            "ALTER TABLE memories ADD COLUMN expired_at TEXT",
            check_column=("memories", "expired_at"),
        ),
    ]
```

- [ ] **Step 7: 写 m005_access_logs.py**

```python
# src/septmuse/storage/migrations/m005_access_logs.py
"""m005: 创建 memory_access_logs 审计日志表。"""
from __future__ import annotations

from septmuse.storage.migrations import MigrationStep

VERSION = "005"
DESCRIPTION = "create memory_access_logs table"


def steps(backend: str = "sqlite") -> list[MigrationStep]:
    """创建访问日志表 + 索引。"""
    return [
        MigrationStep(
            """
            CREATE TABLE IF NOT EXISTS memory_access_logs (
                id           TEXT PRIMARY KEY,
                memory_id    TEXT NOT NULL,
                app_id       TEXT,
                access_type  TEXT NOT NULL,
                metadata     TEXT,
                accessed_at  TEXT NOT NULL
            )
            """
        ),
        MigrationStep(
            "CREATE INDEX IF NOT EXISTS idx_access_logs_memory ON memory_access_logs(memory_id)"
        ),
    ]
```

- [ ] **Step 8: 填充 __init__.py 的 MIGRATIONS 列表**

将 `src/septmuse/storage/migrations/__init__.py` 末尾的 `MIGRATIONS: list[Migration] = []` 替换为：

```python
from septmuse.storage.migrations import (
    m001_initial_schema,
    m002_state_columns,
    m003_session_id,
    m004_temporal,
    m005_access_logs,
)

_MODULES = [m001_initial_schema, m002_state_columns, m003_session_id, m004_temporal, m005_access_logs]

MIGRATIONS: list[Migration] = [
    Migration(version=m.VERSION, description=m.DESCRIPTION, steps=m.steps)
    for m in _MODULES
]
```

- [ ] **Step 9: 运行测试确认通过**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_migrations.py -v`
Expected: PASS（7 测试全通过）

- [ ] **Step 10: ruff 检查**

Run: `ruff check --no-cache src/septmuse/storage/migrations/ tests/unit/test_migrations.py`
Expected: All checks passed!

---

## Task 3: MigrationRunner 完整测试

**Files:**
- Test: `tests/unit/test_migration_runner.py`（追加 Runner 测试）

**Interfaces:**
- Consumes: Task 1 的 `MigrationRunner`，Task 2 的 `MIGRATIONS`
- Produces: 完整的 `MigrationRunner.run()` 验证

- [ ] **Step 1: 追加 Runner 测试**

在 `tests/unit/test_migration_runner.py` 末尾追加：

```python
# ── MigrationRunner 测试 ──


def test_runner_empty_db_full_migration(conn):
    """空 DB → 全量迁移（5 个版本）。"""
    runner = MigrationRunner(conn, "sqlite")
    applied = runner.run()
    assert len(applied) == 5
    assert applied == ["001", "002", "003", "004", "005"]


def test_runner_skips_already_applied(conn):
    """已迁移的版本跳过。"""
    runner = MigrationRunner(conn, "sqlite")
    runner.run()
    # 第二次运行 — 无新增
    applied = runner.run()
    assert applied == []


def test_runner_schema_version_table_created(conn):
    """schema_version 表自动创建。"""
    runner = MigrationRunner(conn, "sqlite")
    runner.run()
    ctx = MigrationContext(conn, "sqlite")
    assert ctx.has_table("schema_version") is True


def test_runner_records_applied_versions(conn):
    """schema_version 表记录已应用版本。"""
    runner = MigrationRunner(conn, "sqlite")
    runner.run()
    cur = conn.execute("SELECT version FROM schema_version ORDER BY version")
    versions = [r[0] for r in cur.fetchall()]
    assert versions == ["001", "002", "003", "004", "005"]


def test_runner_idempotent(conn):
    """幂等 — 多次运行不报错，不重复。"""
    runner = MigrationRunner(conn, "sqlite")
    runner.run()
    runner.run()
    runner.run()
    cur = conn.execute("SELECT COUNT(*) FROM schema_version")
    assert cur.fetchone()[0] == 5


def test_runner_partial_migration(conn):
    """部分迁移 — 只运行未应用的。"""
    # 手动创建 schema_version 表 + 记录 001-003
    conn.execute("CREATE TABLE schema_version (version TEXT PRIMARY KEY, description TEXT, applied_at TEXT)")
    conn.execute("INSERT INTO schema_version (version, description, applied_at) VALUES ('001', 'test', '2024')")
    conn.execute("INSERT INTO schema_version (version, description, applied_at) VALUES ('002', 'test', '2024')")
    conn.execute("INSERT INTO schema_version (version, description, applied_at) VALUES ('003', 'test', '2024')")
    conn.commit()
    # 创建 memories 表（m001 的 DDL）
    conn.execute("CREATE TABLE memories (id TEXT, user_id TEXT, content TEXT, embedding TEXT, metadata TEXT, created_at TEXT, updated_at TEXT, is_deleted INTEGER)")
    conn.execute("CREATE TABLE history (id TEXT)")
    conn.commit()

    runner = MigrationRunner(conn, "sqlite")
    applied = runner.run()
    assert applied == ["004", "005"]

    # 检查 004 的列已添加
    cols = [r[1] for r in conn.execute("PRAGMA table_info(memories)").fetchall()]
    assert "valid_at" in cols
    assert "invalid_at" in cols
    assert "expired_at" in cols

    # 检查 005 的表已创建
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    assert "memory_access_logs" in tables
```

- [ ] **Step 2: 运行测试确认通过**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_migration_runner.py -v`
Expected: PASS（11 测试全通过 = 5 Context + 6 Runner）

- [ ] **Step 3: ruff 检查**

Run: `ruff check --no-cache src/septmuse/storage/migrations/ tests/unit/test_migration_runner.py`
Expected: All checks passed!

---

## Task 4: Store 集成（SQLiteMemoryStore + AsyncSQLiteMemoryStore）

**Files:**
- Modify: `src/septmuse/storage/sqlite/store.py`
- Modify: `src/septmuse/storage/async_sqlite/store.py`

**Interfaces:**
- Consumes: Task 1-3 的 `MigrationRunner`，`MIGRATIONS`
- Produces: Store init 时自动运行迁移

- [ ] **Step 1: 修改 SQLiteMemoryStore.__init__**

在 `src/septmuse/storage/sqlite/store.py` 中，找到 `__init__` 方法。当前代码：

```python
        self._create_tables()
        self._migrate_add_state_columns()
        self._migrate_add_session_id_column()
        self._migrate_add_temporal_columns()
        self._create_access_logs_table()
```

替换为：

```python
        self._create_tables()
        # 轻量级迁移：版本追踪 + 有序迁移（替代 _migrate_add_* 方法）
        from septmuse.storage.migrations.runner import MigrationRunner
        MigrationRunner(self.conn, "sqlite").run()
```

注意：保留 `_create_tables()`（CREATE IF NOT EXISTS memories + history 基础表）。`_migrate_add_*` 和 `_create_access_logs_table` 方法本身保留（向后兼容，但不从 `__init__` 调用）。

- [ ] **Step 2: 修改 AsyncSQLiteMemoryStore._init_dual_write**

在 `src/septmuse/storage/async_sqlite/store.py` 中，找到 `_init_dual_write` 方法。当前代码：

```python
    def _init_dual_write(self) -> None:
        """初始化双写组件（sync，在 to_thread 中调用）。"""
        sync_conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._vector_store = SQLiteVectorStore(conn=sync_conn)
        self._keyword_index = SQLiteBM25Index(db_path=self._db_path)
```

替换为：

```python
    def _init_dual_write(self) -> None:
        """初始化双写组件（sync，在 to_thread 中调用）。"""
        sync_conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._vector_store = SQLiteVectorStore(conn=sync_conn)
        self._keyword_index = SQLiteBM25Index(db_path=self._db_path)
        # 轻量级迁移：在 sync 连接上运行（DDL，快，一次性）
        from septmuse.storage.migrations.runner import MigrationRunner
        MigrationRunner(sync_conn, "sqlite").run()
```

- [ ] **Step 3: 运行 sync store 测试确认不破坏**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_sqlite_store.py -q --tb=line 2>&1 | Select-Object -Last 5`
Expected: 通过或失败不超过之前基线

- [ ] **Step 4: 运行 async store 测试确认不破坏**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_async_sqlite_store.py tests/unit/test_async_memory.py -v`
Expected: PASS（6 + 5 = 11 测试全通过）

- [ ] **Step 5: 运行 async 权限 + 日志测试**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_async_permissions.py tests/unit/test_async_access_log.py tests/unit/test_async_store_base.py -v`
Expected: PASS（5 + 3 + 3 = 11 测试全通过）

- [ ] **Step 6: 运行 REST 测试确认不破坏**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_api_permission_integration.py tests/unit/test_rbac_rest_openai.py -q --tb=line 2>&1 | Select-Object -Last 5`
Expected: 37 passed

- [ ] **Step 7: ruff 检查**

Run: `ruff check --no-cache src/septmuse/storage/sqlite/store.py src/septmuse/storage/async_sqlite/store.py`
Expected: All checks passed!

---

## Task 5: CLI migrate 命令 + 全量验证

**Files:**
- Modify: `src/septmuse/cli/main.py`

**Interfaces:**
- Consumes: Task 3 的 `MigrationRunner`
- Produces: `septmuse migrate` CLI 命令

- [ ] **Step 1: 在 CLI 中加 migrate 子命令**

在 `src/septmuse/cli/main.py` 中，找到 argparse subparsers 部分。在现有子命令（如 `backends`、`config`）后面加：

```python
    # migrate 子命令
    p_migrate = subparsers.add_parser("migrate", help="运行数据库迁移")
    p_migrate.add_argument("--db-path", default=None, help="SQLite 数据库路径（默认 ~/.septmuse/septmuse.db）")
    p_migrate.set_defaults(cmd="migrate")
```

在命令处理函数部分加：

```python
def _cmd_migrate(args) -> None:
    """运行数据库迁移。"""
    import sqlite3
    from septmuse.storage.migrations.runner import MigrationRunner

    db_path = args.db_path
    if db_path is None:
        from septmuse.configs.defaults import default_config
        config = default_config()
        db_path = str(config.db_path)

    conn = sqlite3.connect(db_path)
    try:
        runner = MigrationRunner(conn, "sqlite")
        applied = runner.run()
        if applied:
            print(f"已应用 {len(applied)} 个迁移:")
            for v in applied:
                # 从 MIGRATIONS 查描述
                from septmuse.storage.migrations import MIGRATIONS
                desc = next((m.description for m in MIGRATIONS if m.version == v), "")
                print(f"  {v} - {desc}")
        else:
            print("所有迁移已应用，无需操作")
        print(f"schema_version: {len(MIGRATIONS)} migrations total")
    finally:
        conn.close()
```

- [ ] **Step 2: 运行 CLI migrate 命令验证**

Run: `$env:PYTHONPATH="src"; python -m septmuse.cli.main migrate --db-path "$env:TEMP\test_migrate.db"`
Expected: 输出 `已应用 5 个迁移`

- [ ] **Step 3: 再次运行确认幂等**

Run: `$env:PYTHONPATH="src"; python -m septmuse.cli.main migrate --db-path "$env:TEMP\test_migrate.db"`
Expected: `所有迁移已应用，无需操作`

- [ ] **Step 4: 全量 ruff**

Run: `ruff check --no-cache src/ tests/`
Expected: All checks passed!

- [ ] **Step 5: 全量 pytest**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/ tests/e2e/ -q --tb=line 2>&1 | Select-Object -Last 5`
Expected: 失败不超过 23（之前基线），passed 不低于 1058 + 新增 18 个迁移测试

- [ ] **Step 6: CLI backends 验证（不破坏）**

Run: `$env:PYTHONPATH="src"; python -m septmuse.cli.main backends`
Expected: 8 个 LLM 后端输出

- [ ] **Step 7: AsyncMemory 零配置验证（不破坏）**

Run: `$env:PYTHONPATH="src"; python -c "import asyncio; from septmuse.memory.async_main import AsyncMemory; from septmuse.embedders.hash import HashEmbedder; import tempfile, os; db=os.path.join(tempfile.mkdtemp(), 't.db'); from septmuse.storage.async_sqlite.store import AsyncSQLiteMemoryStore; s=AsyncSQLiteMemoryStore(db_path=db); m=AsyncMemory(embedder=HashEmbedder(), store=s); r=asyncio.run(m.add('hello', user_id='test')); print('OK', r['results'][0]['id']); asyncio.run(m.close())"`
Expected: `OK mem-...`

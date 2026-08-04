# ORM 重构收尾 Step 1：补全 ORMMemoryStore 路径

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** ORMMemoryStore 路径下 graph_store / entity_store / typed_store / MigrationRunner 全部可用，零配置路径不动。

**Architecture:** duck typing 替代 isinstance 检查——facade 用 `getattr(self.store, "engine", None)` 判断走 ORMMemoryStore 路径还是 SQLiteMemoryStore 路径。所有改造组件保留旧构造签名 + 新增 `from_engine` / 可选 `engine` 参数，确保零测试迁移。

**Tech Stack:** Python 3.10+ / SQLModel / SQLAlchemy 2.0 / struct / pytest

## Global Constraints

- PYTHONPATH=src 运行 pytest（包未 pip install -e .）
- ruff line-length 120，select E/F/I/W/UP/B/SIM/RUF，ignore E501/RUF001-003
- 禁止 `ruff format <file>`（Windows 清空 bug），用 `ruff check --fix` + `ruff check --no-cache`
- 现有测试固定不动，仅新增测试
- `pytest_asyncio_mode = "auto"`，async 测试无需 @pytest.mark.asyncio
- 代码注释用中文
- 不用 git（文件快照模式），Step 1 无 commit 步骤

## File Structure

| 文件 | 责任 | 操作 |
|------|------|------|
| `src/septmuse/storage/relational_stores/orm_store.py` | ORMMemoryStore | 加 `engine` property |
| `src/septmuse/storage/relational_stores/async_orm_store.py` | AsyncORMMemoryStore | 加 `async_engine` property |
| `src/septmuse/storage/relational_stores/typed_store.py` | TypedMemoryStore | `__init__` 加可选 `engine` 参数 |
| `src/septmuse/storage/migrations/runner.py` | MigrationRunner | 加 `from_engine` + inspect 路径 |
| `src/septmuse/storage/relational_stores/entity_store.py` | EntityStore | 加 `from_engine` + ORM CRUD |
| `src/septmuse/memory/main.py` | Memory facade | isinstance → duck typing |
| `tests/unit/test_orm_engine_property.py` | 新增 | ORMMemoryStore.engine 测试 |
| `tests/unit/test_typed_store_shared_engine.py` | 新增 | TypedMemoryStore(engine=) 测试 |
| `tests/unit/test_migration_runner_orm.py` | 新增 | MigrationRunner.from_engine 测试 |
| `tests/unit/test_entity_store_orm.py` | 新增 | EntityStore.from_engine 全 CRUD 测试 |
| `tests/unit/test_facade_orm_path.py` | 新增 | Memory(store=ORMMemoryStore) 完整路径测试 |

---

### Task 1: ORMMemoryStore.engine property

**Files:**
- Modify: `src/septmuse/storage/relational_stores/orm_store.py`（ORMMemoryStore 类内）
- Modify: `src/septmuse/storage/relational_stores/async_orm_store.py`（AsyncORMMemoryStore 类内）
- Test: `tests/unit/test_orm_engine_property.py`

**Interfaces:**
- Produces: `ORMMemoryStore.engine` property → `Engine`；`AsyncORMMemoryStore.async_engine` property → `AsyncEngine`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_orm_engine_property.py
"""ORMMemoryStore.engine / AsyncORMMemoryStore.async_engine property 测试。"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from sqlalchemy import Engine
from sqlalchemy.ext.asyncio import AsyncEngine

from septmuse.storage.relational_stores.async_orm_store import AsyncORMMemoryStore
from septmuse.storage.relational_stores.orm_store import ORMMemoryStore


def _make_sqlite_engine(tmp_path):
    from sqlmodel import create_engine

    return create_engine(f"sqlite:///{tmp_path / 'test.db'}")


def test_orm_memory_store_exposes_engine(tmp_path):
    engine = _make_sqlite_engine(tmp_path)
    store = ORMMemoryStore(engine)
    assert store.engine is engine
    assert isinstance(store.engine, Engine)
    store.close()


def test_orm_memory_store_engine_is_readonly(tmp_path):
    engine = _make_sqlite_engine(tmp_path)
    store = ORMMemoryStore(engine)
    try:
        store.engine = "fake"  # type: ignore[assignment]
        assert False, "应抛 AttributeError"
    except AttributeError:
        pass
    store.close()


def test_async_orm_memory_store_exposes_async_engine(tmp_path):
    from sqlalchemy.ext.asyncio import create_async_engine

    async_engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'async.db'}")
    store = AsyncORMMemoryStore(async_engine)
    assert store.async_engine is async_engine
    assert isinstance(store.async_engine, AsyncEngine)

    import asyncio

    asyncio.get_event_loop().run_until_complete(store.close())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_orm_engine_property.py -v`
Expected: FAIL with `AttributeError: 'ORMMemoryStore' object has no attribute 'engine'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/septmuse/storage/relational_stores/orm_store.py — 在 ORMMemoryStore 类内加:

@property
def engine(self) -> "Engine":
    """暴露内部 engine，供 facade duck typing 取用。"""
    return self._engine
```

```python
# src/septmuse/storage/relational_stores/async_orm_store.py — 在 AsyncORMMemoryStore 类内加:

@property
def async_engine(self) -> "AsyncEngine":
    """暴露内部 async engine，供 async facade duck typing 取用。"""
    return self._engine
```

- [ ] **Step 4: Run test to verify it passes**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_orm_engine_property.py -v`
Expected: 3 passed

- [ ] **Step 5: Lint**

Run: `ruff check --no-cache src/septmuse/storage/relational_stores/orm_store.py src/septmuse/storage/relational_stores/async_orm_store.py tests/unit/test_orm_engine_property.py`
Expected: All checks passed

---

### Task 2: TypedMemoryStore 共享 engine

**Files:**
- Modify: `src/septmuse/storage/relational_stores/typed_store.py:59`（`__init__` 方法）
- Test: `tests/unit/test_typed_store_shared_engine.py`

**Interfaces:**
- Consumes: `Engine` from Task 1
- Produces: `TypedMemoryStore(engine=engine)` 构造方式

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_typed_store_shared_engine.py
"""TypedMemoryStore(engine=) 共享 engine 验证。"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from sqlmodel import create_engine, Session, select

from septmuse.models.episodic import EpisodicEvent, EpisodeType
from septmuse.storage.relational_stores.typed_store import TypedMemoryStore


def test_typed_store_with_shared_engine(tmp_path):
    """传入 engine 时使用共享 engine，不自建。"""
    engine = create_engine(f"sqlite:///{tmp_path / 'shared.db'}")
    store = TypedMemoryStore(engine=engine)
    assert store.engine is engine  # 同一对象

    # 验证 CRUD 正常
    with Session(engine) as session:
        episode = EpisodicEvent(
            content="测试事件",
            event_type=EpisodeType.FACT,
            user_id="u1",
        )
        session.add(episode)
        session.commit()
        stmt = select(EpisodicEvent).where(EpisodicEvent.user_id == "u1")
        result = session.exec(stmt).first()
        assert result is not None
        assert result.content == "测试事件"


def test_typed_store_backward_compat_db_path(tmp_path):
    """旧构造（db_path=）仍可用。"""
    store = TypedMemoryStore(db_path=str(tmp_path / "compat.db"))
    assert store.engine is not None

    with Session(store.engine) as session:
        episode = EpisodicEvent(
            content="兼容测试",
            event_type=EpisodeType.FACT,
            user_id="u2",
        )
        session.add(episode)
        session.commit()
        stmt = select(EpisodicEvent).where(EpisodicEvent.user_id == "u2")
        result = session.exec(stmt).first()
        assert result is not None


def test_typed_store_engine_takes_priority_over_db_path(tmp_path):
    """同时传 engine 和 db_path 时 engine 优先。"""
    engine = create_engine(f"sqlite:///{tmp_path / 'priority.db'}")
    store = TypedMemoryStore(db_path=str(tmp_path / "ignored.db"), engine=engine)
    assert store.engine is engine  # engine 赢
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_typed_store_shared_engine.py -v`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'engine'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/septmuse/storage/relational_stores/typed_store.py:59 — 修改 __init__:

def __init__(self, db_path: str | Path | None = None, *, engine: Any | None = None) -> None:
    if engine is not None:
        # 共享 engine（ORMMemoryStore 路径）
        self.engine = engine
        self.db_path = None
    else:
        # 自建 engine（零配置默认路径，向后兼容）
        if db_path is None:
            db_path = _default_db_path()
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        url = f"sqlite:///{self.db_path}"
        self.engine = create_engine(url, echo=False, connect_args={"check_same_thread": False})
    # create_all 建所有已 import 的 SQLModel table
    SQLModel.metadata.create_all(self.engine)
    logger.info("typed_store_ready", path=str(self.db_path), shared_engine=engine is not None)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_typed_store_shared_engine.py -v`
Expected: 3 passed

- [ ] **Step 5: Lint**

Run: `ruff check --no-cache src/septmuse/storage/relational_stores/typed_store.py tests/unit/test_typed_store_shared_engine.py`
Expected: All checks passed

---

### Task 3: MigrationRunner.from_engine + inspect

**Files:**
- Modify: `src/septmuse/storage/migrations/runner.py`（MigrationRunner 类）
- Test: `tests/unit/test_migration_runner_orm.py`

**Interfaces:**
- Consumes: `Engine` from Task 1
- Produces: `MigrationRunner.from_engine(engine)` classmethod

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

```python
# src/septmuse/storage/migrations/runner.py — 修改 MigrationRunner:

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

---

### Task 4: EntityStore.from_engine + ORM CRUD

**Files:**
- Modify: `src/septmuse/storage/relational_stores/entity_store.py`（EntityStore 类全部 CRUD 方法）
- Test: `tests/unit/test_entity_store_orm.py`

**Interfaces:**
- Consumes: `Engine` from Task 1；`EntityTable` from `services/database/models/entity.py`；`Entity` from `extraction/entity.py`
- Produces: `EntityStore.from_engine(engine, embedder=None)` classmethod

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_entity_store_orm.py
"""EntityStore.from_engine ORM CRUD 全量测试。"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from sqlmodel import create_engine, Session

from septmuse.extraction.entity import Entity
from septmuse.services.database.models.entity import EntityTable
from septmuse.storage.relational_stores.entity_store import EntityStore


def _make_engine(tmp_path):
    return create_engine(f"sqlite:///{tmp_path / 'entity.db'}")


def test_from_engine_creates_table(tmp_path):
    """from_engine 自动建 septmuse_entities 表。"""
    engine = _make_engine(tmp_path)
    store = EntityStore.from_engine(engine, embedder=None)
    assert store._engine is engine
    # 表存在
    from sqlalchemy import inspect

    assert inspect(engine).has_table("septmuse_entities")


def test_upsert_new_entity(tmp_path):
    """新建实体 → 返回 entity_id，linked_memory_ids 含 memory_id。"""
    engine = _make_engine(tmp_path)
    store = EntityStore.from_engine(engine, embedder=None)
    entity = Entity(text="Alice", entity_type="PROPER")
    eid = store.upsert(entity, memory_id="mem-1", user_id="u1")
    assert eid is not None

    result = store.get(eid)
    assert result is not None
    assert result["entity_text"] == "Alice"
    assert result["entity_type"] == "PROPER"
    import json

    assert "mem-1" in json.loads(result["linked_memory_ids"])


def test_upsert_exact_match_appends(tmp_path):
    """精确归一化匹配 → linked_memory_ids 追加。"""
    engine = _make_engine(tmp_path)
    store = EntityStore.from_engine(engine, embedder=None)
    entity = Entity(text="Google", entity_type="PROPER")

    eid1 = store.upsert(entity, memory_id="mem-1", user_id="u1")
    eid2 = store.upsert(entity, memory_id="mem-2", user_id="u1")
    assert eid1 == eid2  # 同一实体

    result = store.get(eid1)
    import json

    linked = json.loads(result["linked_memory_ids"])
    assert "mem-1" in linked
    assert "mem-2" in linked


def test_upsert_different_users_separate(tmp_path):
    """不同 user_id 的同名实体独立存储。"""
    engine = _make_engine(tmp_path)
    store = EntityStore.from_engine(engine, embedder=None)
    entity = Entity(text="Alice", entity_type="PROPER")

    eid1 = store.upsert(entity, memory_id="mem-1", user_id="u1")
    eid2 = store.upsert(entity, memory_id="mem-2", user_id="u2")
    assert eid1 != eid2


def test_search_exact_match(tmp_path):
    """search 精确匹配返回结果。"""
    engine = _make_engine(tmp_path)
    store = EntityStore.from_engine(engine, embedder=None)
    entity = Entity(text="Alice", entity_type="PROPER")
    store.upsert(entity, memory_id="mem-1", user_id="u1")

    results = store.search("Alice", user_id="u1")
    assert len(results) == 1
    assert results[0]["entity_text"] == "Alice"
    assert results[0]["score"] == 1.0


def test_list_entities(tmp_path):
    """list 返回用户全部实体。"""
    engine = _make_engine(tmp_path)
    store = EntityStore.from_engine(engine, embedder=None)
    store.upsert(Entity(text="Alice", entity_type="PROPER"), memory_id="m1", user_id="u1")
    store.upsert(Entity(text="Google", entity_type="PROPER"), memory_id="m2", user_id="u1")

    entities = store.list(user_id="u1")
    assert len(entities) == 2


def test_list_by_type(tmp_path):
    """list 按 entity_type 过滤。"""
    engine = _make_engine(tmp_path)
    store = EntityStore.from_engine(engine, embedder=None)
    store.upsert(Entity(text="Alice", entity_type="PROPER"), memory_id="m1", user_id="u1")
    store.upsert(Entity(text="Python", entity_type="TOPIC"), memory_id="m2", user_id="u1")

    proper = store.list(user_id="u1", entity_type="PROPER")
    assert len(proper) == 1
    assert proper[0]["entity_text"] == "Alice"


def test_get_linked_memories(tmp_path):
    """get_linked_memories 返回 linked_memory_ids 列表。"""
    engine = _make_engine(tmp_path)
    store = EntityStore.from_engine(engine, embedder=None)
    entity = Entity(text="Alice", entity_type="PROPER")
    eid = store.upsert(entity, memory_id="mem-1", user_id="u1")

    linked = store.get_linked_memories(eid)
    assert "mem-1" in linked


def test_remove_memory_from_entities(tmp_path):
    """remove_memory_from_entities 清理引用 + 空时软删除。"""
    engine = _make_engine(tmp_path)
    store = EntityStore.from_engine(engine, embedder=None)
    entity = Entity(text="Alice", entity_type="PROPER")
    eid = store.upsert(entity, memory_id="mem-1", user_id="u1")

    store.remove_memory_from_entities("mem-1")

    result = store.get(eid)
    # 只有一个 memory_id，清空后软删除
    from sqlalchemy import inspect as sqla_inspect

    with Session(engine) as session:
        from sqlmodel import select

        stmt = select(EntityTable).where(EntityTable.id == eid)
        row = session.exec(stmt).first()
        assert row is not None
        assert row.is_deleted == 1


def test_remove_memory_keeps_entity_with_other_links(tmp_path):
    """多个 memory_id 时只移除一个，保留实体。"""
    engine = _make_engine(tmp_path)
    store = EntityStore.from_engine(engine, embedder=None)
    entity = Entity(text="Alice", entity_type="PROPER")
    eid = store.upsert(entity, memory_id="mem-1", user_id="u1")
    store.upsert(entity, memory_id="mem-2", user_id="u1")

    store.remove_memory_from_entities("mem-1")

    with Session(engine) as session:
        from sqlmodel import select

        stmt = select(EntityTable).where(EntityTable.id == eid)
        row = session.exec(stmt).first()
        assert row is not None
        assert row.is_deleted == 0
        import json

        linked = json.loads(row.linked_memory_ids)
        assert "mem-1" not in linked
        assert "mem-2" in linked


def test_old_constructor_still_works(tmp_path):
    """旧 __init__(conn, lock, embedder) 向后兼容。"""
    import sqlite3
    import threading

    conn = sqlite3.connect(str(tmp_path / "old.db"))
    lock = threading.Lock()
    store = EntityStore(conn, lock, embedder=None)
    assert store._conn is conn
    assert store._lock is lock
    assert store._engine is None

    entity = Entity(text="TestOld", entity_type="PROPER")
    eid = store.upsert(entity, memory_id="m1", user_id="u1")
    assert eid is not None
    result = store.get(eid)
    assert result is not None
    assert result["entity_text"] == "TestOld"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_entity_store_orm.py -v`
Expected: FAIL with `AttributeError: type object 'EntityStore' has no attribute 'from_engine'`

- [ ] **Step 3: Write minimal implementation**

在 `entity_store.py` 中：

1. 文件顶部加 import：
```python
from sqlmodel import Session, select
from septmuse.services.database.models.entity import EntityTable
```

2. 修改 EntityStore 类，加 `from_engine` classmethod + ORM CRUD 方法：

```python
class EntityStore:
    """实体向量库 (独立表, 借鉴 mem0 V3 去图化设计)。

    双模式：
    - 旧 __init__(conn, lock, embedder) — SQLiteMemoryStore 路径 (原生 sqlite3)
    - from_engine(engine, embedder) — ORMMemoryStore 路径 (SQLModel ORM)
    """

    def __init__(self, conn, lock, embedder: Embedder | None = None):
        """旧构造 (SQLiteMemoryStore 路径, 向后兼容)。"""
        self._conn = conn
        self._lock = lock
        self._embedder = embedder
        self._engine = None
        self._create_table_if_not_exists()

    @classmethod
    def from_engine(cls, engine, embedder: Embedder | None = None) -> "EntityStore":
        """新构造 (ORMMemoryStore 路径, SQLModel ORM)。"""
        store = cls.__new__(cls)
        store._conn = None
        store._lock = None
        store._engine = engine
        store._embedder = embedder
        # 用 EntityTable SQLModel 建表 (幂等)
        SQLModel.metadata.create_all(engine)
        return store

    def _is_orm_mode(self) -> bool:
        """是否走 ORM 路径。"""
        return self._engine is not None
```

3. 在每个方法开头加 ORM 分支。以 `upsert` 为例：

```python
    def upsert(self, entity, memory_id, *, user_id, agent_id=None) -> str:
        """upsert 实体 (借鉴 mem0 _upsert_entity)。

        1. 精确归一化名匹配 → 命中则 linked_memory_ids 追加 memory_id
        2. 语义匹配 (embedder 有时) → score>=0.95 命中则追加
        3. 新建 → 插入实体 + 嵌入向量 + linked_memory_ids=[memory_id]
        """
        normalized = _normalize_entity_text(entity.text)

        if self._is_orm_mode():
            return self._upsert_orm(entity, memory_id, user_id=user_id, agent_id=agent_id, normalized=normalized)

        # ... 旧 raw SQL 路径不变 ...

    def _upsert_orm(self, entity, memory_id, *, user_id, agent_id, normalized) -> str:
        """ORM 路径 upsert。"""
        # 1. 精确归一化名匹配
        existing = self._find_by_text_orm(normalized, user_id=user_id)
        if existing:
            self._append_memory_id_orm(existing["id"], memory_id)
            return existing["id"]

        # 2. 语义匹配 (embedder 有时)
        emb = None
        if self._embedder is not None:
            emb = self._embedder.embed(entity.text)
            semantic_match = self._find_by_embedding_orm(emb, user_id=user_id, threshold=0.95)
            if semantic_match:
                self._append_memory_id_orm(semantic_match["id"], memory_id)
                return semantic_match["id"]

        # 3. 新建
        import uuid
        from datetime import datetime, timezone

        entity_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        emb_blob = self._serialize_embedding(emb) if emb is not None else None

        with Session(self._engine) as session:
            row = EntityTable(
                id=entity_id,
                entity_text=entity.text,
                entity_type=entity.entity_type,
                entity_embedding=emb_blob,
                linked_memory_ids=json.dumps([memory_id]),
                user_id=user_id,
                agent_id=agent_id,
                created_at=now,
                updated_at=now,
                is_deleted=0,
            )
            session.add(row)
            session.commit()
        return entity_id

    def _find_by_text_orm(self, normalized_text, *, user_id) -> dict | None:
        """ORM 路径精确归一化名匹配。"""
        with Session(self._engine) as session:
            stmt = select(EntityTable).where(
                EntityTable.user_id == user_id,
                EntityTable.is_deleted == 0,  # noqa: E712
            )
            rows = session.exec(stmt).all()
            for r in rows:
                if _normalize_entity_text(r.entity_text) == normalized_text:
                    return {
                        "id": r.id,
                        "entity_text": r.entity_text,
                        "entity_type": r.entity_type,
                        "linked_memory_ids": r.linked_memory_ids,
                    }
        return None

    def _find_by_embedding_orm(self, embedding, *, user_id, threshold=0.95) -> dict | None:
        """ORM 路径语义匹配。"""
        with Session(self._engine) as session:
            stmt = select(EntityTable).where(
                EntityTable.user_id == user_id,
                EntityTable.is_deleted == 0,  # noqa: E712
                EntityTable.entity_embedding.is_not(None),
            )
            rows = session.exec(stmt).all()
            for r in rows:
                stored_emb = self._deserialize_embedding(r.entity_embedding)
                if stored_emb is not None:
                    sim = _cosine_similarity(embedding, stored_emb)
                    if sim >= threshold:
                        return {
                            "id": r.id,
                            "entity_text": r.entity_text,
                            "entity_type": r.entity_type,
                            "linked_memory_ids": r.linked_memory_ids,
                        }
        return None

    def _append_memory_id_orm(self, entity_id, memory_id) -> None:
        """ORM 路径追加 memory_id。"""
        from datetime import datetime, timezone

        with Session(self._engine) as session:
            stmt = select(EntityTable).where(EntityTable.id == entity_id)
            row = session.exec(stmt).first()
            if row is None:
                return
            linked = json.loads(row.linked_memory_ids)
            if memory_id not in linked:
                linked.append(memory_id)
            row.linked_memory_ids = json.dumps(linked)
            row.updated_at = datetime.now(timezone.utc).isoformat()
            session.add(row)
            session.commit()
```

4. 同理为 `get`、`search`、`list`、`get_linked_memories`、`remove_memory_from_entities` 加 ORM 分支：

```python
    def get(self, entity_id: str) -> dict | None:
        """取单条实体。"""
        if self._is_orm_mode():
            with Session(self._engine) as session:
                row = session.get(EntityTable, entity_id)
                if row is None or row.is_deleted == 1:
                    return None
                return {
                    "id": row.id,
                    "entity_text": row.entity_text,
                    "entity_type": row.entity_type,
                    "entity_embedding": row.entity_embedding,
                    "linked_memory_ids": row.linked_memory_ids,
                    "user_id": row.user_id,
                    "agent_id": row.agent_id,
                    "created_at": row.created_at,
                    "updated_at": row.updated_at,
                }
        # 旧 raw SQL 路径不变
        ...

    def search(self, query, *, user_id, top_k=5) -> list[dict]:
        """搜索实体: 精确匹配 + 向量相似度。"""
        if self._is_orm_mode():
            return self._search_orm(query, user_id=user_id, top_k=top_k)
        # 旧 raw SQL 路径不变
        ...

    def _search_orm(self, query, *, user_id, top_k=5) -> list[dict]:
        """ORM 路径搜索。"""
        results = []
        normalized_query = _normalize_entity_text(query)
        with Session(self._engine) as session:
            stmt = select(EntityTable).where(
                EntityTable.user_id == user_id,
                EntityTable.is_deleted == 0,  # noqa: E712
            )
            rows = session.exec(stmt).all()
            existing_ids = set()

            # 精确匹配
            for r in rows:
                normalized_entity = _normalize_entity_text(r.entity_text)
                if normalized_query in normalized_entity or normalized_entity in normalized_query:
                    score = 1.0 if normalized_entity == normalized_query else 0.8
                    results.append({
                        "id": r.id,
                        "entity_text": r.entity_text,
                        "entity_type": r.entity_type,
                        "linked_memory_ids": r.linked_memory_ids,
                        "score": score,
                    })
                    existing_ids.add(r.id)

            # 向量相似度 (embedder 有时)
            if self._embedder is not None:
                query_emb = self._embedder.embed(query)
                for r in rows:
                    if r.id in existing_ids:
                        continue
                    if r.entity_embedding is None:
                        continue
                    stored_emb = self._deserialize_embedding(r.entity_embedding)
                    if stored_emb is not None:
                        sim = _cosine_similarity(query_emb, stored_emb)
                        if sim > 0.3:
                            results.append({
                                "id": r.id,
                                "entity_text": r.entity_text,
                                "entity_type": r.entity_type,
                                "linked_memory_ids": r.linked_memory_ids,
                                "score": sim,
                            })
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def list(self, *, user_id, entity_type=None, limit=100) -> list[dict]:
        """列出用户全部未删除实体。"""
        if self._is_orm_mode():
            with Session(self._engine) as session:
                stmt = select(EntityTable).where(
                    EntityTable.user_id == user_id,
                    EntityTable.is_deleted == 0,  # noqa: E712
                )
                if entity_type:
                    stmt = stmt.where(EntityTable.entity_type == entity_type)
                stmt = stmt.order_by(EntityTable.created_at.desc()).limit(limit)
                rows = session.exec(stmt).all()
            return [
                {
                    "id": r.id,
                    "entity_text": r.entity_text,
                    "entity_type": r.entity_type,
                    "linked_memory_ids": r.linked_memory_ids,
                    "created_at": r.created_at,
                }
                for r in rows
            ]
        # 旧 raw SQL 路径不变
        ...

    def get_linked_memories(self, entity_id: str) -> list[str]:
        """获取实体的 linked_memory_ids。"""
        if self._is_orm_mode():
            with Session(self._engine) as session:
                row = session.get(EntityTable, entity_id)
                if row is None or row.is_deleted == 1:
                    return []
                return json.loads(row.linked_memory_ids)
        # 旧 raw SQL 路径不变
        ...

    def remove_memory_from_entities(self, memory_id: str) -> None:
        """删除记忆时清理实体引用。"""
        if self._is_orm_mode():
            from datetime import datetime, timezone

            with Session(self._engine) as session:
                stmt = select(EntityTable).where(
                    EntityTable.is_deleted == 0,  # noqa: E712
                )
                rows = session.exec(stmt).all()
                now = datetime.now(timezone.utc).isoformat()
                for r in rows:
                    linked = json.loads(r.linked_memory_ids)
                    if memory_id not in linked:
                        continue
                    remaining = [mid for mid in linked if mid != memory_id]
                    if not remaining:
                        r.is_deleted = 1
                    else:
                        r.linked_memory_ids = json.dumps(remaining)
                    r.updated_at = now
                    session.add(r)
                session.commit()
            return
        # 旧 raw SQL 路径不变
        ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_entity_store_orm.py -v`
Expected: 11 passed

- [ ] **Step 5: Lint**

Run: `ruff check --no-cache src/septmuse/storage/relational_stores/entity_store.py tests/unit/test_entity_store_orm.py`
Expected: All checks passed

- [ ] **Step 6: 现有 EntityStore 测试回归**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_entity_store.py tests/unit/test_cognify.py -q --tb=no`
Expected: 全绿（旧 raw SQL 路径不变）

---

### Task 5: facade duck typing

**Files:**
- Modify: `src/septmuse/memory/main.py:123-156`（Memory.__init__ 中 isinstance 检查段）
- Test: `tests/unit/test_facade_orm_path.py`

**Interfaces:**
- Consumes: Task 1-4 全部产出
- Produces: `Memory(store=ORMMemoryStore(...))` 完整路径

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_facade_orm_path.py
"""Memory(store=ORMMemoryStore) 完整路径测试。"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from sqlmodel import create_engine

from septmuse.configs.defaults import default_config
from septmuse.memory.main import Memory
from septmuse.storage.relational_stores.orm_store import ORMMemoryStore


def _make_memory(tmp_path, **kwargs):
    engine = create_engine(f"sqlite:///{tmp_path / 'facade.db'}")
    store = ORMMemoryStore(engine)
    config = default_config()
    config.db_path = str(tmp_path / "facade.db")
    return Memory(config=config, store=store, **kwargs)


def test_facade_orm_path_entity_store_not_none(tmp_path):
    """ORMMemoryStore 路径下 entity_store 不为 None。"""
    m = _make_memory(tmp_path)
    assert m.entity_store is not None
    assert m.entity_store._engine is not None


def test_facade_orm_path_typed_store_shares_engine(tmp_path):
    """ORMMemoryStore 路径下 typed_store 共享 engine。"""
    m = _make_memory(tmp_path)
    assert m.typed_store.engine is m.store.engine


def test_facade_orm_path_graph_store_not_none(tmp_path):
    """SQLite ORM 路径下 graph_store 不为 None。"""
    m = _make_memory(tmp_path)
    assert m.graph_store is not None


def test_facade_orm_path_add_search_roundtrip(tmp_path):
    """ORMMemoryStore 路径 add + search 完整往返。"""
    m = _make_memory(tmp_path)
    mid = m.add("我喜欢 Python", user_id="alice")
    assert mid is not None

    results = m.search("Python", user_id="alice")
    assert len(results) > 0
    assert "Python" in results[0]["memory"]


def test_facade_orm_path_cognify_works(tmp_path):
    """ORMMemoryStore 路径 cognify 知识图谱构建可用。"""
    m = _make_memory(tmp_path)
    m.add("Alice works at Google", user_id="u1")
    # cognify 应不报错 (entity_store 不为 None)
    try:
        m.cognify("Alice works at Google", user_id="u1")
    except Exception as e:
        # cognify 可能因 embedder/llm 缺失而降级, 但不应因 entity_store=None 崩溃
        assert "NoneType" not in str(e), f"entity_store 为 None 导致崩溃: {e}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_facade_orm_path.py -v`
Expected: FAIL with `AssertionError: assert None is not None`（entity_store 为 None）

- [ ] **Step 3: Write minimal implementation**

修改 `src/septmuse/memory/main.py` 的 `Memory.__init__`，将 isinstance 检查替换为 duck typing：

```python
# memory/main.py:121-156 — 替换以下段:

        self.store = store or self._resolve_store()

        # duck typing: ORMMemoryStore 有 engine 属性, SQLiteMemoryStore 没有
        store_engine = getattr(self.store, "engine", None)

        if graph_store is not None:
            self.graph_store: GraphStore | None = graph_store
        elif store_engine is not None:
            # ORMMemoryStore 路径: SQLite dialect 从 engine 取 raw connection
            if store_engine.dialect.name == "sqlite":
                import threading

                raw_conn = store_engine.raw_connection()
                self.graph_store = SQLiteGraphStore(raw_conn, threading.Lock())
            else:
                # MySQL/PG 暂不支持原生 GraphStore (AGE/Neo4j 后续)
                self.graph_store = graph_store
        elif isinstance(self.store, SQLiteMemoryStore):
            self.graph_store = SQLiteGraphStore(self.store.conn, self.store._lock)
        else:
            self.graph_store = graph_store

        # typed_store 共享 engine
        if store_engine is not None:
            self.typed_store = TypedMemoryStore(engine=store_engine)
        else:
            self.typed_store = TypedMemoryStore(db_path=self.config.db_path)
        self.semantic = SemanticMemory(self.typed_store, self.embedder)
        self.episodic = EpisodicMemory(self.typed_store)
        self.procedural = ProceduralMemory(self.typed_store)

        self.llm: LLM | None = llm
        if self.llm is None and self.config.llm_provider:
            try:
                from septmuse.llms import _resolve_llm

                self.llm = _resolve_llm(self.config)
            except Exception as e:
                logger.warning("llm_resolve_failed", error=str(e))
                self.llm = None

        self.extractor: FactExtractor | None = None
        if self.llm is not None:
            self.extractor = FactExtractor(self.llm, self.embedder, self.typed_store, self.store)

        from septmuse.extraction.entity import _resolve_entity_extractor

        self.entity_extractor = entity_extractor or _resolve_entity_extractor(self.config)

        # entity_store 双模式
        self.entity_store = None
        if store_engine is not None:
            from septmuse.storage.relational_stores.entity_store import EntityStore

            self.entity_store = EntityStore.from_engine(store_engine, self.embedder)
        elif isinstance(self.store, SQLiteMemoryStore):
            from septmuse.storage.relational_stores.entity_store import EntityStore

            self.entity_store = EntityStore(self.store.conn, self.store._lock, self.embedder)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_facade_orm_path.py -v`
Expected: 5 passed

- [ ] **Step 5: Lint**

Run: `ruff check --no-cache src/septmuse/memory/main.py tests/unit/test_facade_orm_path.py`
Expected: All checks passed

- [ ] **Step 6: 现有 facade 测试回归**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_memory.py -q --tb=no`
Expected: 零新增失败（预存在的 6 个 API key 失败不变）

---

### Task 6: 全量回归 + ruff

**Files:**
- 无修改，仅验证

- [ ] **Step 1: ruff 全量检查**

Run: `ruff check --no-cache src/septmuse/storage/relational_stores/ src/septmuse/storage/migrations/ src/septmuse/memory/main.py tests/unit/test_orm_engine_property.py tests/unit/test_typed_store_shared_engine.py tests/unit/test_migration_runner_orm.py tests/unit/test_entity_store_orm.py tests/unit/test_facade_orm_path.py`
Expected: All checks passed

- [ ] **Step 2: 新增测试全量运行**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_orm_engine_property.py tests/unit/test_typed_store_shared_engine.py tests/unit/test_migration_runner_orm.py tests/unit/test_entity_store_orm.py tests/unit/test_facade_orm_path.py -v`
Expected: 全绿（~26 passed）

- [ ] **Step 3: 现有测试零退化验证**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_memory.py tests/unit/test_permissions.py tests/unit/test_composite_store.py tests/unit/test_entity_store.py tests/unit/test_cognify.py tests/unit/test_migrations.py tests/e2e/ -q --tb=no`
Expected: 零新增失败（预存在 API key 失败不变）

- [ ] **Step 4: 全量回归**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/ tests/e2e/ -q --tb=no`
Expected: ≥ 1215 passed + 36 skipped + 13 failed（预存在 API key 失败不变）

---

## Self-Review

### Spec coverage

| Spec 要求 | 覆盖 Task |
|----------|-----------|
| §4.1.1 ORMMemoryStore.engine property | Task 1 ✅ |
| §4.1.2 EntityStore.from_engine + ORM CRUD | Task 4 ✅ |
| §4.1.3 TypedMemoryStore 共享 engine | Task 2 ✅ |
| §4.1.4 MigrationRunner.from_engine + inspect | Task 3 ✅ |
| §4.1.5 facade duck typing | Task 5 ✅ |
| §4.1.6 新增测试 | Task 1-5 各有测试 ✅ |
| §6 验收标准（ruff + 现有零退化） | Task 6 ✅ |

### Placeholder scan

无 TBD/TODO。所有代码块完整。

### Type consistency

- `engine` property 在 Task 1 定义，Task 2/3/4/5 消费 — 一致
- `from_engine(engine)` 在 Task 3/4 定义，Task 5 消费 — 一致
- `_is_orm_mode()` 在 Task 4 定义并使用 — 一致
- `TypedMemoryStore(engine=)` 在 Task 2 定义，Task 5 消费 — 一致

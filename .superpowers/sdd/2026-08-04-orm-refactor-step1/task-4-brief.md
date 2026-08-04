### Task 4: EntityStore.from_engine + ORM CRUD

**Files:**
- Modify: `src/septmuse/storage/relational_stores/entity_store.py`（EntityStore 类全部 CRUD 方法）
- Test: `tests/unit/test_entity_store_orm.py`

**Interfaces:**
- Consumes: `Engine` from Task 1；`EntityTable` from `services/database/models/entity.py`；`Entity` from `extraction/entity.py`
- Produces: `EntityStore.from_engine(engine, embedder=None)` classmethod

**Global Constraints:**
- PYTHONPATH=src 运行 pytest（包未 pip install -e .）
- ruff line-length 120，select E/F/I/W/UP/B/SIM/RUF，ignore E501/RUF001-003
- 禁止 `ruff format <file>`（Windows 清空 bug），用 `ruff check --fix` + `ruff check --no-cache`
- 现有测试固定不动，仅新增测试
- `pytest_asyncio_mode = "auto"`，async 测试无需 @pytest.mark.asyncio
- 代码注释用中文
- 不用 git（文件快照模式），无 commit 步骤

## Implementation Strategy

EntityStore currently uses raw `sqlite3.Connection` + `?` placeholders. You must add a SECOND mode (ORM) without breaking the first mode (raw SQL). The approach:

1. Add `from_engine(cls, engine, embedder=None)` classmethod that creates an EntityStore with `_engine` set and `_conn=None`
2. Add `_is_orm_mode()` helper: returns `self._engine is not None`
3. For EACH public method (upsert, get, search, list, get_linked_memories, remove_memory_from_entities), add an `if self._is_orm_mode(): return self._method_orm(...)` branch at the top, keeping the old raw SQL path in the `else` branch
4. ORM methods use `Session(engine)` + `select(EntityTable)` / `session.get(EntityTable, id)` / `session.add()`
5. The `EntityTable` SQLModel class is already defined at `src/septmuse/services/database/models/entity.py` with `__tablename__ = "septmuse_entities"` — same column names as the raw SQL

## Step 1: Write the failing test

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

## Step 2: Run test to verify it fails

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_entity_store_orm.py -v`
Expected: FAIL with `AttributeError: type object 'EntityStore' has no attribute 'from_engine'`

## Step 3: Write minimal implementation

Add to the top of `entity_store.py` (after existing imports):
```python
from sqlmodel import Session, select, SQLModel
from septmuse.services.database.models.entity import EntityTable
```

Modify the EntityStore class:

1. Modify `__init__` to set `self._engine = None` (add this line at the end)
2. Add `from_engine` classmethod
3. Add `_is_orm_mode` helper
4. Add ORM branches to each method

### `from_engine` classmethod:
```python
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

### ORM methods to add (as `_method_orm` private methods):

For each public method, add `if self._is_orm_mode(): return self._method_orm(...)` at the top, then implement the `_method_orm` variant.

Key ORM patterns:
- **Query**: `with Session(self._engine) as session: stmt = select(EntityTable).where(...); rows = session.exec(stmt).all()`
- **Get by id**: `session.get(EntityTable, entity_id)`
- **Insert**: `row = EntityTable(...); session.add(row); session.commit()`
- **Update**: modify fields on the fetched row, `session.add(row)`, `session.commit()`
- **Soft delete**: set `row.is_deleted = 1`

The semantic deduplication logic (exact normalized match → semantic match ≥0.95 → new) stays the same — only the storage layer changes from raw SQL to ORM.

## Step 4: Run test to verify it passes

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_entity_store_orm.py -v`
Expected: 11 passed

## Step 5: Lint

Run: `ruff check --no-cache src/septmuse/storage/relational_stores/entity_store.py tests/unit/test_entity_store_orm.py`
Expected: All checks passed

## Step 6: 现有 EntityStore 测试回归

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_entity_store.py tests/unit/test_cognify.py -q --tb=no`
Expected: 全绿（旧 raw SQL 路径不变）

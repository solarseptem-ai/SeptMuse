### Task 5: facade duck typing

**Files:**
- Modify: `src/septmuse/memory/main.py:123-156`（Memory.__init__ 中 isinstance 检查段）
- Test: `tests/unit/test_facade_orm_path.py`

**Interfaces:**
- Consumes: Task 1-4 全部产出
  - `ORMMemoryStore.engine` property (Task 1)
  - `TypedMemoryStore(engine=)` (Task 2)
  - `MigrationRunner.from_engine` (Task 3, not directly used in facade but available)
  - `EntityStore.from_engine(engine, embedder)` (Task 4)
- Produces: `Memory(store=ORMMemoryStore(...))` 完整路径

**Global Constraints:**
- PYTHONPATH=src 运行 pytest（包未 pip install -e .）
- ruff line-length 120，select E/F/I/W/UP/B/SIM/RUF，ignore E501/RUF001-003
- 禁止 `ruff format <file>`（Windows 清空 bug），用 `ruff check --fix` + `ruff check --no-cache`
- 现有测试固定不动，仅新增测试
- `pytest_asyncio_mode = "auto"`，async 测试无需 @pytest.mark.asyncio
- 代码注释用中文
- 不用 git（文件快照模式），无 commit 步骤

**Known issue from Task 2 review**: `TypedMemoryStore.close()` calls `engine.dispose()`. When engine is shared between `store` and `typed_store`, `Memory.close()` will double-dispose. Add `_owns_engine` flag to TypedMemoryStore to prevent this: if engine was passed (not created), don't dispose in close().

## Step 1: Write the failing test

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

## Step 2: Run test to verify it fails

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_facade_orm_path.py -v`
Expected: FAIL with `AssertionError: assert None is not None`（entity_store 为 None）

## Step 3: Write minimal implementation

### 3a: Fix TypedMemoryStore close() double-dispose

In `src/septmuse/storage/relational_stores/typed_store.py`:

1. In `__init__`, add `self._owns_engine = engine is None` (True when self-created, False when shared)
2. In `close()`, only dispose if `self._owns_engine`:

```python
def close(self) -> None:
    if getattr(self, "_owns_engine", True):
        self.engine.dispose()
```

### 3b: Modify Memory.__init__ in `src/septmuse/memory/main.py`

Replace the section at lines 121-156 (from `self.store = store or self._resolve_store()` through the entity_store block) with duck typing:

```python
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

## Step 4: Run test to verify it passes

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_facade_orm_path.py -v`
Expected: 5 passed

## Step 5: Lint

Run: `ruff check --no-cache src/septmuse/memory/main.py src/septmuse/storage/relational_stores/typed_store.py tests/unit/test_facade_orm_path.py`
Expected: All checks passed

## Step 6: 现有 facade 测试回归

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_memory.py -q --tb=no`
Expected: 零新增失败（预存在的 6 个 API key 失败不变）

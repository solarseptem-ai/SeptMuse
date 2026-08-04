### Task 1: ORMMemoryStore.engine property

**Files:**
- Modify: `src/septmuse/storage/relational_stores/orm_store.py`（ORMMemoryStore 类内）
- Modify: `src/septmuse/storage/relational_stores/async_orm_store.py`（AsyncORMMemoryStore 类内）
- Test: `tests/unit/test_orm_engine_property.py`

**Interfaces:**
- Produces: `ORMMemoryStore.engine` property → `Engine`；`AsyncORMMemoryStore.async_engine` property → `AsyncEngine`

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

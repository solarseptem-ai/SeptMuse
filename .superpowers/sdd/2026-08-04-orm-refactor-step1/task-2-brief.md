### Task 2: TypedMemoryStore 共享 engine

**Files:**
- Modify: `src/septmuse/storage/relational_stores/typed_store.py:59`（`__init__` 方法）
- Test: `tests/unit/test_typed_store_shared_engine.py`

**Interfaces:**
- Consumes: `Engine` from Task 1
- Produces: `TypedMemoryStore(engine=engine)` 构造方式

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

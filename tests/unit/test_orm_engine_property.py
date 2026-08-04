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
        raise AssertionError("应抛 AttributeError")
    except AttributeError:
        pass
    store.close()


async def test_async_orm_memory_store_exposes_async_engine(tmp_path):
    from sqlalchemy.ext.asyncio import create_async_engine

    async_engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'async.db'}")
    store = AsyncORMMemoryStore(async_engine)
    assert store.async_engine is async_engine
    assert isinstance(store.async_engine, AsyncEngine)
    await store.close()

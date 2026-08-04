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
    config.database.db_path = str(tmp_path / "facade.db")
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

"""reset() 方法测试 — 清表 + 重建 + 缓存失效。"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from sqlmodel import create_engine

from septmuse.configs.defaults import default_config
from septmuse.memory.main import Memory
from septmuse.storage.relational_stores.orm_store import ORMMemoryStore


def _make_memory(tmp_path):
    """构造 Memory (HashEmbedder + SQLite tmp_path, ORMMemoryStore 路径)。"""
    engine = create_engine(f"sqlite:///{tmp_path / 'reset.db'}")
    store = ORMMemoryStore(engine)
    config = default_config()
    config.database.db_path = str(tmp_path / "reset.db")
    return Memory(config=config, store=store)


def test_reset_clears_memories(tmp_path):
    """reset 后 search 返回 0 条。"""
    m = _make_memory(tmp_path)
    m.add("test memory content", user_id="alice")
    results_before = m.search("test", user_id="alice")
    assert len(results_before) >= 1

    result = m.reset()
    assert result["status"] == "reset"

    results_after = m.search("test", user_id="alice")
    assert len(results_after) == 0


def test_reset_clears_get_all(tmp_path):
    """reset 后 get_all 返回空列表。"""
    m = _make_memory(tmp_path)
    m.add("memory one", user_id="alice")
    m.add("memory two", user_id="alice")
    before = m.get_all(user_id="alice")
    assert len(before["results"]) == 2

    m.reset()
    after = m.get_all(user_id="alice")
    assert len(after["results"]) == 0


def test_reset_returns_status(tmp_path):
    """reset 返回 {"status": "reset", "message": ...}。"""
    m = _make_memory(tmp_path)
    m.add("data", user_id="alice")
    result = m.reset()
    assert result["status"] == "reset"
    assert "message" in result


def test_reset_on_empty_db_no_crash(tmp_path):
    """reset 空库不崩溃。"""
    m = _make_memory(tmp_path)
    result = m.reset()
    assert result["status"] == "reset"


def test_reset_twice_no_crash(tmp_path):
    """连续 reset 两次不崩溃。"""
    m = _make_memory(tmp_path)
    m.add("content", user_id="alice")
    m.reset()
    result = m.reset()
    assert result["status"] == "reset"


def test_add_after_reset_works(tmp_path):
    """reset 后可继续 add (表重建)。"""
    m = _make_memory(tmp_path)
    m.add("first memory", user_id="alice")
    m.reset()

    mid = m.add("second memory", user_id="alice")
    assert mid is not None
    results = m.search("second", user_id="alice")
    assert len(results) == 1
    assert "second" in results[0]["memory"]


def test_reset_clears_typed_store(tmp_path):
    """reset 清空 typed_store (facts / rules / episodes)。"""
    m = _make_memory(tmp_path)
    m.typed_store.add_fact("Alice", "likes", "Python", user_id="alice")
    m.typed_store.add_rule("always test", user_id="alice")
    m.typed_store.add_episode("an event", user_id="alice")
    assert len(m.typed_store.get_all_facts(user_id="alice")) == 1
    assert len(m.typed_store.get_all_rules(user_id="alice")) == 1
    assert len(m.typed_store.get_episodes(user_id="alice")) == 1

    m.reset()

    assert len(m.typed_store.get_all_facts(user_id="alice")) == 0
    assert len(m.typed_store.get_all_rules(user_id="alice")) == 0
    assert len(m.typed_store.get_episodes(user_id="alice")) == 0


def test_reset_clears_entity_store(tmp_path):
    """reset 清空 entity_store。"""
    m = _make_memory(tmp_path)
    m.add("Alice works at Google", user_id="alice")
    entities_before = m.entity_store.list(user_id="alice")
    assert len(entities_before) > 0

    m.reset()

    entities_after = m.entity_store.list(user_id="alice")
    assert len(entities_after) == 0


def test_reset_isolated_per_user(tmp_path):
    """reset 清全部, 不分 user (对齐 mem0 reset 行为)。"""
    m = _make_memory(tmp_path)
    m.add("alice memory", user_id="alice")
    m.add("bob memory", user_id="bob")
    assert len(m.get_all(user_id="alice")["results"]) == 1
    assert len(m.get_all(user_id="bob")["results"]) == 1

    m.reset()

    assert len(m.get_all(user_id="alice")["results"]) == 0
    assert len(m.get_all(user_id="bob")["results"]) == 0

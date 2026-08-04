"""P3-2: 高级过滤操作符单元测试。"""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlmodel import create_engine

from septmuse.embedders.hash import HashEmbedder
from septmuse.storage.relational_stores.orm_store import ORMMemoryStore


@pytest.fixture()
def store(tmp_path: Path) -> ORMMemoryStore:
    engine = create_engine(f"sqlite:///{tmp_path / 'test_filters.db'}")
    return ORMMemoryStore(engine)


@pytest.fixture()
def embedder() -> HashEmbedder:
    return HashEmbedder(dim=32)


class TestExactMatch:
    """向后兼容: {"key": "value"} 精确匹配。"""

    def test_exact_match(self, store, embedder):
        emb = embedder.embed("hello")
        store.add("text1", emb, user_id="u1", agent_id="a1")
        store.add("text2", emb, user_id="u1", agent_id="a2")
        results = store.search(emb, user_id="u1", filters={"agent_id": "a1"})
        assert len(results) == 1
        assert results[0]["memory"] == "text1"


class TestOperatorFilters:
    def test_eq_operator(self, store, embedder):
        emb = embedder.embed("x")
        store.add("text1", emb, user_id="u1", agent_id="a1")
        store.add("text2", emb, user_id="u1", agent_id="a2")
        results = store.search(emb, user_id="u1", filters={"agent_id": {"eq": "a1"}})
        assert len(results) == 1

    def test_ne_operator(self, store, embedder):
        emb = embedder.embed("x")
        store.add("text1", emb, user_id="u1", agent_id="a1")
        store.add("text2", emb, user_id="u1", agent_id="a2")
        results = store.search(emb, user_id="u1", filters={"agent_id": {"ne": "a1"}})
        assert len(results) == 1
        assert results[0]["memory"] == "text2"

    def test_in_operator(self, store, embedder):
        emb = embedder.embed("x")
        store.add("t1", emb, user_id="u1", agent_id="a1")
        store.add("t2", emb, user_id="u1", agent_id="a2")
        store.add("t3", emb, user_id="u1", agent_id="a3")
        results = store.search(emb, user_id="u1", filters={"agent_id": {"in": ["a1", "a2"]}})
        assert len(results) == 2

    def test_nin_operator(self, store, embedder):
        emb = embedder.embed("x")
        store.add("t1", emb, user_id="u1", agent_id="a1")
        store.add("t2", emb, user_id="u1", agent_id="a2")
        store.add("t3", emb, user_id="u1", agent_id="a3")
        results = store.search(emb, user_id="u1", filters={"agent_id": {"nin": ["a1"]}})
        assert len(results) == 2

    def test_contains_operator(self, store, embedder):
        emb = embedder.embed("x")
        store.add("hello world", emb, user_id="u1")
        store.add("goodbye", emb, user_id="u1")
        results = store.get_all(user_id="u1", filters={"content": {"contains": "hello"}})
        assert len(results) == 1
        assert "hello" in results[0]["memory"]

    def test_icontains_operator(self, store, embedder):
        emb = embedder.embed("x")
        store.add("Hello World", emb, user_id="u1")
        store.add("Goodbye", emb, user_id="u1")
        results = store.get_all(user_id="u1", filters={"content": {"icontains": "hello"}})
        assert len(results) == 1
        assert "Hello" in results[0]["memory"]

    def test_unknown_operator_ignored(self, store, embedder):
        """未知操作符忽略, 不报错。"""
        emb = embedder.embed("x")
        store.add("t1", emb, user_id="u1", agent_id="a1")
        results = store.search(emb, user_id="u1", filters={"agent_id": {"unknown_op": "a1"}})
        # 未知操作符忽略 → 无过滤 → 返回全部
        assert len(results) == 1

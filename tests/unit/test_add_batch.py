"""P1-1/P1-2: 批量插入 + Hash 去重单元测试。"""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlmodel import create_engine

from septmuse.embedders.hash import HashEmbedder
from septmuse.storage.relational_stores.orm_store import ORMMemoryStore


@pytest.fixture()
def store(tmp_path: Path) -> ORMMemoryStore:
    engine = create_engine(f"sqlite:///{tmp_path / 'test_batch.db'}")
    return ORMMemoryStore(engine)


@pytest.fixture()
def embedder() -> HashEmbedder:
    return HashEmbedder(dim=64)


class TestAddBatch:
    def test_batch_insert_returns_ids(self, store, embedder):
        """add_batch 返回 memory_id 列表, 长度与 records 一致。"""
        emb = embedder.embed("hello")
        records = [("text1", emb), ("text2", emb), ("text3", emb)]
        ids = store.add_batch(records, user_id="u1")
        assert len(ids) == 3
        assert all(mid is not None for mid in ids)
        assert all(mid.startswith("mem-") for mid in ids)

    def test_batch_empty_records(self, store, embedder):
        """空 records 返回空列表。"""
        ids = store.add_batch([], user_id="u1")
        assert ids == []

    def test_batch_single_record(self, store, embedder):
        """单条记录也能用 add_batch。"""
        emb = embedder.embed("hello")
        ids = store.add_batch([("single", emb)], user_id="u1")
        assert len(ids) == 1
        assert store.get(ids[0]) is not None

    def test_batch_memories_searchable(self, store, embedder):
        """批量插入的记忆可被检索。"""
        records = [
            ("I love Python programming", embedder.embed("python")),
            ("Working at Google as engineer", embedder.embed("google")),
        ]
        store.add_batch(records, user_id="alice")
        results = store.search(embedder.embed("python"), user_id="alice", top_k=5)
        assert len(results) >= 1
        assert any("Python" in r["memory"] for r in results)

    def test_batch_history_recorded(self, store, embedder):
        """批量插入记录 history (ADD 事件)。"""
        emb = embedder.embed("hello")
        ids = store.add_batch([("text1", emb), ("text2", emb)], user_id="u1")
        for mid in ids:
            history = store.get_history(mid)
            assert len(history) >= 1
            assert history[0]["event"] == "ADD"


class TestHashDedup:
    def test_batch_dedup_same_content(self, store, embedder):
        """同批次相同内容的记录只存一条 (hash 去重)。"""
        emb = embedder.embed("duplicate")
        records = [("same text", emb), ("same text", emb), ("different", emb)]
        ids = store.add_batch(records, user_id="u1")
        # 第一个和第二个去重, 第二个 id 为 None
        assert ids[0] is not None
        assert ids[1] is None
        assert ids[2] is not None

    def test_batch_dedup_all_same(self, store, embedder):
        """全部相同的记录只存一条。"""
        emb = embedder.embed("same")
        records = [("same", emb), ("same", emb), ("same", emb)]
        ids = store.add_batch(records, user_id="u1")
        assert ids[0] is not None
        assert ids[1] is None
        assert ids[2] is None

    def test_batch_dedup_different_users(self, store, embedder):
        """不同用户的相同内容不去重 (hash 去重是批次内, 不跨 user)。"""
        emb = embedder.embed("same")
        ids1 = store.add_batch([("same", emb)], user_id="alice")
        ids2 = store.add_batch([("same", emb)], user_id="bob")
        assert ids1[0] is not None
        assert ids2[0] is not None
        assert ids1[0] != ids2[0]

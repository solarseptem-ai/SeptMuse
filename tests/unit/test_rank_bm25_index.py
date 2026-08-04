#  Copyright 2026 The sonhhxg0529 Authors. All Rights Reserved.
"""RankBM25Index 测试 (integration, 默认 skip)。"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture()
def rank_bm25_store(tmp_path):
    pytest.importorskip("rank_bm25")
    from septmuse.storage.keyword_stores.rank_bm25 import RankBM25Index

    store = RankBM25Index(db_path=tmp_path / "rank_bm25.db")
    yield store
    store.close()


def test_rank_bm25_add_and_retrieve(rank_bm25_store):
    rank_bm25_store.add_docs(
        {
            "m1": "the quick brown fox",
            "m2": "slow turtle",
        }
    )
    results = rank_bm25_store.retrieve("quick fox", limit=2)
    assert "m1" in results
    assert 0.0 <= results["m1"] <= 1.0


def test_rank_bm25_delete(rank_bm25_store):
    rank_bm25_store.add_docs({"m1": "alpha", "m2": "beta"})
    rank_bm25_store.delete_docs(["m1"])
    assert "m1" not in rank_bm25_store.retrieve("alpha", limit=5)


def test_rank_bm25_clear(rank_bm25_store):
    rank_bm25_store.add_docs({"m1": "alpha"})
    rank_bm25_store.clear()
    assert rank_bm25_store.retrieve("alpha") == {}

#  Copyright 2026 The sonhhxg0529 Authors. All Rights Reserved.
"""KeywordIndexBase ABC 契约 + SQLiteBM25Index 测试。"""

from __future__ import annotations

import pytest

from septmuse.storage.keyword.base import KeywordIndexBase
from septmuse.storage.keyword.sqlite_bm25 import SQLiteBM25Index


def test_abc_cannot_instantiate():
    with pytest.raises(TypeError):
        KeywordIndexBase()


@pytest.fixture()
def bm25_store(tmp_path):
    store = SQLiteBM25Index(db_path=tmp_path / "bm25.db")
    yield store
    store.close()


def test_add_and_retrieve(bm25_store):
    bm25_store.add_docs(
        {
            "m1": "the quick brown fox jumps over the lazy dog",
            "m2": "slow brown turtle crawls under the log",
            "m3": "fast orange fox leaps over the fence",
        }
    )
    results = bm25_store.retrieve("quick fox", limit=2)
    assert "m1" in results
    assert results["m1"] > 0.0


def test_add_docs_overwrite(bm25_store):
    bm25_store.add_docs({"m1": "alpha beta gamma"})
    bm25_store.add_docs({"m1": "delta epsilon zeta"})
    results = bm25_store.retrieve("delta", limit=5)
    assert "m1" in results
    results_alpha = bm25_store.retrieve("alpha", limit=5)
    assert "m1" not in results_alpha  # 已被覆盖


def test_retrieve_empty_query_returns_empty(bm25_store):
    bm25_store.add_docs({"m1": "hello world"})
    assert bm25_store.retrieve("") == {}


def test_retrieve_empty_index_returns_empty(bm25_store):
    assert bm25_store.retrieve("anything") == {}


def test_delete_docs(bm25_store):
    bm25_store.add_docs({"m1": "alpha", "m2": "beta"})
    bm25_store.delete_docs(["m1"])
    results = bm25_store.retrieve("alpha", limit=5)
    assert "m1" not in results
    results_beta = bm25_store.retrieve("beta", limit=5)
    assert "m2" in results_beta


def test_delete_nonexistent_silent(bm25_store):
    bm25_store.delete_docs(["nonexistent"])  # 不报错


def test_clear(bm25_store):
    bm25_store.add_docs({"m1": "alpha", "m2": "beta"})
    bm25_store.clear()
    assert bm25_store.retrieve("alpha") == {}
    assert bm25_store.retrieve("beta") == {}


def test_retrieve_score_normalized(bm25_store):
    bm25_store.add_docs({"m1": "alpha beta gamma", "m2": "alpha beta"})
    results = bm25_store.retrieve("alpha", limit=5)
    for score in results.values():
        assert 0.0 <= score <= 1.0


def test_chinese_tokenization(bm25_store):
    bm25_store.add_docs(
        {
            "m1": "用户喜欢快速的应用程序",
            "m2": "系统响应缓慢",
        }
    )
    results = bm25_store.retrieve("用户应用", limit=2)
    assert "m1" in results

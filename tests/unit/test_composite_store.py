#  Copyright 2026 The sonhhxg0529 Authors. All Rights Reserved.
"""SQLiteCompositeStore (SQLiteMemoryStore 重构后) 集成测试。"""

from __future__ import annotations

import pytest

from septmuse.storage.sqlite.store import SQLiteMemoryStore


@pytest.fixture()
def store(tmp_path):
    s = SQLiteMemoryStore(db_path=tmp_path / "test.db")
    yield s
    s.close()


def test_add_returns_memory_id(store):
    mid = store.add("hello world", [1.0, 0.0, 0.0], user_id="alice")
    assert mid.startswith("mem-")


def test_search_returns_matching_memory(store):
    # threshold=0.0 关闭阈值过滤, 验证 vector_store 委托返回 top_k 条 (含低分项)
    store.add("hello world", [1.0, 0.0, 0.0], user_id="alice")
    store.add("foo bar", [0.0, 1.0, 0.0], user_id="alice")
    results = store.search([1.0, 0.0, 0.0], user_id="alice", top_k=2, threshold=0.0)
    assert len(results) == 2
    assert results[0]["memory"] == "hello world"


def test_keyword_search_returns_matching(store):
    store.add("the quick brown fox", [1.0, 0.0], user_id="alice")
    store.add("slow turtle", [0.0, 1.0], user_id="alice")
    results = store.keyword_search("quick fox", user_id="alice", top_k=5)
    assert any(r["memory"] == "the quick brown fox" for r in results)


def test_hybrid_search_fuses_vector_and_keyword(store):
    store.add("the quick brown fox jumps", [1.0, 0.0, 0.0], user_id="alice")
    store.add("slow turtle crawls", [0.0, 1.0, 0.0], user_id="alice")
    results = store.hybrid_search("quick fox", [1.0, 0.0, 0.0], user_id="alice", top_k=2)
    assert len(results) <= 2
    if results:
        assert results[0]["memory"] == "the quick brown fox jumps"


def test_search_user_isolation(store):
    store.add("alice secret", [1.0, 0.0], user_id="alice")
    store.add("bob secret", [1.0, 0.0], user_id="bob")
    results = store.search([1.0, 0.0], user_id="alice", top_k=5)
    assert len(results) == 1


def test_delete_soft_delete(store):
    mid = store.add("to delete", [1.0, 0.0], user_id="alice")
    store.delete(mid)
    assert store.get(mid) is None
    history = store.get_history(mid)
    assert any(h.get("event") == "DELETE" for h in history)


def test_update_changes_content(store):
    mid = store.add("original", [1.0, 0.0], user_id="alice")
    store.update(mid, "updated", [0.5, 0.5], metadata={"k": "v"})
    mem = store.get(mid)
    assert mem["memory"] == "updated"


def test_add_writes_to_both_vector_store_and_keyword_index(store):
    mid = store.add("alpha beta", [1.0, 0.0], user_id="alice")
    # 验证向量层
    assert store._vector_store.get_vector(mid) is not None
    # 验证关键词层
    kw_results = store._keyword_index.retrieve("alpha", limit=5)
    assert mid in kw_results

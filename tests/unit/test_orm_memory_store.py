"""ORMMemoryStore 测试 — SQLModel ORM 跨方言 CRUD。"""

import pytest
from sqlalchemy import create_engine, inspect

from septmuse.storage.relational_stores.orm_store import ORMMemoryStore


@pytest.fixture
def store():
    """内存 SQLite store，测试自动清理。"""
    engine = create_engine("sqlite://", echo=False)
    s = ORMMemoryStore(engine)
    yield s
    s.close()


def test_orm_store_creates_tables(store):
    """ORMMemoryStore 初始化后自动建 3 张表。"""
    tables = set(inspect(store._engine).get_table_names())
    assert "memories" in tables
    assert "history" in tables
    assert "memory_access_logs" in tables


def test_add_returns_memory_id(store):
    """add 返回 'mem-' 前缀的 UUID。"""
    mid = store.add("hello world", [0.1, 0.2, 0.3], user_id="alice")
    assert mid.startswith("mem-")
    assert len(mid) > 10


def test_get_returns_memory(store):
    """get 返回记忆 dict。"""
    mid = store.add("hello world", [0.1, 0.2, 0.3], user_id="alice")
    mem = store.get(mid)
    assert mem is not None
    assert mem["id"] == mid
    assert mem["memory"] == "hello world"
    assert mem["state"] == "active"


def test_get_returns_none_if_not_found(store):
    """get 不存在返回 None。"""
    assert store.get("nonexistent-id") is None


def test_add_with_metadata(store):
    """add 带 metadata 存入。"""
    mid = store.add("test", [0.1], user_id="alice", metadata={"topic": "science"})
    mem = store.get(mid)
    assert mem["metadata"] == {"topic": "science"}


def test_add_with_valid_at(store):
    """add 带 valid_at 双时态。"""
    mid = store.add("earth is round", [0.1], user_id="alice", valid_at="2024-01-01T00:00:00Z")
    mem = store.get(mid)
    assert mem is not None


def test_search_returns_results(store):
    """search 返回相似度排序结果。"""
    store.add("apple", [1.0, 0.0], user_id="alice")
    store.add("banana", [0.0, 1.0], user_id="alice")
    results = store.search([1.0, 0.0], user_id="alice", top_k=5, threshold=0.0)
    assert len(results) >= 1
    assert results[0]["memory"] == "apple"
    assert results[0]["score"] >= 0.99  # 完全匹配


def test_search_filters_by_user(store):
    """search 按 user_id 隔离。"""
    store.add("alice's memory", [1.0, 0.0], user_id="alice")
    store.add("bob's memory", [1.0, 0.0], user_id="bob")
    results = store.search([1.0, 0.0], user_id="alice", top_k=5, threshold=0.0)
    assert len(results) == 1
    assert results[0]["memory"] == "alice's memory"


def test_search_filters_by_session(store):
    """search 按 session_id 过滤。"""
    store.add("session1", [1.0, 0.0], user_id="alice", session_id="s1")
    store.add("session2", [1.0, 0.0], user_id="alice", session_id="s2")
    results = store.search([1.0, 0.0], user_id="alice", session_id="s1", top_k=5, threshold=0.0)
    assert len(results) == 1
    assert results[0]["memory"] == "session1"


def test_search_threshold_filters(store):
    """search threshold 过滤低相似度。"""
    store.add("orthogonal", [0.0, 1.0], user_id="alice")
    results = store.search([1.0, 0.0], user_id="alice", top_k=5, threshold=0.9)
    assert len(results) == 0


def test_search_excludes_deleted(store):
    """search 排除已删除记忆。"""
    mid = store.add("to delete", [1.0, 0.0], user_id="alice")
    store.delete(mid)
    results = store.search([1.0, 0.0], user_id="alice", top_k=5, threshold=0.0)
    assert len(results) == 0


def test_get_all_returns_user_memories(store):
    """get_all 返回用户全部记忆。"""
    store.add("first", [0.1], user_id="alice")
    store.add("second", [0.2], user_id="alice")
    store.add("bob's", [0.3], user_id="bob")
    mems = store.get_all(user_id="alice")
    assert len(mems) == 2
    assert all(m["memory"] in ("first", "second") for m in mems)


def test_get_all_filters_by_session(store):
    """get_all 按 session_id 过滤。"""
    store.add("s1", [0.1], user_id="alice", session_id="s1")
    store.add("s2", [0.1], user_id="alice", session_id="s2")
    mems = store.get_all(user_id="alice", session_id="s1")
    assert len(mems) == 1
    assert mems[0]["memory"] == "s1"


def test_update_changes_content(store):
    """update 修改 content + embedding + metadata。"""
    mid = store.add("old", [0.1, 0.0], user_id="alice")
    ok = store.update(mid, "new content", [0.0, 1.0], metadata={"updated": True})
    assert ok is True
    mem = store.get(mid)
    assert mem["memory"] == "new content"
    assert mem["metadata"] == {"updated": True}


def test_update_returns_false_if_not_found(store):
    """update 不存在返回 False。"""
    assert store.update("nonexistent", "x", [0.1]) is False


def test_delete_soft_deletes(store):
    """delete 软删除 (is_deleted=1 + state='deleted')。"""
    mid = store.add("to delete", [0.1], user_id="alice")
    store.delete(mid)
    assert store.get(mid) is None
    # get_all 也排除
    assert store.get_all(user_id="alice") == []


def test_delete_records_history(store):
    """delete 记录 DELETE 事件到 history。"""
    mid = store.add("to delete", [0.1], user_id="alice")
    store.delete(mid)
    history = store.get_history(mid)
    events = [h["event"] for h in history]
    assert "ADD" in events
    assert "DELETE" in events


def test_get_history_returns_chronological(store):
    """get_history 返回时间顺序。"""
    mid = store.add("original", [0.1], user_id="alice")
    store.update(mid, "updated", [0.2])
    history = store.get_history(mid)
    assert len(history) >= 2
    assert history[0]["event"] == "ADD"
    assert history[1]["event"] == "UPDATE"


def test_record_access_log(store):
    """_record_access_log 记录访问日志。"""
    mid = store.add("test", [0.1], user_id="alice")
    log_id = store._record_access_log(mid, app_id="app1", access_type="read")
    assert log_id is not None


def test_get_access_logs(store):
    """get_access_logs 返回访问日志。"""
    mid = store.add("test", [0.1], user_id="alice")
    store._record_access_log(mid, app_id="app1", access_type="read")
    store._record_access_log(mid, app_id="app2", access_type="write")
    logs = store.get_access_logs(mid)
    assert len(logs) == 2
    assert all("access_type" in log for log in logs)


def test_get_access_logs_empty(store):
    """get_access_logs 无日志返回空列表。"""
    mid = store.add("test", [0.1], user_id="alice")
    logs = store.get_access_logs(mid)
    assert logs == []


def test_invalidate_sets_invalid_at(store):
    """invalidate 标记事实不再为真。"""
    mid = store.add("earth is flat", [0.1], user_id="alice", valid_at="2020-01-01T00:00:00Z")
    result = store.invalidate(mid, invalid_at="2024-01-01T00:00:00Z")
    assert result["event"] == "INVALIDATE"
    assert result["invalid_at"] == "2024-01-01T00:00:00Z"


def test_invalidate_not_found(store):
    """invalidate 不存在返回 NOT_FOUND。"""
    result = store.invalidate("nonexistent")
    assert result["event"] == "NOT_FOUND"


def test_get_temporal_valid(store):
    """get_temporal_valid 查询某时刻为真的记忆。"""
    store.add("valid fact", [0.1], user_id="alice", valid_at="2020-01-01T00:00:00Z")
    store.add("future fact", [0.1], user_id="alice", valid_at="2025-01-01T00:00:00Z")
    # 2023 年时, valid fact 为真, future fact 还没开始
    results = store.get_temporal_valid("2023-06-01T00:00:00Z", user_id="alice")
    memories = [r["memory"] for r in results]
    assert "valid fact" in memories
    assert "future fact" not in memories


def test_get_temporal_valid_excludes_invalidated(store):
    """get_temporal_valid 排除已 invalidate 的记忆。"""
    mid = store.add("old fact", [0.1], user_id="alice", valid_at="2020-01-01T00:00:00Z")
    store.invalidate(mid, invalid_at="2023-01-01T00:00:00Z")
    # 2024 年时, old fact 已失效
    results = store.get_temporal_valid("2024-06-01T00:00:00Z", user_id="alice")
    memories = [r["memory"] for r in results]
    assert "old fact" not in memories


def test_keyword_search_without_index_returns_empty(store):
    """无 keyword_index 时 keyword_search 返回空。"""
    store.add("hello world", [0.1], user_id="alice")
    results = store.keyword_search("hello", user_id="alice")
    assert results == []


def test_list_agents(store):
    """list_agents 返回用户的 agent_id 去重列表。"""
    store.add("m1", [0.1], user_id="alice", agent_id="agent1")
    store.add("m2", [0.1], user_id="alice", agent_id="agent2")
    store.add("m3", [0.1], user_id="alice", agent_id="agent1")  # 重复
    store.add("m4", [0.1], user_id="alice")  # None, 排除
    agents = store.list_agents("alice")
    assert set(agents) == {"agent1", "agent2"}


def test_list_users(store):
    """list_users 返回 agent 的 user_id 去重列表。"""
    store.add("m1", [0.1], user_id="alice", agent_id="agent1")
    store.add("m2", [0.1], user_id="bob", agent_id="agent1")
    users = store.list_users("agent1")
    assert set(users) == {"alice", "bob"}


def test_get_shared_memories(store):
    """get_shared_memories 返回跨 agent 共享记忆。"""
    store.add("shared1", [0.1], user_id="alice", agent_id="agent1")
    store.add("shared2", [0.1], user_id="alice", agent_id="agent2")
    results = store.get_shared_memories("alice", limit=100)
    assert len(results) == 2

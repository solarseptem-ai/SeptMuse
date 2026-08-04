"""AsyncORMMemoryStore 测试 — SQLModel async ORM CRUD。"""

import pytest
from sqlalchemy import inspect
from sqlalchemy.ext.asyncio import create_async_engine

from septmuse.storage.relational_stores.async_orm_store import AsyncORMMemoryStore


@pytest.fixture
async def store(tmp_path):
    """文件 SQLite async store，测试自动清理。"""
    db_path = tmp_path / "test_async_orm.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}", echo=False)
    s = AsyncORMMemoryStore(engine)
    yield s
    await s.close()


async def test_async_store_creates_tables(store):
    """AsyncORMMemoryStore 初始化后自动建表。"""
    async with store._engine.connect() as conn:
        tables = await conn.run_sync(lambda sync_conn: set(inspect(sync_conn).get_table_names()))
    assert "memories" in tables
    assert "history" in tables
    assert "memory_access_logs" in tables


async def test_add_returns_memory_id(store):
    """add 返回 'mem-' 前缀的 UUID。"""
    mid = await store.add("hello world", [0.1, 0.2, 0.3], user_id="alice")
    assert mid.startswith("mem-")


async def test_get_returns_memory(store):
    """get 返回记忆 dict。"""
    mid = await store.add("hello world", [0.1, 0.2, 0.3], user_id="alice")
    mem = await store.get(mid)
    assert mem is not None
    assert mem["id"] == mid
    assert mem["memory"] == "hello world"
    assert mem["state"] == "active"


async def test_get_returns_none_if_not_found(store):
    """get 不存在返回 None。"""
    assert await store.get("nonexistent-id") is None


async def test_add_with_metadata(store):
    """add 带 metadata 存入。"""
    mid = await store.add("test", [0.1], user_id="alice", metadata={"topic": "science"})
    mem = await store.get(mid)
    assert mem["metadata"] == {"topic": "science"}


async def test_search_returns_results(store):
    """search 返回相似度排序结果。"""
    await store.add("apple", [1.0, 0.0], user_id="alice")
    await store.add("banana", [0.0, 1.0], user_id="alice")
    results = await store.search([1.0, 0.0], user_id="alice", top_k=5, threshold=0.0)
    assert len(results) >= 1
    assert results[0]["memory"] == "apple"


async def test_search_filters_by_user(store):
    """search 按 user_id 隔离。"""
    await store.add("alice's memory", [1.0, 0.0], user_id="alice")
    await store.add("bob's memory", [1.0, 0.0], user_id="bob")
    results = await store.search([1.0, 0.0], user_id="alice", top_k=5, threshold=0.0)
    assert len(results) == 1
    assert results[0]["memory"] == "alice's memory"


async def test_search_filters_by_session(store):
    """search 按 session_id 过滤。"""
    await store.add("session1", [1.0, 0.0], user_id="alice", session_id="s1")
    await store.add("session2", [1.0, 0.0], user_id="alice", session_id="s2")
    results = await store.search([1.0, 0.0], user_id="alice", session_id="s1", top_k=5, threshold=0.0)
    assert len(results) == 1
    assert results[0]["memory"] == "session1"


async def test_get_all_returns_user_memories(store):
    """get_all 返回用户全部记忆。"""
    await store.add("first", [0.1], user_id="alice")
    await store.add("second", [0.2], user_id="alice")
    await store.add("bob's", [0.3], user_id="bob")
    mems = await store.get_all(user_id="alice")
    assert len(mems) == 2


async def test_update_changes_content(store):
    """update 修改 content + embedding + metadata。"""
    mid = await store.add("old", [0.1, 0.0], user_id="alice")
    ok = await store.update(mid, "new content", [0.0, 1.0], metadata={"updated": True})
    assert ok is True
    mem = await store.get(mid)
    assert mem["memory"] == "new content"
    assert mem["metadata"] == {"updated": True}


async def test_update_returns_false_if_not_found(store):
    """update 不存在返回 False。"""
    assert await store.update("nonexistent", "x", [0.1]) is False


async def test_delete_soft_deletes(store):
    """delete 软删除。"""
    mid = await store.add("to delete", [0.1], user_id="alice")
    await store.delete(mid)
    assert await store.get(mid) is None
    assert await store.get_all(user_id="alice") == []


async def test_delete_records_history(store):
    """delete 记录 DELETE 事件。"""
    mid = await store.add("to delete", [0.1], user_id="alice")
    await store.delete(mid)
    history = await store.get_history(mid)
    events = [h["event"] for h in history]
    assert "ADD" in events
    assert "DELETE" in events


async def test_get_history_chronological(store):
    """get_history 返回时间顺序。"""
    mid = await store.add("original", [0.1], user_id="alice")
    await store.update(mid, "updated", [0.2])
    history = await store.get_history(mid)
    assert len(history) >= 2
    assert history[0]["event"] == "ADD"
    assert history[1]["event"] == "UPDATE"


async def test_record_access_log(store):
    """_record_access_log 记录访问日志。"""
    mid = await store.add("test", [0.1], user_id="alice")
    log_id = await store._record_access_log(mid, app_id="app1", access_type="read")
    assert log_id is not None


async def test_get_access_logs(store):
    """get_access_logs 返回访问日志。"""
    mid = await store.add("test", [0.1], user_id="alice")
    await store._record_access_log(mid, app_id="app1", access_type="read")
    await store._record_access_log(mid, app_id="app2", access_type="write")
    logs = await store.get_access_logs(mid)
    assert len(logs) == 2


async def test_invalidate_sets_invalid_at(store):
    """invalidate 标记事实不再为真。"""
    mid = await store.add("earth is flat", [0.1], user_id="alice", valid_at="2020-01-01T00:00:00Z")
    result = await store.invalidate(mid, invalid_at="2024-01-01T00:00:00Z")
    assert result["event"] == "INVALIDATE"


async def test_invalidate_not_found(store):
    """invalidate 不存在返回 NOT_FOUND。"""
    result = await store.invalidate("nonexistent")
    assert result["event"] == "NOT_FOUND"


async def test_get_temporal_valid(store):
    """get_temporal_valid 查询某时刻为真的记忆。"""
    await store.add("valid fact", [0.1], user_id="alice", valid_at="2020-01-01T00:00:00Z")
    await store.add("future fact", [0.1], user_id="alice", valid_at="2025-01-01T00:00:00Z")
    results = await store.get_temporal_valid("2023-06-01T00:00:00Z", user_id="alice")
    memories = [r["memory"] for r in results]
    assert "valid fact" in memories
    assert "future fact" not in memories


async def test_keyword_search_without_index_returns_empty(store):
    """无 keyword_index 时 keyword_search 返回空。"""
    await store.add("hello world", [0.1], user_id="alice")
    results = await store.keyword_search("hello", user_id="alice")
    assert results == []

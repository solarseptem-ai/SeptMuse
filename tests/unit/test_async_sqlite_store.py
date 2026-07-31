"""AsyncSQLiteMemoryStore 测试。"""
import pytest

from septmuse.storage.async_sqlite.store import AsyncSQLiteMemoryStore


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "async_test.db")


async def test_add_and_search(db_path):
    """添加记忆后能检索到。"""
    store = AsyncSQLiteMemoryStore(db_path=db_path)
    try:
        mid = await store.add("hello world", [1.0, 0.0, 0.0], user_id="alice")
        assert mid.startswith("mem-")

        results = await store.search([1.0, 0.0, 0.0], user_id="alice", top_k=5)
        assert len(results) == 1
        assert results[0]["memory"] == "hello world"
        assert results[0]["score"] > 0.5
    finally:
        await store.close()


async def test_get_and_delete(db_path):
    """获取和软删除。"""
    store = AsyncSQLiteMemoryStore(db_path=db_path)
    try:
        mid = await store.add("test memory", [0.5, 0.5], user_id="bob")
        mem = await store.get(mid)
        assert mem is not None
        assert mem["memory"] == "test memory"

        await store.delete(mid)
        mem_after = await store.get(mid)
        assert mem_after is None
    finally:
        await store.close()


async def test_get_all(db_path):
    """列出全部记忆。"""
    store = AsyncSQLiteMemoryStore(db_path=db_path)
    try:
        await store.add("first", [1.0, 0.0], user_id="alice")
        await store.add("second", [0.0, 1.0], user_id="alice")
        all_mems = await store.get_all(user_id="alice")
        assert len(all_mems) == 2
    finally:
        await store.close()


async def test_update(db_path):
    """更新记忆。"""
    store = AsyncSQLiteMemoryStore(db_path=db_path)
    try:
        mid = await store.add("original", [1.0, 0.0], user_id="alice")
        success = await store.update(mid, "updated", [0.0, 1.0])
        assert success is True
        mem = await store.get(mid)
        assert mem["memory"] == "updated"
    finally:
        await store.close()


async def test_get_history(db_path):
    """变更历史。"""
    store = AsyncSQLiteMemoryStore(db_path=db_path)
    try:
        mid = await store.add("original", [1.0, 0.0], user_id="alice")
        await store.update(mid, "updated", [0.0, 1.0])
        await store.delete(mid)
        history = await store.get_history(mid)
        assert len(history) == 3
        events = [h["event"] for h in history]
        assert "ADD" in events
        assert "UPDATE" in events
        assert "DELETE" in events
    finally:
        await store.close()


async def test_user_isolation(db_path):
    """用户隔离 — alice 看不到 bob 的记忆。"""
    store = AsyncSQLiteMemoryStore(db_path=db_path)
    try:
        await store.add("alice memory", [1.0, 0.0], user_id="alice")
        await store.add("bob memory", [0.0, 1.0], user_id="bob")
        alice_results = await store.search([1.0, 0.0], user_id="alice")
        assert len(alice_results) == 1
        assert alice_results[0]["memory"] == "alice memory"
    finally:
        await store.close()

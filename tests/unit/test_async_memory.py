"""AsyncMemory facade 测试。"""
from septmuse.embedders.hash import HashEmbedder
from septmuse.memory.async_main import AsyncMemory
from septmuse.storage.async_sqlite.store import AsyncSQLiteMemoryStore


def _make_memory(tmp_path):
    """用临时 DB 创建 AsyncMemory。"""
    store = AsyncSQLiteMemoryStore(db_path=str(tmp_path / "async_mem.db"))
    return AsyncMemory(embedder=HashEmbedder(), store=store)


async def test_add_and_search(tmp_path):
    """添加记忆后能检索到。"""
    mem = _make_memory(tmp_path)
    try:
        result = await mem.add("hello world", user_id="alice")
        assert "results" in result
        assert len(result["results"]) == 1
        assert result["results"][0]["memory"] == "hello world"

        results = await mem.search("hello", user_id="alice", top_k=5)
        assert len(results) >= 1
        assert results[0]["memory"] == "hello world"
    finally:
        await mem.close()


async def test_get_and_get_all(tmp_path):
    """获取单条和列出全部。"""
    mem = _make_memory(tmp_path)
    try:
        result = await mem.add("test memory", user_id="bob")
        mid = result["results"][0]["id"]

        got = await mem.get(mid)
        assert got is not None
        assert got["memory"] == "test memory"

        all_mems = await mem.get_all(user_id="bob")
        assert len(all_mems) == 1
    finally:
        await mem.close()


async def test_delete(tmp_path):
    """删除记忆。"""
    mem = _make_memory(tmp_path)
    try:
        result = await mem.add("to delete", user_id="alice")
        mid = result["results"][0]["id"]
        await mem.delete(mid)
        got = await mem.get(mid)
        assert got is None
    finally:
        await mem.close()


async def test_update(tmp_path):
    """更新记忆。"""
    mem = _make_memory(tmp_path)
    try:
        result = await mem.add("original", user_id="alice")
        mid = result["results"][0]["id"]
        success = await mem.update(mid, "updated")
        assert success is True
        got = await mem.get(mid)
        assert got["memory"] == "updated"
    finally:
        await mem.close()


async def test_get_history(tmp_path):
    """变更历史。"""
    mem = _make_memory(tmp_path)
    try:
        result = await mem.add("original", user_id="alice")
        mid = result["results"][0]["id"]
        await mem.update(mid, "updated")
        await mem.delete(mid)
        history = await mem.get_history(mid)
        assert len(history) == 3
    finally:
        await mem.close()

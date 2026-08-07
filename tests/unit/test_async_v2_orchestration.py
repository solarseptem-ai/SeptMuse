"""AsyncMemory V2 编排方法 (remember/recall/forget/improve) + update/delete 增强测试。"""
from sqlalchemy.ext.asyncio import create_async_engine

from septmuse.configs.base import MemoryConfig
from septmuse.embedders.hash import HashEmbedder
from septmuse.memory.async_main import AsyncMemory
from septmuse.storage.relational_stores.async_orm_store import AsyncORMMemoryStore


def _make_memory(tmp_path):
    """用临时 DB 创建 AsyncMemory (config 与 store 共享同一 db_path)."""
    db_path = str(tmp_path / "async_v2.db")
    config = MemoryConfig(db_path=db_path)
    async_engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    store = AsyncORMMemoryStore(async_engine)
    return AsyncMemory(config=config, embedder=HashEmbedder(), store=store)


async def test_remember_returns_captured(tmp_path):
    """remember 编排: 捕获 → add → episodic raw_log, 返回 captured=True."""
    mem = _make_memory(tmp_path)
    try:
        await mem.add("hello world", user_id="alice")
        result = await mem.remember("second message", user_id="alice")
        assert result["captured"] is True
    finally:
        await mem.close()


async def test_recall_returns_memories(tmp_path):
    """recall 编排: 检索 → 遗忘加权 → token 预算, 返回 memories 列表."""
    mem = _make_memory(tmp_path)
    try:
        await mem.add("hello world", user_id="alice")
        result = await mem.recall("hello", user_id="alice")
        assert "memories" in result
        assert isinstance(result["memories"], list)
    finally:
        await mem.close()


async def test_forget_returns_event(tmp_path):
    """forget 编排: invalidate → delete → 图清理, 返回 event=FORGET."""
    mem = _make_memory(tmp_path)
    try:
        add_result = await mem.add("to forget", user_id="alice")
        mid = add_result["results"][0]["id"]
        result = await mem.forget(mid, user_id="alice")
        assert result["event"] == "FORGET"
    finally:
        await mem.close()


async def test_improve_returns_keys(tmp_path):
    """improve 编排: dream + reflect + conflict + coverage, 返回四键."""
    mem = _make_memory(tmp_path)
    try:
        await mem.add("some memory", user_id="alice")
        result = await mem.improve(user_id="alice")
        assert "dream" in result
        assert "rules" in result
        assert "conflicts" in result
        assert "coverage" in result
    finally:
        await mem.close()


async def test_update_with_user_id(tmp_path):
    """update 增强: user_id 提供时委托 sync (含实体重链接), 不崩."""
    mem = _make_memory(tmp_path)
    try:
        add_result = await mem.add("hello", user_id="alice")
        mid = add_result["results"][0]["id"]
        result = await mem.update(mid, "world", user_id="alice")
        assert result is not None
    finally:
        await mem.close()


async def test_delete_with_user_id(tmp_path):
    """delete 增强: user_id 提供时委托 sync forget (invalidate + 图清理)."""
    mem = _make_memory(tmp_path)
    try:
        add_result = await mem.add("to delete", user_id="alice")
        mid = add_result["results"][0]["id"]
        result = await mem.delete(mid, user_id="alice")
        assert result is not None
        assert result["event"] == "FORGET"
    finally:
        await mem.close()

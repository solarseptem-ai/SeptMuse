"""AsyncMemory facade 测试。"""
from sqlalchemy.ext.asyncio import create_async_engine

from septmuse.configs.defaults import MemoryConfig
from septmuse.embedders.hash import HashEmbedder
from septmuse.memory.async_main import AsyncMemory
from septmuse.storage.relational_stores.async_orm_store import AsyncORMMemoryStore


def _make_memory(tmp_path):
    """用临时 DB 创建 AsyncMemory（config 与 store 共享同一 db_path）。"""
    db_path = str(tmp_path / "async_mem.db")
    config = MemoryConfig(db_path=db_path)
    async_engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    store = AsyncORMMemoryStore(async_engine)
    return AsyncMemory(config=config, embedder=HashEmbedder(), store=store)


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


# ======================================================================
# 高级方法（to_thread 桥接 sync ExperimentalMemory）
# ======================================================================


async def test_search_at(tmp_path):
    """时态查询：查询某时刻为真的记忆。"""
    mem = _make_memory(tmp_path)
    try:
        await mem.add("old fact", user_id="alice", valid_at="2020-01-01T00:00:00")
        results = await mem.search_at("2025-01-01T00:00:00", "old", user_id="alice", top_k=5)
        assert len(results) >= 1
        assert results[0]["memory"] == "old fact"
    finally:
        await mem.close()


async def test_search_interval(tmp_path):
    """时间区间查询：返回 [start, end) 内为真的记忆。"""
    mem = _make_memory(tmp_path)
    try:
        await mem.add("interval fact", user_id="alice", valid_at="2020-06-01T00:00:00")
        results = await mem.search_interval(
            "2020-01-01T00:00:00", "2025-01-01T00:00:00", "interval", user_id="alice", top_k=5
        )
        assert len(results) >= 1
        assert results[0]["memory"] == "interval fact"
    finally:
        await mem.close()


async def test_get_access_logs(tmp_path):
    """访问日志查询。"""
    mem = _make_memory(tmp_path)
    try:
        result = await mem.add("logged memory", user_id="alice")
        mid = result["results"][0]["id"]
        logs = await mem.get_access_logs(mid)
        # 默认 store 可能无日志记录行为，至少不报错
        assert isinstance(logs, list)
    finally:
        await mem.close()


async def test_get_active_rules(tmp_path):
    """规则系统：获取应注入的规则。"""
    mem = _make_memory(tmp_path)
    try:
        rules = await mem.get_active_rules(user_id="alice")
        assert isinstance(rules, list)
    finally:
        await mem.close()


async def test_rules_to_prompt(tmp_path):
    """规则系统：编译规则为 prompt。"""
    mem = _make_memory(tmp_path)
    try:
        prompt = await mem.rules_to_prompt(user_id="alice")
        assert isinstance(prompt, str)
    finally:
        await mem.close()


async def test_resolve_conflicts_empty(tmp_path):
    """冲突解决：空用户不报错。"""
    mem = _make_memory(tmp_path)
    try:
        result = await mem.resolve_conflicts(user_id="alice")
        assert isinstance(result, dict)
    finally:
        await mem.close()


async def test_deduplicate_entities_empty(tmp_path):
    """实体去重：空用户不报错。"""
    mem = _make_memory(tmp_path)
    try:
        result = await mem.deduplicate_entities(user_id="alice")
        assert isinstance(result, dict)
    finally:
        await mem.close()


async def test_invalidate(tmp_path):
    """标记事实不再为真。"""
    mem = _make_memory(tmp_path)
    try:
        result = await mem.add("temporal fact", user_id="alice")
        mid = result["results"][0]["id"]
        inv = await mem.invalidate(mid)
        assert inv["event"] == "INVALIDATE"
        assert inv["id"] == mid
    finally:
        await mem.close()

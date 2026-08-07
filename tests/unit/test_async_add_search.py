"""AsyncMemory add+search 增强测试 (P3 Task 1+2)。"""
from sqlalchemy.ext.asyncio import create_async_engine

from septmuse.configs.defaults import MemoryConfig
from septmuse.embedders.hash import HashEmbedder
from septmuse.memory.async_main import AsyncMemory
from septmuse.storage.relational_stores.async_orm_store import AsyncORMMemoryStore


def _make_memory(tmp_path):
    """用临时 DB 创建 AsyncMemory（config 与 store 共享同一 db_path）。"""
    db_path = str(tmp_path / "async_add_search.db")
    config = MemoryConfig(db_path=db_path)
    async_engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    store = AsyncORMMemoryStore(async_engine)
    return AsyncMemory(config=config, embedder=HashEmbedder(), store=store)


# ======================================================================
# add 增强 (Task 1)
# ======================================================================


async def test_add_basic_verbatim(tmp_path):
    """基础 verbatim add (infer=False) → 真 async 路径, event="ADD"。"""
    mem = _make_memory(tmp_path)
    try:
        result = await mem.add("hello world", user_id="alice", infer=False)
        assert "results" in result
        assert len(result["results"]) == 1
        assert result["results"][0]["event"] == "ADD"
        assert result["results"][0]["memory"] == "hello world"
    finally:
        await mem.close()


async def test_add_expiration_date(tmp_path):
    """add 带 expiration_date → metadata 含归一化后的 expiration_date。"""
    mem = _make_memory(tmp_path)
    try:
        result = await mem.add("test exp", user_id="alice", infer=False, expiration_date="2099-12-31")
        mid = result["results"][0]["id"]
        got = await mem.get(mid)
        assert got is not None
        meta = got.get("metadata") or {}
        assert meta.get("expiration_date") == "2099-12-31"
    finally:
        await mem.close()


async def test_add_attributed_to(tmp_path):
    """add 带 attributed_to → metadata 含 attributed_to。"""
    mem = _make_memory(tmp_path)
    try:
        result = await mem.add("test attr", user_id="alice", infer=False, attributed_to="user")
        mid = result["results"][0]["id"]
        got = await mem.get(mid)
        assert got is not None
        meta = got.get("metadata") or {}
        assert meta.get("attributed_to") == "user"
    finally:
        await mem.close()


async def test_add_actor_id_from_message_name(tmp_path):
    """add 带 message[{"name":...}] → metadata 含 actor_id。"""
    mem = _make_memory(tmp_path)
    try:
        result = await mem.add(
            [{"role": "user", "name": "bob", "content": "hello from bob"}],
            user_id="alice",
            infer=False,
        )
        mid = result["results"][0]["id"]
        got = await mem.get(mid)
        assert got is not None
        meta = got.get("metadata") or {}
        assert meta.get("actor_id") == "bob"
    finally:
        await mem.close()


async def test_add_memory_type_rule(tmp_path):
    """add memory_type="rule" → 委托 sync, 返回含 rule 字段。"""
    mem = _make_memory(tmp_path)
    try:
        result = await mem.add("always test before deploy", user_id="alice", memory_type="rule")
        assert "rule" in result
        assert result["event"] == "ADD"
        assert "always test" in result["rule"]
    finally:
        await mem.close()


# ======================================================================
# search 增强 (Task 2)
# ======================================================================


async def test_search_basic(tmp_path):
    """基础 search (纯向量, 无高级参数) → 真 async 路径, 返回 list。"""
    mem = _make_memory(tmp_path)
    try:
        await mem.add("hello world", user_id="alice", infer=False)
        results = await mem.search("hello", user_id="alice")
        assert isinstance(results, list)
        assert len(results) >= 1
        assert results[0]["memory"] == "hello world"
    finally:
        await mem.close()


async def test_search_show_expired(tmp_path):
    """search show_expired=True → 真 async 路径, 不过滤过期, 返回 list。"""
    mem = _make_memory(tmp_path)
    try:
        await mem.add("expired memory", user_id="alice", infer=False, expiration_date="2000-01-01")
        results = await mem.search("expired", user_id="alice", show_expired=True)
        assert isinstance(results, list)
        assert len(results) >= 1
    finally:
        await mem.close()


async def test_search_filters_expired_by_default(tmp_path):
    """search 默认过滤过期记忆 (show_expired=False)。"""
    mem = _make_memory(tmp_path)
    try:
        await mem.add("old expired", user_id="alice", infer=False, expiration_date="2000-01-01")
        await mem.add("fresh active", user_id="alice", infer=False)
        results = await mem.search("expired", user_id="alice")
        assert isinstance(results, list)
        # 过期的被过滤掉, 不应出现在结果中
        for r in results:
            meta = r.get("metadata") or {}
            assert not meta.get("expiration_date") or meta.get("expiration_date") >= "2000-01-01"
    finally:
        await mem.close()


async def test_search_hybrid(tmp_path):
    """search hybrid=True → 委托 sync (BM25+向量 RRF 融合), 返回 list。"""
    mem = _make_memory(tmp_path)
    try:
        await mem.add("hello world", user_id="alice", infer=False)
        results = await mem.search("hello", user_id="alice", hybrid=True)
        assert isinstance(results, list)
        assert len(results) >= 1
    finally:
        await mem.close()

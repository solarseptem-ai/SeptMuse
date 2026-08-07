"""Memory 编排方法测试 — remember/recall/forget/improve 委托 self.add/search/delete."""
from __future__ import annotations

import os
import warnings

import pytest


@pytest.fixture
def memory(tmp_path):
    os.environ["SEPTMUSE_EMBEDDER"] = "hash"
    os.environ["SEPTMUSE_TOKENIZER"] = "space"
    from septmuse.configs import MemoryConfig
    from septmuse.memory.main import Memory
    cfg = MemoryConfig(db_path=str(tmp_path / "test.db"))
    return Memory(config=cfg)


@pytest.fixture
def v2_memory_deprecated(memory):
    from septmuse.memory.memory_v2 import V2Memory
    return V2Memory(memory=memory)


def test_remember_delegates_add(memory):
    """remember 委托 add + episodic raw_log."""
    result = memory.remember("I like Python", user_id="alice")
    assert result["captured"] is True
    assert result["raw_id"]  # episodic raw_log 存了
    assert len(result["memory_ids"]) >= 1  # add 存了


def test_remember_empty_text(memory):
    result = memory.remember("", user_id="alice")
    assert result["captured"] is False


def test_remember_dedup_blocks_duplicate(memory):
    """同文本第二次 remember 被去重拒绝."""
    r1 = memory.remember("hello world", user_id="alice")
    assert r1["captured"] is True
    r2 = memory.remember("hello world", user_id="alice")
    assert r2["captured"] is False


def test_recall_returns_real_id(memory):
    """recall 返回真实 memory_id (不是 text[:50])."""
    memory.remember("I like Python programming", user_id="alice")
    result = memory.recall("what do I like", user_id="alice")
    for m in result["memories"]:
        assert m["id"].startswith("mem-")  # 真实 id
    assert "injected_prompt" in result


def test_recall_id_survives_token_budget(memory):
    """token 预算裁剪后 id 仍保留."""
    for i in range(10):
        memory.remember(f"fact number {i} about programming", user_id="alice")
    result = memory.recall("programming", user_id="alice", top_k=3)
    assert len(result["memories"]) <= 3
    for m in result["memories"]:
        assert m["id"].startswith("mem-")


def test_forget_delegates_delete(memory):
    """forget 委托 delete + invalidate."""
    add_result = memory.remember("temp fact here", user_id="alice")
    mid = add_result["memory_ids"][0]
    result = memory.forget(mid, user_id="alice")
    assert result["event"] == "FORGET"
    assert memory.get(mid) is None  # delete 后 get 返回 None


def test_improve_runs(memory):
    """improve 不报错 (dream + reflect + conflict + coverage)."""
    memory.remember("some fact", user_id="alice")
    result = memory.improve(user_id="alice", limit=5)
    assert "dream" in result
    assert "coverage" in result


def test_v2memory_deprecated_warning(v2_memory_deprecated):
    """V2Memory 薄层发 DeprecationWarning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # V2Memory(memory=...) 构造时应发警告
        from septmuse.memory.memory_v2 import V2Memory
        V2Memory(memory=v2_memory_deprecated.mem)
        assert any(issubclass(wi.category, DeprecationWarning) for wi in w)


def test_v2memory_delegates_remember(v2_memory_deprecated):
    """V2.remember 委托 Memory.remember."""
    result = v2_memory_deprecated.remember("test message", user_id="bob")
    assert result["captured"] is True


def test_v2memory_delegates_recall(v2_memory_deprecated):
    """V2.recall 委托 Memory.recall."""
    v2_memory_deprecated.remember("test message", user_id="bob")
    result = v2_memory_deprecated.recall("test", user_id="bob")
    assert "memories" in result

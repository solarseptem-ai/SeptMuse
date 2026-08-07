"""add verbatim 路径语义去重测试."""
from __future__ import annotations

import os

import pytest


@pytest.fixture
def memory(tmp_path):
    os.environ["SEPTMUSE_EMBEDDER"] = "hash"
    os.environ["SEPTMUSE_TOKENIZER"] = "space"
    from septmuse.configs import MemoryConfig
    from septmuse.memory.main import Memory
    return Memory(config=MemoryConfig(db_path=str(tmp_path / "test.db")))


def test_add_verbatim_no_cross_batch_dedup(memory):
    """跨批次精确重复不跳过 (语义去重已移除, HashEmbedder 下误杀)."""
    r1 = memory.add("I like Python", user_id="alice", infer=False)
    assert len(r1["results"]) == 1
    r2 = memory.add("I like Python", user_id="alice", infer=False)
    # 无跨批次去重, 两次都 ADD (批次内 MD5 去重仅对同一 add_batch 调用生效)
    assert len(r2["results"]) == 1
    assert r2["results"][0]["event"] == "ADD"


def test_add_verbatim_different_not_deduped(memory):
    """不同内容不去重."""
    memory.add("I like Python", user_id="alice", infer=False)
    result = memory.add("I live in Tokyo", user_id="alice", infer=False)
    assert len(result["results"]) == 1
    assert result["results"][0]["event"] == "ADD"


def test_add_verbatim_different_users_not_deduped(memory):
    """不同用户同文本不去重."""
    memory.add("I like Python", user_id="alice", infer=False)
    result = memory.add("I like Python", user_id="bob", infer=False)
    assert len(result["results"]) == 1

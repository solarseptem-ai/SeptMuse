"""forgetting + search 组合集成测试。"""

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


def test_recall_passes_forgetting(memory):
    """recall() 内部传 forgetting=True, 返回结果有 final_score。"""
    memory.add("alice likes python", user_id="alice", infer=False)
    result = memory.recall("python", user_id="alice")
    assert "memories" in result
    if len(result["memories"]) > 0:
        assert "final_score" in result["memories"][0]

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


def test_search_forget_then_budget(memory):
    """forgetting 加权后按 final_score 裁剪到 token 预算内。"""
    for i in range(10):
        memory.add(f"fact number {i} about python programming", user_id="alice", infer=False)
    results = memory.search("python", user_id="alice", forgetting=True, token_budget=20)
    total = sum(len(r["memory"]) // 4 for r in results)
    assert total <= 20
    if results:
        assert "final_score" in results[0]


def test_search_inject_with_forgetting(memory):
    """inject_prompt + forgetting 组合返回 dict 含 final_score。"""
    memory.add("alice knows python", user_id="alice", infer=False)
    result = memory.search("python", user_id="alice", forgetting=True, inject_prompt=True)
    assert isinstance(result, dict)
    assert "results" in result
    assert "injected_prompt" in result
    assert "<memory>" in result["injected_prompt"]
    if result["results"]:
        assert "final_score" in result["results"][0]


def test_search_forgetting_degrades_gracefully(memory):
    """forgetting=True 时 self.forgetting 可用, 结果有 final_score。"""
    memory.add("some fact about coding", user_id="alice", infer=False)
    results = memory.search("coding", user_id="alice", forgetting=True)
    assert len(results) > 0
    assert "final_score" in results[0]

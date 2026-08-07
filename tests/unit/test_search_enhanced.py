"""search 参数化测试 — forgetting/token_budget/inject_prompt."""
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


def test_search_forgetting_param(memory):
    """search(forgetting=True) 不报错, 返回结果."""
    memory.add("old fact about python", user_id="alice", infer=False)
    results = memory.search("python", user_id="alice", forgetting=True)
    assert len(results) > 0
    # forgetting 加权后应有 score 或 final_score
    assert "score" in results[0] or "final_score" in results[0]


def test_search_token_budget(memory):
    """search(token_budget=N) 裁剪到预算内."""
    for i in range(20):
        memory.add(f"fact number {i} " * 20, user_id="alice", infer=False)
    results = memory.search("fact", user_id="alice", token_budget=50)
    # 总 token (chars/4) 不超 50
    total = sum(len(r["memory"]) // 4 for r in results)
    assert total <= 50


def test_search_inject_prompt(memory):
    """search(inject_prompt=True) 返回 dict 含 injected_prompt."""
    memory.add("test fact", user_id="alice", infer=False)
    result = memory.search("test", user_id="alice", inject_prompt=True)
    assert isinstance(result, dict)
    assert "results" in result
    assert "injected_prompt" in result


def test_search_default_returns_list(memory):
    """默认 search 返回 list (向后兼容)."""
    memory.add("test fact", user_id="alice", infer=False)
    results = memory.search("test", user_id="alice")
    assert isinstance(results, list)


def test_search_no_llm_forgetting_works(memory):
    """无 LLM 时 forgetting 仍可用 (不依赖 LLM)."""
    memory.add("some fact", user_id="alice", infer=False)
    results = memory.search("some", user_id="alice", forgetting=True)
    assert len(results) > 0

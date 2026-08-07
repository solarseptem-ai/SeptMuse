"""上下文感知查询改写测试."""
from __future__ import annotations

import os

import pytest


class _MockLLM:
    def __init__(self, response=""):
        self._response = response

    def set_response(self, r):
        self._response = r

    def complete(self, system_prompt, user_prompt):
        return self._response


@pytest.fixture
def memory_no_llm(tmp_path):
    os.environ["SEPTMUSE_EMBEDDER"] = "hash"
    os.environ["SEPTMUSE_TOKENIZER"] = "space"
    from septmuse.configs import MemoryConfig
    from septmuse.memory.main import Memory
    return Memory(config=MemoryConfig(db_path=str(tmp_path / "test.db")))


@pytest.fixture
def memory_with_llm(tmp_path):
    os.environ["SEPTMUSE_EMBEDDER"] = "hash"
    os.environ["SEPTMUSE_TOKENIZER"] = "space"
    from septmuse.configs import MemoryConfig
    from septmuse.memory.main import Memory
    llm = _MockLLM()
    mem = Memory(config=MemoryConfig(db_path=str(tmp_path / "test.db")), llm=llm)
    return mem, llm


def test_query_rewrite_disabled(memory_no_llm):
    """query_rewrite=False (默认) → 用原文 query."""
    memory_no_llm.add("I like Python", user_id="alice", infer=False)
    results = memory_no_llm.search("Python", user_id="alice")
    assert len(results) > 0


def test_query_rewrite_no_llm_falls_back(memory_no_llm):
    """query_rewrite=True 但无 LLM → 降级原文."""
    memory_no_llm.add("I like Python", user_id="alice", session_id="s1", infer=False)
    results = memory_no_llm.search("Python", user_id="alice", query_rewrite=True, session_id="s1")
    assert len(results) > 0


def test_query_rewrite_no_session_falls_back(memory_with_llm):
    """query_rewrite=True + LLM 但无 session_id → 降级原文."""
    mem, _ = memory_with_llm
    mem.add("I like Python", user_id="alice", infer=False)
    results = mem.search("Python", user_id="alice", query_rewrite=True)
    assert len(results) > 0


def test_query_rewrite_with_context(memory_with_llm):
    """query_rewrite=True + LLM + session → LLM 改写 query."""
    mem, llm = memory_with_llm
    # 存一些 episodic 上下文
    mem.episodic.add_raw_log("User discussed a project called ProjectX", user_id="alice", session_id="s1")
    mem.add("ProjectX is about AI memory", user_id="alice", session_id="s1", infer=False)
    # LLM 改写 "那个项目" → "ProjectX"
    llm.set_response("ProjectX AI memory project")
    results = mem.search("那个项目", user_id="alice", query_rewrite=True, session_id="s1")
    # 应该能检索到 (改写后的 query 更贴近记忆内容)
    assert len(results) > 0


def test_query_rewrite_prompt_exists():
    from septmuse.prompts.rewrite import QUERY_REWRITE_PROMPT
    assert QUERY_REWRITE_PROMPT
    assert len(QUERY_REWRITE_PROMPT) > 50

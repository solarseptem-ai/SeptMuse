"""HyDE 假设文档嵌入单元测试。"""

import os
from unittest.mock import MagicMock

import pytest


def test_generate_hypothetical_document_with_llm():
    """有 LLM 时应生成假设文档。"""
    from septmuse.retrieval.hyde import generate_hypothetical_document

    llm = MagicMock()
    llm.complete.return_value = "张三是公司CTO，负责技术架构决策。"
    result = generate_hypothetical_document("张三负责什么", llm)
    assert result == "张三是公司CTO，负责技术架构决策。"
    assert llm.complete.call_count == 1


def test_generate_hypothetical_document_no_llm():
    """无 LLM 时应返回 None。"""
    from septmuse.retrieval.hyde import generate_hypothetical_document

    result = generate_hypothetical_document("张三负责什么", None)
    assert result is None


def test_generate_hypothetical_document_llm_error():
    """LLM 异常时应返回 None（降级，不崩溃）。"""
    from septmuse.retrieval.hyde import generate_hypothetical_document

    llm = MagicMock()
    llm.complete.side_effect = Exception("Ollama connection refused")
    result = generate_hypothetical_document("张三负责什么", llm)
    assert result is None


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


def test_hyde_disabled_uses_original_query(memory_no_llm):
    """hyde=False (默认) → 用原文 query 检索 (不调 LLM)."""
    memory_no_llm.add("I like Python programming", user_id="alice", infer=False)
    results = memory_no_llm.search("Python", user_id="alice")
    assert len(results) > 0


def test_hyde_no_llm_falls_back(memory_no_llm):
    """hyde=True 但无 LLM → 降级原文检索 (不报错)."""
    memory_no_llm.add("I like Python", user_id="alice", infer=False)
    results = memory_no_llm.search("Python", user_id="alice", hyde=True)
    assert len(results) > 0


def test_hyde_with_llm_uses_hypothetical(memory_with_llm):
    """hyde=True + LLM → 用假设答案 embedding 检索."""
    mem, llm = memory_with_llm
    mem.add("User likes Python programming language", user_id="alice", infer=False)
    llm.set_response("The user enjoys programming in Python and prefers it for development.")
    results = mem.search("what do I like", user_id="alice", hyde=True)
    assert len(results) > 0


def test_hyde_prompt_exists():
    """HYDE_PROMPT 存在且非空."""
    from septmuse.prompts.hyde import HYDE_PROMPT

    assert HYDE_PROMPT
    assert len(HYDE_PROMPT) > 50

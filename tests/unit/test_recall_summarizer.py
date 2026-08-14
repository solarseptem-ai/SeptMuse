"""Tests for RecallSummarizer."""

from __future__ import annotations

from unittest.mock import MagicMock


def test_summarize_with_llm():
    """LLM 可用时：调 LLM 压缩记忆列表为摘要。"""
    llm = MagicMock()
    llm.complete.return_value = "用户计划下周去上海出差，涉及项目X。"
    memories = [
        {"id": "m1", "memory": "用户下周要出差去上海", "score": 0.9},
        {"id": "m2", "memory": "用户参与了项目X", "score": 0.8},
    ]
    from septmuse.retrieval.recall_summarizer import RecallSummarizer

    s = RecallSummarizer(memories, llm=llm)
    result = s.summarize()
    assert "上海" in result
    assert "项目X" in result
    llm.complete.assert_called_once()


def test_summarize_no_llm():
    """LLM 不可用时：取 top-3 拼接截断 200 字。"""
    memories = [
        {"id": "m1", "memory": "用户下周要出差去上海", "score": 0.9},
        {"id": "m2", "memory": "用户参与了项目X", "score": 0.8},
        {"id": "m3", "memory": "用户上周说可能不去上海了", "score": 0.7},
        {"id": "m4", "memory": "无关内容", "score": 0.5},
    ]
    from septmuse.retrieval.recall_summarizer import RecallSummarizer

    s = RecallSummarizer(memories, llm=None)
    result = s.summarize()
    assert "上海" in result
    assert len(result) <= 200
    assert "无关内容" not in result  # top-3 only


def test_summarize_empty_memories():
    """空记忆列表返回空字符串。"""
    from septmuse.retrieval.recall_summarizer import RecallSummarizer

    s = RecallSummarizer([], llm=None)
    assert s.summarize() == ""

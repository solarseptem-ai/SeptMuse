"""Tests for SemanticDeduplicator."""

from __future__ import annotations

from unittest.mock import MagicMock

from septmuse.capture.dedup import SemanticDeduplicator


def test_find_duplicate_high_similarity():
    """cosine > 0.92 → 判定重复。"""
    embedder = MagicMock()
    embedder.embed.return_value = [0.1, 0.2, 0.3]
    store = MagicMock()
    store.search.return_value = [
        {"id": "mem-old", "memory": "用户下周去上海出差", "score": 0.95},
    ]
    dedup = SemanticDeduplicator(embedder, store, llm=None)
    result = dedup.find_duplicate("用户下周要去上海出差", user_id="alice")
    assert result is not None
    assert result.memory_id == "mem-old"
    assert result.similarity >= 0.92


def test_find_duplicate_no_match():
    """cosine < 0.92 → 不重复。"""
    embedder = MagicMock()
    embedder.embed.return_value = [0.1, 0.2, 0.3]
    store = MagicMock()
    store.search.return_value = [
        {"id": "mem-other", "memory": "完全不同的内容关于天气", "score": 0.3},
    ]
    dedup = SemanticDeduplicator(embedder, store, llm=None)
    result = dedup.find_duplicate("用户下周要去上海出差", user_id="alice")
    assert result is None


def test_find_duplicate_empty_results():
    """无已有记忆 → 不重复。"""
    embedder = MagicMock()
    embedder.embed.return_value = [0.1, 0.2, 0.3]
    store = MagicMock()
    store.search.return_value = []
    dedup = SemanticDeduplicator(embedder, store, llm=None)
    result = dedup.find_duplicate("用户下周要去上海出差", user_id="alice")
    assert result is None


def test_find_duplicate_llm_gray_zone():
    """cosine 0.85-0.92 灰色地带 + LLM → 让 LLM 判定。"""
    embedder = MagicMock()
    embedder.embed.return_value = [0.1, 0.2, 0.3]
    store = MagicMock()
    store.search.return_value = [
        {"id": "mem-gray", "memory": "用户要去上海", "score": 0.88},
    ]
    llm = MagicMock()
    llm.complete.return_value = "yes"
    dedup = SemanticDeduplicator(embedder, store, llm=llm)
    result = dedup.find_duplicate("用户下周要去上海出差", user_id="alice")
    assert result is not None
    assert result.memory_id == "mem-gray"
    llm.complete.assert_called_once()


def test_find_duplicate_llm_says_no():
    """cosine 灰色地带 + LLM 说不是重复 → 不重复。"""
    embedder = MagicMock()
    embedder.embed.return_value = [0.1, 0.2, 0.3]
    store = MagicMock()
    store.search.return_value = [
        {"id": "mem-gray", "memory": "用户要去上海", "score": 0.88},
    ]
    llm = MagicMock()
    llm.complete.return_value = "no, different context"
    dedup = SemanticDeduplicator(embedder, store, llm=llm)
    result = dedup.find_duplicate("用户下周要去上海出差", user_id="alice")
    assert result is None

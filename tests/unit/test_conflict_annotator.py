"""Tests for ConflictAnnotator."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

from septmuse.models.semantic import SemanticFact
from septmuse.retrieval.conflict_annotator import ConflictAnnotator


def _make_fact(subject: str, predicate: str, obj: str, created_at: datetime) -> SemanticFact:
    return SemanticFact(
        subject=subject,
        predicate=predicate,
        object=obj,
        user_id="alice",
        created_at=created_at,
    )


def test_annotate_no_conflicts():
    """无矛盾事实返回空列表。"""
    now = datetime.now(timezone.utc)
    facts = [
        _make_fact("用户", "喜欢", "咖啡", now),
        _make_fact("用户", "住在", "北京", now),
    ]
    memories = [{"id": "m1", "memory": "用户喜欢咖啡", "score": 0.9}]
    annotator = ConflictAnnotator(llm=None)
    result = annotator.annotate(memories, facts)
    assert result == []


def test_annotate_rule_based_conflict():
    """规则检测：相同 subject+predicate 不同 object。"""
    now = datetime.now(timezone.utc)
    old = now - timedelta(days=7)
    facts = [
        _make_fact("用户", "去", "上海", old),
        _make_fact("用户", "去", "北京", now),
    ]
    memories = [
        {"id": "m1", "memory": "用户去了上海", "score": 0.9},
        {"id": "m2", "memory": "用户去了北京", "score": 0.8},
    ]
    annotator = ConflictAnnotator(llm=None)
    result = annotator.annotate(memories, facts)
    assert len(result) == 1
    assert result[0]["topic"] == "用户 去"
    assert "上海" in result[0]["current"]
    assert "北京" in result[0]["contradicting"]
    assert result[0]["suggestion"] != ""


def test_annotate_with_llm():
    """LLM 可用时：候选冲突对喂给 LLM 判定。"""
    now = datetime.now(timezone.utc)
    old = now - timedelta(days=7)
    facts = [
        _make_fact("用户", "去", "上海", old),
        _make_fact("用户", "去", "北京", now),
    ]
    memories = [
        {"id": "m1", "memory": "用户去了上海", "score": 0.9},
        {"id": "m2", "memory": "用户去了北京", "score": 0.8},
    ]
    llm = MagicMock()
    llm.complete.return_value = "yes"
    annotator = ConflictAnnotator(llm=llm)
    result = annotator.annotate(memories, facts)
    assert len(result) == 1
    llm.complete.assert_called_once()


def test_annotate_llm_says_no():
    """LLM 判定不是真矛盾时：不返回冲突。"""
    now = datetime.now(timezone.utc)
    old = now - timedelta(days=7)
    facts = [
        _make_fact("用户", "去", "上海", old),
        _make_fact("用户", "去", "北京", now),
    ]
    memories = [
        {"id": "m1", "memory": "用户去了上海", "score": 0.9},
        {"id": "m2", "memory": "用户去了北京", "score": 0.8},
    ]
    llm = MagicMock()
    llm.complete.return_value = "no, these are about different trips"
    annotator = ConflictAnnotator(llm=llm)
    result = annotator.annotate(memories, facts)
    assert result == []


def test_annotate_empty():
    """空输入返回空列表。"""
    annotator = ConflictAnnotator(llm=None)
    assert annotator.annotate([], []) == []

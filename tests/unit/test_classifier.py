"""Tests for MemoryClassifier."""

from __future__ import annotations

from unittest.mock import MagicMock

from septmuse.capture.classifier import MemoryClassifier


def test_classify_procedural():
    """规则：含步骤/操作关键词 → procedural。"""
    clf = MemoryClassifier(llm=None)
    types = clf.classify("第一步打开终端，第二步输入命令")
    assert "procedural" in types


def test_classify_episodic_with_time():
    """规则：含时间+动作 → episodic + semantic 双存。"""
    clf = MemoryClassifier(llm=None)
    types = clf.classify("用户下周要去上海出差")
    assert "episodic" in types
    assert "semantic" in types


def test_classify_semantic_preference():
    """规则：含偏好关键词 → semantic。"""
    clf = MemoryClassifier(llm=None)
    types = clf.classify("用户喜欢喝咖啡")
    assert "semantic" in types


def test_classify_default_semantic():
    """默认 → semantic。"""
    clf = MemoryClassifier(llm=None)
    types = clf.classify("天空是蓝色的")
    assert types == ["semantic"]


def test_classify_with_llm():
    """LLM 可用时：prompt 分类。"""
    llm = MagicMock()
    llm.complete.return_value = "semantic"
    clf = MemoryClassifier(llm=llm)
    types = clf.classify("用户下周要去上海出差")
    assert "semantic" in types
    llm.complete.assert_called_once()


def test_classify_llm_failure_fallback():
    """LLM 调用失败 → 降级为规则。"""
    llm = MagicMock()
    llm.complete.side_effect = Exception("API error")
    clf = MemoryClassifier(llm=llm)
    types = clf.classify("用户喜欢喝咖啡")
    assert "semantic" in types


def test_classify_empty():
    """空文本返回空列表。"""
    clf = MemoryClassifier(llm=None)
    assert clf.classify("") == []

"""Tests for ImportanceAssessor."""

from __future__ import annotations

from unittest.mock import MagicMock

from septmuse.capture.importance import ImportanceAssessor


def test_assess_high_value_with_llm():
    """LLM 打分：高价值文本得分 > 0.5。"""
    llm = MagicMock()
    llm.complete.return_value = "0.9"
    assessor = ImportanceAssessor(llm=llm)
    result = assessor.assess("用户下周要出差去上海，涉及项目X的验收")
    assert result.score > 0.5
    llm.complete.assert_called_once()


def test_assess_low_value_with_llm():
    """LLM 打分：低价值文本得分 < 0.3。"""
    llm = MagicMock()
    llm.complete.return_value = "0.1"
    assessor = ImportanceAssessor(llm=llm)
    result = assessor.assess("嗯好的")
    assert result.score < 0.3


def test_assess_rule_based_high():
    """无 LLM：含实体+数字+时间 → 高分。"""
    assessor = ImportanceAssessor(llm=None)
    result = assessor.assess("用户下周要去上海出差3天")
    assert result.score >= 0.7


def test_assess_rule_based_medium():
    """无 LLM：含偏好关键词 → 中分。"""
    assessor = ImportanceAssessor(llm=None)
    result = assessor.assess("用户喜欢喝咖啡")
    assert result.score >= 0.5


def test_assess_rule_based_low():
    """无 LLM：无信号 → 低分。"""
    assessor = ImportanceAssessor(llm=None)
    result = assessor.assess("嗯好的知道了")
    assert result.score < 0.5


def test_assess_llm_failure_fallback():
    """LLM 调用失败 → 降级为规则评估。"""
    llm = MagicMock()
    llm.complete.side_effect = Exception("API error")
    assessor = ImportanceAssessor(llm=llm)
    result = assessor.assess("用户下周要去上海出差")
    assert result.score >= 0.5  # 规则兜底

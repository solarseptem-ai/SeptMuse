"""ImportanceAssessor — 写入时给记忆打价值分 (0-1)。

LLM 可用：prompt 打分。
LLM 不可用：规则——含实体/数字/时间 → 0.7+；含偏好关键词 → 0.6+；其余 → 0.3。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from septmuse.core.logging import get_logger

if TYPE_CHECKING:
    from septmuse.llms.base import LLM

logger = get_logger(__name__)

_THRESHOLD = 0.3

_SYSTEM_PROMPT = (
    "你是一个记忆价值评估助手。评估以下文本是否值得记住。只回答一个 0 到 1 的小数。1.0=关键信息，0.5=有用，0.0=噪音。"
)
_USER_TEMPLATE = "文本: {text}\n\n价值评分:"

_ENTITY_PATTERN = re.compile(r"[\u4e00-\u9fff]{2,}|[A-Z][a-z]+")
_NUMBER_PATTERN = re.compile(r"\d+")
_TIME_PATTERN = re.compile(r"今天|明天|下周|这周|上周|下午|上午|\d+月|\d+日|\d+点|\d+年")
_PLAN_PATTERN = re.compile(r"计划|打算|将要|准备|安排|要去|要做|会去")
_PREF_PATTERN = re.compile(r"喜欢|偏好|总是|习惯|讨厌|爱吃|爱喝|常用")


@dataclass
class ImportanceResult:
    """价值评估结果。"""

    score: float
    method: str  # "llm" | "rule"


class ImportanceAssessor:
    """给文本打价值分。

    Args:
        llm: LLM 实例（可选，None 时用规则）
    """

    def __init__(self, llm: LLM | None = None) -> None:
        self._llm = llm

    def assess(self, text: str) -> ImportanceResult:
        """返回价值评分 (0-1)。"""
        if not text or not text.strip():
            return ImportanceResult(score=0.0, method="rule")

        if self._llm is not None:
            return self._assess_with_llm(text)
        return self._assess_rule(text)

    def _assess_with_llm(self, text: str) -> ImportanceResult:
        user_prompt = _USER_TEMPLATE.format(text=text[:500])
        try:
            response = self._llm.complete(_SYSTEM_PROMPT, user_prompt)
            score = self._parse_score(response)
            logger.debug("importance_llm_done", score=score)
            return ImportanceResult(score=score, method="llm")
        except Exception as e:
            logger.warning("importance_llm_failed", error=str(e), fallback="rule")
            return self._assess_rule(text)

    def _assess_rule(self, text: str) -> ImportanceResult:
        has_entity = bool(_ENTITY_PATTERN.search(text))
        has_number = bool(_NUMBER_PATTERN.search(text))
        has_time = bool(_TIME_PATTERN.search(text))
        has_plan = bool(_PLAN_PATTERN.search(text))
        has_pref = bool(_PREF_PATTERN.search(text))

        density = sum([has_entity, has_number, has_time]) / 3.0  # 信息密度 0-1
        relevance = 0.6 if (has_plan or has_pref) else 0.3  # 持续相关性

        score = 0.4 * density + 0.6 * relevance
        if has_pref:
            score = max(score, 0.6)  # 偏好关键词 → 0.6+（spec 保证）
        logger.debug(
            "importance_rule_done",
            score=score,
            entity=has_entity,
            number=has_number,
            time=has_time,
            plan=has_plan,
            pref=has_pref,
        )
        return ImportanceResult(score=round(score, 2), method="rule")

    @staticmethod
    def _parse_score(response: str) -> float:
        """从 LLM 响应中解析分数。"""
        for token in response.strip().split():
            try:
                val = float(token)
                if 0.0 <= val <= 1.0:
                    return val
            except ValueError:
                continue
        return 0.5  # 无法解析时默认中分

"""自动判定记忆类型。"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from septmuse.core.logging import get_logger

if TYPE_CHECKING:
    from septmuse.llms.base import LLM

logger = get_logger(__name__)

_SYSTEM_PROMPT = "你是一个记忆分类助手。判断以下文本属于哪些类型，用逗号分隔返回。可选类型：semantic(事实陈述), episodic(情节事件), procedural(操作步骤)。"
_USER_TEMPLATE = "文本: {text}\n\n类型:"

_PROC_PATTERN = re.compile(r"步骤|操作|方法|如何|第一步|第二步|输入命令|运行|执行")
_TIME_PATTERN = re.compile(r"今天|明天|下周|这周|上周|刚才|刚刚|\d+月|\d+日|\d+点")
_ACTION_PATTERN = re.compile(r"去|来|做|说|买|看|吃|开|关|跑|走|出差|开会|汇报")
_PREF_PATTERN = re.compile(r"喜欢|偏好|总是|习惯|讨厌|爱吃|爱喝|常用|一般|通常")


class MemoryClassifier:
    """自动判定记忆类型。"""

    def __init__(self, llm: LLM | None = None) -> None:
        self._llm = llm

    def classify(self, text: str) -> list[str]:
        """返回记忆类型列表（可多类型双存）。空文本返回空列表。"""
        if not text or not text.strip():
            return []

        if self._llm is not None:
            return self._classify_with_llm(text)
        return self._classify_rule(text)

    def _classify_with_llm(self, text: str) -> list[str]:
        user_prompt = _USER_TEMPLATE.format(text=text[:500])
        try:
            response = self._llm.complete(_SYSTEM_PROMPT, user_prompt)
            types = [t.strip().lower() for t in response.split(",")]
            valid = [t for t in types if t in ("semantic", "episodic", "procedural")]
            if valid:
                return valid
            logger.warning("classifier_llm_invalid_response", response=response[:200], fallback="rule")
            return self._classify_rule(text)
        except Exception as e:
            logger.warning("classifier_llm_failed", error=str(e), fallback="rule")
            return self._classify_rule(text)

    def _classify_rule(self, text: str) -> list[str]:
        types: list[str] = []

        is_procedural = bool(_PROC_PATTERN.search(text))
        has_time = bool(_TIME_PATTERN.search(text))
        has_action = bool(_ACTION_PATTERN.search(text))
        has_pref = bool(_PREF_PATTERN.search(text))

        if is_procedural:
            types.append("procedural")
        if has_time and has_action:
            types.append("episodic")
            types.append("semantic")
        if has_pref:
            types.append("semantic")
        if not types:
            types.append("semantic")

        logger.debug(
            "classifier_rule_done",
            types=types,
            procedural=is_procedural,
            time=has_time,
            action=has_action,
            pref=has_pref,
        )
        return types

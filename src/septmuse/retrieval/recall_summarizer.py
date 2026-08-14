"""RecallSummarizer — LLM 压缩 N 条记忆为一段摘要。

LLM 可用：prompt 压缩，保留关键事实和时间线索。
LLM 不可用：取 top-3 记忆 memory 字段拼接，截断 200 字。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from septmuse.core.logging import get_logger

if TYPE_CHECKING:
    from septmuse.llms.base import LLM

logger = get_logger(__name__)

_MAX_FALLBACK_LEN = 200
_TOP_N_FALLBACK = 3

_SUMMARY_SYSTEM_PROMPT = "你是一个记忆摘要助手。将以下记忆条目压缩为一段简洁摘要，保留关键事实和时间线索。"
_SUMMARY_USER_TEMPLATE = "记忆条目：\n{items}\n\n请压缩为一段摘要："


class RecallSummarizer:
    """压缩记忆列表为摘要。

    Args:
        memories: recall 返回的记忆列表，每个 dict 有 memory/score 字段
        llm: LLM 实例（可选，None 时降级为拼接）
    """

    def __init__(self, memories: list[dict[str, Any]], llm: LLM | None = None) -> None:
        self._memories = memories
        self._llm = llm

    def summarize(self) -> str:
        """返回摘要字符串。空记忆返回空字符串。"""
        if not self._memories:
            return ""

        if self._llm is not None:
            return self._summarize_with_llm()
        return self._summarize_fallback()

    def _summarize_with_llm(self) -> str:
        items = "\n".join(f"- {m.get('memory', '')}" for m in self._memories)
        user_prompt = _SUMMARY_USER_TEMPLATE.format(items=items)
        try:
            result = self._llm.complete(_SUMMARY_SYSTEM_PROMPT, user_prompt)
            logger.debug("recall_summary_llm_done", memories=len(self._memories), length=len(result))
            return result
        except Exception as e:
            logger.warning("recall_summary_llm_failed", error=str(e), fallback="concat")
            return self._summarize_fallback()

    def _summarize_fallback(self) -> str:
        sorted_memories = sorted(self._memories, key=lambda m: m.get("score", 0.0), reverse=True)
        top = sorted_memories[:_TOP_N_FALLBACK]
        texts = [m.get("memory", "") for m in top]
        joined = " ".join(texts)
        return joined[:_MAX_FALLBACK_LEN]

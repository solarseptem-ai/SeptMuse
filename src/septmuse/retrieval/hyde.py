"""HyDE — Hypothetical Document Embeddings。

用 LLM 根据查询生成假设答案，用假设答案的向量（而非原始查询）做向量检索。
假设答案在向量空间更接近已存储的记忆。
"""

from __future__ import annotations

from septmuse.core.logging import get_logger
from septmuse.prompts.hyde import HYDE_PROMPT

logger = get_logger(__name__)


def generate_hypothetical_document(query: str, llm: object | None) -> str | None:
    """生成假设文档。

    Args:
        query: 用户查询。
        llm: SeptMuse LLM 实例（有 complete(system, user) -> str）。None 时返回 None。

    Returns:
        假设答案文本，或 None（无 LLM 或出错时）。
    """
    if llm is None:
        return None
    try:
        prompt = HYDE_PROMPT.format(query=query)
        result = llm.complete("", prompt)
        if result and result.strip():
            return result.strip()
        return None
    except Exception as e:
        logger.warning("hyde_generation_failed", error=str(e))
        return None

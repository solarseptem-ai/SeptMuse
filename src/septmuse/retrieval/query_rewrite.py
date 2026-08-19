"""查询重写 — 用 LLM 生成查询变体，多路检索后 RRF 融合。"""

from __future__ import annotations

from septmuse.core.logging import get_logger

logger = get_logger(__name__)

REWRITE_SYSTEM_PROMPT = "You are a search query rewriter. Rewrite the given question into 2 different phrasings that ask for the same information. Number them 1. and 2. Be concise."

REWRITE_USER_TEMPLATE = "Original question: {query}\n\nWrite 2 rewritten versions:"


def rewrite_query(query: str, llm: object | None) -> list[str]:
    """重写查询，返回原始 + 变体列表。

    Args:
        query: 原始查询。
        llm: SeptMuse LLM 实例。None 时只返回原始查询。

    Returns:
        查询列表，第一个是原始查询，后续是变体。无 LLM 或出错时只含原始查询。
    """
    if llm is None:
        return [query]
    try:
        prompt = REWRITE_USER_TEMPLATE.format(query=query)
        result = llm.complete(REWRITE_SYSTEM_PROMPT, prompt)
        if not result or not result.strip():
            return [query]
        variants = _parse_variants(result, query)
        return variants if len(variants) > 1 else [query]
    except Exception as e:
        logger.warning("query_rewrite_failed", error=str(e))
        return [query]


def _parse_variants(raw: str, original: str) -> list[str]:
    """从 LLM 输出解析变体列表。"""
    variants = [original]
    for line in raw.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        for prefix in ("1.", "2.", "3.", "- ", "* "):
            if line.startswith(prefix):
                line = line[len(prefix) :].strip()
                break
        if line and line != original:
            variants.append(line)
    return variants

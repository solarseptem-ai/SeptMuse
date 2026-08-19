"""语义去重。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from septmuse.core.logging import get_logger

if TYPE_CHECKING:
    from septmuse.embedders.base import Embedder
    from septmuse.llms.base import LLM
    from septmuse.storage.base import MemoryStore

logger = get_logger(__name__)

_HARD_THRESHOLD = 0.92
_GRAY_LOW = 0.85
_GRAY_HIGH = 0.92

_SYSTEM_PROMPT = "你是一个文本去重助手。判断以下两条文本是否表达同一信息。回答 'yes' 或 'no'。"
_USER_TEMPLATE = "文本1: {text1}\n文本2: {text2}\n\n是否表达同一信息？"


@dataclass
class DedupResult:
    """去重检测结果。"""

    memory_id: str
    similarity: float
    method: str  # "hard" | "llm"


class SemanticDeduplicator:
    """语义去重。"""

    def __init__(self, embedder: Embedder, store: MemoryStore, llm: LLM | None = None) -> None:
        self._embedder = embedder
        self._store = store
        self._llm = llm

    def find_duplicate(self, text: str, *, user_id: str) -> DedupResult | None:
        """返回重复记忆的 ID + 相似度，无重复返回 None。"""
        emb = self._embedder.embed(text)
        try:
            results = self._store.search(emb, user_id=user_id, top_k=5, threshold=0.1)
        except Exception as e:
            logger.warning("dedup_search_failed", error=str(e))
            return None

        for r in results:
            score = r.get("score", 0.0)
            if score >= _HARD_THRESHOLD:
                logger.debug("dedup_hard_match", memory_id=r["id"], score=score)
                return DedupResult(memory_id=r["id"], similarity=score, method="hard")

            if (
                _GRAY_LOW <= score < _GRAY_HIGH
                and self._llm is not None
                and self._llm_says_duplicate(text, r.get("memory", ""))
            ):
                logger.debug("dedup_llm_match", memory_id=r["id"], score=score)
                return DedupResult(memory_id=r["id"], similarity=score, method="llm")

        return None

    def _llm_says_duplicate(self, text1: str, text2: str) -> bool:
        user_prompt = _USER_TEMPLATE.format(text1=text1[:200], text2=text2[:200])
        try:
            response = self._llm.complete(_SYSTEM_PROMPT, user_prompt)
            return "yes" in response.strip().lower()
        except Exception as e:
            logger.warning("dedup_llm_failed", error=str(e))
            return False

"""ConflictAnnotator — 在检索结果中找矛盾并标注。

规则：相同 (subject, predicate) 不同 object → 冲突。
LLM 可用时：候选冲突对喂给 LLM 判定是否真矛盾。
LLM 不可用时：只做规则匹配。

只在 recall 返回的 top-K 结果对应的事实里找，不全库扫描。
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any

from septmuse.core.logging import get_logger

if TYPE_CHECKING:
    from septmuse.llms.base import LLM
    from septmuse.models.semantic import SemanticFact

logger = get_logger(__name__)

_CONFLICT_SYSTEM_PROMPT = "你是一个矛盾判定助手。判断以下两条事实是否真的矛盾。回答 'yes' 或 'no'。"
_CONFLICT_USER_TEMPLATE = "事实1: {fact1}\n事实2: {fact2}\n\n这两条事实是否矛盾？"


class ConflictAnnotator:
    """在检索结果中标注矛盾事实。

    Args:
        llm: LLM 实例（可选，None 时只做规则匹配）
    """

    def __init__(self, llm: LLM | None = None) -> None:
        self._llm = llm

    def annotate(
        self,
        memories: list[dict[str, Any]],
        facts: list[SemanticFact],
    ) -> list[dict[str, Any]]:
        """在事实列表中找矛盾并返回冲突标注。

        Args:
            memories: recall 返回的记忆列表（用于确定范围）
            facts: 对应的 SemanticFact 列表（从 typed_store 获取）

        Returns:
            [{"topic": str, "current": str, "contradicting": str,
              "contradicting_memory_id": str, "suggestion": str}]
        """
        if not facts:
            return []

        groups: dict[tuple[str, str], list[Any]] = defaultdict(list)
        for f in facts:
            if getattr(f, "is_deleted", False):
                continue
            key = (f.subject, f.predicate)
            groups[key].append(f)

        conflicts: list[dict[str, Any]] = []
        for (subject, predicate), group_facts in groups.items():
            objects = {f.object for f in group_facts}
            if len(objects) <= 1:
                continue

            group_facts_sorted = sorted(group_facts, key=lambda f: f.created_at)
            earliest = group_facts_sorted[0]

            for newer_fact in group_facts_sorted[1:]:
                fact1 = f"{earliest.subject} {earliest.predicate} {earliest.object}"
                fact2 = f"{newer_fact.subject} {newer_fact.predicate} {newer_fact.object}"

                if self._llm is not None and not self._llm_says_conflict(fact1, fact2):
                    continue

                conflicts.append(
                    {
                        "topic": f"{subject} {predicate}",
                        "current": fact1,
                        "contradicting": fact2,
                        "contradicting_memory_id": newer_fact.id,
                        "suggestion": f"建议询问{subject}的{predicate}是否已变更",
                    }
                )

        logger.info("conflicts_annotated", count=len(conflicts))
        return conflicts

    def _llm_says_conflict(self, fact1: str, fact2: str) -> bool:
        """LLM 判定两条事实是否真矛盾。"""
        user_prompt = _CONFLICT_USER_TEMPLATE.format(fact1=fact1, fact2=fact2)
        try:
            response = self._llm.complete(_CONFLICT_SYSTEM_PROMPT, user_prompt)
            return "yes" in response.strip().lower()
        except Exception as e:
            logger.warning("llm_conflict_check_failed", error=str(e))
            return True

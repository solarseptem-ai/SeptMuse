"""RecallPipeline — 召回管线（深化模块）。

拥有 recall() 的全 7 步链路: compress → search → budget → prompt → route → strategy → return。
Memory.recall() 委托给本管线, 不再持有召回编排逻辑。

架构决策见: grilling 共识 Q1-Q7 (RecallPipeline)
"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace
from typing import Any

from septmuse.core.logging import get_logger

logger = get_logger(__name__)


class RecallPipeline:
    """召回管线 — compress → search → budget → prompt → route → strategy。

    Memory.recall() 委托给本管线。管线持有 SearchPipeline 实例 (Q2),
    调 search_pipeline.search(forgetting=True) 获取检索结果,
    然后做 token budget fit + prompt assembly + metacognition route/strategy。
    """

    def __init__(
        self,
        search_pipeline: Any,
        meta: Any | None = None,
        token_budget: Any | None = None,
        working_memory: Any | None = None,
        summarizer: Any | None = None,
        typed_store: Any | None = None,
        llm: Any | None = None,
    ) -> None:
        self._search = search_pipeline
        self._meta = meta
        self._token_budget = token_budget
        self._working_memory = working_memory
        self._summarizer = summarizer
        self._typed_store = typed_store
        self._llm = llm

    def recall(
        self,
        query: str,
        *,
        user_id: str,
        session_id: str | None = None,
        top_k: int | None = None,
        auto_compress: bool = False,
        recipe: str | None = None,
        inject_profile: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """召回: compress → search → budget → prompt → route → strategy。

        Returns:
            {"memories": list, "injected_prompt": str, "route": dict, "strategy": StrategyResult|None}
        """
        # [1] compress (if auto_compress + summarizer available)
        if auto_compress and self._summarizer is not None:
            with contextlib.suppress(Exception):
                self._summarizer.compress(user_id=user_id)

        # [2] pop params that recall() handles itself (Q5: 移到管线, 不是 hack)
        forgetting = kwargs.pop("forgetting", True)
        kwargs.pop("inject_prompt", None)
        kwargs.pop("token_budget", None)

        # [3] search (with forgetting=True default)
        memories = self._search.search(
            query,
            user_id=user_id,
            session_id=session_id,
            top_k=top_k,
            recipe=recipe,
            forgetting=forgetting,
            **kwargs,
        )

        # [4] token budget fit (recall's own, distinct from search()'s token_budget param)
        from septmuse.retrieval.token_budget import BudgetItem

        items = [BudgetItem(text=m.get("memory", ""), score=m.get("score", 0.0), id=m.get("id")) for m in memories]
        if self._token_budget is not None:
            budget_result = self._token_budget.fit(items)
        else:
            budget_result = SimpleNamespace(texts=[item.text for item in items], used_tokens=0)

        # [4.5] summarize
        from septmuse.retrieval.recall_summarizer import RecallSummarizer

        summarizer = RecallSummarizer(memories, llm=self._llm)
        summary = summarizer.summarize()

        # [4.6] conflict annotation
        conflicts: list[dict[str, Any]] = []
        if self._typed_store is not None:
            from septmuse.retrieval.conflict_annotator import ConflictAnnotator

            try:
                facts = self._typed_store.get_all_facts(user_id=user_id)
                annotator = ConflictAnnotator(llm=self._llm)
                conflicts = annotator.annotate(memories, facts)
            except Exception as e:
                logger.warning("recall_conflict_annotate_failed", error=str(e))

        # [5] prompt assembly (with working_memory prefix)
        wm_prompt = ""
        if self._working_memory is not None:
            wm_prompt = self._working_memory.compile_to_prompt() + "\n"

        memory_texts = "\n".join(f"- {t}" for t in budget_result.texts)
        injected_prompt = f"{wm_prompt}<memory>\n{memory_texts}\n</memory>"

        # [6] metacognition route (L0)
        route_result = None
        if self._meta is not None:
            try:
                route_result = self._meta.route(query)
            except Exception as e:
                logger.warning("recall_route_failed", error=str(e))

        # [7] strategy adaptation (L2, requires coverage facts)
        strategy = None
        if self._meta is not None and self._typed_store is not None:
            coverage_facts = self._typed_store.get_all_facts(user_id=user_id)
            has_coverage = any("meta" in (f.tags or []) and "coverage" in (f.tags or []) for f in coverage_facts)
            if has_coverage:
                try:
                    report = self._meta.analyze_coverage(user_id=user_id)
                    strategy = self._meta.adapt_strategy(report)
                except Exception as e:
                    logger.warning("recall_strategy_failed", error=str(e))

        logger.info("recall_done", user_id=user_id, hits=len(memories), tokens=budget_result.used_tokens)
        return {
            "memories": memories,
            "injected_prompt": injected_prompt,
            "route": route_result,
            "strategy": strategy,
            "summary": summary,
            "conflicts": conflicts,
        }

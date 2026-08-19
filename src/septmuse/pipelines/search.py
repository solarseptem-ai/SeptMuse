"""SearchPipeline — 检索管线（深化模块）。

拥有 search() 的 6 步链路: recipe 解析 → 检索 → reranker → 过期过滤 → 后处理。
Memory.search() 委托给本管线, 不再持有检索编排逻辑。

架构决策见: docs/superpowers/specs/ (grilling 共识 Q1-Q11)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from septmuse.core.logging import get_logger
from septmuse.core.utils import is_expired as _is_expired
from septmuse.retrieval.hybrid import HybridResult

logger = get_logger(__name__)


# ------------------------------------------------------------------
# 参数分组 dataclass — 消除 search() 17+ 参数的爆炸
# ------------------------------------------------------------------


@dataclass
class SearchConfig:
    """检索配置 — 影响检索方式和结果质量。"""

    top_k: int | None = None
    threshold: float | None = None
    hybrid: bool = True
    reranker: str | None = None
    recipe: str | None = None
    explain: bool = False
    filters: dict[str, Any] | None = None
    search_filter: dict[str, Any] | None = None
    show_expired: bool = False
    hyde: bool = False
    query_rewrite: bool = False


@dataclass
class PostProcessConfig:
    """后处理配置 — 检索结果的后处理链。"""

    forgetting: bool = False
    token_budget: int | None = None
    inject_prompt: bool = False


# ------------------------------------------------------------------
# Retriever Protocol (Q2/Q5: 统一 hybrid/纯向量分支)
# ------------------------------------------------------------------


class Retriever(Protocol):
    """检索器接口 — 一个 search() 方法, 返回 list[HybridResult]。

    两个实现证明 seam 为真:
    - HybridRetriever (向量 + BM25 RRF 融合)
    - VectorRetrieverAdapter (纯向量, dict→HybridResult 适配)
    """

    def search(
        self,
        query: str,
        *,
        user_id: str,
        session_id: str | None = None,
        top_k: int = 5,
        threshold: float = 0.0,
        explain: bool = False,
        filters: dict[str, Any] | None = None,
    ) -> list[HybridResult]: ...


# ------------------------------------------------------------------
# VectorRetrieverAdapter (Q2/Q5: 纯向量分支适配)
# ------------------------------------------------------------------


class VectorRetrieverAdapter:
    """纯向量检索适配器 — 包装 store.search(), 返回 list[HybridResult]。

    dict → HybridResult 映射: vector_score=score, bm25_score=0.0, entity_boost=0.0。
    """

    def __init__(self, store: Any, embedder: Any) -> None:
        self._store = store
        self._embedder = embedder

    def search(
        self,
        query: str,
        *,
        user_id: str,
        session_id: str | None = None,
        top_k: int = 5,
        threshold: float = 0.0,
        explain: bool = False,
        filters: dict[str, Any] | None = None,
    ) -> list[HybridResult]:
        emb = self._embedder.embed(query)
        results = self._store.search(
            emb,
            user_id=user_id,
            session_id=session_id,
            top_k=top_k,
            threshold=threshold,
            filters=filters,
        )
        return [
            HybridResult(
                id=r["id"],
                memory=r["memory"],
                score=r["score"],
                vector_score=r.get("score", 0.0),
                bm25_score=0.0,
                entity_boost=0.0,
                metadata=r.get("metadata", {}),
                created_at=r.get("created_at"),
            )
            for r in results
        ]


# ------------------------------------------------------------------
# SearchPipeline (Q1: 拥有全部 6 步链路)
# ------------------------------------------------------------------


class SearchPipeline:
    """检索管线 — recipe → 检索 → reranker → 过期过滤 → 后处理。

    Memory.search() 委托给本管线。管线是纯编排者, 不持有 config, 不拥有组件。
    组件由 Memory 构造并注入。
    """

    def __init__(
        self,
        hybrid_retriever: Retriever,
        vector_retriever: Retriever,
        reranker_factory: Callable[[str], Any],
        embedder: Any,
        forgetting: Any | None = None,
        default_top_k: int = 5,
        default_threshold: float = 0.1,
    ) -> None:
        self._hybrid_retriever = hybrid_retriever
        self._vector_retriever = vector_retriever
        self._reranker_factory = reranker_factory
        self._embedder = embedder
        self._forgetting = forgetting
        self._default_top_k = default_top_k
        self._default_threshold = default_threshold

    def search(
        self,
        query: str,
        *,
        user_id: str,
        session_id: str | None = None,
        config: SearchConfig | None = None,
        post_process: PostProcessConfig | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]] | dict[str, Any]:
        """检索记忆 (hybrid BM25+向量 RRF 融合 + 后处理链)。

        支持两种调用方式:
        - 新方式: config=SearchConfig(...), post_process=PostProcessConfig(...)
        - 旧方式: top_k=5, forgetting=True, ... (自动映射到 config/post_process)
        """
        # 向后兼容: 从 kwargs 构建 config/post_process
        if kwargs and config is None:
            search_keys = set(SearchConfig.__dataclass_fields__)
            post_keys = set(PostProcessConfig.__dataclass_fields__)
            config = SearchConfig(**{k: v for k, v in kwargs.items() if k in search_keys})
            post_process = PostProcessConfig(**{k: v for k, v in kwargs.items() if k in post_keys})

        cfg = config or SearchConfig()
        pp = post_process or PostProcessConfig()

        # [1] recipe 解析
        if cfg.recipe is not None:
            from septmuse.retrieval.recipes import get_recipe

            r = get_recipe(cfg.recipe)
            cfg.hybrid = r.hybrid
            if r.reranker != "noop":
                cfg.reranker = r.reranker
            cfg.explain = r.explain
            if r.forgetting:
                pp.forgetting = True
            if r.graph_bfs:
                logger.info("recipe_graph_bfs_not_routed", hint="GraphSearcher not wired into pipeline (future)")
            if r.progressive:
                logger.info(
                    "recipe_progressive_not_routed", hint="ProgressiveRetriever not wired into pipeline (future)"
                )
            if r.hyde:
                logger.info("recipe_hyde_not_enabled", hint="requires LLM + prompt engineering (future)")
            if r.query_rewrite:
                logger.info("recipe_query_rewrite_not_enabled", hint="requires LLM (future)")

        # [2] HyDE / query_rewrite no-op
        if cfg.hyde:
            logger.info("hyde_not_enabled", hint="requires LLM + prompt engineering (future)")
        if cfg.query_rewrite:
            logger.info("query_rewrite_not_enabled", hint="requires LLM (future)")

        # [3] 默认值 resolve (Q11: 管线持有 default_top_k/default_threshold)
        tk = cfg.top_k or self._default_top_k
        fetch_k = tk * 2 if not cfg.show_expired else tk
        if cfg.hybrid:
            th = 0.0 if cfg.threshold is None else cfg.threshold
        else:
            th = cfg.threshold if cfg.threshold is not None else self._default_threshold

        # [4] 检索 (统一路径, Q2: Retriever protocol)
        retriever = self._hybrid_retriever if cfg.hybrid else self._vector_retriever
        results = retriever.search(
            query,
            user_id=user_id,
            session_id=session_id,
            top_k=fetch_k,
            threshold=th,
            explain=cfg.explain,
            filters=cfg.filters,
        )

        # [5] reranker (统一一份, 消除重复)
        if cfg.reranker is not None and cfg.reranker != "noop":
            from septmuse.rerankers.strategies import RerankerStrategyFactory

            try:
                reranker_instance = self._reranker_factory(cfg.reranker)
                strategy = RerankerStrategyFactory.create("full_memory")
                tracker, documents = strategy.prepare(results)
                scored = reranker_instance.rerank(query, documents, top_k=fetch_k)
                results = strategy.reconstruct(scored, tracker, results, fetch_k, cfg.search_filter)
            except Exception as e:
                logger.warning("reranker_failed_fallback", reranker=cfg.reranker, error=str(e))

        # [6] dict 转换 + 过期过滤 (hybrid 含 vector_score/bm25_score, 纯向量不含——向后兼容)
        if cfg.hybrid:
            hybrid_return = [
                {
                    "id": r.id,
                    "memory": r.memory,
                    "score": r.score,
                    "vector_score": r.vector_score,
                    "bm25_score": r.bm25_score,
                    "metadata": r.metadata,
                    "created_at": r.created_at,
                }
                for r in results
            ]
        else:
            hybrid_return = [
                {
                    "id": r.id,
                    "memory": r.memory,
                    "score": r.score,
                    "metadata": r.metadata,
                    "created_at": r.created_at,
                }
                for r in results
            ]
        if not cfg.show_expired:
            hybrid_return = [r for r in hybrid_return if not _is_expired(r.get("metadata"))]

        logger.info("search_done", user_id=user_id, query=query[:50], hits=len(hybrid_return))

        # [7] 后处理链: forgetting → token_budget → inject_prompt
        return self._post_process(
            hybrid_return[:tk],
            user_id=user_id,
            forgetting=pp.forgetting,
            token_budget=pp.token_budget,
            inject_prompt=pp.inject_prompt,
        )

    def _post_process(
        self,
        results: list[dict[str, Any]],
        *,
        user_id: str,
        forgetting: bool = False,
        token_budget: int | None = None,
        inject_prompt: bool = False,
    ) -> list[dict[str, Any]] | dict[str, Any]:
        """后处理链: forgetting → token_budget → inject_prompt。

        后处理失败不阻塞检索——吞错降级, 返回原始结果。
        """
        # [1] forgetting: 强度加权 + 测试效应
        if forgetting and self._forgetting is not None:
            try:
                weighted = self._forgetting.apply_strength(results, user_id=user_id)
                original_map = {r.get("id"): r for r in results}
                results = [
                    {
                        "id": w.id,
                        "memory": w.memory,
                        "score": w.final_score,
                        "final_score": w.final_score,
                        "strength": w.strength,
                        "metadata": w.metadata,
                        "created_at": original_map.get(w.id, {}).get("created_at"),
                    }
                    for w in weighted
                ]
            except Exception as e:
                logger.warning("forgetting_apply_failed", error=str(e))
        elif forgetting and self._forgetting is None:
            logger.warning("forgetting_component_unavailable")

        # [2] token_budget: 按加权后 score 贪心裁剪
        if token_budget is not None and results:
            try:
                from septmuse.retrieval.token_budget import BudgetItem, TokenBudget

                items = [
                    BudgetItem(text=r.get("memory", ""), score=r.get("score", 0.0), id=r.get("id")) for r in results
                ]
                budget = TokenBudget(budget=token_budget)
                fitted = budget.fit(items)
                chosen_ids = {item.id for item in fitted.items}
                results = [r for r in results if r.get("id") in chosen_ids]
            except Exception as e:
                logger.warning("token_budget_failed", error=str(e))

        # [3] inject_prompt: 返回 dict
        if inject_prompt:
            memory_texts = "\n".join(f"- {r.get('memory', '')}" for r in results)
            injected_prompt = f"<memory>\n{memory_texts}\n</memory>"
            return {"results": results, "injected_prompt": injected_prompt}

        return results

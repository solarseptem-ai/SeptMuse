#  Copyright 2026 The sonhhxg0529 Authors. All Rights Reserved.
"""SearchModule — 检索领域模块（search / search_hybrid / search_progressive / search_with_strength / search_graph / search_graph_fused / search_facts / apply_token_budget / redact）。"""

from __future__ import annotations

from typing import Any

from septmuse.embedders.base import Embedder
from septmuse.llms.base import LLM
from septmuse.storage.base import MemoryStore
from septmuse.storage.relational_stores.typed_store import TypedMemoryStore


class SearchModule:
    """检索领域模块 — 向量 + BM25 + 图 + 时态 + 遗忘曲线。"""

    def __init__(
        self,
        store: MemoryStore,
        embedder: Embedder,
        typed_store: TypedMemoryStore,
        llm: LLM | None = None,
        entity_store: Any | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.typed_store = typed_store
        self.llm = llm
        self.entity_store = entity_store

    def search_hybrid(
        self, query: str, *, user_id: str, top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """混合检索（向量 + BM25 RRF 融合）。"""
        from septmuse.retrieval.hybrid import HybridRetriever

        retriever = HybridRetriever(self.store, self.embedder)
        results = retriever.search(query, user_id=user_id, top_k=top_k)
        return [
            {"id": r.id, "memory": r.memory, "score": r.score, "vector_score": r.vector_score,
             "bm25_score": getattr(r, "bm25_score", 0.0)}
            for r in results
        ]

    def search_progressive(
        self, query: str, *, user_id: str, top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """渐进三层检索。"""
        from septmuse.retrieval.progressive import ProgressiveRetriever

        retriever = ProgressiveRetriever(self.store, self.typed_store, self.embedder)
        results = retriever.retrieve(query, user_id=user_id, top_k=top_k)
        return [{"id": r.id, "memory": r.memory, "score": r.score, "memory_type": r.memory_type} for r in results]

    def search_with_strength(
        self, query: str, *, user_id: str, top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """遗忘曲线加权检索。"""
        emb = self.embedder.embed(query)
        results = self.store.search(emb, user_id=user_id, top_k=top_k * 2)
        from septmuse.retrieval.forgetting import ForgettingRetriever

        fr = ForgettingRetriever(self.typed_store)
        weighted = fr.apply_strength(results, user_id=user_id)
        return [
            {"id": w.id, "memory": w.memory, "relevance": w.relevance,
             "strength": w.strength, "final_score": w.final_score}
            for w in weighted[:top_k]
        ]

    def search_graph(
        self, seed_memory_id: str, *, max_depth: int = 2,
    ) -> list[dict[str, Any]]:
        """BFS 图遍历检索。"""
        from septmuse.retrieval.graph_search import GraphSearcher

        searcher = GraphSearcher(self.store, self.typed_store)
        return searcher.search_graph(seed_memory_id, max_depth)

    def search_graph_fused(
        self, query: str, *, user_id: str, seed_memory_id: str | None = None,
        max_depth: int = 2, top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """图 + 向量 RRF 融合检索。"""
        emb = self.embedder.embed(query)
        vector_results = self.store.search(emb, user_id=user_id, top_k=top_k)
        from septmuse.core.utils import rrf_fuse

        if seed_memory_id is not None:
            from septmuse.retrieval.graph_search import GraphSearcher

            graph_results = GraphSearcher(self.store, self.typed_store).search_graph(seed_memory_id, max_depth)
            return rrf_fuse(vector_results, graph_results, top_k=top_k)
        return vector_results[:top_k]

    def search_facts(
        self, query: str, *, user_id: str, top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """搜索语义事实。"""
        facts = self.typed_store.search_facts(query, user_id=user_id, top_k=top_k)
        return [
            {"id": f.id, "subject": f.subject, "predicate": f.predicate,
             "object": f.object, "confidence": f.confidence}
            for f in facts
        ]

    def apply_token_budget(
        self, texts: list[str], scores: list[float] | None = None, budget: int = 2000,
    ) -> list[str]:
        """token 预算裁剪。"""
        from septmuse.retrieval.token_budget import TokenBudget

        return TokenBudget(budget=budget).fit_texts(texts, scores)

    def redact(self, text: str) -> str:
        """隐私脱敏。"""
        from septmuse.capture.sanitize import PrivacyFilter

        return PrivacyFilter().redact(text)

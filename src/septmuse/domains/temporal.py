#  Copyright 2026 The sonhhxg0529 Authors. All Rights Reserved.
"""TemporalModule — 时态领域模块（search_at / search_interval / search_natural）。"""

from __future__ import annotations

from typing import Any

from septmuse.embedders.base import Embedder
from septmuse.llms.base import LLM
from septmuse.storage.base import MemoryStore, TemporalStore


class TemporalModule:
    """时态领域模块 — 双时态查询 + 自然语言时间抽取。"""

    def __init__(
        self,
        store: MemoryStore,
        embedder: Embedder,
        llm: LLM | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.llm = llm

    def search_at(
        self, reference_time: str, query: str, *, user_id: str, top_k: int = 5
    ) -> list[dict[str, Any]]:
        """时态查询: 在 reference_time 为真的记忆。"""
        if not isinstance(self.store, TemporalStore):
            return []
        emb = self.embedder.embed(query)
        results = self.store.search_at(reference_time, emb, user_id=user_id, top_k=top_k)
        return [
            {
                "id": r["id"],
                "memory": r["memory"],
                "score": r.get("score", 0.0),
                "valid_at": r.get("valid_at"),
                "invalid_at": r.get("invalid_at"),
            }
            for r in results
        ]

    def search_interval(
        self, start: str, end: str, query: str, *, user_id: str, top_k: int = 5
    ) -> list[dict[str, Any]]:
        """区间时态查询: [start, end) 内为真的记忆。"""
        from septmuse.retrieval.temporal import TemporalRetriever

        retriever = TemporalRetriever(self.store, self.embedder, self.llm)
        return retriever.search_interval(start, end, query, user_id=user_id)

    def search_natural(
        self, query: str, *, user_id: str, top_k: int = 5
    ) -> list[dict[str, Any]]:
        """自然语言时态查询: 先抽时间区间 → 时态过滤 → 回退普通检索。"""
        from septmuse.retrieval.temporal import TemporalRetriever

        retriever = TemporalRetriever(self.store, self.embedder, self.llm)
        return retriever.search_natural(query, user_id=user_id)

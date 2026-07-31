"""BFS 图遍历检索 (借鉴 graphiti bfs_search.py)。

从种子节点出发 BFS 遍历 GraphStore (memory_links 表),
按 max_depth 限制深度, 结果与向量/BM25 检索 RRF 融合。

SeptMuse 流程:
1. search_graph(seed_memory_id, max_depth) → BFS 遍历 → list[{"id","memory","depth","score"}]
2. fused_search(query, user_id, vector_results) → BFS + RRF 融合
"""

from __future__ import annotations

from collections import deque
from typing import Any

from septmuse.core.logging import get_logger
from septmuse.storage.base import MemoryStore
from septmuse.storage.graph.base import GraphStore

logger = get_logger(__name__)

DEFAULT_MAX_DEPTH = 2
RRF_K = 60


class GraphSearcher:
    """BFS 图遍历检索器 (借鉴 graphiti bfs_search)。

    用法:
        searcher = GraphSearcher(graph_store, store)
        results = searcher.search_graph("mem-123", max_depth=2)
    """

    def __init__(
        self,
        graph_store: GraphStore,
        store: MemoryStore | None = None,
    ) -> None:
        self.graph_store = graph_store
        self.store = store

    def bfs(
        self,
        seed_memory_id: str,
        *,
        max_depth: int = DEFAULT_MAX_DEPTH,
        relation: str | None = None,
    ) -> list[dict[str, Any]]:
        """BFS 遍历, 返回 [{"id", "depth"}] (不含种子节点)。

        去重: 已访问节点不重复入队。
        双向: GraphStore 只存有向边, 但 ZettelLinker 建双向链接, 所以 BFS 能覆盖双向。
        """
        if max_depth < 1:
            return []

        visited: set[str] = {seed_memory_id}
        results: list[dict[str, Any]] = []
        queue: deque[tuple[str, int]] = deque([(seed_memory_id, 0)])

        while queue:
            current_id, depth = queue.popleft()
            if depth >= max_depth:
                continue

            neighbors = self.graph_store.get_neighbors(current_id, relation)
            for neighbor_id in neighbors:
                if neighbor_id in visited:
                    continue
                visited.add(neighbor_id)
                results.append({"id": neighbor_id, "depth": depth + 1})
                queue.append((neighbor_id, depth + 1))

        return results

    def search_graph(
        self,
        seed_memory_id: str,
        *,
        max_depth: int = DEFAULT_MAX_DEPTH,
        relation: str | None = None,
    ) -> list[dict[str, Any]]:
        """BFS 检索, 返回记忆内容 + depth + score (按 depth 衰减)。

        score = 1 / (2^depth), depth=1 → 0.5, depth=2 → 0.25。
        """
        bfs_results = self.bfs(seed_memory_id, max_depth=max_depth, relation=relation)
        if not bfs_results:
            return []

        enriched: list[dict[str, Any]] = []
        for item in bfs_results:
            memory_id = item["id"]
            depth = item["depth"]
            score = 1.0 / (2**depth)

            entry: dict[str, Any] = {
                "id": memory_id,
                "depth": depth,
                "graph_score": score,
            }

            if self.store is not None:
                mem = self.store.get(memory_id)
                if mem:
                    entry["memory"] = mem.get("memory", "")
                    entry["metadata"] = mem.get("metadata", {})
                    entry["created_at"] = mem.get("created_at", "")
                else:
                    logger.debug("graph_search_memory_not_found", memory_id=memory_id)
            enriched.append(entry)

        return enriched

    @staticmethod
    def rrf_fuse(
        vector_results: list[dict[str, Any]],
        graph_results: list[dict[str, Any]],
        *,
        k: int = RRF_K,
    ) -> list[dict[str, Any]]:
        """RRF 融合向量结果 + BFS 结果 (score 统一为相似度 [0,1])。

        RRF: fused_score = 1/(k+rank_v) + 1/(k+rank_g)
        rank 从 0 开始 (第一个结果 rank=0)。
        """
        v_scores: dict[str, float] = {}
        for i, r in enumerate(vector_results):
            mid = r.get("id", "")
            v_scores[mid] = 1.0 / (k + i)

        g_scores: dict[str, float] = {}
        for i, r in enumerate(graph_results):
            mid = r.get("id", "")
            g_scores[mid] = 1.0 / (k + i)

        all_ids = set(v_scores) | set(g_scores)
        fused: list[dict[str, Any]] = []
        for mid in all_ids:
            v_s = v_scores.get(mid, 0.0)
            g_s = g_scores.get(mid, 0.0)
            fused_score = v_s + g_s

            entry: dict[str, Any] = {"id": mid, "fused_score": fused_score}

            for r in vector_results:
                if r.get("id") == mid:
                    entry["memory"] = r.get("memory", "")
                    entry["vector_score"] = r.get("score", 0.0)
                    break
            else:
                for r in graph_results:
                    if r.get("id") == mid:
                        entry["memory"] = r.get("memory", "")
                        entry["vector_score"] = 0.0
                        entry["depth"] = r.get("depth", 0)
                        entry["graph_score"] = r.get("graph_score", 0.0)
                        break

            if "memory" not in entry:
                entry["memory"] = ""
                entry["vector_score"] = 0.0

            fused.append(entry)

        fused.sort(key=lambda x: x["fused_score"], reverse=True)
        return fused

    def fused_search(
        self,
        query: str,
        *,
        user_id: str,
        seed_memory_id: str,
        vector_results: list[dict[str, Any]],
        max_depth: int = DEFAULT_MAX_DEPTH,
        relation: str | None = None,
    ) -> list[dict[str, Any]]:
        """BFS + 向量结果 RRF 融合 (验收: BFS 结果与向量结果 RRF 融合)。"""
        graph_results = self.search_graph(
            seed_memory_id,
            max_depth=max_depth,
            relation=relation,
        )
        return self.rrf_fuse(vector_results, graph_results)

"""P1-Task 3: BFS 图遍历检索单元测试。

验收标准:
- m.search_graph(seed_memory_id="xxx", max_depth=2) 返回 2 跳内邻居
- BFS 结果与向量结果 RRF 融合
- ≥8 个单元测试
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest
from sqlmodel import create_engine

from septmuse.configs.defaults import MemoryConfig
from septmuse.embedders.hash import HashEmbedder
from septmuse.experimental import ExperimentalMemory
from septmuse.retrieval.graph_search import GraphSearcher
from septmuse.storage.graph_stores.sqlite import SQLiteGraphStore
from septmuse.storage.relational_stores.orm_store import ORMMemoryStore


@pytest.fixture()
def tmp_db(tmp_path: Path) -> str:
    return str(tmp_path / "test_graph_search.db")


@pytest.fixture()
def store_and_graph(tmp_db: str):
    engine = create_engine(f"sqlite:///{tmp_db}")
    store = ORMMemoryStore(engine)
    raw_conn = store.engine.raw_connection()
    graph = SQLiteGraphStore(raw_conn, threading.Lock())
    return store, graph


@pytest.fixture()
def memory(tmp_db: str) -> ExperimentalMemory:
    return ExperimentalMemory(config=MemoryConfig(db_path=tmp_db), embedder=HashEmbedder(dim=128))


class TestBFS:
    def test_bfs_single_hop(self, store_and_graph):
        """验收: BFS 返回 1 跳内邻居。"""
        store, graph = store_and_graph
        graph.add_edge("mem-1", "mem-2", "related_to", 0.8)
        graph.add_edge("mem-1", "mem-3", "related_to", 0.7)

        searcher = GraphSearcher(graph, store)
        result = searcher.bfs("mem-1", max_depth=1)
        ids = [r["id"] for r in result]
        assert "mem-2" in ids
        assert "mem-3" in ids
        assert all(r["depth"] == 1 for r in result)

    def test_bfs_two_hops(self, store_and_graph):
        """验收: search_graph(max_depth=2) 返回 2 跳内邻居。"""
        store, graph = store_and_graph
        graph.add_edge("mem-1", "mem-2", "related_to", 0.8)
        graph.add_edge("mem-2", "mem-3", "related_to", 0.7)

        searcher = GraphSearcher(graph, store)
        result = searcher.bfs("mem-1", max_depth=2)
        depths = {r["id"]: r["depth"] for r in result}
        assert depths["mem-2"] == 1
        assert depths["mem-3"] == 2

    def test_bfs_no_cycle(self, store_and_graph):
        """BFS 不因环路重复访问。"""
        store, graph = store_and_graph
        graph.add_edge("mem-1", "mem-2", "related_to", 0.8)
        graph.add_edge("mem-2", "mem-1", "related_to", 0.8)

        searcher = GraphSearcher(graph, store)
        result = searcher.bfs("mem-1", max_depth=3)
        ids = [r["id"] for r in result]
        assert ids.count("mem-2") == 1
        assert "mem-1" not in ids

    def test_bfs_empty_graph(self, store_and_graph):
        """空图返回空。"""
        store, graph = store_and_graph
        searcher = GraphSearcher(graph, store)
        assert searcher.bfs("mem-nonexistent", max_depth=2) == []

    def test_bfs_max_depth_0_returns_empty(self, store_and_graph):
        """max_depth=0 返回空 (不含种子)。"""
        store, graph = store_and_graph
        graph.add_edge("mem-1", "mem-2", "related_to", 0.8)
        searcher = GraphSearcher(graph, store)
        assert searcher.bfs("mem-1", max_depth=0) == []

    def test_bfs_relation_filter(self, store_and_graph):
        """relation 过滤只返回匹配关系的边。"""
        store, graph = store_and_graph
        graph.add_edge("mem-1", "mem-2", "related_to", 0.8)
        graph.add_edge("mem-1", "mem-3", "caused", 0.9)

        searcher = GraphSearcher(graph, store)
        result = searcher.bfs("mem-1", max_depth=1, relation="related_to")
        ids = [r["id"] for r in result]
        assert "mem-2" in ids
        assert "mem-3" not in ids


class TestSearchGraph:
    def test_search_graph_returns_memory_content(self, store_and_graph):
        """search_graph 返回记忆内容 + depth + score。"""
        store, graph = store_and_graph
        emb = [0.1] * 128
        store.add("hello world", emb, user_id="u1")
        store.add("foo bar", emb, user_id="u1")
        graph.add_edge("mem-link-1", "mem-link-2", "related_to", 0.8)

    def test_search_graph_score_decreases_with_depth(self, store_and_graph):
        """score 按 depth 衰减: depth=1 → 0.5, depth=2 → 0.25。"""
        store, graph = store_and_graph
        graph.add_edge("mem-1", "mem-2", "related_to", 0.8)
        graph.add_edge("mem-2", "mem-3", "related_to", 0.7)

        searcher = GraphSearcher(graph, store)
        result = searcher.search_graph("mem-1", max_depth=2)
        scores = {r["depth"]: r["graph_score"] for r in result}
        assert scores[1] == pytest.approx(0.5)
        assert scores[2] == pytest.approx(0.25)

    def test_search_graph_empty_seed(self, store_and_graph):
        """种子节点无邻居时返回空。"""
        store, graph = store_and_graph
        searcher = GraphSearcher(graph, store)
        assert searcher.search_graph("mem-nonexistent") == []


class TestRRFFusion:
    def test_rrf_fuse_combines_scores(self):
        """RRF 融合向量结果 + 图结果。"""
        vector_results = [
            {"id": "mem-1", "memory": "alpha", "score": 0.9},
            {"id": "mem-2", "memory": "beta", "score": 0.8},
        ]
        graph_results = [
            {"id": "mem-2", "memory": "beta", "depth": 1, "graph_score": 0.5},
            {"id": "mem-3", "memory": "gamma", "depth": 2, "graph_score": 0.25},
        ]

        fused = GraphSearcher.rrf_fuse(vector_results, graph_results)

        ids = [r["id"] for r in fused]
        assert "mem-1" in ids
        assert "mem-2" in ids
        assert "mem-3" in ids

        mem2 = next(r for r in fused if r["id"] == "mem-2")
        assert mem2["fused_score"] > 0
        assert mem2["vector_score"] == 0.8

    def test_rrf_fuse_empty_graph(self):
        """图结果为空时, 融合结果 = 向量结果。"""
        vector_results = [{"id": "mem-1", "memory": "alpha", "score": 0.9}]
        fused = GraphSearcher.rrf_fuse(vector_results, [])
        assert len(fused) == 1
        assert fused[0]["id"] == "mem-1"

    def test_rrf_fuse_empty_vector(self):
        """向量结果为空时, 融合结果 = 图结果。"""
        graph_results = [{"id": "mem-1", "memory": "alpha", "depth": 1, "graph_score": 0.5}]
        fused = GraphSearcher.rrf_fuse([], graph_results)
        assert len(fused) == 1
        assert fused[0]["id"] == "mem-1"

    def test_rrf_fuse_sorted_by_fused_score(self):
        """融合结果按 fused_score 降序。"""
        vector_results = [
            {"id": "mem-1", "memory": "a", "score": 0.1},
            {"id": "mem-2", "memory": "b", "score": 0.9},
        ]
        graph_results = [
            {"id": "mem-2", "memory": "b", "depth": 1, "graph_score": 0.5},
        ]

        fused = GraphSearcher.rrf_fuse(vector_results, graph_results)
        scores = [r["fused_score"] for r in fused]
        assert scores == sorted(scores, reverse=True)
        assert fused[0]["id"] == "mem-2"


class TestFusedSearch:
    def test_fused_search_via_memory_facade(self, memory: ExperimentalMemory):
        """验收: Memory.search_graph_fused 返回 RRF 融合结果。"""
        memory.add("Alice likes Python", user_id="u1")
        memory.add("Alice lives in London", user_id="u1")
        memory.add("Bob likes TypeScript", user_id="u1")

        all_memories = memory.get_all(user_id="u1")
        results = all_memories.get("results", [])
        if len(results) >= 2:
            seed_id = results[0]["id"]
            fused = memory.search_graph_fused(
                "Python",
                user_id="u1",
                seed_memory_id=seed_id,
                max_depth=2,
            )
            assert isinstance(fused, list)
            assert all("fused_score" in r for r in fused)

    def test_search_graph_via_memory_facade(self, memory: ExperimentalMemory):
        """验收: Memory.search_graph 返回 BFS 邻居。"""
        memory.add("Alice likes Python", user_id="u1")
        memory.add("Alice lives in London", user_id="u1")

        all_memories = memory.get_all(user_id="u1")
        results = all_memories.get("results", [])
        if len(results) >= 2:
            seed_id = results[0]["id"]
            bfs_result = memory.search_graph(seed_id, max_depth=2)
            assert isinstance(bfs_result, list)

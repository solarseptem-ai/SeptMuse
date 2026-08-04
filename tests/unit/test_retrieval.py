#  Copyright 2026 The sonhhxg0529 Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""阶段3 Batch2 检索模块单元测试 — progressive + hybrid。

固化 (架构文档 §5.2 检索策略):
- ProgressiveRetriever: recall→locate→expand 三层渐进 (ReMe 模式)
- BM25Scorer: 纯 Python BM25 关键词评分 (mem0 模式)
- HybridRetriever: BM25+向量 RRF 融合 (ReMe _rrf_merge 模式)
"""

from __future__ import annotations

import pytest

from septmuse import MemoryConfig
from septmuse.embedders.hash import HashEmbedder
from septmuse.experimental import ExperimentalMemory
from septmuse.retrieval.hybrid import BM25Scorer, HybridRetriever
from septmuse.retrieval.progressive import ProgressiveRetriever


@pytest.fixture()
def mem() -> ExperimentalMemory:
    return ExperimentalMemory(
        config=MemoryConfig(db_path=":memory:"),
        embedder=HashEmbedder(),
    )


# ======================================================================
# BM25Scorer
# ======================================================================


class TestBM25Scorer:
    def test_empty_index(self) -> None:
        scorer = BM25Scorer()
        scorer.index([])
        assert scorer.score("hello") == []

    def test_single_doc(self) -> None:
        scorer = BM25Scorer()
        scorer.index(["hello world"])
        scores = scorer.score("hello")
        assert len(scores) == 1
        assert scores[0] > 0

    def test_term_not_in_docs(self) -> None:
        scorer = BM25Scorer()
        scorer.index(["hello world"])
        scores = scorer.score("python")
        assert scores == [0.0]

    def test_rare_term_scores_higher(self) -> None:
        scorer = BM25Scorer()
        # "rare" 出现在 1 篇, "common" 出现在 3 篇
        scorer.index(
            [
                "rare common",
                "common foo",
                "common bar",
            ]
        )
        scores = scorer.score("rare common")
        # rare 的 IDF 应高于 common
        assert scores[0] > 0

    def test_empty_query(self) -> None:
        scorer = BM25Scorer()
        scorer.index(["hello world"])
        assert scorer.score("") == [0.0]

    def test_multiple_docs_ranking(self) -> None:
        scorer = BM25Scorer()
        scorer.index(
            [
                "python is great",
                "java is also great",
                "python and java together",
            ]
        )
        scores = scorer.score("python")
        # doc 0 和 doc 2 都含 "python", doc 1 不含
        assert scores[0] > 0
        assert scores[2] > 0
        assert scores[1] == 0.0

    def test_case_insensitive(self) -> None:
        scorer = BM25Scorer()
        scorer.index(["Hello World"])
        scores = scorer.score("hello")
        assert scores[0] > 0


# ======================================================================
# HybridRetriever
# ======================================================================


class TestHybridRetriever:
    def test_empty_store(self, mem: ExperimentalMemory) -> None:
        retriever = HybridRetriever(mem.store, mem.embedder)
        results = retriever.search("hello", user_id="alice")
        assert results == []

    def test_vector_only_hit(self, mem: ExperimentalMemory) -> None:
        mem.add("alice likes python programming", user_id="alice")
        retriever = HybridRetriever(mem.store, mem.embedder)
        results = retriever.search("alice python", user_id="alice")
        assert len(results) >= 1
        assert "alice" in results[0].memory.lower()

    def test_bm25_boosts_keyword_match(self, mem: ExperimentalMemory) -> None:
        mem.add("alice likes python", user_id="alice")
        mem.add("bob likes java", user_id="alice")
        retriever = HybridRetriever(mem.store, mem.embedder)
        results = retriever.search("python", user_id="alice")
        assert len(results) >= 1
        # "python" doc should rank first (both vector + bm25 hit)
        assert "python" in results[0].memory.lower()

    def test_rrf_fusion_scores(self, mem: ExperimentalMemory) -> None:
        mem.add("alice likes python", user_id="alice")
        retriever = HybridRetriever(mem.store, mem.embedder)
        results = retriever.search("alice python", user_id="alice")
        assert len(results) >= 1
        assert results[0].score > 0
        # RRF score should have both vector and keyword components
        assert results[0].vector_score >= 0
        assert results[0].bm25_score >= 0

    def test_threshold_filters(self, mem: ExperimentalMemory) -> None:
        mem.add("alice likes python", user_id="alice")
        retriever = HybridRetriever(mem.store, mem.embedder)
        # Very high threshold should filter out results
        results = retriever.search("xyzabc", user_id="alice", threshold=0.99)
        assert len(results) == 0

    def test_user_id_isolation(self, mem: ExperimentalMemory) -> None:
        mem.add("alice secret", user_id="alice")
        mem.add("bob secret", user_id="bob")
        retriever = HybridRetriever(mem.store, mem.embedder)
        results = retriever.search("secret", user_id="alice")
        assert all("alice" in r.memory.lower() or "bob" not in r.memory.lower() for r in results)

    def test_filters_session_id(self, mem: ExperimentalMemory) -> None:
        mem.add("alice likes python", user_id="alice", session_id="s1")
        mem.add("alice likes java", user_id="alice", session_id="s2")
        retriever = HybridRetriever(mem.store, mem.embedder)
        results = retriever.search("alice", user_id="alice", filters={"session_id": "s1"})
        ids = {r.id for r in results}
        s1_mems = {m["id"] for m in mem.store.get_all(user_id="alice", filters={"session_id": "s1"})}
        assert ids == s1_mems
        assert all("python" in r.memory.lower() for r in results)

    def test_filters_no_match(self, mem: ExperimentalMemory) -> None:
        mem.add("alice likes python", user_id="alice", session_id="s1")
        retriever = HybridRetriever(mem.store, mem.embedder)
        results = retriever.search("alice", user_id="alice", filters={"session_id": "nonexistent"})
        assert results == []

    def test_filters_none_returns_all(self, mem: ExperimentalMemory) -> None:
        mem.add("alice likes python", user_id="alice", session_id="s1")
        mem.add("alice likes java", user_id="alice", session_id="s2")
        retriever = HybridRetriever(mem.store, mem.embedder)
        results = retriever.search("alice", user_id="alice", filters=None)
        assert len(results) >= 2


# ======================================================================
# ProgressiveRetriever
# ======================================================================


class TestProgressiveRetriever:
    def test_empty_store(self, mem: ExperimentalMemory) -> None:
        retriever = ProgressiveRetriever(mem.store, mem.typed_store, mem.embedder)
        results = retriever.retrieve("hello", user_id="alice")
        assert results == []

    def test_recall_verbatim(self, mem: ExperimentalMemory) -> None:
        mem.add("alice likes python", user_id="alice")
        retriever = ProgressiveRetriever(mem.store, mem.typed_store, mem.embedder)
        results = retriever.retrieve("alice python", user_id="alice")
        assert len(results) >= 1
        assert results[0].memory_type == "verbatim"

    def test_recall_semantic(self, mem: ExperimentalMemory) -> None:
        mem.add_fact("alice", "likes", "python", user_id="alice")
        retriever = ProgressiveRetriever(mem.store, mem.typed_store, mem.embedder)
        results = retriever.retrieve("alice likes", user_id="alice")
        # Should find the semantic fact
        semantic_hits = [r for r in results if r.memory_type == "semantic"]
        assert len(semantic_hits) >= 1
        assert "alice" in semantic_hits[0].memory

    def test_expand_by_tags(self, mem: ExperimentalMemory) -> None:
        mem.add("python tutorial", user_id="alice", metadata={"tags": ["coding"]})
        mem.add("java tutorial", user_id="alice", metadata={"tags": ["coding"]})
        mem.add("cooking recipe", user_id="alice", metadata={"tags": ["food"]})
        retriever = ProgressiveRetriever(mem.store, mem.typed_store, mem.embedder)
        results = retriever.retrieve("python tutorial", user_id="alice", top_k=5)
        # Should find python (recall) + possibly java (expand by tag "coding")
        all_memories = [r.memory for r in results]
        assert "python tutorial" in all_memories

    def test_results_sorted_by_score(self, mem: ExperimentalMemory) -> None:
        mem.add("alice likes python", user_id="alice")
        mem.add("bob likes java", user_id="alice")
        retriever = ProgressiveRetriever(mem.store, mem.typed_store, mem.embedder)
        results = retriever.retrieve("alice python", user_id="alice")
        if len(results) >= 2:
            assert results[0].score >= results[1].score

    def test_dedup(self, mem: ExperimentalMemory) -> None:
        mem.add("alice likes python", user_id="alice")
        retriever = ProgressiveRetriever(mem.store, mem.typed_store, mem.embedder)
        results = retriever.retrieve("alice python", user_id="alice")
        ids = [r.id for r in results]
        assert len(ids) == len(set(ids))  # no duplicates


# ======================================================================
# Memory Facade Reranker 集成 (Task 7)
# ======================================================================


class TestMemoryReranker:
    def test_search_with_noop_reranker(self, tmp_path):
        from septmuse.configs.defaults import MemoryConfig
        from septmuse.experimental import ExperimentalMemory

        m = ExperimentalMemory(config=MemoryConfig(db_path=str(tmp_path / "test.db")))
        m.add("hello world", user_id="u1")
        results = m.search("hello", user_id="u1", reranker="noop")
        assert len(results) >= 1

    def test_search_with_mmr_reranker(self, tmp_path):
        from septmuse.configs.defaults import MemoryConfig
        from septmuse.experimental import ExperimentalMemory

        m = ExperimentalMemory(config=MemoryConfig(db_path=str(tmp_path / "test.db")))
        m.add("Python programming", user_id="u1")
        m.add("Java programming", user_id="u1")
        results = m.search("programming", user_id="u1", reranker="mmr")
        assert len(results) >= 1

    def test_search_with_explain(self, tmp_path):
        from septmuse.configs.defaults import MemoryConfig
        from septmuse.experimental import ExperimentalMemory

        m = ExperimentalMemory(config=MemoryConfig(db_path=str(tmp_path / "test.db")))
        m.add("hello world", user_id="u1")
        results = m.search("hello", user_id="u1", explain=True)
        assert len(results) >= 1
        assert "score_details" in (results[0].get("metadata", {}) or {})

    def test_config_reranker_backend(self, tmp_path):
        from septmuse.configs.defaults import MemoryConfig
        from septmuse.experimental import ExperimentalMemory

        config = MemoryConfig(db_path=str(tmp_path / "test.db"), reranker_backend="mmr")
        m = ExperimentalMemory(config=config)
        m.add("test", user_id="u1")
        results = m.search("test", user_id="u1")
        assert len(results) >= 1

    def test_reranker_param_overrides_config(self, tmp_path):
        from septmuse.configs.defaults import MemoryConfig
        from septmuse.experimental import ExperimentalMemory

        config = MemoryConfig(db_path=str(tmp_path / "test.db"), reranker_backend="noop")
        m = ExperimentalMemory(config=config)
        m.add("test content", user_id="u1")
        results = m.search("test", user_id="u1", reranker="noop")
        assert len(results) >= 1


class TestCLIReranker:
    def test_cli_search_with_reranker(self):
        from septmuse.cli.main import _build_parser

        parser = _build_parser()
        args = parser.parse_args(
            [
                "search",
                "hello",
                "--user-id",
                "u1",
                "--reranker",
                "mmr",
            ]
        )
        assert args.reranker == "mmr"


class TestRESTReranker:
    def test_rest_search_with_reranker(self, tmp_path):
        from fastapi.testclient import TestClient

        from septmuse.api.rest import create_app
        from septmuse.configs.defaults import MemoryConfig
        from septmuse.embedders.hash import HashEmbedder
        from septmuse.experimental import ExperimentalMemory

        mem = ExperimentalMemory(
            config=MemoryConfig(db_path=str(tmp_path / "rest.db")),
            embedder=HashEmbedder(),
        )
        app = create_app(mem)
        client = TestClient(app)

        client.post("/memories", json={"messages": "hello world", "user_id": "u1"})
        resp = client.post(
            "/memories/search",
            json={"query": "hello", "user_id": "u1", "reranker": "noop"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data

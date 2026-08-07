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
"""P0-Task 3: 三路并发检索降级测试。

验证:
- 单路异常不阻塞其他路
- 向量检索异常 → 返回空
- BM25 异常 → 降级候选集 BM25
- Entity boost 异常 → 跳过 boost
"""

from __future__ import annotations

import time

from septmuse.retrieval.hybrid import HybridRetriever


class _SlowEmbedder:
    """慢 embedder, 模拟向量检索超时。"""

    def embed(self, text: str) -> list[float]:
        time.sleep(10)  # 超过 SEARCH_TIMEOUT=5s
        return [1.0, 0.0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


class _SimpleEmbedder:
    def embed(self, text: str) -> list[float]:
        return [float(len(text)), 1.0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


class _MockStore:
    """可配置异常的 mock store。"""

    def __init__(
        self,
        memories: list[dict] | None = None,
        search_exc: Exception | None = None,
        keyword_exc: Exception | None = None,
    ) -> None:
        self._memories = memories or []
        self._search_exc = search_exc
        self._keyword_exc = keyword_exc

    def search(self, query_embedding, *, user_id, session_id=None, top_k=5, threshold=0.1, filters=None):
        if self._search_exc is not None:
            raise self._search_exc
        return [
            {"id": m["id"], "memory": m["memory"], "score": 0.5, "metadata": {}, "created_at": "2026-01-01"}
            for m in self._memories
        ]

    def keyword_search(self, query, *, user_id, top_k=5):
        if self._keyword_exc is not None:
            raise self._keyword_exc
        return [
            {"id": m["id"], "score": 0.3}
            for m in self._memories
            if m["memory"].lower().count(query.lower()) > 0
        ]


class TestConcurrentFallback:
    """三路并发降级测试。"""

    def test_vector_search_exception_returns_empty(self):
        """向量检索抛异常 → 返回空结果。"""
        store = _MockStore(
            memories=[{"id": "m0", "memory": "hello world"}],
            search_exc=RuntimeError("vector store down"),
        )
        retriever = HybridRetriever(store, _SimpleEmbedder())
        results = retriever.search("hello", user_id="u1")
        assert results == []

    def test_keyword_search_exception_fallbacks_to_candidate_bm25(self):
        """BM25 keyword_search 抛异常 → 降级到候选集 BM25 (仍返回结果)。"""
        store = _MockStore(
            memories=[{"id": "m0", "memory": "hello python world"}],
            keyword_exc=RuntimeError("keyword index down"),
        )
        retriever = HybridRetriever(store, _SimpleEmbedder())
        results = retriever.search("python", user_id="u1")
        assert len(results) >= 1
        # 降级 BM25 应该有分数
        assert results[0].bm25_score >= 0

    def test_vector_search_timeout_returns_empty(self):
        """向量检索超时 → 返回空结果 (不阻塞其他路)。"""
        store = _MockStore(memories=[{"id": "m0", "memory": "hello world"}])
        retriever = HybridRetriever(store, _SlowEmbedder())
        results = retriever.search("hello", user_id="u1")
        assert results == []

    def test_all_three_paths_succeed(self):
        """三路都成功 → 正常 RRF 融合结果。"""
        store = _MockStore(
            memories=[
                {"id": "m0", "memory": "alice likes python"},
                {"id": "m1", "memory": "bob likes java"},
            ]
        )
        retriever = HybridRetriever(store, _SimpleEmbedder())
        results = retriever.search("python", user_id="alice", top_k=5)
        assert len(results) >= 1
        assert results[0].score > 0

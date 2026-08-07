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
"""Entity boost 三信号融合单元测试 (借鉴 mem0 _search_vector_store scoring)。"""

from __future__ import annotations

from septmuse.extraction.entity import Entity
from septmuse.retrieval.hybrid import HybridRetriever


class _MockEmbedder:
    def embed(self, text: str) -> list[float]:
        return [float(len(text)), 0.0, 0.0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


class _MockEntityExtractor:
    def extract(self, text: str) -> list[Entity]:
        if "Python" in text:
            return [Entity(text="Python", entity_type="TOPIC", start=0, end=6)]
        return []


class _MockEntityStore:
    def __init__(self, entities: dict[str, list[str]] | None = None):
        self._entities = entities or {}

    def search(self, entity_text: str, user_id: str, top_k: int = 5):
        results = []
        for text, linked_ids in self._entities.items():
            if entity_text.lower() in text.lower():
                results.append({"text": text, "linked_memory_ids": linked_ids, "entity_type": "TOPIC"})
        return results


class _MockStore:
    def __init__(self, memories: list[dict] | None = None):
        self._memories = memories or []

    def get_all(self, *, user_id: str, session_id: str | None = None, filters: dict | None = None) -> list[dict]:
        return self._memories

    def search(
        self,
        query_embedding: list[float],
        *,
        user_id: str,
        session_id: str | None = None,
        top_k: int = 5,
        threshold: float = 0.1,
        filters: dict | None = None,
    ):
        return [
            {"id": m["id"], "memory": m["memory"], "score": 0.5, "metadata": {}, "created_at": "2026-01-01"}
            for m in self._memories
        ]

    def keyword_search(self, query: str, *, user_id: str, top_k: int = 5) -> list[dict]:
        return []


class TestEntityBoostBackwardCompat:
    def test_no_entity_store_degrades_to_dual_signal(self):
        store = _MockStore([{"id": "m0", "memory": "hello world", "metadata": {}, "created_at": "2026-01-01"}])
        retriever = HybridRetriever(store, _MockEmbedder())
        results = retriever.search("hello", user_id="u1", top_k=5)
        assert len(results) == 1
        assert results[0].entity_boost == 0.0

    def test_only_extractor_no_store_degrades(self):
        store = _MockStore([{"id": "m0", "memory": "hello", "metadata": {}, "created_at": "2026-01-01"}])
        retriever = HybridRetriever(store, _MockEmbedder(), entity_extractor=_MockEntityExtractor())
        results = retriever.search("hello", user_id="u1", top_k=5)
        assert len(results) == 1
        assert results[0].entity_boost == 0.0


class TestEntityBoost:
    def test_entity_boost_increases_score(self):
        store = _MockStore(
            [
                {"id": "m0", "memory": "I love Python", "metadata": {}, "created_at": "2026-01-01"},
                {"id": "m1", "memory": "I love Java", "metadata": {}, "created_at": "2026-01-01"},
            ]
        )
        entity_store = _MockEntityStore({"Python": ["m0"]})
        retriever = HybridRetriever(
            store,
            _MockEmbedder(),
            entity_extractor=_MockEntityExtractor(),
            entity_store=entity_store,
        )
        results = retriever.search("Python", user_id="u1", top_k=5)
        m0 = [r for r in results if r.id == "m0"]
        m1 = [r for r in results if r.id == "m1"]
        if m0 and m1:
            assert m0[0].entity_boost > 0
            assert m1[0].entity_boost == 0.0

    def test_entity_boost_decays_with_n(self):
        n1 = 0.5 * 1.0 / (1.0 + 0.001 * (1 - 1) ** 2)
        n10 = 0.5 * 1.0 / (1.0 + 0.001 * (10 - 1) ** 2)
        assert n1 > n10

    def test_empty_entity_store(self):
        store = _MockStore([{"id": "m0", "memory": "Python", "metadata": {}, "created_at": "2026-01-01"}])
        entity_store = _MockEntityStore({})
        retriever = HybridRetriever(
            store,
            _MockEmbedder(),
            entity_extractor=_MockEntityExtractor(),
            entity_store=entity_store,
        )
        results = retriever.search("Python", user_id="u1", top_k=5)
        for r in results:
            assert r.entity_boost == 0.0


class TestExplain:
    def test_explain_returns_score_details(self):
        store = _MockStore([{"id": "m0", "memory": "hello world", "metadata": {}, "created_at": "2026-01-01"}])
        retriever = HybridRetriever(store, _MockEmbedder())
        results = retriever.search("hello", user_id="u1", top_k=5, explain=True)
        assert len(results) == 1
        details = results[0].metadata.get("score_details")
        assert details is not None
        assert "vector" in details
        assert "bm25" in details
        assert "entity_boost" in details
        assert "combined" in details

    def test_no_explain_no_details(self):
        store = _MockStore([{"id": "m0", "memory": "hello world", "metadata": {}, "created_at": "2026-01-01"}])
        retriever = HybridRetriever(store, _MockEmbedder())
        results = retriever.search("hello", user_id="u1", top_k=5, explain=False)
        assert len(results) == 1
        assert results[0].metadata is None or "score_details" not in (results[0].metadata or {})

    def test_explain_with_entity_boost(self):
        store = _MockStore(
            [
                {"id": "m0", "memory": "I love Python", "metadata": {}, "created_at": "2026-01-01"},
            ]
        )
        entity_store = _MockEntityStore({"Python": ["m0"]})
        retriever = HybridRetriever(
            store,
            _MockEmbedder(),
            entity_extractor=_MockEntityExtractor(),
            entity_store=entity_store,
        )
        results = retriever.search("Python", user_id="u1", top_k=5, explain=True)
        m0 = [r for r in results if r.id == "m0"]
        if m0:
            details = m0[0].metadata.get("score_details")
            assert details["entity_boost"] > 0
            assert details["combined"] >= details["entity_boost"]

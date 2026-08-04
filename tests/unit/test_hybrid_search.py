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
"""MemoryStore.keyword_search + hybrid_search + RRF 融合测试。"""

from __future__ import annotations

from septmuse.storage.base import _rrf_fuse


def test_rrf_empty_inputs():
    assert _rrf_fuse([], []) == []


def test_rrf_vec_only():
    vec = [{"id": "m1", "memory": "a", "score": 0.9}]
    result = _rrf_fuse(vec, [], alpha=1.0)
    assert len(result) == 1
    assert result[0]["id"] == "m1"
    assert result[0]["score"] > 0


def test_rrf_kw_only():
    kw = [{"id": "m2", "memory": "b", "score": 0.8}]
    result = _rrf_fuse([], kw, alpha=0.0)
    assert len(result) == 1
    assert result[0]["id"] == "m2"


def test_rrf_fuses_and_reranks():
    vec = [
        {"id": "m1", "memory": "a", "score": 0.9},
        {"id": "m2", "memory": "b", "score": 0.7},
    ]
    kw = [
        {"id": "m2", "memory": "b", "score": 0.8},
        {"id": "m3", "memory": "c", "score": 0.6},
    ]
    result = _rrf_fuse(vec, kw, alpha=0.5)
    ids = [r["id"] for r in result]
    assert set(ids) == {"m1", "m2", "m3"}
    # m2 出现在两边, RRF 应该排第一
    assert ids[0] == "m2"


def test_rrf_alpha_pure_vec():
    vec = [{"id": "m1", "memory": "a", "score": 0.9}]
    kw = [{"id": "m2", "memory": "b", "score": 0.8}]
    result = _rrf_fuse(vec, kw, alpha=1.0)
    assert len(result) == 1
    assert result[0]["id"] == "m1"


def test_rrf_alpha_pure_kw():
    vec = [{"id": "m1", "memory": "a", "score": 0.9}]
    kw = [{"id": "m2", "memory": "b", "score": 0.8}]
    result = _rrf_fuse(vec, kw, alpha=0.0)
    assert len(result) == 1
    assert result[0]["id"] == "m2"


def test_rrf_preserves_metadata():
    vec = [{"id": "m1", "memory": "alpha", "score": 0.9, "metadata": {"k": "v"}}]
    result = _rrf_fuse(vec, [], alpha=1.0)
    assert result[0]["metadata"] == {"k": "v"}
    assert result[0]["memory"] == "alpha"


from septmuse.storage.base import MemoryStore  # noqa: E402


class _StubStore(MemoryStore):
    """最小 MemoryStore 实现, 用于测试 keyword_search/hybrid_search 默认行为。"""

    def add(self, content, embedding, *, user_id, agent_id=None, metadata=None):
        return "stub"

    def search(self, query_embedding, *, user_id, session_id=None, top_k=5, threshold=0.1, filters=None):
        return [{"id": "m1", "memory": "a", "score": 0.9, "metadata": {}, "created_at": "t"}]

    def get_all(self, *, user_id):
        return []

    def get(self, memory_id):
        return None

    def delete(self, memory_id):
        pass

    def update(self, memory_id, content, embedding, *, metadata=None):
        return True

    def get_history(self, memory_id):
        return []

    def close(self):
        pass

    def list_agents(self, user_id):
        return []

    def list_users(self, agent_id):
        return []

    def get_shared_memories(self, user_id, limit=100):
        return []


def test_keyword_search_default_returns_empty():
    store = _StubStore()
    assert store.keyword_search("query", user_id="u") == []


def test_hybrid_search_default_uses_search_and_rrf():
    store = _StubStore()
    results = store.hybrid_search("query", [0.1, 0.2], user_id="u", top_k=5)
    # StubStore.search 返回 m1, keyword_search 返回 [], RRF 融合后应只有 m1
    assert len(results) == 1
    assert results[0]["id"] == "m1"

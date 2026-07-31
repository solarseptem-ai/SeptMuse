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
"""Reranker 单元测试 (借鉴 mem0 test_rerankers + graphiti reranker tests)。"""

from __future__ import annotations

import pytest

from septmuse.retrieval.hybrid import HybridResult
from septmuse.retrieval.reranker import (
    CrossEncoderReranker,
    LLMReranker,
    MMRReranker,
    NoopReranker,
    Reranker,
    _resolve_reranker,
)


def _make_results(n: int = 5) -> list[HybridResult]:
    return [
        HybridResult(
            id=f"m{i}",
            memory=f"memory content {i}",
            score=0.5 - i * 0.1,
            vector_score=0.5 - i * 0.1,
            bm25_score=0.0,
            metadata={},
            created_at="2026-01-01",
        )
        for i in range(n)
    ]


class TestNoopReranker:
    def test_passthrough_preserves_order(self):
        results = _make_results(3)
        reranker = NoopReranker()
        out = reranker.rerank("query", results)
        assert len(out) == 3
        assert [r.id for r in out] == ["m0", "m1", "m2"]

    def test_top_k_truncation(self):
        results = _make_results(5)
        reranker = NoopReranker()
        out = reranker.rerank("query", results, top_k=2)
        assert len(out) == 2

    def test_empty_input(self):
        reranker = NoopReranker()
        out = reranker.rerank("query", [])
        assert out == []

    def test_scores_unchanged(self):
        results = _make_results(3)
        reranker = NoopReranker()
        out = reranker.rerank("query", results)
        for orig, reranked in zip(results, out, strict=True):
            assert orig.score == reranked.score


class TestRerankerABC:
    def test_noop_is_subclass_of_reranker(self):
        assert issubclass(NoopReranker, Reranker)

    def test_reranker_is_abstract(self):
        assert getattr(Reranker.rerank, "__isabstractmethod__", False)


class TestResolveReranker:
    def test_noop_backend(self):
        r = _resolve_reranker("noop")
        assert isinstance(r, NoopReranker)

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown reranker"):
            _resolve_reranker("nonexistent")

    def test_default_is_noop(self):
        r = _resolve_reranker()
        assert isinstance(r, NoopReranker)


class TestMMRReranker:
    def test_dedup_high_similarity(self):
        """相似度 >0.9 的结果只保留一个。"""

        # Mock embedder: 所有文本返回相同向量 → 相似度=1.0
        class MockEmbedder:
            def embed(self, text: str) -> list[float]:
                return [1.0, 0.0, 0.0]

        results = [
            HybridResult(id="m0", memory="doc A", score=0.9, vector_score=0.9),
            HybridResult(id="m1", memory="doc B", score=0.8, vector_score=0.8),
        ]
        reranker = MMRReranker(embedder=MockEmbedder(), lambda_param=0.5)
        out = reranker.rerank("query", results, top_k=2)
        # MMR 会选择第一个, 然后第二个与第一个完全相似 → 被去重
        assert len(out) == 1

    def test_lambda_maximizes_relevance(self):
        """lambda=1.0 时最大化相关性, 不考虑多样性。"""

        class MockEmbedder:
            def embed(self, text: str) -> list[float]:
                # query 和 doc0 同向量, doc1 不同
                if "A" in text or "query" in text:
                    return [1.0, 0.0]
                return [0.0, 1.0]

        results = [
            HybridResult(id="m0", memory="doc A", score=0.5),
            HybridResult(id="m1", memory="doc B", score=0.9),
        ]
        reranker = MMRReranker(embedder=MockEmbedder(), lambda_param=1.0)
        out = reranker.rerank("query", results, top_k=2)
        assert len(out) == 2

    def test_empty_input(self):
        class MockEmbedder:
            def embed(self, text: str) -> list[float]:
                return [0.0]

        reranker = MMRReranker(embedder=MockEmbedder())
        out = reranker.rerank("query", [])
        assert out == []

    def test_top_k_truncation(self):
        class MockEmbedder:
            def embed(self, text: str) -> list[float]:
                # 正交向量避免触发去重, 专注测试 top_k 截断
                vec = [0.0] * 5
                tail = text.rsplit(" ", 1)[-1]
                vec[int(tail) if tail.isdigit() else 0] = 1.0
                return vec

        results = _make_results(5)
        reranker = MMRReranker(embedder=MockEmbedder(), lambda_param=0.7)
        out = reranker.rerank("query", results, top_k=2)
        assert len(out) == 2

    def test_preserves_fields(self):
        class MockEmbedder:
            def embed(self, text: str) -> list[float]:
                return [1.0, 0.0, 0.0]

        results = [
            HybridResult(id="m0", memory="doc A", score=0.5, metadata={"k": "v"}, created_at="2026-01-01"),
        ]
        reranker = MMRReranker(embedder=MockEmbedder(), lambda_param=1.0)
        out = reranker.rerank("query", results, top_k=1)
        assert out[0].id == "m0"
        assert out[0].memory == "doc A"
        assert out[0].metadata == {"k": "v"}
        assert out[0].created_at == "2026-01-01"


class TestCrossEncoderReranker:
    def test_degrades_without_onnxruntime(self, monkeypatch):
        """onnxruntime 不可用时降级为 Noop + 警告。"""
        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "onnxruntime":
                raise ImportError("No module named 'onnxruntime'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        reranker = CrossEncoderReranker(model_cache_dir=None)
        results = _make_results(3)
        out = reranker.rerank("query", results, top_k=3)
        # 降级为 Noop: 顺序不变
        assert len(out) == 3
        assert [r.id for r in out] == ["m0", "m1", "m2"]

    def test_empty_input(self):
        reranker = CrossEncoderReranker(model_cache_dir=None)
        out = reranker.rerank("query", [])
        assert out == []

    def test_resolve_cross_encoder(self):
        r = _resolve_reranker("cross_encoder", model_cache_dir=None)
        assert isinstance(r, CrossEncoderReranker)


class _MockLLM:
    """Mock LLM for testing (implements LLM ABC)."""

    def __init__(self, response: str = "0.8"):
        self._response = response
        self.calls: list[tuple[str, str]] = []

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        return self._response


class TestLLMReranker:
    def test_scores_from_llm(self):
        llm = _MockLLM("0.8")
        reranker = LLMReranker(llm=llm)
        results = _make_results(2)
        out = reranker.rerank("query", results, top_k=2)
        assert len(out) == 2
        assert out[0].score == 0.8
        assert len(llm.calls) == 2

    def test_no_llm_raises(self):
        reranker = LLMReranker(llm=None)
        with pytest.raises(ValueError, match="requires an LLM instance"):
            reranker.rerank("query", _make_results(1))

    def test_extract_score_clamps_high(self):
        llm = _MockLLM("2.5")
        reranker = LLMReranker(llm=llm)
        out = reranker.rerank("query", _make_results(1))
        assert out[0].score == 1.0

    def test_extract_score_clamps_low(self):
        llm = _MockLLM("-0.5")
        reranker = LLMReranker(llm=llm)
        out = reranker.rerank("query", _make_results(1))
        assert out[0].score == 0.0

    def test_extract_score_no_number(self):
        llm = _MockLLM("no number here")
        reranker = LLMReranker(llm=llm)
        out = reranker.rerank("query", _make_results(1))
        assert out[0].score == 0.5

    def test_empty_input(self):
        llm = _MockLLM()
        reranker = LLMReranker(llm=llm)
        out = reranker.rerank("query", [])
        assert out == []

    def test_preserves_fields(self):
        llm = _MockLLM("0.9")
        reranker = LLMReranker(llm=llm)
        results = [
            HybridResult(id="m0", memory="doc A", score=0.5, metadata={"k": "v"}, created_at="2026-01-01"),
        ]
        out = reranker.rerank("query", results, top_k=1)
        assert out[0].id == "m0"
        assert out[0].memory == "doc A"
        assert out[0].metadata == {"k": "v"}

    def test_resolve_llm(self):
        r = _resolve_reranker("llm", llm=_MockLLM())
        assert isinstance(r, LLMReranker)

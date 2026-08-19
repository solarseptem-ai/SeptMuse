"""SearchPipeline 单元测试 — mock retriever + mock reranker, 验证链路逻辑。

不构造 Memory, 隔离快速。测试覆盖:
- recipe 解析 (覆盖 hybrid/reranker/explain)
- forgetting 加权 (mock ForgettingRetriever)
- token_budget 裁剪
- inject_prompt 返回 dict
- hybrid vs 纯向量选择正确 retriever
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from septmuse.retrieval.hybrid import HybridResult
from septmuse.retrieval.search_pipeline import SearchPipeline

# ------------------------------------------------------------------
# Mock fixtures
# ------------------------------------------------------------------


class MockRetriever:
    """Mock retriever that returns canned HybridResults, records calls."""

    def __init__(self, results: list[HybridResult]) -> None:
        self._results = results
        self.calls: list[dict[str, Any]] = []

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
        self.calls.append(
            {
                "query": query,
                "user_id": user_id,
                "session_id": session_id,
                "top_k": top_k,
                "threshold": threshold,
                "explain": explain,
                "filters": filters,
            }
        )
        return self._results


@dataclass
class MockStrengthWeighted:
    """Mock StrengthWeightedResult."""

    id: str
    memory: str
    relevance: float
    strength: float
    final_score: float
    metadata: dict[str, Any]


class MockForgetting:
    """Mock ForgettingRetriever, records apply_strength calls."""

    def __init__(self) -> None:
        self.apply_calls: list[dict[str, Any]] = []

    def apply_strength(self, results: list[dict[str, Any]], *, user_id: str) -> list[MockStrengthWeighted]:
        self.apply_calls.append({"results": results, "user_id": user_id})
        return [
            MockStrengthWeighted(
                id=r["id"],
                memory=r["memory"],
                relevance=r["score"],
                strength=0.5,
                final_score=r["score"] * 0.7 + 0.5 * 0.3,
                metadata=r.get("metadata", {}),
            )
            for r in results
        ]


def _make_results(n: int = 3) -> list[HybridResult]:
    """Generate n canned HybridResult objects."""
    return [
        HybridResult(
            id=f"mem-{i}",
            memory=f"memory content {i}",
            score=1.0 - i * 0.2,
            vector_score=1.0 - i * 0.2,
            bm25_score=0.5,
            metadata={},
            created_at="2024-01-01T00:00:00",
        )
        for i in range(n)
    ]


def _make_pipeline(
    hybrid_retriever: MockRetriever | None = None,
    vector_retriever: MockRetriever | None = None,
    forgetting: MockForgetting | None = None,
    default_top_k: int = 5,
    default_threshold: float = 0.1,
) -> SearchPipeline:
    """Construct a SearchPipeline with mock dependencies."""
    if hybrid_retriever is None:
        hybrid_retriever = MockRetriever(_make_results())
    if vector_retriever is None:
        vector_retriever = MockRetriever(_make_results())
    return SearchPipeline(
        hybrid_retriever=hybrid_retriever,
        vector_retriever=vector_retriever,
        reranker_factory=lambda name: None,
        embedder=None,
        forgetting=forgetting,
        default_top_k=default_top_k,
        default_threshold=default_threshold,
    )


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


class TestSearchPipelineRecipe:
    """recipe 解析: 覆盖 hybrid/reranker/explain。"""

    def test_recipe_overrides_hybrid(self) -> None:
        """GRAPH_BFS recipe sets hybrid=False."""
        hybrid_ret = MockRetriever(_make_results())
        vector_ret = MockRetriever(_make_results())
        pipeline = _make_pipeline(hybrid_ret, vector_ret)
        pipeline.search("q", user_id="u", recipe="GRAPH_BFS")
        assert len(vector_ret.calls) == 1
        assert len(hybrid_ret.calls) == 0

    def test_recipe_overrides_reranker(self) -> None:
        """HYBRID_RRF_CROSS_ENCODER sets reranker='cross_encoder'."""
        reranker_names: list[str] = []
        hybrid_ret = MockRetriever(_make_results())
        pipeline = SearchPipeline(
            hybrid_retriever=hybrid_ret,
            vector_retriever=MockRetriever(_make_results()),
            reranker_factory=lambda name: reranker_names.append(name) or None,
            embedder=None,
        )
        pipeline.search("q", user_id="u", recipe="HYBRID_RRF_CROSS_ENCODER")
        assert "cross_encoder" in reranker_names


class TestSearchPipelineForgetting:
    """forgetting 加权: apply_strength 被调用, 结果包含 final_score/strength。"""

    def test_forgetting_calls_apply_strength(self) -> None:
        forgetting = MockForgetting()
        pipeline = _make_pipeline(forgetting=forgetting)
        results = pipeline.search("q", user_id="alice", forgetting=True)
        assert len(forgetting.apply_calls) == 1
        assert forgetting.apply_calls[0]["user_id"] == "alice"
        assert all("final_score" in r for r in results)
        assert all("strength" in r for r in results)

    def test_forgetting_none_logs_warning(self) -> None:
        """forgetting=True but component unavailable — returns results unchanged."""
        pipeline = _make_pipeline(forgetting=None)
        results = pipeline.search("q", user_id="u", forgetting=True)
        assert isinstance(results, list)
        assert len(results) == 3


class TestSearchPipelineTokenBudget:
    """token_budget 裁剪: 结果数减少。"""

    def test_token_budget_trims_results(self) -> None:
        """Small budget trims results (each memory ~14 chars, budget=10 trims all)."""
        pipeline = _make_pipeline()
        results = pipeline.search("q", user_id="u", token_budget=10)
        assert isinstance(results, list)
        # BudgetItem.estimate_tokens = len(text)//4, "memory content 0" = 16 chars = 4 tokens
        # budget=10 fits ~2 items, so at least 1 is trimmed
        assert len(results) < 3

    def test_no_token_budget_returns_all(self) -> None:
        pipeline = _make_pipeline()
        results = pipeline.search("q", user_id="u")
        assert len(results) == 3


class TestSearchPipelineInjectPrompt:
    """inject_prompt: 返回 dict with injected_prompt。"""

    def test_inject_prompt_returns_dict(self) -> None:
        pipeline = _make_pipeline()
        result = pipeline.search("q", user_id="u", inject_prompt=True)
        assert isinstance(result, dict)
        assert "results" in result
        assert "injected_prompt" in result
        assert "<memory>" in result["injected_prompt"]
        assert "memory content 0" in result["injected_prompt"]


class TestSearchPipelineRetrieverSelection:
    """hybrid vs 纯向量: 选择正确 retriever。"""

    def test_hybrid_uses_hybrid_retriever(self) -> None:
        hybrid_ret = MockRetriever(_make_results())
        vector_ret = MockRetriever(_make_results())
        pipeline = _make_pipeline(hybrid_ret, vector_ret)
        pipeline.search("q", user_id="u", hybrid=True)
        assert len(hybrid_ret.calls) == 1
        assert len(vector_ret.calls) == 0

    def test_pure_vector_uses_vector_retriever(self) -> None:
        hybrid_ret = MockRetriever(_make_results())
        vector_ret = MockRetriever(_make_results())
        pipeline = _make_pipeline(hybrid_ret, vector_ret)
        pipeline.search("q", user_id="u", hybrid=False)
        assert len(hybrid_ret.calls) == 0
        assert len(vector_ret.calls) == 1

    def test_threshold_hybrid_default_zero(self) -> None:
        """Hybrid branch uses 0.0 threshold default."""
        hybrid_ret = MockRetriever(_make_results())
        pipeline = _make_pipeline(hybrid_retriever=hybrid_ret)
        pipeline.search("q", user_id="u", hybrid=True)
        assert hybrid_ret.calls[0]["threshold"] == 0.0

    def test_threshold_pure_vector_uses_default(self) -> None:
        """Pure vector branch uses default_threshold."""
        vector_ret = MockRetriever(_make_results())
        pipeline = _make_pipeline(vector_retriever=vector_ret, default_threshold=0.42)
        pipeline.search("q", user_id="u", hybrid=False)
        assert vector_ret.calls[0]["threshold"] == 0.42

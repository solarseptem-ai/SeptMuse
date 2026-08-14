"""RecallPipeline 单元测试 — mock search_pipeline + mock meta, 验证链路逻辑。

不构造 Memory, 隔离快速。测试覆盖:
- search_pipeline.search 被调 with forgetting=True
- token_budget.fit 被调
- prompt 含 working_memory 前缀
- meta.route 被调
- 返回 dict 结构正确
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

from septmuse.retrieval.recall_pipeline import RecallPipeline

# ------------------------------------------------------------------
# Mock fixtures
# ------------------------------------------------------------------


@dataclass
class MockBudgetResult:
    items: list[Any] = None
    used_tokens: int = 0
    budget: int = 2000
    dropped: int = 0

    @property
    def texts(self) -> list[str]:
        return [item.text for item in (self.items or [])]


class MockTokenBudget:
    def __init__(self) -> None:
        self.fit_calls: list[dict[str, Any]] = []

    def fit(self, items: list[Any]) -> MockBudgetResult:
        self.fit_calls.append({"items": items})
        return MockBudgetResult(items=items, used_tokens=len(items) * 4)


class MockSearchPipeline:
    def __init__(self, results: list[dict[str, Any]] | None = None) -> None:
        self._results = results or [
            {"id": f"mem-{i}", "memory": f"content {i}", "score": 1.0 - i * 0.2} for i in range(3)
        ]
        self.search_calls: list[dict[str, Any]] = []

    def search(self, query: str, **kwargs: Any) -> list[dict[str, Any]]:
        self.search_calls.append({"query": query, **kwargs})
        return self._results


class MockWorkingMemory:
    def compile_to_prompt(self) -> str:
        return "<memory><block label='notes'>notes here</block></memory>"


class MockMeta:
    def __init__(self) -> None:
        self.route_calls: list[str] = []
        self.coverage_calls: list[str] = []
        self.strategy_calls: list[Any] = []

    def route(self, query: str) -> dict[str, Any]:
        self.route_calls.append(query)
        return {"namespaces": ["default"], "fallback": None, "scores": [0.9]}

    def analyze_coverage(self, *, user_id: str) -> dict[str, Any]:
        self.coverage_calls.append(user_id)
        return {"overall_score": 0.8, "weak_areas": [], "strong_areas": ["default"]}

    def adapt_strategy(self, report: Any) -> dict[str, Any]:
        self.strategy_calls.append(report)
        return {"overall_action": "maintain", "recommendations": []}


class MockTypedStore:
    def __init__(self, has_coverage: bool = False) -> None:
        self._has_coverage = has_coverage

    def get_all_facts(self, *, user_id: str) -> list[Any]:
        if not self._has_coverage:
            return []

        @dataclass
        class MockFact:
            tags: list[str]

        return [MockFact(tags=["meta", "coverage"])]


class MockSummarizer:
    def __init__(self) -> None:
        self.compress_calls: list[dict[str, Any]] = []

    def compress(self, *, user_id: str, mode: str = "static", buffer_size: int = 20) -> dict[str, Any]:
        self.compress_calls.append({"user_id": user_id, "mode": mode, "buffer_size": buffer_size})
        return {"compressed": True, "evicted": 5, "kept": 15, "summary_id": "sum-1"}


def _make_pipeline(
    search_pipeline: MockSearchPipeline | None = None,
    meta: MockMeta | None = None,
    token_budget: MockTokenBudget | None = None,
    working_memory: MockWorkingMemory | None = None,
    summarizer: MockSummarizer | None = None,
    typed_store: MockTypedStore | None = None,
) -> RecallPipeline:
    if search_pipeline is None:
        search_pipeline = MockSearchPipeline()
    return RecallPipeline(
        search_pipeline=search_pipeline,
        meta=meta,
        token_budget=token_budget,
        working_memory=working_memory,
        summarizer=summarizer,
        typed_store=typed_store,
    )


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


class TestRecallPipelineSearch:
    """search_pipeline.search 被调 with forgetting=True。"""

    def test_search_called_with_forgetting_true(self) -> None:
        search = MockSearchPipeline()
        pipeline = _make_pipeline(search_pipeline=search)
        pipeline.recall("query", user_id="alice")
        assert len(search.search_calls) == 1
        assert search.search_calls[0]["forgetting"] is True

    def test_search_called_with_recipe(self) -> None:
        search = MockSearchPipeline()
        pipeline = _make_pipeline(search_pipeline=search)
        pipeline.recall("query", user_id="alice", recipe="HYBRID_RRF_MMR")
        assert search.search_calls[0]["recipe"] == "HYBRID_RRF_MMR"

    def test_forgetting_false_passed_through(self) -> None:
        search = MockSearchPipeline()
        pipeline = _make_pipeline(search_pipeline=search)
        pipeline.recall("query", user_id="alice", forgetting=False)
        assert search.search_calls[0]["forgetting"] is False


class TestRecallPipelineBudget:
    """token_budget.fit 被调。"""

    def test_token_budget_fit_called(self) -> None:
        token_budget = MockTokenBudget()
        pipeline = _make_pipeline(token_budget=token_budget)
        result = pipeline.recall("query", user_id="alice")
        assert len(token_budget.fit_calls) == 1
        assert result["strategy"] is None  # no meta

    def test_no_token_budget_fallback(self) -> None:
        """token_budget=None → SimpleNamespace fallback, still works."""
        pipeline = _make_pipeline(token_budget=None)
        result = pipeline.recall("query", user_id="alice")
        assert "memories" in result
        assert "injected_prompt" in result


class TestRecallPipelinePrompt:
    """prompt 含 working_memory 前缀。"""

    def test_prompt_has_working_memory_prefix(self) -> None:
        wm = MockWorkingMemory()
        pipeline = _make_pipeline(working_memory=wm)
        result = pipeline.recall("query", user_id="alice")
        assert result["injected_prompt"].startswith("<memory><block")
        assert "<memory>\n" in result["injected_prompt"]
        assert "content 0" in result["injected_prompt"]

    def test_prompt_without_working_memory(self) -> None:
        pipeline = _make_pipeline(working_memory=None)
        result = pipeline.recall("query", user_id="alice")
        assert result["injected_prompt"].startswith("<memory>\n")


class TestRecallPipelineMeta:
    """meta.route 被调, strategy adaptation。"""

    def test_route_called(self) -> None:
        meta = MockMeta()
        pipeline = _make_pipeline(meta=meta, typed_store=MockTypedStore(has_coverage=False))
        result = pipeline.recall("query", user_id="alice")
        assert len(meta.route_calls) == 1
        assert result["route"] is not None
        assert result["strategy"] is None  # no coverage facts

    def test_strategy_called_with_coverage(self) -> None:
        meta = MockMeta()
        typed_store = MockTypedStore(has_coverage=True)
        pipeline = _make_pipeline(meta=meta, typed_store=typed_store)
        result = pipeline.recall("query", user_id="alice")
        assert len(meta.coverage_calls) == 1
        assert len(meta.strategy_calls) == 1
        assert result["strategy"] is not None

    def test_no_meta_skips_route_and_strategy(self) -> None:
        pipeline = _make_pipeline(meta=None, typed_store=None)
        result = pipeline.recall("query", user_id="alice")
        assert result["route"] is None
        assert result["strategy"] is None


class TestRecallPipelineCompress:
    """auto_compress + summarizer。"""

    def test_compress_called(self) -> None:
        summarizer = MockSummarizer()
        pipeline = _make_pipeline(summarizer=summarizer)
        pipeline.recall("query", user_id="alice", auto_compress=True)
        assert len(summarizer.compress_calls) == 1
        assert summarizer.compress_calls[0]["user_id"] == "alice"

    def test_compress_not_called_without_flag(self) -> None:
        summarizer = MockSummarizer()
        pipeline = _make_pipeline(summarizer=summarizer)
        pipeline.recall("query", user_id="alice")
        assert len(summarizer.compress_calls) == 0


class TestRecallPipelineReturnStructure:
    """返回 dict 结构正确。"""

    def test_return_has_all_keys(self) -> None:
        pipeline = _make_pipeline()
        result = pipeline.recall("query", user_id="alice")
        assert set(result.keys()) == {"memories", "injected_prompt", "route", "strategy", "summary", "conflicts"}

    def test_memories_is_list(self) -> None:
        pipeline = _make_pipeline()
        result = pipeline.recall("query", user_id="alice")
        assert isinstance(result["memories"], list)
        assert len(result["memories"]) == 3


def test_recall_returns_summary_and_conflicts():
    """recall 返回包含 summary 和 conflicts 字段。"""
    mock_search = MagicMock()
    mock_search.search.return_value = [
        {"id": "m1", "memory": "用户喜欢咖啡", "score": 0.9, "metadata": {}, "created_at": "2026-01-01T00:00:00Z"},
    ]
    pipeline = RecallPipeline(mock_search)
    result = pipeline.recall("咖啡", user_id="alice")

    assert "summary" in result
    assert "conflicts" in result
    assert isinstance(result["summary"], str)
    assert isinstance(result["conflicts"], list)


def test_recall_summary_fallback_no_llm():
    """无 LLM 时 summary 是降级拼接（非空）。"""
    mock_search = MagicMock()
    mock_search.search.return_value = [
        {"id": "m1", "memory": "用户喜欢咖啡", "score": 0.9, "metadata": {}, "created_at": "2026-01-01T00:00:00Z"},
    ]
    pipeline = RecallPipeline(mock_search)
    result = pipeline.recall("咖啡", user_id="alice")

    assert result["summary"] != ""
    assert "咖啡" in result["summary"]

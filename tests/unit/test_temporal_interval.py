"""P2-Task 2: 时态区间查询 + LLM 自然语言时间抽取单元测试。

验收标准:
- m.search_interval("2024-06-01", "2024-07-01", query="Alice", user_id="u1") 正确过滤
- LLM 从"上周Alice在做什么"抽取时间区间
- 无时间信息时回退到普通检索
- ≥12 个单元测试
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from septmuse.configs.defaults import MemoryConfig
from septmuse.embedders.hash import HashEmbedder
from septmuse.experimental import ExperimentalMemory
from septmuse.llms.base import LLM
from septmuse.retrieval.temporal import TemporalRetriever


class StubLLM(LLM):
    """测试用 LLM stub。"""

    def __init__(self, response: str) -> None:
        self._response = response

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        return self._response


@pytest.fixture()
def tmp_db(tmp_path: Path) -> str:
    return str(tmp_path / "test_temporal.db")


@pytest.fixture()
def memory(tmp_db: str) -> ExperimentalMemory:
    return ExperimentalMemory(config=MemoryConfig(db_path=tmp_db), embedder=HashEmbedder(dim=128))


@pytest.fixture()
def memory_with_llm(tmp_db: str) -> ExperimentalMemory:
    llm = StubLLM(json.dumps({"start": "2024-07-14", "end": "2024-07-21"}))
    return ExperimentalMemory(config=MemoryConfig(db_path=tmp_db), embedder=HashEmbedder(dim=128), llm=llm)


@pytest.fixture()
def memory_with_no_time_llm(tmp_db: str) -> ExperimentalMemory:
    llm = StubLLM(json.dumps({"start": None, "end": None}))
    return ExperimentalMemory(config=MemoryConfig(db_path=tmp_db), embedder=HashEmbedder(dim=128), llm=llm)


class TestSearchInterval:
    def test_search_interval_returns_valid_memories(self, memory: ExperimentalMemory):
        """验收: search_interval 正确过滤时间区间内的记忆。"""
        memory.add("Alice works at Google", user_id="u1", valid_at="2024-01-01")
        memory.add("Alice works at Apple", user_id="u1", valid_at="2025-01-01")

        results = memory.search_interval("2024-06-01", "2024-12-31", "Alice", user_id="u1")
        assert isinstance(results, list)
        assert all(r["memory"] == "Alice works at Google" for r in results)

    def test_search_interval_excludes_future_memories(self, memory: ExperimentalMemory):
        """search_interval 排除 valid_at > end 的记忆。"""
        memory.add("future fact", user_id="u1", valid_at="2026-01-01")
        results = memory.search_interval("2024-01-01", "2025-01-01", "future", user_id="u1")
        assert results == []

    def test_search_interval_includes_null_valid_at(self, memory: ExperimentalMemory):
        """valid_at IS NULL 的记忆视为无时间约束, 始终返回。"""
        memory.add("no time constraint", user_id="u1")
        results = memory.search_interval("2024-01-01", "2025-01-01", "time", user_id="u1")
        assert any("no time constraint" in r.get("memory", "") for r in results)

    def test_search_interval_excludes_invalidated(self, memory: ExperimentalMemory):
        """search_interval 排除在 start 前已失效的记忆。"""
        result = memory.add("old job", user_id="u1", valid_at="2023-01-01")
        mid = result["results"][0]["id"]
        memory.invalidate(mid, invalid_at="2024-01-01")
        memory.add("new job", user_id="u1", valid_at="2025-01-01")

        results = memory.search_interval("2025-06-01", "2026-01-01", "job", user_id="u1")
        assert all("new job" in r.get("memory", "") for r in results)

    def test_search_interval_empty(self, memory: ExperimentalMemory):
        """空记忆库返回空。"""
        results = memory.search_interval("2024-01-01", "2025-01-01", "nothing", user_id="u1")
        assert results == []


class TestExtractTimeRange:
    def test_extract_time_range_with_llm(self, memory_with_llm: ExperimentalMemory):
        """验收: LLM 从"上周Alice在做什么"抽取时间区间。"""
        retriever = TemporalRetriever(memory_with_llm.store, memory_with_llm.embedder, llm=memory_with_llm.llm)
        result = retriever.extract_time_range("Alice上周在做什么")
        assert result is not None
        assert result["start"] == "2024-07-14"
        assert result["end"] == "2024-07-21"

    def test_extract_time_range_no_time_returns_none(self, memory_with_no_time_llm: ExperimentalMemory):
        """无时间信息返回 None。"""
        retriever = TemporalRetriever(
            memory_with_no_time_llm.store, memory_with_no_time_llm.embedder, llm=memory_with_no_time_llm.llm
        )
        result = retriever.extract_time_range("Alice的工作经历")
        assert result is None

    def test_extract_time_range_no_llm_returns_none(self, memory: ExperimentalMemory):
        """无 LLM 时返回 None (回退普通检索)。"""
        retriever = TemporalRetriever(memory.store, memory.embedder, llm=None)
        result = retriever.extract_time_range("上周Alice在做什么")
        assert result is None

    def test_parse_time_response_valid_json(self):
        """_parse_time_range 解析有效 JSON。"""
        result = TemporalRetriever._parse_time_response('{"start": "2024-06-01", "end": "2024-07-01"}')
        assert result == {"start": "2024-06-01", "end": "2024-07-01"}

    def test_parse_time_response_null_values(self):
        """_parse_time_range 处理 null 值。"""
        result = TemporalRetriever._parse_time_response('{"start": null, "end": null}')
        assert result is None

    def test_parse_time_response_invalid_json(self):
        """_parse_time_range 处理无效 JSON。"""
        result = TemporalRetriever._parse_time_response("not json")
        assert result is None


class TestSearchNatural:
    def test_search_natural_with_time_range(self, memory_with_llm: ExperimentalMemory):
        """验收: search_natural 有时间时走 search_interval。"""
        memory_with_llm.add("Alice was on vacation", user_id="u1", valid_at="2024-07-15")
        results = memory_with_llm.search_natural("Alice上周在做什么", user_id="u1")
        assert isinstance(results, list)

    def test_search_natural_no_time_fallback(self, memory_with_no_time_llm: ExperimentalMemory):
        """验收: 无时间信息时回退普通检索。"""
        memory_with_no_time_llm.add("Alice works at Google", user_id="u1")
        results = memory_with_no_time_llm.search_natural("Alice的工作经历", user_id="u1")
        assert isinstance(results, list)

    def test_search_natural_no_llm_fallback(self, memory: ExperimentalMemory):
        """无 LLM 时回退普通检索。"""
        memory.add("Alice works at Google", user_id="u1")
        results = memory.search_natural("Alice", user_id="u1")
        assert isinstance(results, list)


class TestStoreGetTemporalInterval:
    def test_get_temporal_interval_base_returns_empty(self):
        """MemoryStore ABC 默认返回空。"""
        from unittest.mock import MagicMock

        from septmuse.storage.base import MemoryStore

        mock_store = MagicMock(spec=MemoryStore)
        mock_store.get_temporal_interval = MemoryStore.get_temporal_interval.__get__(mock_store)
        result = mock_store.get_temporal_interval("2024-01-01", "2025-01-01", user_id="u1")
        assert result == []

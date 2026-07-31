"""P0-Task 2: 三元组 LLM 联合抽取单元测试。

验收标准:
- extract_triplets("Alice works at Google") 返回 [("Alice", "works_at", "Google")]
- 孤儿实体被丢弃
- 无 LLM 时 fallback 到 EntityExtractor
- ≥8 个单元测试
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from septmuse.extraction.entity import Entity, RegexEntityExtractor
from septmuse.extraction.triplet import (
    Triplet,
    TripletExtractor,
    _drop_orphans,
    _normalize_relation,
    _parse_triplet_response,
    extract_triplets,
)
from septmuse.llms.base import LLM


class StubLLM(LLM):
    """测试用 LLM stub, 返回预设 JSON。"""

    def __init__(self, response: str) -> None:
        self._response = response

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        return self._response


def _make_llm_response(entities: list[str], edges: list[dict]) -> str:
    """构造 LLM JSON 响应字符串。"""
    return json.dumps({"entities": entities, "edges": edges})


class TestParseTripletResponse:
    def test_valid_json(self):
        raw = _make_llm_response(
            ["Alice", "Google"],
            [{"source": "Alice", "relation": "works_at", "target": "Google"}],
        )
        entities, edges = _parse_triplet_response(raw)
        assert entities == ["Alice", "Google"]
        assert len(edges) == 1
        assert edges[0]["source"] == "Alice"
        assert edges[0]["relation"] == "works_at"
        assert edges[0]["target"] == "Google"

    def test_markdown_code_block(self):
        raw = "```json\n" + _make_llm_response(["A", "B"], []) + "\n```"
        entities, edges = _parse_triplet_response(raw)
        assert entities == ["A", "B"]
        assert edges == []

    def test_invalid_json(self):
        entities, edges = _parse_triplet_response("not json at all")
        assert entities == []
        assert edges == []

    def test_empty_entities_and_edges(self):
        raw = _make_llm_response([], [])
        entities, edges = _parse_triplet_response(raw)
        assert entities == []
        assert edges == []

    def test_filters_incomplete_edges(self):
        raw = json.dumps(
            {
                "entities": ["A", "B"],
                "edges": [
                    {"source": "A", "relation": "", "target": "B"},
                    {"source": "", "relation": "x", "target": "B"},
                    {"source": "A", "relation": "x", "target": ""},
                    {"source": "A", "relation": "r", "target": "B"},
                ],
            }
        )
        entities, edges = _parse_triplet_response(raw)
        assert len(entities) == 2
        assert len(edges) == 1
        assert edges[0]["source"] == "A"
        assert edges[0]["relation"] == "r"
        assert edges[0]["target"] == "B"


class TestDropOrphans:
    def test_drops_orphan_entities(self):
        entities = ["Alice", "Google", "Orphan"]
        edges = [{"source": "Alice", "relation": "works_at", "target": "Google"}]
        result = _drop_orphans(entities, edges)
        assert result == ["Alice", "Google"]
        assert "Orphan" not in result

    def test_keeps_all_connected(self):
        entities = ["A", "B", "C"]
        edges = [
            {"source": "A", "relation": "x", "target": "B"},
            {"source": "B", "relation": "y", "target": "C"},
        ]
        result = _drop_orphans(entities, edges)
        assert set(result) == {"A", "B", "C"}

    def test_empty_edges_drops_all(self):
        entities = ["A", "B"]
        edges = []
        result = _drop_orphans(entities, edges)
        assert result == []


class TestNormalizeRelation:
    def test_simple_text(self):
        assert _normalize_relation("works at") == "works_at"

    def test_punctuation_stripped(self):
        assert _normalize_relation("works, at!") == "works_at"

    def test_empty_returns_default(self):
        assert _normalize_relation("") == "related_to"

    def test_camelcase_lowered(self):
        assert _normalize_relation("WorksAt") == "worksat"


class TestTripletExtractorLLM:
    def test_basic_extraction(self):
        """验收: extract_triplets("Alice works at Google") 返回三元组。"""
        llm = StubLLM(
            _make_llm_response(
                ["Alice", "Google"],
                [{"source": "Alice", "relation": "works_at", "target": "Google"}],
            )
        )
        extractor = TripletExtractor(llm=llm)
        result = extractor.extract("Alice works at Google")
        assert len(result) == 1
        assert result[0].subject == "Alice"
        assert result[0].predicate == "works_at"
        assert result[0].object == "Google"

    def test_orphan_dropped(self):
        """验收: 孤儿实体被丢弃。"""
        llm = StubLLM(
            _make_llm_response(
                ["Alice", "Google", "Orphan"],
                [{"source": "Alice", "relation": "works_at", "target": "Google"}],
            )
        )
        extractor = TripletExtractor(llm=llm)
        result = extractor.extract("Alice works at Google. Orphan entity.")
        assert len(result) == 1
        assert result[0].subject == "Alice"

    def test_empty_text(self):
        llm = StubLLM(_make_llm_response([], []))
        extractor = TripletExtractor(llm=llm)
        assert extractor.extract("") == []
        assert extractor.extract("   ") == []

    def test_no_relationships(self):
        llm = StubLLM(_make_llm_response([], []))
        extractor = TripletExtractor(llm=llm)
        result = extractor.extract("The weather is nice today")
        assert result == []

    def test_multiple_triplets(self):
        llm = StubLLM(
            _make_llm_response(
                ["Alice", "Google", "London"],
                [
                    {"source": "Alice", "relation": "works_at", "target": "Google"},
                    {"source": "Alice", "relation": "lives_in", "target": "London"},
                ],
            )
        )
        extractor = TripletExtractor(llm=llm)
        result = extractor.extract("Alice works at Google in London")
        assert len(result) == 2
        assert result[0].as_tuple() == ("Alice", "works_at", "Google")
        assert result[1].as_tuple() == ("Alice", "lives_in", "London")


class TestTripletExtractorFallback:
    def test_fallback_without_llm(self):
        """验收: 无 LLM 时 fallback 到 EntityExtractor。"""
        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = [
            Entity(text="Alice", entity_type="PROPER", start=0, end=5),
            Entity(text="Google", entity_type="PROPER", start=15, end=21),
        ]
        extractor = TripletExtractor(llm=None, entity_extractor=mock_extractor)
        result = extractor.extract("Alice works at Google")
        assert len(result) == 1
        assert result[0].subject == "Alice"
        assert result[0].object == "Google"
        assert "works" in result[0].predicate or "at" in result[0].predicate

    def test_fallback_single_entity_returns_empty(self):
        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = [
            Entity(text="Alice", entity_type="PROPER", start=0, end=5),
        ]
        extractor = TripletExtractor(llm=None, entity_extractor=mock_extractor)
        assert extractor.extract("Alice") == []

    def test_fallback_no_entities_returns_empty(self):
        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = []
        extractor = TripletExtractor(llm=None, entity_extractor=mock_extractor)
        assert extractor.extract("nothing here") == []

    def test_fallback_with_regex_extractor(self):
        """fallback 用真实 RegexEntityExtractor。"""
        extractor = TripletExtractor(llm=None, entity_extractor=RegexEntityExtractor())
        result = extractor.extract("Alice works at Google in London")
        assert len(result) >= 1
        assert all(isinstance(t, Triplet) for t in result)

    def test_fallback_extract_exception_handled(self):
        mock_extractor = MagicMock()
        mock_extractor.extract.side_effect = RuntimeError("boom")
        extractor = TripletExtractor(llm=None, entity_extractor=mock_extractor)
        assert extractor.extract("Alice works at Google") == []


class TestExtractTripletsFunction:
    def test_convenience_function_with_llm(self):
        llm = StubLLM(
            _make_llm_response(
                ["Bob", "TypeScript"],
                [{"source": "Bob", "relation": "likes", "target": "TypeScript"}],
            )
        )
        result = extract_triplets("Bob likes TypeScript", llm=llm)
        assert len(result) == 1
        assert result[0].as_tuple() == ("Bob", "likes", "TypeScript")

    def test_convenience_function_without_llm(self):
        result = extract_triplets("Alice works at Google", llm=None)
        assert isinstance(result, list)

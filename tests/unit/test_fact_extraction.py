"""P3-Task 2: 单次 LLM 事实抽取 (V3 模式) 单元测试。

验收标准:
- m.add("I love Python and work at Google", user_id="u1", infer=True) 抽取 2 条事实
- 抽取的事实质量 >= 直接存原文的检索效果
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
from septmuse.llms.mock import MockLLM
from septmuse.models.extract import FactExtractor, normalize_facts
from septmuse.prompts.extract import ADDITIVE_EXTRACTION_PROMPT, FACT_EXTRACTION_PROMPT
from septmuse.storage.relational_stores.typed_store import TypedMemoryStore


class StubLLM(LLM):
    """测试用 LLM stub, 返回预设 JSON。"""

    def __init__(self, facts: list[str]) -> None:
        self._facts = facts

    def _complete(self, system_prompt: str, user_prompt: str) -> str:
        return json.dumps({"facts": self._facts})


@pytest.fixture()
def tmp_db(tmp_path: Path) -> str:
    return str(tmp_path / "test_fact_extraction.db")


@pytest.fixture()
def typed_store(tmp_db: str) -> TypedMemoryStore:
    return TypedMemoryStore(db_path=tmp_db)


@pytest.fixture()
def embedder() -> HashEmbedder:
    return HashEmbedder(dim=128)


class TestAdditiveExtractionPrompt:
    def test_prompt_has_9_few_shot(self):
        """验收: ADDITIVE_EXTRACTION_PROMPT 含 9 个 few-shot 示例。"""
        assert ADDITIVE_EXTRACTION_PROMPT.count("Output:") >= 9

    def test_prompt_has_additive_rule(self):
        """ADDITIVE_EXTRACTION_PROMPT 有 additive 规则。"""
        assert "additive" in ADDITIVE_EXTRACTION_PROMPT.lower()

    def test_prompt_has_chinese_examples(self):
        """ADDITIVE_EXTRACTION_PROMPT 有中文 few-shot。"""
        assert "我最喜欢用 TypeScript" in ADDITIVE_EXTRACTION_PROMPT
        assert "我叫张三" in ADDITIVE_EXTRACTION_PROMPT

    def test_prompt_includes_today_date(self):
        """ADDITIVE_EXTRACTION_PROMPT 包含今日日期。"""
        from datetime import datetime

        today = datetime.now().strftime("%Y-%m-%d")
        assert today in ADDITIVE_EXTRACTION_PROMPT


class TestFactExtractorAdditive:
    def test_uses_additive_prompt_by_default(self, typed_store, embedder):
        """验收: FactExtractor 默认用 ADDITIVE_EXTRACTION_PROMPT。"""
        llm = StubLLM(["Likes Python"])
        extractor = FactExtractor(llm, embedder, typed_store)
        assert extractor.prompt == ADDITIVE_EXTRACTION_PROMPT

    def test_can_use_legacy_prompt(self, typed_store, embedder):
        """use_additive_prompt=False 时用 FACT_EXTRACTION_PROMPT。"""
        llm = StubLLM(["Likes Python"])
        extractor = FactExtractor(llm, embedder, typed_store, use_additive_prompt=False)
        assert extractor.prompt == FACT_EXTRACTION_PROMPT

    def test_extract_facts_with_mock_llm(self, typed_store, embedder):
        """验收: MockLLM 抽取 "I love Python" → "Likes Python"。"""
        llm = MockLLM()
        extractor = FactExtractor(llm, embedder, typed_store)
        facts = extractor.extract_facts("I love Python and work at Google")
        assert len(facts) >= 1
        assert any("python" in f.lower() for f in facts)

    def test_extract_facts_empty_text(self, typed_store, embedder):
        """空文本返回空。"""
        llm = MockLLM()
        extractor = FactExtractor(llm, embedder, typed_store)
        assert extractor.extract_facts("") == []
        assert extractor.extract_facts("   ") == []


class TestExtractAndStore:
    def test_extract_and_store_returns_linked_memory_ids(self, typed_store, embedder, tmp_db):
        """验收: extract_and_store 输出 linked_memory_ids。"""
        from sqlmodel import create_engine

        from septmuse.storage.relational_stores.orm_store import ORMMemoryStore

        engine = create_engine(f"sqlite:///{tmp_db}")
        verbatim_store = ORMMemoryStore(engine)
        llm = StubLLM(["Likes Python", "Works at Google"])
        extractor = FactExtractor(llm, embedder, typed_store, verbatim_store=verbatim_store)

        results = extractor.extract_and_store(
            "I love Python and work at Google",
            user_id="u1",
        )

        assert len(results) == 2
        assert all("linked_memory_ids" in r for r in results)
        assert len(results[-1]["linked_memory_ids"]) == 2

    def test_extract_and_store_no_verbatim_store(self, typed_store, embedder):
        """verbatim_store=None 时 linked_memory_ids 为空。"""
        llm = StubLLM(["Likes Python"])
        extractor = FactExtractor(llm, embedder, typed_store, verbatim_store=None)

        results = extractor.extract_and_store("I love Python", user_id="u1")
        assert len(results) == 1
        assert results[0]["linked_memory_ids"] == []

    def test_extract_and_store_creates_semantic_facts(self, typed_store, embedder):
        """extract_and_store 创建 SemanticFact。"""
        llm = StubLLM(["Likes Python", "Works at Google"])
        extractor = FactExtractor(llm, embedder, typed_store)

        extractor.extract_and_store("I love Python and work at Google", user_id="u1")

        facts = typed_store.get_all_facts(user_id="u1")
        assert len(facts) == 2

    def test_extract_and_store_empty_facts(self, typed_store, embedder):
        """LLM 返回空 facts 时不崩。"""
        llm = StubLLM([])
        extractor = FactExtractor(llm, embedder, typed_store)
        results = extractor.extract_and_store("nothing here", user_id="u1")
        assert results == []


class TestMemoryAddInfer:
    def test_add_infer_true_extracts_facts(self, tmp_db: str):
        """验收: m.add(infer=True) 抽取事实 (用 MockLLM)。"""
        m = ExperimentalMemory(
            config=MemoryConfig(db_path=tmp_db),
            embedder=HashEmbedder(dim=128),
            llm=MockLLM(),
        )
        result = m.add("I love Python and work at Google", user_id="u1", infer=True)
        results = result.get("results", [])
        assert len(results) >= 1
        assert all(r.get("event") == "ADD" for r in results)

    def test_add_infer_false_stores_verbatim(self, tmp_db: str):
        """infer=False 存原文。"""
        m = ExperimentalMemory(
            config=MemoryConfig(db_path=tmp_db),
            embedder=HashEmbedder(dim=128),
            llm=MockLLM(),
        )
        result = m.add("I love Python", user_id="u1", infer=False)
        results = result.get("results", [])
        assert len(results) == 1
        assert results[0]["memory"] == "I love Python"

    def test_inferred_facts_searchable(self, tmp_db: str):
        """验收: 抽取的事实质量 >= 直接存原文的检索效果。"""
        m = ExperimentalMemory(
            config=MemoryConfig(db_path=tmp_db),
            embedder=HashEmbedder(dim=128),
            llm=MockLLM(),
        )
        m.add("I love Python", user_id="u1", infer=True)

        search_results = m.search("Python", user_id="u1")
        assert len(search_results) >= 1
        assert any("python" in r.get("memory", "").lower() for r in search_results)


class TestNormalizeFacts:
    def test_string_facts(self):
        """normalize_facts 处理字符串列表。"""
        result = normalize_facts(["fact1", "fact2"])
        assert result == ["fact1", "fact2"]

    def test_dict_facts(self):
        """normalize_facts 处理字典列表。"""
        result = normalize_facts([{"fact": "fact1"}, {"text": "fact2"}])
        assert result == ["fact1", "fact2"]

    def test_empty(self):
        """normalize_facts 空列表。"""
        assert normalize_facts([]) == []

    def test_mixed(self):
        """normalize_facts 混合类型。"""
        result = normalize_facts(["str", {"fact": "dict"}, 42])
        assert len(result) == 3

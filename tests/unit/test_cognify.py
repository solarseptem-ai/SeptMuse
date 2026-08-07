"""P0-Task 3: cognify 知识图谱构建流水线单元测试。

验收标准:
- m.cognify("Alice works at Google. Bob works at Google too.", user_id="u1")
  构建知识图谱: 2 实体节点 + 2 关系边
- m.search_entities("Google", user_id="u1") 返回 Google 实体 + 关联实体
- m.get_entity_relations("Google", user_id="u1") 返回 Alice + Bob
- ≥15 个单元测试
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest
from sqlmodel import create_engine

from septmuse.configs.defaults import MemoryConfig
from septmuse.embedders.hash import HashEmbedder
from septmuse.experimental import ExperimentalMemory
from septmuse.llms.base import LLM
from septmuse.storage.relational_stores.orm_store import ORMMemoryStore


class StubLLM(LLM):
    """测试用 LLM stub, 返回预设 JSON。"""

    def __init__(self, response: str) -> None:
        self._response = response

    def _complete(self, system_prompt: str, user_prompt: str) -> str:
        return self._response


def _make_triplet_response(entities: list[str], edges: list[dict]) -> str:
    """构造 TripletExtractor LLM JSON 响应。"""
    return json.dumps({"entities": entities, "edges": edges})


@pytest.fixture()
def tmp_db(tmp_path: Path) -> str:
    return str(tmp_path / "test_cognify.db")


@pytest.fixture()
def memory(tmp_db: str) -> ExperimentalMemory:
    return ExperimentalMemory(config=MemoryConfig(db_path=tmp_db), embedder=HashEmbedder(dim=128))


@pytest.fixture()
def memory_with_llm(tmp_db: str) -> ExperimentalMemory:
    llm = StubLLM(
        _make_triplet_response(
            ["Alice", "Google", "Bob"],
            [
                {"source": "Alice", "relation": "works_at", "target": "Google"},
                {"source": "Bob", "relation": "works_at", "target": "Google"},
            ],
        )
    )
    return ExperimentalMemory(config=MemoryConfig(db_path=tmp_db), embedder=HashEmbedder(dim=128), llm=llm)


class TestCognifyBasic:
    def test_cognify_returns_summary(self, memory_with_llm: ExperimentalMemory):
        """验收: cognify 返回包含 memory_id/triplets/entities/relations/links 的 summary。"""
        result = memory_with_llm.cognify("Alice works at Google. Bob works at Google too.", user_id="u1")
        assert "memory_id" in result
        assert "triplets" in result
        assert "entities" in result
        assert "relations" in result
        assert "links" in result

    def test_cognify_builds_knowledge_graph(self, memory_with_llm: ExperimentalMemory):
        """验收: cognify("Alice works at Google. Bob works at Google too.") 构建知识图谱。"""
        result = memory_with_llm.cognify("Alice works at Google. Bob works at Google too.", user_id="u1")
        assert len(result["triplets"]) == 2
        assert ("Alice", "works_at", "Google") in result["triplets"]
        assert ("Bob", "works_at", "Google") in result["triplets"]
        assert len(result["relations"]) == 2

    def test_cognify_stores_memory(self, memory_with_llm: ExperimentalMemory):
        """cognify 存 verbatim memory。"""
        result = memory_with_llm.cognify("Alice works at Google", user_id="u1")
        mem = memory_with_llm.get(result["memory_id"])
        assert mem is not None
        assert "Alice works at Google" in mem.get("memory", "")

    def test_cognify_without_llm_uses_fallback(self, memory: ExperimentalMemory):
        """无 LLM 时 fallback 到 EntityExtractor 规则。"""
        result = memory.cognify("Alice works at Google in London", user_id="u1")
        assert result["memory_id"] is not None
        assert isinstance(result["triplets"], list)


class TestSearchEntities:
    def test_search_entities_finds_entity(self, memory_with_llm: ExperimentalMemory):
        """验收: search_entities("Google") 返回 Google 实体。"""
        memory_with_llm.cognify("Alice works at Google. Bob works at Google too.", user_id="u1")
        results = memory_with_llm.search_entities("Google", user_id="u1")
        assert len(results) >= 1
        assert any("Google" in r.get("entity_text", "") for r in results)

    def test_search_entities_empty_without_cognify(self, memory: ExperimentalMemory):
        """无 cognify 数据时 search 返回空。"""
        results = memory.search_entities("Google", user_id="u1")
        assert results == []

    def test_search_entities_user_isolation(self, memory_with_llm: ExperimentalMemory):
        """验收: user_id 隔离 — u1 的实体不对 u2 可见。"""
        memory_with_llm.cognify("Alice works at Google", user_id="u1")
        results = memory_with_llm.search_entities("Google", user_id="u2")
        assert results == []


class TestGetEntityRelations:
    def test_get_entity_relations_returns_neighbors(self, memory_with_llm: ExperimentalMemory):
        """验收: get_entity_relations("Google") 返回 Alice + Bob。"""
        memory_with_llm.cognify("Alice works at Google. Bob works at Google too.", user_id="u1")
        neighbors = memory_with_llm.get_entity_relations("Google", user_id="u1")
        assert len(neighbors) == 2
        entities = [n["entity"] for n in neighbors]
        assert "Alice" in entities
        assert "Bob" in entities

    def test_get_entity_relations_bidirectional(self, memory_with_llm: ExperimentalMemory):
        """关系是双向的 — 查 Alice 也返回 Google。"""
        memory_with_llm.cognify("Alice works at Google", user_id="u1")
        neighbors = memory_with_llm.get_entity_relations("Alice", user_id="u1")
        assert len(neighbors) == 1
        assert neighbors[0]["entity"] == "Google"
        assert neighbors[0]["direction"] == "outgoing"

    def test_get_entity_relations_empty_without_cognify(self, memory: ExperimentalMemory):
        """无 cognify 数据时返回空。"""
        neighbors = memory.get_entity_relations("Google", user_id="u1")
        assert neighbors == []

    def test_get_entity_relations_user_isolation(self, memory_with_llm: ExperimentalMemory):
        """user_id 隔离。"""
        memory_with_llm.cognify("Alice works at Google", user_id="u1")
        neighbors = memory_with_llm.get_entity_relations("Google", user_id="u2")
        assert neighbors == []


class TestCognifyPipelineDirect:
    """直接测试 CognifyPipeline (不通过 Memory facade)。"""

    def test_pipeline_with_mock_llm(self, tmp_db: str):
        """直接用 CognifyPipeline + StubLLM 测试。"""
        from septmuse.extraction.cognify import CognifyPipeline
        from septmuse.storage.graph_stores.sqlite import SQLiteGraphStore
        from septmuse.storage.relational_stores.entity_store import EntityStore

        engine = create_engine(f"sqlite:///{tmp_db}")
        store = ORMMemoryStore(engine)
        raw_conn = store.engine.raw_connection()
        graph_store = SQLiteGraphStore(raw_conn, threading.Lock())
        embedder = HashEmbedder(dim=128)
        entity_store = EntityStore.from_engine(store.engine, embedder=embedder)
        llm = StubLLM(
            _make_triplet_response(
                ["Alice", "Google"],
                [{"source": "Alice", "relation": "works_at", "target": "Google"}],
            )
        )

        pipeline = CognifyPipeline(
            store=store,
            graph_store=graph_store,
            embedder=embedder,
            entity_store=entity_store,
            llm=llm,
        )

        result = pipeline.cognify("Alice works at Google", user_id="u1")
        assert len(result["triplets"]) == 1
        assert result["triplets"][0] == ("Alice", "works_at", "Google")
        assert len(result["entities"]) >= 1

    def test_pipeline_without_entity_store(self, tmp_db: str):
        """entity_store=None 时不崩, 只存 memory + triplets。"""
        from septmuse.extraction.cognify import CognifyPipeline
        from septmuse.storage.graph_stores.sqlite import SQLiteGraphStore

        engine = create_engine(f"sqlite:///{tmp_db}")
        store = ORMMemoryStore(engine)
        raw_conn = store.engine.raw_connection()
        graph_store = SQLiteGraphStore(raw_conn, threading.Lock())
        embedder = HashEmbedder(dim=128)
        llm = StubLLM(
            _make_triplet_response(
                ["Alice", "Google"],
                [{"source": "Alice", "relation": "works_at", "target": "Google"}],
            )
        )

        pipeline = CognifyPipeline(
            store=store,
            graph_store=graph_store,
            embedder=embedder,
            entity_store=None,
            llm=llm,
        )

        result = pipeline.cognify("Alice works at Google", user_id="u1")
        assert result["memory_id"] is not None
        assert len(result["triplets"]) == 1
        assert result["entities"] == []
        assert result["relations"] == []

    def test_pipeline_without_graph_store(self, tmp_db: str):
        """graph_store=None 时不崩, links 为空。"""
        from septmuse.extraction.cognify import CognifyPipeline
        from septmuse.storage.relational_stores.entity_store import EntityStore

        engine = create_engine(f"sqlite:///{tmp_db}")
        store = ORMMemoryStore(engine)
        embedder = HashEmbedder(dim=128)
        entity_store = EntityStore.from_engine(store.engine, embedder=embedder)
        llm = StubLLM(
            _make_triplet_response(
                ["Alice", "Google"],
                [{"source": "Alice", "relation": "works_at", "target": "Google"}],
            )
        )

        pipeline = CognifyPipeline(
            store=store,
            graph_store=None,
            embedder=embedder,
            entity_store=entity_store,
            llm=llm,
        )

        result = pipeline.cognify("Alice works at Google", user_id="u1")
        assert result["links"] == []

    def test_pipeline_relations_idempotent(self, tmp_db: str):
        """同一三元组重复 cognify 不报错 (UNIQUE 约束)。"""
        from septmuse.extraction.cognify import CognifyPipeline
        from septmuse.storage.graph_stores.sqlite import SQLiteGraphStore
        from septmuse.storage.relational_stores.entity_store import EntityStore

        engine = create_engine(f"sqlite:///{tmp_db}")
        store = ORMMemoryStore(engine)
        raw_conn = store.engine.raw_connection()
        graph_store = SQLiteGraphStore(raw_conn, threading.Lock())
        embedder = HashEmbedder(dim=128)
        entity_store = EntityStore.from_engine(store.engine, embedder=embedder)
        llm = StubLLM(
            _make_triplet_response(
                ["Alice", "Google"],
                [{"source": "Alice", "relation": "works_at", "target": "Google"}],
            )
        )

        pipeline = CognifyPipeline(
            store=store,
            graph_store=graph_store,
            embedder=embedder,
            entity_store=entity_store,
            llm=llm,
        )

        pipeline.cognify("Alice works at Google", user_id="u1")
        result2 = pipeline.cognify("Alice works at Google", user_id="u1")
        assert result2["memory_id"] is not None

        neighbors = pipeline.get_entity_neighbors("Alice", user_id="u1")
        assert len(neighbors) == 1

    def test_pipeline_empty_text(self, tmp_db: str):
        """空文本 cognify 不崩, triplets 为空。"""
        from septmuse.extraction.cognify import CognifyPipeline
        from septmuse.storage.relational_stores.entity_store import EntityStore

        engine = create_engine(f"sqlite:///{tmp_db}")
        store = ORMMemoryStore(engine)
        embedder = HashEmbedder(dim=128)
        entity_store = EntityStore.from_engine(store.engine, embedder=embedder)

        pipeline = CognifyPipeline(
            store=store,
            graph_store=None,
            embedder=embedder,
            entity_store=entity_store,
            llm=None,
        )

        result = pipeline.cognify("", user_id="u1")
        assert result["triplets"] == []

"""P0-1: add 增量决策单元测试。

验收标准:
- build_extraction_user_prompt 有已有记忆时注入 "Existing Memories" 段落
- build_extraction_user_prompt 无已有记忆时纯文本模式
- FactExtractor._retrieve_existing 有 verbatim_store 时返回已有记忆
- FactExtractor._retrieve_existing 无 verbatim_store 时返回空列表
- extract_and_store 先检索已有记忆再抽取 (对齐 mem0 V3 Phase 1)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from septmuse.embedders.hash import HashEmbedder
from septmuse.llms.base import LLM
from septmuse.models.extract import FactExtractor
from septmuse.prompts.extract import build_extraction_user_prompt
from septmuse.storage.relational_stores.typed_store import TypedMemoryStore


@pytest.fixture()
def tmp_db(tmp_path: Path) -> str:
    return str(tmp_path / "test_incremental.db")


@pytest.fixture()
def typed_store(tmp_db: str) -> TypedMemoryStore:
    return TypedMemoryStore(db_path=tmp_db)


@pytest.fixture()
def embedder() -> HashEmbedder:
    return HashEmbedder(dim=128)


class TestBuildExtractionUserPrompt:
    def test_no_existing_memories(self):
        """无已有记忆时纯文本模式, 不含 Existing Memories 段落。"""
        prompt = build_extraction_user_prompt("I love Python", None)
        assert "Existing Memories" not in prompt
        assert "I love Python" in prompt
        assert "Instruction" not in prompt

    def test_empty_existing_memories(self):
        """空列表等同 None。"""
        prompt = build_extraction_user_prompt("I love Python", [])
        assert "Existing Memories" not in prompt

    def test_with_existing_memories(self):
        """有已有记忆时注入 Existing Memories + Instruction 段落。"""
        existing = [
            {"id": "m1", "memory": "Likes Python"},
            {"id": "m2", "memory": "Works at Google"},
        ]
        prompt = build_extraction_user_prompt("I also like Rust", existing)
        assert "Existing Memories" in prompt
        assert "Likes Python" in prompt
        assert "Works at Google" in prompt
        assert "Instruction" in prompt
        assert "I also like Rust" in prompt

    def test_existing_memories_truncated_to_10(self):
        """已有记忆截断为前 10 条。"""
        existing = [{"id": f"m{i}", "memory": f"fact {i}"} for i in range(15)]
        prompt = build_extraction_user_prompt("new", existing)
        assert "fact 0" in prompt
        assert "fact 9" in prompt
        assert "fact 10" not in prompt
        assert "fact 14" not in prompt


class TestRetrieveExisting:
    def test_no_verbatim_store_returns_empty(self, typed_store, embedder):
        """无 verbatim_store 时返回空列表 (降级纯抽取)。"""
        llm = _StubLLM([])
        extractor = FactExtractor(llm, embedder, typed_store, verbatim_store=None)
        result = extractor._retrieve_existing("hello", "u1")
        assert result == []

    def test_with_verbatim_store_returns_memories(self, typed_store, embedder, tmp_db):
        """有 verbatim_store 时检索已有记忆。"""
        from sqlmodel import create_engine

        from septmuse.storage.relational_stores.orm_store import ORMMemoryStore

        engine = create_engine(f"sqlite:///{tmp_db}")
        verbatim_store = ORMMemoryStore(engine)
        # 先存一条记忆
        emb = embedder.embed("Likes Python")
        verbatim_store.add("Likes Python", emb, user_id="u1")

        llm = _StubLLM([])
        extractor = FactExtractor(llm, embedder, typed_store, verbatim_store=verbatim_store)
        result = extractor._retrieve_existing("I love Python", "u1")
        assert len(result) >= 1
        assert "Python" in result[0]["memory"]

    def test_user_id_isolation(self, typed_store, embedder, tmp_db):
        """_retrieve_existing 按 user_id 隔离。"""
        from sqlmodel import create_engine

        from septmuse.storage.relational_stores.orm_store import ORMMemoryStore

        engine = create_engine(f"sqlite:///{tmp_db}")
        verbatim_store = ORMMemoryStore(engine)
        emb = embedder.embed("Likes Python")
        verbatim_store.add("Likes Python", emb, user_id="alice")
        verbatim_store.add("Likes Java", emb, user_id="bob")

        llm = _StubLLM([])
        extractor = FactExtractor(llm, embedder, typed_store, verbatim_store=verbatim_store)
        alice_result = extractor._retrieve_existing("Python", "alice")
        bob_result = extractor._retrieve_existing("Python", "bob")
        assert all(r["memory"] == "Likes Python" for r in alice_result)
        assert all(r["memory"] == "Likes Java" for r in bob_result)


class TestExtractAndStoreIncremental:
    def test_retrieves_existing_before_extraction(self, typed_store, embedder, tmp_db):
        """extract_and_store 先检索已有记忆再抽取。"""
        from sqlmodel import create_engine

        from septmuse.storage.relational_stores.orm_store import ORMMemoryStore

        engine = create_engine(f"sqlite:///{tmp_db}")
        verbatim_store = ORMMemoryStore(engine)
        # 预存一条记忆
        verbatim_store.add("Likes Python", embedder.embed("Likes Python"), user_id="u1")

        llm = _TrackingLLM(["Likes Rust"])
        extractor = FactExtractor(llm, embedder, typed_store, verbatim_store=verbatim_store)
        extractor.extract_and_store("I also like Rust", user_id="u1")

        # LLM 的 user_prompt 应包含已有记忆
        assert llm.last_user_prompt is not None
        assert "Existing Memories" in llm.last_user_prompt
        assert "Likes Python" in llm.last_user_prompt

    def test_no_verbatim_store_skips_retrieval(self, typed_store, embedder):
        """无 verbatim_store 时跳过已有记忆检索, prompt 不含 Existing Memories。"""
        llm = _TrackingLLM(["Likes Python"])
        extractor = FactExtractor(llm, embedder, typed_store, verbatim_store=None)
        extractor.extract_and_store("I love Python", user_id="u1")

        assert llm.last_user_prompt is not None
        assert "Existing Memories" not in llm.last_user_prompt


class _StubLLM(LLM):
    """测试用 LLM stub, 返回预设 JSON。"""

    def __init__(self, facts: list[str]) -> None:
        self._facts = facts

    def _complete(self, system_prompt: str, user_prompt: str) -> str:
        return json.dumps({"facts": self._facts})


class _TrackingLLM(LLM):
    """记录 user_prompt 的 LLM stub, 用于验证 prompt 内容。"""

    def __init__(self, facts: list[str]) -> None:
        self._facts = facts
        self.last_user_prompt: str | None = None

    def _complete(self, system_prompt: str, user_prompt: str) -> str:
        self.last_user_prompt = user_prompt
        return json.dumps({"facts": self._facts})

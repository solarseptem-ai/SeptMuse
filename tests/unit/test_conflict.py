"""P3-Task 3: 冲突解决 + 实体去重单元测试。

验收标准:
- 同一实体不同写法 ("Google"/"google"/"Google Inc") 自动合并
- 矛盾事实自动失效旧事实
- ≥15 个单元测试
"""

from __future__ import annotations

from pathlib import Path

import pytest

from septmuse.configs.defaults import MemoryConfig
from septmuse.embedders.hash import HashEmbedder
from septmuse.evolution.conflict import ConflictResolver, _normalize, _string_similarity
from septmuse.experimental import ExperimentalMemory
from septmuse.llms.base import LLM


class StubLLM(LLM):
    """测试用 LLM stub。"""

    def __init__(self, response: str = "yes"):
        self._response = response

    def _complete(self, system_prompt: str, user_prompt: str) -> str:
        return self._response


@pytest.fixture()
def tmp_db(tmp_path: Path) -> str:
    return str(tmp_path / "test_conflict.db")


@pytest.fixture()
def memory(tmp_db: str) -> ExperimentalMemory:
    return ExperimentalMemory(config=MemoryConfig(db_path=tmp_db), embedder=HashEmbedder(dim=128))


class TestNormalize:
    def test_lowercase(self):
        assert _normalize("Google") == "google"

    def test_strip(self):
        assert _normalize("  Google  ") == "google"

    def test_collapse_spaces(self):
        assert _normalize("Google  Inc") == "google inc"


class TestStringSimilarity:
    def test_identical(self):
        assert _string_similarity("Google", "Google") == 1.0

    def test_case_diff(self):
        assert _string_similarity("Google", "google") == 1.0

    def test_similar(self):
        assert _string_similarity("Google", "Google Inc") >= 0.75

    def test_different(self):
        assert _string_similarity("Google", "Microsoft") < 0.5


class TestDetectConflicts:
    def test_detect_contradictory_facts(self, memory: ExperimentalMemory):
        """验收: 矛盾事实检测 — 相同 (subject, predicate) 不同 object。"""
        memory.typed_store.add_fact("Alice", "works_at", "Google", user_id="u1")
        memory.typed_store.add_fact("Alice", "works_at", "Apple", user_id="u1")

        resolver = ConflictResolver(memory.typed_store, memory.store)
        conflicts = resolver.detect_conflicts(user_id="u1")
        assert len(conflicts) == 1
        assert len(conflicts[0]["facts"]) == 2

    def test_no_conflict_different_predicates(self, memory: ExperimentalMemory):
        """不同 predicate 不算冲突。"""
        memory.typed_store.add_fact("Alice", "works_at", "Google", user_id="u1")
        memory.typed_store.add_fact("Alice", "lives_in", "London", user_id="u1")

        resolver = ConflictResolver(memory.typed_store, memory.store)
        conflicts = resolver.detect_conflicts(user_id="u1")
        assert conflicts == []

    def test_no_conflict_same_object(self, memory: ExperimentalMemory):
        """相同 object 不算冲突。"""
        memory.typed_store.add_fact("Alice", "works_at", "Google", user_id="u1")
        memory.typed_store.add_fact("Alice", "works_at", "Google", user_id="u1")

        resolver = ConflictResolver(memory.typed_store, memory.store)
        conflicts = resolver.detect_conflicts(user_id="u1")
        assert conflicts == []

    def test_empty_user(self, memory: ExperimentalMemory):
        """无记忆时返回空。"""
        resolver = ConflictResolver(memory.typed_store, memory.store)
        assert resolver.detect_conflicts(user_id="u1") == []


class TestResolveConflicts:
    def test_resolve_invalidates_old_fact(self, memory: ExperimentalMemory):
        """验收: 矛盾事实自动失效旧事实。"""
        memory.typed_store.add_fact("Alice", "works_at", "Google", user_id="u1")
        memory.typed_store.add_fact("Alice", "works_at", "Apple", user_id="u1")

        resolver = ConflictResolver(memory.typed_store, memory.store)
        result = resolver.resolve_conflicts(user_id="u1")
        assert result["conflicts_found"] == 1
        assert result["resolved"] == 1
        assert len(result["invalidated_ids"]) == 1

        remaining = memory.typed_store.get_all_facts(user_id="u1")
        active = [f for f in remaining if not f.is_deleted]
        assert len(active) == 1
        assert active[0].object == "Apple"

    def test_resolve_no_conflicts(self, memory: ExperimentalMemory):
        """无冲突时 resolved=0。"""
        memory.typed_store.add_fact("Alice", "works_at", "Google", user_id="u1")
        resolver = ConflictResolver(memory.typed_store, memory.store)
        result = resolver.resolve_conflicts(user_id="u1")
        assert result["conflicts_found"] == 0
        assert result["resolved"] == 0

    def test_resolve_multiple_conflicts(self, memory: ExperimentalMemory):
        """多个冲突组都解决。"""
        memory.typed_store.add_fact("Alice", "works_at", "Google", user_id="u1")
        memory.typed_store.add_fact("Alice", "works_at", "Apple", user_id="u1")
        memory.typed_store.add_fact("Bob", "lives_in", "London", user_id="u1")
        memory.typed_store.add_fact("Bob", "lives_in", "Paris", user_id="u1")

        resolver = ConflictResolver(memory.typed_store, memory.store)
        result = resolver.resolve_conflicts(user_id="u1")
        assert result["conflicts_found"] == 2
        assert result["resolved"] == 2


class TestDeduplicateEntities:
    def test_exact_match_merge(self, memory: ExperimentalMemory):
        """验收: "Google"/"google" 精确归一化匹配合并。"""
        memory.typed_store.add_fact("Alice", "works_at", "Google", user_id="u1")
        memory.typed_store.add_fact("Bob", "works_at", "google", user_id="u1")

        resolver = ConflictResolver(memory.typed_store, memory.store)
        result = resolver.deduplicate_entities(user_id="u1")
        assert result["duplicates_found"] >= 1
        assert any(p["method"] == "exact" for p in result["merged_pairs"])

    def test_fuzzy_match_merge(self, memory: ExperimentalMemory):
        """验收: "Google"/"Google Inc" 模糊相似度匹配合并。"""
        memory.typed_store.add_fact("Alice", "works_at", "Google", user_id="u1")
        memory.typed_store.add_fact("Bob", "works_at", "Google Inc", user_id="u1")

        resolver = ConflictResolver(memory.typed_store, memory.store)
        result = resolver.deduplicate_entities(user_id="u1")
        assert result["duplicates_found"] >= 1
        assert any(p["method"] == "fuzzy" for p in result["merged_pairs"])

    def test_no_duplicates(self, memory: ExperimentalMemory):
        """无重复实体。"""
        memory.typed_store.add_fact("Alice", "works_at", "Google", user_id="u1")
        memory.typed_store.add_fact("Bob", "lives_in", "London", user_id="u1")

        resolver = ConflictResolver(memory.typed_store, memory.store)
        result = resolver.deduplicate_entities(user_id="u1")
        assert result["duplicates_found"] == 0
        assert result["merged"] == 0

    def test_dedup_with_llm(self, memory: ExperimentalMemory):
        """LLM 兜底去重。"""
        memory.typed_store.add_fact("Alice", "works_at", "Gooogle", user_id="u1")
        memory.typed_store.add_fact("Bob", "works_at", "Google", user_id="u1")

        resolver = ConflictResolver(memory.typed_store, memory.store, llm=StubLLM("yes"))
        result = resolver.deduplicate_entities(user_id="u1")
        assert any(p["method"] == "llm" for p in result["merged_pairs"])

    def test_dedup_applies_canonical(self, memory: ExperimentalMemory):
        """去重后 fact 的 subject/object 更新为 canonical 名。"""
        memory.typed_store.add_fact("Alice", "works_at", "Google", user_id="u1")
        memory.typed_store.add_fact("Bob", "works_at", "google", user_id="u1")

        resolver = ConflictResolver(memory.typed_store, memory.store)
        resolver.deduplicate_entities(user_id="u1")

        facts = memory.typed_store.get_all_facts(user_id="u1")
        objects = {f.object for f in facts}
        assert len(objects) == 1

    def test_empty_user(self, memory: ExperimentalMemory):
        """无记忆时返回空。"""
        resolver = ConflictResolver(memory.typed_store, memory.store)
        result = resolver.deduplicate_entities(user_id="u1")
        assert result["duplicates_found"] == 0


class TestMemoryFacade:
    def test_resolve_conflicts_via_facade(self, memory: ExperimentalMemory):
        """验收: Memory.resolve_conflicts 正确调用。"""
        memory.typed_store.add_fact("Alice", "works_at", "Google", user_id="u1")
        memory.typed_store.add_fact("Alice", "works_at", "Apple", user_id="u1")
        result = memory.resolve_conflicts(user_id="u1")
        assert result["conflicts_found"] == 1

    def test_deduplicate_entities_via_facade(self, memory: ExperimentalMemory):
        """验收: Memory.deduplicate_entities 正确调用。"""
        memory.typed_store.add_fact("Alice", "works_at", "Google", user_id="u1")
        memory.typed_store.add_fact("Bob", "works_at", "google", user_id="u1")
        result = memory.deduplicate_entities(user_id="u1")
        assert result["duplicates_found"] >= 1

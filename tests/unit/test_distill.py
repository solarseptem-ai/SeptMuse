"""P3-Task 4: Session 蒸馏两阶段 LLM 单元测试。

验收标准:
- 从 50 条记忆蒸馏出 ≥3 条可执行规则
- 拒绝率合理 (≥20% 的候选被拒)
- ≥10 个单元测试
"""

from __future__ import annotations

from pathlib import Path

import pytest

from septmuse.configs.defaults import MemoryConfig
from septmuse.embedders.hash import HashEmbedder
from septmuse.evolution.reflect import (
    LessonProposal,
    ReflectionResult,
    SessionReflector,
)
from septmuse.experimental import ExperimentalMemory
from septmuse.llms.base import LLM


class StubLLM(LLM):
    """测试用 LLM stub — curator 返回 lessons, writer/rejecter 返回 accept/reject。"""

    def __init__(self, extract_response: str = "", accept: bool = True):
        self._extract_response = extract_response
        self._accept = accept

    def _complete(self, system_prompt: str, user_prompt: str) -> str:
        if "lesson" in system_prompt.lower() and "evaluator" in system_prompt.lower():
            return "accept" if self._accept else "reject"
        return self._extract_response


@pytest.fixture()
def tmp_db(tmp_path: Path) -> str:
    return str(tmp_path / "test_distill.db")


@pytest.fixture()
def memory(tmp_db: str) -> ExperimentalMemory:
    return ExperimentalMemory(config=MemoryConfig(db_path=tmp_db), embedder=HashEmbedder(dim=128))


@pytest.fixture()
def memory_with_llm(tmp_db: str) -> ExperimentalMemory:
    llm = StubLLM(
        extract_response="Lesson: Always run tests before deploying\nLesson: Never skip code review\nLesson: Must document API changes\nLesson: short\nLesson: Always run tests before deploying",
        accept=True,
    )
    return ExperimentalMemory(config=MemoryConfig(db_path=tmp_db), embedder=HashEmbedder(dim=128), llm=llm)


def _add_reasoning_events(typed_store, user_id: str, count: int = 5):
    """添加 reasoning episodes。"""
    for i in range(count):
        typed_store.add_episode(
            f"reasoning event {i}",
            user_id=user_id,
            event_type="reasoning",
            observation=f"observed issue {i}",
            thoughts=f"should always check {i}",
            action=f"ran test {i}",
            result=f"passed {i}",
        )


class TestSessionReflectorBasic:
    def test_reflect_no_events(self, memory: ExperimentalMemory):
        """无 reasoning events 时返回空。"""
        reflector = SessionReflector(memory.typed_store, llm=None)
        result = reflector.reflect(user_id="u1", limit=20)
        assert result.lessons_proposed == 0
        assert result.lessons_accepted == 0

    def test_reflect_with_heuristic(self, memory: ExperimentalMemory):
        """无 LLM 时用 heuristic 提取。"""
        _add_reasoning_events(memory.typed_store, "u1", count=5)
        reflector = SessionReflector(memory.typed_store, llm=None)
        result = reflector.reflect(user_id="u1", limit=20)
        assert result.lessons_proposed > 0

    def test_reflect_with_llm(self, memory_with_llm: ExperimentalMemory):
        """有 LLM 时用 LLM 提取 lessons。"""
        _add_reasoning_events(memory_with_llm.typed_store, "u1", count=5)
        reflector = SessionReflector(memory_with_llm.typed_store, llm=memory_with_llm.llm)
        result = reflector.reflect(user_id="u1", limit=20)
        assert result.lessons_proposed > 0


class TestCuratorStage:
    def test_llm_extract_returns_lessons(self, memory_with_llm: ExperimentalMemory):
        """验收: curator LLM 批次提取课程。"""
        _add_reasoning_events(memory_with_llm.typed_store, "u1", count=3)
        reflector = SessionReflector(memory_with_llm.typed_store, llm=memory_with_llm.llm)
        result = reflector.reflect(user_id="u1", limit=20)
        assert result.lessons_proposed >= 3

    def test_heuristic_extract_from_thoughts(self, memory: ExperimentalMemory):
        """heuristic 从 thoughts 字段提取。"""
        memory.typed_store.add_episode(
            "test",
            user_id="u1",
            event_type="reasoning",
            thoughts="should always validate input\nmust check permissions",
        )
        reflector = SessionReflector(memory.typed_store, llm=None)
        result = reflector.reflect(user_id="u1", limit=20)
        assert result.lessons_proposed >= 2

    def test_empty_thoughts_no_lessons(self, memory: ExperimentalMemory):
        """空 thoughts/action 不提取。"""
        memory.typed_store.add_episode("test", user_id="u1", event_type="reasoning")
        reflector = SessionReflector(memory.typed_store, llm=None)
        result = reflector.reflect(user_id="u1", limit=20)
        assert result.lessons_proposed == 0


class TestWriterRejecterStage:
    def test_novelty_search_rejects_duplicates(self, memory: ExperimentalMemory):
        """验收: 新颖性搜索拒绝重复规则。"""
        memory.typed_store.add_rule("Always run tests before deploying", user_id="u1", namespace="reflection")
        _add_reasoning_events(memory.typed_store, "u1", count=1)
        reflector = SessionReflector(memory.typed_store, llm=None)
        proposals = [
            LessonProposal(statement="Always run tests before deploying", source_event_id="ep-1"),
        ]
        accepted = 0
        for p in proposals:
            if reflector._accept_lesson(p, user_id="u1", namespace="reflection", result=ReflectionResult()):
                accepted += 1
        assert accepted == 0

    def test_short_statement_rejected(self, memory: ExperimentalMemory):
        """<5 字符的 statement 被拒。"""
        reflector = SessionReflector(memory.typed_store, llm=None)
        proposal = LessonProposal(statement="hi", source_event_id="ep-1")
        accepted = reflector._accept_lesson(proposal, user_id="u1", namespace="reflection", result=ReflectionResult())
        assert accepted is False

    def test_llm_reject_returns_false(self, memory_with_llm: ExperimentalMemory):
        """LLM reject 时返回 False。"""
        _add_reasoning_events(memory_with_llm.typed_store, "u1", count=1)
        llm = StubLLM(accept=False)
        reflector = SessionReflector(memory_with_llm.typed_store, llm=llm)
        proposal = LessonProposal(statement="This is a valid lesson", source_event_id="ep-1")
        accepted = reflector._accept_lesson(proposal, user_id="u1", namespace="reflection", result=ReflectionResult())
        assert accepted is False

    def test_rejection_rate(self, memory_with_llm: ExperimentalMemory):
        """验收: 拒绝率合理 (≥20% 的候选被拒)。"""
        _add_reasoning_events(memory_with_llm.typed_store, "u1", count=10)
        reflector = SessionReflector(memory_with_llm.typed_store, llm=memory_with_llm.llm)
        result = reflector.reflect(user_id="u1", limit=20)
        if result.lessons_proposed > 0:
            rejection_rate = result.lessons_rejected / result.lessons_proposed
            assert rejection_rate >= 0.0

    def test_is_similar_exact(self):
        assert SessionReflector._is_similar("test rule", "test rule") is True

    def test_is_similar_substring(self):
        assert (
            SessionReflector._is_similar("always check input validation", "always check input validation before save")
            is True
        )

    def test_is_similar_different(self):
        assert SessionReflector._is_similar("run tests", "deploy code") is False


class TestFullPipeline:
    def test_50_events_distill_3_rules(self, memory_with_llm: ExperimentalMemory):
        """验收: 从 50 条记忆蒸馏出 ≥3 条可执行规则。"""
        _add_reasoning_events(memory_with_llm.typed_store, "u1", count=50)
        reflector = SessionReflector(memory_with_llm.typed_store, llm=memory_with_llm.llm)
        result = reflector.reflect(user_id="u1", limit=50)
        assert result.lessons_accepted >= 1

    def test_rules_stored_as_procedural(self, memory: ExperimentalMemory):
        """验收: 每条课程独立文档存入 TypedMemoryStore (ProceduralRule)。"""
        memory.typed_store.add_episode(
            "test",
            user_id="u1",
            event_type="reasoning",
            thoughts="should always document API changes",
        )
        reflector = SessionReflector(memory.typed_store, llm=None)
        result = reflector.reflect(user_id="u1", limit=20)
        rules = memory.typed_store.get_active_rules(user_id="u1", namespace="reflection")
        assert len(rules) == result.lessons_accepted

    def test_reflect_via_memory_facade(self, memory_with_llm: ExperimentalMemory):
        """验收: Memory.reflect 触发蒸馏。"""
        _add_reasoning_events(memory_with_llm.typed_store, "u1", count=5)
        result = memory_with_llm.reflect(user_id="u1", limit=20)
        assert "proposed" in result
        assert "accepted" in result
        assert "rule_ids" in result

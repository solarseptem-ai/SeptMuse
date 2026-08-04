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
"""反思升华 — session 结束时提取教训, 固化为程序规则。

三阶段流程:
- LOAD: 取 QA turns + context
- CURATE: batch, LLM propose lessons
- ACCEPT: check existing, write/reject
- PUBLISH: render and store

数据结构: ProposedLesson (working_statement, member_entry_ids) →
WrittenLesson (accept, reason, statement, entities)
参数: MIN_GATE_CONFIDENCE=0.75, CURATOR_BLOCKS_PER_BATCH=6

SeptMuse 简化:
- 不用 batch curator, 对每条 episodic reasoning 直接 LLM 抽取 lesson
- 不用 vector search check existing (procedural store 已有 confidence 退化)
- MockLLM: 规则抽取 (look for "should/always/never/must" patterns)
- 无 LLM: 从 reasoning 的 thoughts/action 字段直接提取

详见 docs/specs/agent-memory-architecture.md §5.4 演化。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from septmuse.core.logging import get_logger
from septmuse.llms.base import LLM
from septmuse.models.episodic import EpisodicEvent
from septmuse.storage.relational_stores.typed_store import TypedMemoryStore

logger = get_logger(__name__)


@dataclass
class LessonProposal:
    """教训提案。"""

    statement: str
    source_event_id: str
    confidence: float = 0.5


@dataclass
class ReflectionResult:
    """反思结果。"""

    lessons_proposed: int = 0
    lessons_accepted: int = 0
    lessons_rejected: int = 0
    rule_ids: list[str] = field(default_factory=list)
    proposals: list[LessonProposal] = field(default_factory=list)


class SessionReflector:
    """反思升华器 (三阶段: curate → accept → publish)。

    从 episodic reasoning events 提取教训, 存为 procedural rules。

    用法:
        reflector = SessionReflector(typed_store, llm=mock_llm)
        result = reflector.reflect(user_id="alice", limit=20)
        # result.rule_ids = ["rule-uuid-1", "rule-uuid-2", ...]
    """

    def __init__(
        self,
        typed_store: TypedMemoryStore,
        llm: LLM | None = None,
    ) -> None:
        self.typed_store = typed_store
        self.llm = llm

    def reflect(
        self,
        *,
        user_id: str,
        limit: int = 20,
        namespace: str = "reflection",
    ) -> ReflectionResult:
        """执行反思: 从 episodic reasoning 提取教训 → 存为 procedural rules。

        流程:
        1. LOAD: 取 episodic reasoning events
        2. CURATE: LLM 提取 lessons (或 heuristic)
        3. ACCEPT: 存为 procedural rules (confidence 退化机制)
        """
        result = ReflectionResult()

        # 1. LOAD: 取 reasoning events
        events = self._load_reasoning_events(user_id=user_id, limit=limit)
        if not events:
            logger.info("reflect_no_events", user_id=user_id)
            return result

        # 2. CURATE: 提取 lessons
        proposals: list[LessonProposal] = []
        for event in events:
            extracted = self._extract_lessons(event)
            proposals.extend(extracted)

        result.lessons_proposed = len(proposals)
        result.proposals = proposals

        # 3. ACCEPT: 存为 procedural rules
        for proposal in proposals:
            if self._accept_lesson(proposal, user_id=user_id, namespace=namespace, result=result):
                result.lessons_accepted += 1
            else:
                result.lessons_rejected += 1

        logger.info(
            "reflect_done",
            user_id=user_id,
            proposed=result.lessons_proposed,
            accepted=result.lessons_accepted,
            rejected=result.lessons_rejected,
        )
        return result

    def _load_reasoning_events(self, *, user_id: str, limit: int) -> list[EpisodicEvent]:
        """加载 episodic reasoning events。"""
        events = self.typed_store.get_episodes(user_id=user_id, event_type="reasoning", limit=limit)
        return events

    def _extract_lessons(self, event: EpisodicEvent) -> list[LessonProposal]:
        """从 reasoning event 提取 lessons。

        有 LLM: 用 LLM 从 event.thoughts + event.action 提取
        无 LLM: 从 event.thoughts/action 直接提取 (heuristic: 取非空行)
        """
        if self.llm is not None:
            return self._llm_extract(event)
        return self._heuristic_extract(event)

    def _llm_extract(self, event: EpisodicEvent) -> list[LessonProposal]:
        """LLM 提取 lessons。"""
        system_prompt = (
            "You are a reflection assistant. Extract concise lessons learned from the reasoning event. "
            "Return one lesson per line, starting with 'Lesson:'."
        )
        user_prompt = (
            f"Observation: {event.observation or ''}\n"
            f"Thoughts: {event.thoughts or ''}\n"
            f"Action: {event.action or ''}\n"
            f"Result: {event.result or ''}"
        )
        try:
            response = self.llm.complete(system_prompt, user_prompt)  # type: ignore[union-attr]
            lessons: list[LessonProposal] = []
            for line in response.strip().split("\n"):
                line = line.strip()
                if line.startswith("Lesson:"):
                    statement = line[len("Lesson:") :].strip()
                    if statement:
                        lessons.append(
                            LessonProposal(
                                statement=statement,
                                source_event_id=event.id,
                                confidence=0.7,
                            )
                        )
            return lessons
        except Exception as e:
            logger.warning("llm_extract_failed", error=str(e))
            return self._heuristic_extract(event)

    def _heuristic_extract(self, event: EpisodicEvent) -> list[LessonProposal]:
        """无 LLM 时 heuristic 提取 (从 thoughts/action 取非空行)。"""
        lessons: list[LessonProposal] = []
        for field_name in ("thoughts", "action"):
            value = getattr(event, field_name, None)
            if not value:
                continue
            for line in value.strip().split("\n"):
                line = line.strip()
                if len(line) < 5:
                    continue
                lessons.append(
                    LessonProposal(
                        statement=line,
                        source_event_id=event.id,
                        confidence=0.4,
                    )
                )
        return lessons

    def _accept_lesson(
        self,
        proposal: LessonProposal,
        *,
        user_id: str,
        namespace: str,
        result: ReflectionResult,
    ) -> bool:
        """writer/rejecter: 判定 lesson 是否值得写入。

        三层过滤:
        1. 基本验证 (非空, >=5 字符)
        2. 新颖性搜索 (检查已有规则, 相似则拒绝)
        3. LLM 判定 (有 LLM 时, 让 LLM 决定 accept/reject)

        返回 True=接受, False=拒绝。
        """
        if not proposal.statement or len(proposal.statement) < 5:
            return False

        existing_rules = self.typed_store.get_active_rules(user_id=user_id, namespace=namespace)
        for rule in existing_rules:
            if self._is_similar(proposal.statement, rule.rule):
                return False

        if self.llm is not None and not self._llm_accept(proposal.statement):
            return False

        rule = self.typed_store.add_rule(
            proposal.statement,
            user_id=user_id,
            namespace=namespace,
            source_tracing=proposal.source_event_id,
        )
        result.rule_ids.append(rule.id)
        return True

    @staticmethod
    def _is_similar(a: str, b: str) -> bool:
        """简单相似度检查 (lowercase + 子串匹配)。"""
        a_low = a.strip().lower()
        b_low = b.strip().lower()
        if a_low == b_low:
            return True
        if len(a_low) > 10 and a_low in b_low:
            return True
        return len(b_low) > 10 and b_low in a_low

    def _llm_accept(self, statement: str) -> bool:
        """LLM writer/rejecter: 判定 lesson 是否值得写入。"""
        prompt = (
            f"Decide if this lesson is worth saving as a rule. Answer ONLY 'accept' or 'reject'.\nLesson: {statement}"
        )
        try:
            response = self.llm.complete("You are a lesson evaluator.", prompt)  # type: ignore[union-attr]
            lower = response.strip().lower()
            return not ("reject" in lower and "accept" not in lower)
        except Exception as e:
            logger.warning("llm_accept_failed", error=str(e))
            return True

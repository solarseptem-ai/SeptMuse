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
"""检索后自动评估规则是否帮助了当前任务。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from septmuse.core.logging import get_logger
from septmuse.llms.base import LLM
from septmuse.storage.relational_stores.typed_store import TypedMemoryStore

logger = get_logger(__name__)


@dataclass
class RuleFeedback:
    """单条规则反馈。"""

    rule_id: str
    rule_text: str
    helpful: bool  # True=规则帮助了当前任务
    relevance_score: float = 0.0  # 0-1 相关性
    reason: str = ""  # 判断理由


@dataclass
class RetrievalFeedbackResult:
    """检索反馈结果。"""

    query: str
    feedbacks: list[RuleFeedback] = field(default_factory=list)
    total_rules_evaluated: int = 0
    helpful_count: int = 0
    harmful_count: int = 0
    summary: str = ""


class RetrievalFeedback:
    """检索后自动反馈评估器。"""

    def __init__(
        self,
        typed_store: TypedMemoryStore,
        llm: LLM | None = None,
    ) -> None:
        self.typed_store = typed_store
        self.llm = llm

    def evaluate(
        self,
        query: str,
        *,
        retrieved_memories: list[dict[str, Any]],
        user_id: str,
        agent_id: str | None = None,
        namespace: str = "default",
        auto_record: bool = True,
    ) -> RetrievalFeedbackResult:
        """评估检索中注入的规则是否帮助了当前任务。"""
        # 获取当前活跃规则
        rules = self.typed_store.get_active_rules(user_id=user_id, namespace=namespace)
        if not rules:
            return RetrievalFeedbackResult(query=query, total_rules_evaluated=0)

        result = RetrievalFeedbackResult(
            query=query,
            total_rules_evaluated=len(rules),
        )

        # 提取检索记忆文本
        memory_texts = [m.get("memory", "") for m in retrieved_memories if m.get("memory")]

        # 逐条评估规则
        for rule in rules:
            if self.llm is not None:
                feedback = self._llm_evaluate(query, rule.rule, memory_texts, rule.id)
            else:
                feedback = self._heuristic_evaluate(query, rule.rule, memory_texts, rule.id)

            result.feedbacks.append(feedback)
            if feedback.helpful:
                result.helpful_count += 1
            else:
                result.harmful_count += 1

            # 自动记录
            if auto_record:
                try:
                    self.typed_store.record_rule_outcome(rule.id, feedback.helpful)
                except Exception as e:
                    logger.warning("auto_record_rule_outcome_failed", rule_id=rule.id, error=str(e))

        result.summary = (
            f"Evaluated {result.total_rules_evaluated} rules: "
            f"{result.helpful_count} helpful, {result.harmful_count} harmful"
        )
        logger.info(
            "retrieval_feedback_done",
            user_id=user_id,
            query=query[:50],
            helpful=result.helpful_count,
            harmful=result.harmful_count,
        )
        return result

    def _llm_evaluate(
        self,
        query: str,
        rule_text: str,
        memory_texts: list[str],
        rule_id: str,
    ) -> RuleFeedback:
        """LLM 评估单条规则与查询+检索结果的相关性。"""
        system_prompt = (
            "You are a rule relevance evaluator. Given a user query, retrieved memories, "
            "and a procedural rule, determine if the rule is helpful for answering the query. "
            "Answer ONLY 'helpful' or 'harmful' followed by a brief reason."
        )
        memories_str = "\n".join(f"- {t[:100]}" for t in memory_texts[:5])
        user_prompt = (
            f"Query: {query}\n"
            f"Retrieved memories:\n{memories_str}\n"
            f"Rule: {rule_text}\n"
            f"Is this rule helpful for answering the query? (helpful/harmful):"
        )
        try:
            response = self.llm.complete(system_prompt, user_prompt)  # type: ignore[union-attr]
            lower = response.strip().lower()
            helpful = "helpful" in lower and "harmful" not in lower
            return RuleFeedback(
                rule_id=rule_id,
                rule_text=rule_text,
                helpful=helpful,
                relevance_score=0.7 if helpful else 0.3,
                reason=response.strip()[:200],
            )
        except Exception as e:
            logger.warning("llm_evaluate_failed", error=str(e))
            return self._heuristic_evaluate(query, rule_text, memory_texts, rule_id)

    @staticmethod
    def _heuristic_evaluate(
        query: str,
        rule_text: str,
        memory_texts: list[str],
        rule_id: str,
    ) -> RuleFeedback:
        """无 LLM 时 heuristic 评估。"""
        # 简单单词重叠
        query_words = set(query.lower().split())
        rule_words = set(rule_text.lower().split())
        overlap = len(query_words & rule_words)

        # 记忆重叠
        memory_overlap = 0
        for mt in memory_texts[:3]:
            mt_words = set(mt.lower().split())
            memory_overlap += len(rule_words & mt_words)

        relevance = min(1.0, (overlap + memory_overlap) / max(1, len(rule_words)))

        helpful = relevance > 0.1
        return RuleFeedback(
            rule_id=rule_id,
            rule_text=rule_text,
            helpful=helpful,
            relevance_score=relevance,
            reason=f"Heuristic: word overlap={overlap}, memory overlap={memory_overlap}",
        )

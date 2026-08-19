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
"""Agent 自进化闭环编排器。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from sqlmodel import Session, select

from septmuse.core.logging import get_logger
from septmuse.embedders.base import Embedder
from septmuse.evolution.feedback import RetrievalFeedback, RetrievalFeedbackResult
from septmuse.evolution.refine import RefinementResult, RuleRefiner
from septmuse.evolution.reflect import SessionReflector
from septmuse.llms.base import LLM
from septmuse.meta.coverage import CoverageAnalyzer, CoverageReport
from septmuse.models.self_evolution import AgentSelfModel
from septmuse.storage.base import MemoryStore
from septmuse.storage.graph_stores.base import GraphStore
from septmuse.storage.relational_stores.typed_store import TypedMemoryStore

logger = get_logger(__name__)


@dataclass
class EvolutionCycleResult:
    """一次自进化循环的结果。"""

    user_id: str
    agent_id: str | None = None
    # reflect
    lessons_proposed: int = 0
    lessons_accepted: int = 0
    rule_ids: list[str] = field(default_factory=list)
    # refine
    refined_rules: list[RefinementResult] = field(default_factory=list)
    # feedback
    feedback: RetrievalFeedbackResult | None = None
    # self model
    self_model_updated: bool = False
    self_model: dict[str, Any] | None = None
    # coverage
    coverage: dict[str, Any] | None = None
    # 摘要
    summary: str = ""


class SelfEvolutionLoop:
    """Agent 自进化闭环编排器。"""

    def __init__(
        self,
        store: MemoryStore,
        graph_store: GraphStore | None,
        embedder: Embedder,
        typed_store: TypedMemoryStore,
        llm: LLM | None = None,
    ) -> None:
        self.store = store
        self.graph_store = graph_store
        self.embedder = embedder
        self.typed_store = typed_store
        self.llm = llm

        self.reflector = SessionReflector(typed_store, llm=llm)
        self.refiner = RuleRefiner(typed_store, llm=llm)
        self.feedback = RetrievalFeedback(typed_store, llm=llm)
        self.coverage = CoverageAnalyzer(store, typed_store)

    # ------------------------------------------------------------------
    # 闭环核心方法
    # ------------------------------------------------------------------

    def evaluate_recall(
        self,
        query: str,
        retrieved_memories: list[dict[str, Any]],
        *,
        user_id: str,
        agent_id: str | None = None,
        namespace: str = "default",
        auto_record: bool = True,
    ) -> RetrievalFeedbackResult:
        """检索后自动评估规则是否帮助了当前任务。"""
        return self.feedback.evaluate(
            query,
            retrieved_memories=retrieved_memories,
            user_id=user_id,
            agent_id=agent_id,
            namespace=namespace,
            auto_record=auto_record,
        )

    def improve(
        self,
        *,
        user_id: str,
        agent_id: str | None = None,
        limit: int = 20,
        refine: bool = True,
    ) -> EvolutionCycleResult:
        """执行一次完整自进化循环。"""
        result = EvolutionCycleResult(user_id=user_id, agent_id=agent_id)

        # 1. reflect: 提取教训
        if self.llm is not None:
            try:
                reflect_result = self.reflector.reflect(user_id=user_id, limit=limit)
                result.lessons_proposed = reflect_result.lessons_proposed
                result.lessons_accepted = reflect_result.lessons_accepted
                result.rule_ids = reflect_result.rule_ids
            except Exception as e:
                logger.warning("self_loop_reflect_failed", error=str(e))

        # 2. refine: 精炼规则
        if refine:
            try:
                result.refined_rules = self.refiner.refine_all(user_id=user_id)
            except Exception as e:
                logger.warning("self_loop_refine_failed", error=str(e))

        # 3. coverage: 覆盖分析
        coverage_report: CoverageReport | None = None
        try:
            coverage_report = self.coverage.analyze(user_id=user_id)
            result.coverage = {
                "overall_score": coverage_report.overall_score,
                "weak_areas": coverage_report.weak_areas,
                "strong_areas": coverage_report.strong_areas,
                "namespaces": [
                    {
                        "namespace": ns.namespace,
                        "count": ns.count,
                        "avg_confidence": ns.avg_confidence,
                        "coverage_score": ns.coverage_score,
                    }
                    for ns in coverage_report.namespaces
                ],
            }
        except Exception as e:
            logger.warning("self_loop_coverage_failed", error=str(e))

        # 4. self_model: 更新能力画像
        try:
            model = self._update_self_model(
                user_id=user_id,
                agent_id=agent_id,
                coverage=coverage_report,
            )
            result.self_model_updated = True
            result.self_model = model
        except Exception as e:
            logger.warning("self_loop_self_model_failed", error=str(e))

        # 5. 摘要
        parts = []
        if result.lessons_accepted > 0:
            parts.append(f"extracted {result.lessons_accepted} lessons")
        if result.refined_rules:
            refined_count = len([r for r in result.refined_rules if r.refined])
            if refined_count > 0:
                parts.append(f"refined {refined_count} rules")
        if result.self_model_updated:
            parts.append("updated self-model")
        result.summary = ", ".join(parts) or "no evolution occurred"

        logger.info(
            "self_evolution_cycle_done",
            user_id=user_id,
            agent_id=agent_id,
            lessons=result.lessons_accepted,
            refined=len([r for r in result.refined_rules if r.refined]),
            self_model=result.self_model_updated,
        )
        return result

    # ------------------------------------------------------------------
    # AgentSelfModel 管理
    # ------------------------------------------------------------------

    def _update_self_model(
        self,
        *,
        user_id: str,
        agent_id: str | None = None,
        coverage: CoverageReport | None = None,
    ) -> dict[str, Any]:
        """更新 AgentSelfModel。"""
        aid = agent_id or "default"

        # 分析覆盖 (若未提供)
        if coverage is None:
            coverage = self.coverage.analyze(user_id=user_id)

        # 提取强弱领域
        strengths = [ns.namespace for ns in coverage.namespaces if ns.coverage_score > 0.5]
        weaknesses = [ns.namespace for ns in coverage.namespaces if ns.coverage_score < 0.3]

        # 技能分布
        skill_distribution = {}
        for ns in coverage.namespaces:
            skill_distribution[ns.namespace] = {
                "count": ns.count,
                "avg_confidence": round(ns.avg_confidence, 3),
                "coverage_score": round(ns.coverage_score, 3),
            }

        # 最佳规则 (helpful 计数最高)
        top_rules = self._get_top_rules(user_id=user_id, top_n=5)

        # 查找或创建 self model
        model = self._get_or_create(user_id=user_id, agent_id=aid)

        model.strengths = strengths
        model.weaknesses = weaknesses
        model.skill_distribution = skill_distribution
        model.top_rules = top_rules
        model.last_improved_at = datetime.now(timezone.utc)
        model.version += 1
        model.build_evolution_summary()
        model.touch()

        with Session(self.typed_store.engine) as session:
            session.add(model)
            session.commit()
            session.refresh(model)

        logger.info(
            "self_model_updated",
            user_id=user_id,
            agent_id=aid,
            strengths=len(strengths),
            weaknesses=len(weaknesses),
            version=model.version,
        )
        return self._serialize_model(model)

    def _get_or_create(self, *, user_id: str, agent_id: str) -> AgentSelfModel:
        """查找或创建 AgentSelfModel。"""
        with Session(self.typed_store.engine) as session:
            stmt = select(AgentSelfModel).where(
                AgentSelfModel.user_id == user_id,
                AgentSelfModel.agent_id == agent_id,
            )
            existing = session.exec(stmt).first()
            if existing:
                return existing
            model = AgentSelfModel(user_id=user_id, agent_id=agent_id)
            session.add(model)
            session.commit()
            session.refresh(model)
            return model

    def _get_top_rules(self, *, user_id: str, top_n: int = 5) -> list[dict[str, Any]]:
        """获取表现最好的规则 (helpful 计数最高)。"""
        rules = self.typed_store.get_all_rules(user_id=user_id, include_deprecated=False)
        active = [r for r in rules if not r.deprecated and not r.is_deleted]
        active.sort(key=lambda r: (r.helpful_count, r.confidence), reverse=True)
        return [
            {
                "rule_id": r.id,
                "rule": r.rule[:100],
                "confidence": round(r.confidence, 3),
                "helpful_count": r.helpful_count,
                "harmful_count": r.harmful_count,
            }
            for r in active[:top_n]
        ]

    @staticmethod
    def _serialize_model(model: AgentSelfModel) -> dict[str, Any]:
        """序列化 AgentSelfModel 为 dict。"""
        return {
            "agent_id": model.agent_id,
            "user_id": model.user_id,
            "strengths": list(model.strengths),
            "weaknesses": list(model.weaknesses),
            "skill_distribution": dict(model.skill_distribution),
            "top_rules": list(model.top_rules),
            "evolution_summary": model.evolution_summary,
            "recall_success_rate": model.recall_success_rate,
            "total_recalls": model.total_recalls,
            "successful_recalls": model.successful_recalls,
            "version": model.version,
        }

    # ------------------------------------------------------------------
    # 查询 API
    # ------------------------------------------------------------------

    def get_self_model(self, *, user_id: str, agent_id: str | None = None) -> dict[str, Any] | None:
        """获取 agent 的能力画像。"""
        aid = agent_id or "default"
        with Session(self.typed_store.engine) as session:
            stmt = select(AgentSelfModel).where(
                AgentSelfModel.user_id == user_id,
                AgentSelfModel.agent_id == aid,
            )
            model = session.exec(stmt).first()
            if model is None:
                return None
            return self._serialize_model(model)

    def get_rule_evolution_chain(self, rule_id: str) -> list[dict[str, Any]]:
        """获取规则的完整演化链。"""
        snapshots = self.refiner.get_evolution_chain(rule_id)
        return [
            {
                "snapshot_id": s.id,
                "rule_id": s.rule_id,
                "parent_rule_id": s.parent_rule_id,
                "rule": s.rule,
                "version": s.version,
                "change_description": s.change_description,
                "refinement_reason": s.refinement_reason,
                "confidence_at_snapshot": s.confidence_at_snapshot,
            }
            for s in snapshots
        ]

    def record_recall_outcome(self, *, user_id: str, agent_id: str | None = None, success: bool) -> None:
        """记录一次检索结果并更新成功率。"""
        aid = agent_id or "default"
        model = self._get_or_create(user_id=user_id, agent_id=aid)
        model.record_recall(success)
        with Session(self.typed_store.engine) as session:
            session.add(model)
            session.commit()

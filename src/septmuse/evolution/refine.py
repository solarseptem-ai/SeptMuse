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
"""规则版本化精炼引擎。"""

from __future__ import annotations

from dataclasses import dataclass

from sqlmodel import Session

from septmuse.core.logging import get_logger
from septmuse.llms.base import LLM
from septmuse.models.procedural import ProceduralRule
from septmuse.models.self_evolution import RuleSnapshot
from septmuse.storage.relational_stores.typed_store import TypedMemoryStore

logger = get_logger(__name__)

# 精炼阈值
MIN_OUTCOMES_FOR_REFINEMENT = 3  # 至少 3 次应用才考虑精炼
MIXED_RATIO_THRESHOLD = 0.3  # helpful/harmful 比例 < 0.7 视为 mixed


@dataclass
class RefinementCandidate:
    """精炼候选规则。"""

    rule: ProceduralRule
    reason: str  # 精炼原因
    mixed_score: float = 0.0  # 0=纯好, 1=纯坏


@dataclass
class RefinementResult:
    """精炼结果。"""

    original_rule_id: str
    new_rule_id: str | None = None
    snapshot_id: str | None = None
    refined: bool = False
    reason: str = ""
    change_description: str = ""
    new_rule_text: str = ""


class RuleRefiner:
    """规则版本化精炼引擎。"""

    def __init__(
        self,
        typed_store: TypedMemoryStore,
        llm: LLM | None = None,
    ) -> None:
        self.typed_store = typed_store
        self.llm = llm

    def find_candidates(self, *, user_id: str) -> list[RefinementCandidate]:
        """查找精炼候选规则。"""
        rules = self.typed_store.get_all_rules(user_id=user_id, include_deprecated=False)
        candidates: list[RefinementCandidate] = []

        for rule in rules:
            if rule.is_deleted or rule.deprecated:
                continue

            total = rule.helpful_count + rule.harmful_count
            if total < MIN_OUTCOMES_FOR_REFINEMENT:
                continue

            # mixed: 既有 helpful 又有 harmful
            if rule.helpful_count > 0 and rule.harmful_count > 0:
                ratio = rule.helpful_count / total if total > 0 else 0.5
                if ratio < (1 - MIXED_RATIO_THRESHOLD) or ratio > MIXED_RATIO_THRESHOLD:
                    # mixed but not extreme
                    mixed_score = abs(0.5 - ratio) * 2  # 0=纯好/纯坏, 1=完全混合
                    candidates.append(
                        RefinementCandidate(
                            rule=rule,
                            reason=f"Mixed outcomes: {rule.helpful_count} helpful, {rule.harmful_count} harmful",
                            mixed_score=mixed_score,
                        )
                    )

            # harmful 快速增长 (最近 3 次中 2 次 harmful)
            if rule.harmful_count >= 2 and rule.harmful_count > rule.helpful_count:
                candidates.append(
                    RefinementCandidate(
                        rule=rule,
                        reason=f"Harmful growth: {rule.harmful_count} harmful vs {rule.helpful_count} helpful",
                        mixed_score=0.8,
                    )
                )

        candidates.sort(key=lambda c: c.mixed_score, reverse=True)
        return candidates

    def refine(
        self,
        candidate: RefinementCandidate,
        *,
        user_id: str,
        source_event_ids: list[str] | None = None,
    ) -> RefinementResult:
        """精炼单条规则。"""
        rule = candidate.rule
        result = RefinementResult(
            original_rule_id=rule.id,
            reason=candidate.reason,
        )

        # 1. 创建快照 (保存当前版本)
        snapshot = self._create_snapshot(rule, user_id=user_id, reason=candidate.reason)
        result.snapshot_id = snapshot.id

        # 2. 生成精炼版本
        if self.llm is not None:
            refined_text = self._llm_refine(rule, candidate)
        else:
            refined_text = self._heuristic_refine(rule, candidate)

        if not refined_text or refined_text.strip() == rule.rule.strip():
            result.refined = False
            result.reason = "No meaningful refinement produced"
            return result

        # 3. 创建新版本规则
        new_rule = self.typed_store.add_rule(
            refined_text.strip(),
            user_id=user_id,
            namespace=rule.namespace,
            source_tracing=rule.source_tracing,
            tags=list(rule.tags) if rule.tags else [],
        )
        result.new_rule_id = new_rule.id
        result.new_rule_text = refined_text.strip()
        result.change_description = f"Refined from {rule.rule[:50]}... → {refined_text[:50]}..."
        result.refined = True

        # 4. 更新快照的 parent 链接
        snapshot.parent_rule_id = new_rule.id
        if source_event_ids:
            snapshot.source_event_ids = list(source_event_ids)
        with Session(self.typed_store.engine) as session:
            session.add(snapshot)
            session.commit()

        # 5. 标记原规则为 deprecated (让新版本生效)
        with Session(self.typed_store.engine) as session:
            r = session.get(ProceduralRule, rule.id)
            if r:
                r.deprecated = True
                r.touch()
                session.add(r)
                session.commit()

        logger.info(
            "rule_refined",
            original=rule.id,
            new=new_rule.id,
            reason=candidate.reason,
        )
        return result

    def _create_snapshot(
        self,
        rule: ProceduralRule,
        *,
        user_id: str,
        reason: str = "",
    ) -> RuleSnapshot:
        """创建规则当前状态的快照。"""
        # 查找已有版本号
        existing_versions = self._get_version_count(rule.id)
        snapshot = RuleSnapshot(
            rule_id=rule.id,
            parent_rule_id=None,  # 后续更新
            rule=rule.rule,
            version=existing_versions + 1,
            change_description="",
            refinement_reason=reason,
            confidence_at_snapshot=rule.confidence,
            helpful_count=rule.helpful_count,
            harmful_count=rule.harmful_count,
            user_id=user_id,
        )
        with Session(self.typed_store.engine) as session:
            session.add(snapshot)
            session.commit()
            session.refresh(snapshot)
        return snapshot

    def _get_version_count(self, rule_id: str) -> int:
        """获取规则已有版本数。"""
        from sqlmodel import select

        with Session(self.typed_store.engine) as session:
            stmt = select(RuleSnapshot).where(RuleSnapshot.rule_id == rule_id)
            return len(list(session.exec(stmt).all()))

    def _llm_refine(self, rule: ProceduralRule, candidate: RefinementCandidate) -> str:
        """LLM 分析规则并生成精炼版本。"""
        system_prompt = (
            "You are a rule refinement assistant. A rule has been partially helpful and partially harmful. "
            "Analyze why the rule sometimes fails and produce a refined version that keeps what works "
            "and fixes what doesn't. Return ONLY the refined rule text, nothing else."
        )
        user_prompt = (
            f"Original rule: {rule.rule}\n"
            f"Helpful count: {rule.helpful_count}\n"
            f"Harmful count: {rule.harmful_count}\n"
            f"Reason for refinement: {candidate.reason}\n"
            f"Please output the refined rule:"
        )
        try:
            response = self.llm.complete(system_prompt, user_prompt)  # type: ignore[union-attr]
            return response.strip()
        except Exception as e:
            logger.warning("llm_refine_failed", error=str(e))
            return self._heuristic_refine(rule, candidate)

    @staticmethod
    def _heuristic_refine(rule: ProceduralRule, candidate: RefinementCandidate) -> str:
        """无 LLM 时 heuristic 精炼: 在原文后加条件限定。

        简单策略: 如果 harmful > helpful, 在规则后加 "(when applicable)" 限定。
        """
        if rule.harmful_count > rule.helpful_count:
            return f"{rule.rule} (when applicable)"
        return rule.rule

    def get_evolution_chain(self, rule_id: str) -> list[RuleSnapshot]:
        """获取规则的完整演化链 (所有版本快照)。"""
        from sqlmodel import select

        with Session(self.typed_store.engine) as session:
            stmt = select(RuleSnapshot).where(
                RuleSnapshot.rule_id == rule_id
            ).order_by(RuleSnapshot.version)
            return list(session.exec(stmt).all())

    def refine_all(
        self,
        *,
        user_id: str,
        max_refinements: int = 3,
    ) -> list[RefinementResult]:
        """批量精炼候选规则。"""
        candidates = self.find_candidates(user_id=user_id)
        results: list[RefinementResult] = []
        for candidate in candidates[:max_refinements]:
            try:
                result = self.refine(candidate, user_id=user_id)
                results.append(result)
            except Exception as e:
                logger.warning("refine_all_item_failed", rule_id=candidate.rule.id, error=str(e))
        logger.info(
            "refine_all_done",
            user_id=user_id,
            candidates=len(candidates),
            refined=len([r for r in results if r.refined]),
        )
        return results

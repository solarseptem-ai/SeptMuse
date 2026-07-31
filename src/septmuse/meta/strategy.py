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
"""L2 策略自调 — 基于覆盖报告自调检索策略 (架构文档 §6.3 自研)。

基于 L1 覆盖报告, 自调检索策略:
- 覆盖薄弱 → 加深检索 (lower threshold, increase top_k)
- 覆盖极弱 → 触发澄清提问 (ask user for more info)
- 覆盖充分 → 维持当前策略

驱动: 检索前先查 L0 路由 + L1 覆盖, 若覆盖薄弱 → L2 触发"主动澄清提问"或"加深检索"。

详见 docs/specs/agent-memory-architecture.md §6.3 元认知。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from septmuse.core.logging import get_logger
from septmuse.meta.coverage import CoverageReport

logger = get_logger(__name__)


class StrategyAction(str, Enum):
    """L2 策略动作 (架构文档 §6.3)。"""

    DEEPEN_RETRIEVAL = "deepen_retrieval"  # 加深检索 (lower threshold, increase top_k)
    TRIGGER_CLARIFICATION = "trigger_clarification"  # 触发澄清提问 (ask user)
    SWITCH_SOURCE = "switch_source"  # 换源 (查其他命名空间)
    MAINTAIN = "maintain"  # 维持当前策略 (覆盖充分)


@dataclass
class StrategyRecommendation:
    """策略推荐。"""

    action: StrategyAction
    namespace: str = ""
    reason: str = ""
    suggested_top_k: int = 5
    suggested_threshold: float = 0.1
    clarification_question: str | None = None


@dataclass
class StrategyResult:
    """策略自调结果。"""

    recommendations: list[StrategyRecommendation] = field(default_factory=list)
    overall_action: StrategyAction = StrategyAction.MAINTAIN

    @property
    def needs_clarification(self) -> bool:
        """是否需要向用户提问。"""
        return any(r.action == StrategyAction.TRIGGER_CLARIFICATION for r in self.recommendations)

    @property
    def needs_deepened_retrieval(self) -> bool:
        """是否需要加深检索。"""
        return any(r.action == StrategyAction.DEEPEN_RETRIEVAL for r in self.recommendations)


class StrategyAdapter:
    """L2 策略自调器 (架构文档 §6.3 自研)。

    基于覆盖报告, 自调检索策略。

    用法:
        adapter = StrategyAdapter()
        result = adapter.adapt(report)
        if result.needs_clarification:
            ask_user(clarification_question)
        if result.needs_deepened_retrieval:
            search(top_k=10, threshold=0.05)  # 加深
    """

    # 覆盖阈值
    WEAK_THRESHOLD = 0.3  # < 0.3 → deepen
    VERY_WEAK_THRESHOLD = 0.1  # < 0.1 → trigger clarification
    STRONG_THRESHOLD = 0.5  # > 0.5 → maintain

    def adapt(self, report: CoverageReport) -> StrategyResult:
        """基于覆盖报告自调策略 (架构文档 §6.3)。"""
        result = StrategyResult()
        weakest_score = 1.0

        for ns in report.namespaces:
            if ns.coverage_score < self.VERY_WEAK_THRESHOLD:
                # 极弱 → 触发澄清提问
                rec = StrategyRecommendation(
                    action=StrategyAction.TRIGGER_CLARIFICATION,
                    namespace=ns.namespace,
                    reason=f"Coverage very weak ({ns.coverage_score:.2f}), need user input",
                    clarification_question=f"I don't have much information about {ns.namespace}. Can you tell me more?",
                )
                result.recommendations.append(rec)
                weakest_score = min(weakest_score, ns.coverage_score)
            elif ns.coverage_score < self.WEAK_THRESHOLD:
                # 弱 → 加深检索
                rec = StrategyRecommendation(
                    action=StrategyAction.DEEPEN_RETRIEVAL,
                    namespace=ns.namespace,
                    reason=f"Coverage weak ({ns.coverage_score:.2f}), deepening retrieval",
                    suggested_top_k=10,
                    suggested_threshold=0.05,
                )
                result.recommendations.append(rec)
                weakest_score = min(weakest_score, ns.coverage_score)
            else:
                # 充分 → 维持
                rec = StrategyRecommendation(
                    action=StrategyAction.MAINTAIN,
                    namespace=ns.namespace,
                    reason=f"Coverage sufficient ({ns.coverage_score:.2f})",
                )
                result.recommendations.append(rec)

        # 总体策略: 最弱的决定
        if weakest_score < self.VERY_WEAK_THRESHOLD:
            result.overall_action = StrategyAction.TRIGGER_CLARIFICATION
        elif weakest_score < self.WEAK_THRESHOLD:
            result.overall_action = StrategyAction.DEEPEN_RETRIEVAL
        else:
            result.overall_action = StrategyAction.MAINTAIN

        logger.info(
            "strategy_adapted",
            overall_action=result.overall_action.value,
            recommendations=len(result.recommendations),
            needs_clarification=result.needs_clarification,
        )
        return result

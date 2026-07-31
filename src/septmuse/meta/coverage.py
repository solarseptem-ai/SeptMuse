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
"""L1 覆盖自描述 — "我记住了什么/记不住什么" (架构文档 §6.3 自研)。

ReMe 仅 L0 路由命名空间, agent 不"知道"自己记住了什么。
SeptMuse 扩展元认知为三层, L1 扫描所有命名空间生成覆盖报告。

CoverageReport 存为语义记忆, 打 `meta` + `coverage` 标签, 跨会话累积。

详见 docs/specs/agent-memory-architecture.md §6.3 元认知。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from septmuse.core.logging import get_logger
from septmuse.storage.base import MemoryStore
from septmuse.storage.typed_store import TypedMemoryStore

logger = get_logger(__name__)

# 覆盖满分阈值 (10 条记忆 = 满分覆盖)
COVERAGE_FULL_THRESHOLD = 10


@dataclass
class NamespaceStats:
    """单命名空间覆盖统计。"""

    namespace: str
    count: int = 0
    avg_confidence: float = 0.0
    coverage_score: float = 0.0  # 0-1, 越高越好
    sample_topics: list[str] = field(default_factory=list)  # 代表性内容摘要


@dataclass
class CoverageReport:
    """记忆覆盖报告 (L1 自描述, 架构文档 §6.3)。"""

    user_id: str
    namespaces: list[NamespaceStats] = field(default_factory=list)
    overall_score: float = 0.0
    weak_areas: list[str] = field(default_factory=list)  # 覆盖薄弱的命名空间
    strong_areas: list[str] = field(default_factory=list)  # 覆盖充分的命名空间
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def get_namespace(self, name: str) -> NamespaceStats | None:
        """获取指定命名空间统计。"""
        return next((ns for ns in self.namespaces if ns.namespace == name), None)

    def summary(self) -> str:
        """自然语言摘要 (用于注入 prompt 或存为语义记忆)。"""
        lines = [f"Coverage report for {self.user_id}:"]
        for ns in self.namespaces:
            status = "strong" if ns.coverage_score > 0.5 else "weak"
            lines.append(f"  {ns.namespace}: {ns.count} items, score={ns.coverage_score:.2f} ({status})")
        if self.weak_areas:
            lines.append(f"  Weak: {', '.join(self.weak_areas)}")
        if self.strong_areas:
            lines.append(f"  Strong: {', '.join(self.strong_areas)}")
        return "\n".join(lines)


class CoverageAnalyzer:
    """L1 覆盖分析器 (架构文档 §6.3 自研)。

    扫描所有命名空间, 按主题/时间/置信度统计覆盖, 生成自述报告。

    用法:
        analyzer = CoverageAnalyzer(store, typed_store)
        report = analyzer.analyze(user_id="alice")
        print(report.summary())
        # "Coverage report for alice:
        #  verbatim: 5 items, score=0.50 (weak)
        #  semantic: 3 items, score=0.30 (weak)
        #  ..."
    """

    def __init__(self, store: MemoryStore, typed_store: TypedMemoryStore) -> None:
        self.store = store
        self.typed_store = typed_store

    def analyze(self, *, user_id: str) -> CoverageReport:
        """生成覆盖报告 (扫描全部命名空间)。"""
        report = CoverageReport(user_id=user_id)

        # 1. verbatim memories
        verbatim = self.store.get_all(user_id=user_id)
        verbatim_topics = [m["memory"][:30] for m in verbatim[:5]]
        report.namespaces.append(
            NamespaceStats(
                namespace="verbatim",
                count=len(verbatim),
                avg_confidence=1.0,
                coverage_score=min(1.0, len(verbatim) / COVERAGE_FULL_THRESHOLD),
                sample_topics=verbatim_topics,
            )
        )

        # 2. semantic facts
        facts = self.typed_store.get_all_facts(user_id=user_id)
        avg_conf = sum(f.confidence for f in facts) / len(facts) if facts else 0.0
        fact_topics = [f"{f.subject} {f.predicate} {f.object}"[:30] for f in facts[:5]]
        report.namespaces.append(
            NamespaceStats(
                namespace="semantic",
                count=len(facts),
                avg_confidence=avg_conf,
                coverage_score=min(1.0, len(facts) / COVERAGE_FULL_THRESHOLD) * avg_conf,
                sample_topics=fact_topics,
            )
        )

        # 3. episodic events
        episodes = self.typed_store.get_episodes(user_id=user_id, limit=1000)
        ep_topics = [e.content[:30] for e in episodes[:5] if e.content]
        report.namespaces.append(
            NamespaceStats(
                namespace="episodic",
                count=len(episodes),
                avg_confidence=0.8,  # episodic 默认 0.8
                coverage_score=min(1.0, len(episodes) / COVERAGE_FULL_THRESHOLD),
                sample_topics=ep_topics,
            )
        )

        # 4. procedural rules
        rules = self.typed_store.get_all_rules(user_id=user_id, include_deprecated=True)
        active_rules = [r for r in rules if not r.deprecated and not r.is_deleted]
        avg_rule_conf = sum(r.confidence for r in active_rules) / len(active_rules) if active_rules else 0.0
        rule_topics = [r.rule[:30] for r in active_rules[:5]]
        report.namespaces.append(
            NamespaceStats(
                namespace="procedural",
                count=len(active_rules),
                avg_confidence=avg_rule_conf,
                coverage_score=min(1.0, len(active_rules) / COVERAGE_FULL_THRESHOLD) * max(0.1, avg_rule_conf),
                sample_topics=rule_topics,
            )
        )

        # 5. 计算总体分数 + 强弱区域
        if report.namespaces:
            report.overall_score = sum(ns.coverage_score for ns in report.namespaces) / len(report.namespaces)
        report.weak_areas = [ns.namespace for ns in report.namespaces if ns.coverage_score < 0.3]
        report.strong_areas = [ns.namespace for ns in report.namespaces if ns.coverage_score > 0.5]

        logger.info(
            "coverage_analyzed",
            user_id=user_id,
            overall_score=report.overall_score,
            weak=len(report.weak_areas),
            strong=len(report.strong_areas),
        )
        return report

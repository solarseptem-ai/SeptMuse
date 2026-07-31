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
"""阶段4 §6.3 元认知 L1/L2 单元测试 — 覆盖自描述 + 策略自调。

固化 (架构文档 §6.3 自研):
- CoverageAnalyzer: L1 扫描命名空间, 生成覆盖报告
- StrategyAdapter: L2 基于覆盖报告自调检索策略
- L0 路由已在阶段3 实现 (concerns/metacognition/router.py)
"""

from __future__ import annotations

import pytest

from septmuse import MemoryConfig
from septmuse.embedders.hash import HashEmbedder
from septmuse.experimental import ExperimentalMemory
from septmuse.meta.coverage import CoverageAnalyzer, CoverageReport, NamespaceStats
from septmuse.meta.strategy import StrategyAction, StrategyAdapter, StrategyResult
from septmuse.prompts.coverage_report import COVERAGE_REPORT_PROMPT, build_coverage_report_message


@pytest.fixture()
def mem() -> ExperimentalMemory:
    return ExperimentalMemory(
        config=MemoryConfig(db_path=":memory:"),
        embedder=HashEmbedder(),
    )


# ======================================================================
# CoverageAnalyzer (L1)
# ======================================================================


class TestCoverageAnalyzer:
    def test_empty_store(self, mem: ExperimentalMemory) -> None:
        analyzer = CoverageAnalyzer(mem.store, mem.typed_store)
        report = analyzer.analyze(user_id="alice")
        assert len(report.namespaces) == 4
        assert all(ns.count == 0 for ns in report.namespaces)
        assert report.overall_score == 0.0
        assert all(ns.namespace in report.weak_areas for ns in report.namespaces)

    def test_with_verbatim(self, mem: ExperimentalMemory) -> None:
        mem.add("alice likes python", user_id="alice")
        mem.add("alice uses fastapi", user_id="alice")
        analyzer = CoverageAnalyzer(mem.store, mem.typed_store)
        report = analyzer.analyze(user_id="alice")
        verbatim = report.get_namespace("verbatim")
        assert verbatim is not None
        assert verbatim.count == 2
        assert verbatim.coverage_score > 0

    def test_with_semantic(self, mem: ExperimentalMemory) -> None:
        mem.add_fact("alice", "likes", "python", user_id="alice", confidence=0.9)
        analyzer = CoverageAnalyzer(mem.store, mem.typed_store)
        report = analyzer.analyze(user_id="alice")
        semantic = report.get_namespace("semantic")
        assert semantic is not None
        assert semantic.count == 1
        assert semantic.avg_confidence == pytest.approx(0.9)

    def test_with_episodic(self, mem: ExperimentalMemory) -> None:
        mem.add_episode("debugging session", user_id="alice", event_type="fact")
        analyzer = CoverageAnalyzer(mem.store, mem.typed_store)
        report = analyzer.analyze(user_id="alice")
        episodic = report.get_namespace("episodic")
        assert episodic is not None
        assert episodic.count == 1

    def test_with_procedural(self, mem: ExperimentalMemory) -> None:
        mem.add_rule("always test edge cases", user_id="alice")
        analyzer = CoverageAnalyzer(mem.store, mem.typed_store)
        report = analyzer.analyze(user_id="alice")
        procedural = report.get_namespace("procedural")
        assert procedural is not None
        assert procedural.count == 1

    def test_strong_areas(self, mem: ExperimentalMemory) -> None:
        # Add > 5 verbatim memories → strong
        for i in range(6):
            mem.add(f"memory {i}", user_id="alice")
        analyzer = CoverageAnalyzer(mem.store, mem.typed_store)
        report = analyzer.analyze(user_id="alice")
        assert "verbatim" in report.strong_areas

    def test_weak_areas(self, mem: ExperimentalMemory) -> None:
        mem.add("one memory", user_id="alice")  # only 1 verbatim
        analyzer = CoverageAnalyzer(mem.store, mem.typed_store)
        report = analyzer.analyze(user_id="alice")
        assert "semantic" in report.weak_areas
        assert "episodic" in report.weak_areas
        assert "procedural" in report.weak_areas

    def test_summary(self, mem: ExperimentalMemory) -> None:
        mem.add("hello", user_id="alice")
        analyzer = CoverageAnalyzer(mem.store, mem.typed_store)
        report = analyzer.analyze(user_id="alice")
        summary = report.summary()
        assert "alice" in summary
        assert "verbatim" in summary

    def test_get_namespace_not_found(self, mem: ExperimentalMemory) -> None:
        analyzer = CoverageAnalyzer(mem.store, mem.typed_store)
        report = analyzer.analyze(user_id="alice")
        assert report.get_namespace("nonexistent") is None


# ======================================================================
# StrategyAdapter (L2)
# ======================================================================


class TestStrategyAdapter:
    def test_maintain_when_strong(self) -> None:
        report = CoverageReport(
            user_id="alice",
            namespaces=[
                NamespaceStats(namespace="semantic", count=10, coverage_score=0.8),
            ],
        )
        adapter = StrategyAdapter()
        result = adapter.adapt(report)
        assert result.overall_action == StrategyAction.MAINTAIN
        assert not result.needs_clarification
        assert not result.needs_deepened_retrieval

    def test_deepen_when_weak(self) -> None:
        report = CoverageReport(
            user_id="alice",
            namespaces=[
                NamespaceStats(namespace="semantic", count=2, coverage_score=0.2),
            ],
        )
        adapter = StrategyAdapter()
        result = adapter.adapt(report)
        assert result.overall_action == StrategyAction.DEEPEN_RETRIEVAL
        assert result.needs_deepened_retrieval
        assert not result.needs_clarification

    def test_clarification_when_very_weak(self) -> None:
        report = CoverageReport(
            user_id="alice",
            namespaces=[
                NamespaceStats(namespace="episodic", count=0, coverage_score=0.0),
            ],
        )
        adapter = StrategyAdapter()
        result = adapter.adapt(report)
        assert result.overall_action == StrategyAction.TRIGGER_CLARIFICATION
        assert result.needs_clarification

    def test_mixed_coverage(self) -> None:
        report = CoverageReport(
            user_id="alice",
            namespaces=[
                NamespaceStats(namespace="verbatim", count=10, coverage_score=0.8),
                NamespaceStats(namespace="semantic", count=0, coverage_score=0.0),
            ],
        )
        adapter = StrategyAdapter()
        result = adapter.adapt(report)
        # Overall = weakest (semantic is very weak)
        assert result.overall_action == StrategyAction.TRIGGER_CLARIFICATION
        assert result.needs_clarification

    def test_deepened_retrieval_params(self) -> None:
        report = CoverageReport(
            user_id="alice",
            namespaces=[
                NamespaceStats(namespace="semantic", count=2, coverage_score=0.2),
            ],
        )
        adapter = StrategyAdapter()
        result = adapter.adapt(report)
        deepen_recs = [r for r in result.recommendations if r.action == StrategyAction.DEEPEN_RETRIEVAL]
        assert len(deepen_recs) >= 1
        assert deepen_recs[0].suggested_top_k == 10
        assert deepen_recs[0].suggested_threshold == 0.05

    def test_clarification_question(self) -> None:
        report = CoverageReport(
            user_id="alice",
            namespaces=[
                NamespaceStats(namespace="episodic", count=0, coverage_score=0.0),
            ],
        )
        adapter = StrategyAdapter()
        result = adapter.adapt(report)
        clarify_recs = [r for r in result.recommendations if r.action == StrategyAction.TRIGGER_CLARIFICATION]
        assert len(clarify_recs) >= 1
        assert clarify_recs[0].clarification_question is not None
        assert "episodic" in clarify_recs[0].clarification_question

    def test_strategy_result_dataclass(self) -> None:
        result = StrategyResult()
        assert result.recommendations == []
        assert result.overall_action == StrategyAction.MAINTAIN


# ======================================================================
# CoverageReport Prompt
# ======================================================================


class TestCoverageReportPrompt:
    def test_prompt_exists(self) -> None:
        assert "metacognitive assistant" in COVERAGE_REPORT_PROMPT

    def test_build_message(self) -> None:
        report = CoverageReport(
            user_id="alice",
            namespaces=[
                NamespaceStats(namespace="semantic", count=5, avg_confidence=0.8, coverage_score=0.5),
            ],
            overall_score=0.5,
            weak_areas=[],
            strong_areas=["semantic"],
        )
        msg = build_coverage_report_message(report)
        assert "semantic" in msg
        assert "0.50" in msg
        assert "Strong areas: semantic" in msg

    def test_build_message_with_weak(self) -> None:
        report = CoverageReport(
            user_id="alice",
            namespaces=[
                NamespaceStats(namespace="episodic", count=0, coverage_score=0.0),
            ],
            weak_areas=["episodic"],
        )
        msg = build_coverage_report_message(report)
        assert "Weak areas: episodic" in msg

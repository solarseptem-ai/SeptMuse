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
"""覆盖报告提示 — LLM 生成自然语言覆盖自述 (架构文档 §6.3 自研)。

CoverageReport 存为语义记忆, 打 `meta` + `coverage` 标签, 跨会话累积。
LLM 可用此提示生成自然语言自述, 注入 agent 的 system prompt。
"""

from __future__ import annotations

from septmuse.meta.coverage import CoverageReport

COVERAGE_REPORT_PROMPT = """You are a metacognitive assistant. Given a coverage report of an AI agent's memory system, generate a natural language self-description of what the agent remembers and what it doesn't.

The self-description should:
1. Acknowledge what the agent knows well (strong areas)
2. Admit what the agent has limited knowledge about (weak areas)
3. Suggest what information would help fill the gaps

Keep it concise (3-5 sentences) and first-person ("I remember...", "I don't have much information about...").
"""


def build_coverage_report_message(report: CoverageReport) -> str:
    """构建覆盖报告的 user message (用于 LLM 生成自然语言自述)。"""
    lines = ["Coverage report data:"]
    for ns in report.namespaces:
        lines.append(
            f"  {ns.namespace}: {ns.count} items, "
            f"avg_confidence={ns.avg_confidence:.2f}, "
            f"coverage_score={ns.coverage_score:.2f}"
        )
        if ns.sample_topics:
            lines.append(f"    samples: {', '.join(ns.sample_topics[:3])}")
    if report.weak_areas:
        lines.append(f"\nWeak areas: {', '.join(report.weak_areas)}")
    if report.strong_areas:
        lines.append(f"Strong areas: {', '.join(report.strong_areas)}")
    lines.append(f"\nOverall coverage score: {report.overall_score:.2f}")
    return "\n".join(lines)

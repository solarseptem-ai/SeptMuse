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
"""因果抽取提示 — LLM 从事件序列识别因果关系 (架构文档 §6.1 自研)。

在 cognify 流水线增加 extract_causal_edges 阶段,
LLM 从事件序列中识别因果, 输出结构化因果边。
"""

from __future__ import annotations

CAUSAL_EXTRACTION_PROMPT = """You are a causal reasoning assistant. Analyze the sequence of events and identify causal relationships between them.

For each causal relationship, output a JSON object with:
- cause_event_id: the ID of the cause event
- effect_event_id: the ID of the effect event
- relation: one of "enables" (X makes Y possible), "prevents" (X stops Y), "causes" (X directly leads to Y), "inhibits" (X partially blocks Y)
- confidence: 0.0 to 1.0 (how certain you are)

Only identify relationships where there is clear evidence of causality, not mere correlation or temporal sequence.

Return a JSON array of causal edges. If no causal relationships exist, return an empty array [].
"""


def build_causal_extraction_message(events: list[dict]) -> str:
    """构建因果抽取的 user message (事件序列)。

    Args:
        events: 事件列表, 每个含 id, content, observation, action, result
    """
    lines = ["Analyze these events for causal relationships:\n"]
    for i, e in enumerate(events, 1):
        lines.append(f"Event {i} (id={e.get('id', '?')}):")
        if e.get("content"):
            lines.append(f"  Content: {e['content']}")
        if e.get("observation"):
            lines.append(f"  Observation: {e['observation']}")
        if e.get("action"):
            lines.append(f"  Action: {e['action']}")
        if e.get("result"):
            lines.append(f"  Result: {e['result']}")
        lines.append("")
    lines.append("Return JSON array of causal edges:")
    return "\n".join(lines)


COUNTERFACTUAL_PROMPT = """You are a counterfactual reasoning assistant. Given a causal chain, determine whether the effect would still have occurred without the cause.

Analyze:
1. The cause event and its relation to the effect
2. Whether there are alternative causal paths to the same effect
3. Whether the cause was necessary or sufficient

Output JSON:
{
  "would_still_occur": true/false,
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation"
}
"""


def build_counterfactual_message(
    cause_content: str,
    effect_content: str,
    relation: str,
    alternative_paths: list[str] | None = None,
) -> str:
    """构建反事实查询的 user message。"""
    lines = [
        f"Cause event: {cause_content}",
        f"Effect event: {effect_content}",
        f"Relation: {relation}",
    ]
    if alternative_paths:
        lines.append(f"Alternative causal paths to the same effect: {alternative_paths}")
    else:
        lines.append("No alternative causal paths found.")
    lines.append("\nWould the effect still have occurred without the cause?")
    return "\n".join(lines)

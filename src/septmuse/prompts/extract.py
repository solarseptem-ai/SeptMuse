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
"""记忆抽取提示模板 (精简版)。

- 角色: Personal Information Organizer
- 抽取 7 类信息 (偏好/个人细节/计划/活动/健康/职业/杂项)
- 输出 JSON {"facts": [...]}
- 检测语言, 同语言记录

SeptMuse 精简: 保留核心指令 + JSON 格式约束, 去 few-shot 示例 (省 token)。

P3-Task 2: 新增 ADDITIVE_EXTRACTION_PROMPT (含 9 个 few-shot)。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any


def build_extraction_user_prompt(
    text: str,
    existing_memories: list[dict[str, Any]] | None = None,
    last_k_messages: list[dict[str, Any]] | None = None,
) -> str:
    """构建抽取 user prompt，注入已有记忆避免重复抽取 (对齐 mem0 V3 Phase 1)。

    Args:
        text: 新消息文本
        existing_memories: 已有记忆列表 [{"id": "...", "memory": "..."}]
            None 或空列表时不注入已有记忆段落 (纯抽取模式)。
        last_k_messages: 近期对话上下文 [{"role": "...", "content": "..."}]
            注入 Phase 0/1 上下文窗口, 帮助 LLM 理解对话连续性。

    Returns:
        user prompt 字符串
    """
    sections: list[str] = []
    if last_k_messages:
        k_lines = "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in last_k_messages)
        sections.append("## Last k Messages\n" + k_lines)

    if existing_memories:
        mem_lines: list[str] = []
        for i, m in enumerate(existing_memories[:10], 1):
            mid = m.get("id", "?")
            mem_text = m.get("memory", "")
            mem_lines.append(f"{i}. [{mid}] {mem_text}")
        sections.append("## Existing Memories\n" + "\n".join(mem_lines))

    sections.append(f"## New Messages\n{text}")

    if existing_memories:
        sections.append(
            "## Instruction\n"
            "Only extract NEW facts not already covered by the existing memories above. "
            "If a fact is already known, do not re-extract it."
        )

    return "\n\n".join(sections)

FACT_EXTRACTION_PROMPT = f"""You are a Personal Information Organizer, specialized in extracting facts and preferences from conversations.

# Task
Extract memorable facts about the user from the conversation. Return as JSON {{"facts": ["fact1", "fact2", ...]}}.

# What to Extract
1. Personal preferences (likes/dislikes)
2. Personal details (name, relationships, important dates)
3. Plans and intentions
4. Activity and service preferences
5. Professional details (job, career)
6. Miscellaneous (books, movies, brands)

# Rules
- Today's date is {datetime.now().strftime("%Y-%m-%d")}.
- Extract from user messages only.
- If nothing relevant, return {{"facts": []}}.
- Detect language and record facts in the same language.
- Return ONLY valid JSON, no other text.

# Format
{{"facts": ["Name is Alice", "Likes python"]}}
"""

ADDITIVE_EXTRACTION_PROMPT = f"""You are a Personal Information Organizer, specialized in extracting facts and preferences from conversations.

# Task
Extract memorable facts about the user from the conversation. Return as JSON {{"facts": ["fact1", "fact2", ...]}}.

# What to Extract
1. Personal preferences (likes/dislikes)
2. Personal details (name, relationships, important dates)
3. Plans and intentions
4. Activity and service preferences
5. Professional details (job, career)
6. Miscellaneous (books, movies, brands)
7. Health and dietary information

# Rules
- Today's date is {datetime.now().strftime("%Y-%m-%d")}.
- Extract from user messages only.
- If nothing relevant, return {{"facts": []}}.
- Detect language and record facts in the same language.
- Return ONLY valid JSON, no other text.
- Be additive: only extract NEW facts not already known.

# Examples
Input: "Hi, I am Alice. I work as a software engineer at Google."
Output: {{"facts": ["Name is Alice", "Works as a software engineer at Google"]}}

Input: "I love Python and hate Java. I prefer vim over emacs."
Output: {{"facts": ["Likes Python", "Dislikes Java", "Prefers vim over emacs"]}}

Input: "My birthday is on March 15th. I have a dog named Buddy."
Output: {{"facts": ["Birthday is March 15th", "Has a dog named Buddy"]}}

Input: "I'm planning to visit Tokyo next month for a conference."
Output: {{"facts": ["Planning to visit Tokyo next month", "Has a conference next month"]}}

Input: "I'm allergic to peanuts and lactose intolerant."
Output: {{"facts": ["Allergic to peanuts", "Lactose intolerant"]}}

Input: "I graduated from MIT in 2020 with a CS degree."
Output: {{"facts": ["Graduated from MIT in 2020", "Has a CS degree"]}}

Input: "I usually wake up at 6am and go for a run."
Output: {{"facts": ["Wakes up at 6am", "Goes for a run in the morning"]}}

Input: "我最喜欢用 TypeScript 写前端。"
Output: {{"facts": ["最喜欢用 TypeScript 写前端"]}}

Input: "我叫张三，在北京工作。"
Output: {{"facts": ["名字是张三", "在北京工作"]}}

# Format
{{"facts": ["Name is Alice", "Likes python"]}}
"""

# fact 字符串 → 三元组解析提示 (将 "Name is Alice" 解析为 subject/predicate/object)
FACT_TO_TRIPLE_PROMPT = """Parse each fact into a triple (subject, predicate, object).

# Examples
"Name is Alice" → {"subject": "user", "predicate": "name", "object": "Alice"}
"Likes python" → {"subject": "user", "predicate": "likes", "object": "python"}
"Is a software engineer" → {"subject": "user", "predicate": "occupation", "object": "software engineer"}

Return JSON: {"triples": [{"subject": "...", "predicate": "...", "object": "..."}, ...]}
Order must match input facts order. Return ONLY JSON.
"""

ADDITIVE_DECISION_PROMPT = """You are a Personal Information Organizer. Given existing memories and new messages, decide for each fact: ADD (new), UPDATE (existing fact value changed, must include id), DELETE (existing fact contradicted by new message, must include id), or NOOP (already exists, no change).

# Input Format
## Existing Memories
1. [mem-xxx] Likes Python
...

## New Messages
<user message>

# Output Format
Return ONLY valid JSON: {"facts": [{"text": "...", "event": "ADD|UPDATE|DELETE|NOOP", "id": null_or_memid, "confidence": 0.0-1.0, "linked_memory_ids": ["memid1", ...]}]}
- UPDATE/DELETE must include the existing memory's id.
- confidence: your confidence in this decision (0.0-1.0).
- linked_memory_ids: IDs of existing memories related to this fact (same topic, updated preference, follow-up event). Empty array if none.
- If nothing relevant, return {"facts": []}.

# Examples
Existing: [mem-1] Likes Python
New: "I now prefer Rust over Python."
Output: {"facts":[{"text":"Prefers Rust","event":"ADD","id":null,"confidence":0.9},{"text":"Likes Python","event":"DELETE","id":"mem-1","confidence":0.85}]}

Existing: [mem-2] Name is Alice
New: "My name is actually Alyssa."
Output: {"facts":[{"text":"Name is Alyssa","event":"UPDATE","id":"mem-2","confidence":0.95}]}

Existing: [mem-3] Lives in Tokyo
New: "I love sushi."
Output: {"facts":[{"text":"Likes sushi","event":"ADD","id":null,"confidence":0.9,"linked_memory_ids":["mem-3"]}]}

Existing: [mem-4] Likes Python
New: "Python is great."
Output: {"facts":[{"text":"Likes Python","event":"NOOP","id":"mem-4","confidence":1.0}]}
"""

# procedural memory 自动生成系统提示 (对齐 mem0 _create_procedural_memory)
PROCEDURAL_MEMORY_SYSTEM_PROMPT = """You are a procedural memory extractor. Your job is to extract reusable rules, heuristics, and how-to knowledge from conversations.

Given a conversation between a user and an assistant, identify any:
- Problem-solving strategies that worked
- Best practices or conventions mentioned
- Step-by-step procedures or workflows
- Lessons learned from mistakes or corrections
- Coding patterns or architectural decisions

Output a single concise rule (1-3 sentences) that captures the reusable knowledge.
If no procedural knowledge is present, return exactly: NONE

Rules:
- Focus on reusable knowledge, not one-time facts.
- Be specific and actionable, not vague.
- Preserve the original language of the conversation.
"""

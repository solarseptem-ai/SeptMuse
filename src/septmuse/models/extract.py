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
"""cognify 抽取流水线 — LLM 抽取事实 + 三元组存储 (架构文档 §3.2.2)。

源码参考 (实证):
- mem0 FACT_RETRIEVAL_PROMPT + normalize_facts (LLM 输出 {"facts": [...]})
- Cognee cognify 流水线 (classify → extract → summarize → store)
- mem0 add infer=True (LLM 抽取 + 存储)

SeptMuse 流程:
1. parse_messages → 文本
2. LLM complete(FACT_EXTRACTION_PROMPT) → {"facts": [...]}
3. normalize_facts (对齐 mem0, 处理 str/dict 变体)
4. fact → 三元组 (简单规则解析, 避免二次 LLM)
5. 存为 SemanticFact (subject/predicate/object) + verbatim memory (向量检索)
"""

from __future__ import annotations

import json
import re
from typing import Any

from septmuse.core.logging import get_logger
from septmuse.embedders.base import Embedder
from septmuse.llms.base import LLM
from septmuse.prompts.extract import ADDITIVE_EXTRACTION_PROMPT, FACT_EXTRACTION_PROMPT
from septmuse.storage.base import MemoryStore
from septmuse.storage.typed_store import TypedMemoryStore

logger = get_logger(__name__)


def parse_messages(messages: Any) -> str:
    """解析 messages 为文本 (对齐 mem0 parse_messages)。"""
    if isinstance(messages, str):
        return messages
    parts: list[str] = []
    for m in messages:
        if isinstance(m, dict):
            role = m.get("role", "")
            content = m.get("content", "")
            if content:
                parts.append(f"{role}: {content}")
        elif isinstance(m, str):
            parts.append(m)
    return "\n".join(parts)


def normalize_facts(raw_facts: list[Any]) -> list[str]:
    """归一化 facts (源码参考 mem0 normalize_facts)。

    处理 str / {"fact": ...} / {"text": ...} 变体。
    """
    if not raw_facts:
        return []
    normalized: list[str] = []
    for item in raw_facts:
        if isinstance(item, str):
            fact = item
        elif isinstance(item, dict):
            fact = item.get("fact") or item.get("text")
            if fact is None:
                continue
        else:
            fact = str(item)
        if fact:
            normalized.append(fact)
    return normalized


def fact_to_triple(fact: str, user_id: str) -> tuple[str, str, str]:
    """简单规则解析 fact → (subject, predicate, object)。

    避免二次 LLM 调用。模式:
    - "Name is X" → (user_id, "name", X)
    - "Likes X" → (user_id, "likes", X)
    - "Is a X" → (user_id, "occupation", X)
    - "Dislikes X" → (user_id, "dislikes", X)
    - 默认 → (user_id, "fact", fact)
    """
    f = fact.strip()
    low = f.lower()

    if low.startswith("name is"):
        return user_id, "name", f[len("name is") :].strip()
    if low.startswith("likes"):
        return user_id, "likes", f[len("likes") :].strip()
    if low.startswith("dislikes"):
        return user_id, "dislikes", f[len("dislikes") :].strip()
    if low.startswith("is a"):
        return user_id, "occupation", f[len("is a") :].strip()
    # 默认: 整句作为 object
    return user_id, "fact", f


class FactExtractor:
    """cognify 抽取流水线 (架构文档 §3.2.2, 借鉴 Cognee + mem0)。

    依赖注入 LLM + Embedder + stores, 便于测试 (注入 MockLLM)。
    """

    def __init__(
        self,
        llm: LLM,
        embedder: Embedder,
        typed_store: TypedMemoryStore,
        verbatim_store: MemoryStore | None = None,
        use_additive_prompt: bool = True,
    ) -> None:
        self.llm = llm
        self.embedder = embedder
        self.typed_store = typed_store
        self.verbatim_store = verbatim_store
        self.prompt = ADDITIVE_EXTRACTION_PROMPT if use_additive_prompt else FACT_EXTRACTION_PROMPT

    def extract_facts(self, messages: Any) -> list[str]:
        """LLM 抽取 fact 字符串列表 (对齐 mem0 add infer=True 抽取阶段)。

        P3-Task 2: 默认用 ADDITIVE_EXTRACTION_PROMPT (含 9 个 few-shot, 对齐 mem0 V3)。
        """
        text = parse_messages(messages)
        if not text.strip():
            return []

        raw = self.llm.complete(self.prompt, f"Input:\n{text}")
        facts = self._parse_facts_response(raw)
        logger.info("facts_extracted", count=len(facts))
        return facts

    def extract_and_store(
        self,
        messages: Any,
        *,
        user_id: str,
        provenance: str = "inferred",
    ) -> list[dict[str, Any]]:
        """完整 cognify 流水线: 抽取 → 三元组 → 存储 (对齐 Cognee cognify + mem0 add)。

        P3-Task 2: 输出 linked_memory_ids (跨记忆链接, 对齐 mem0 V3)。
        返回 [{"id", "memory", "triple", "event": "ADD", "linked_memory_ids": [...]}]。
        """
        facts = self.extract_facts(messages)
        results: list[dict[str, Any]] = []
        linked_memory_ids: list[str] = []

        for fact in facts:
            subject, predicate, object_ = fact_to_triple(fact, user_id)
            fact_obj = self.typed_store.add_fact(
                subject,
                predicate,
                object_,
                user_id=user_id,
                confidence=0.7,
                provenance=provenance,
                tags=[],
                embedding=self.embedder.embed(f"{subject} {predicate} {object_}"),
            )
            verbatim_id = None
            if self.verbatim_store is not None:
                verbatim_id = self.verbatim_store.add(
                    fact,
                    self.embedder.embed(fact),
                    user_id=user_id,
                    metadata={"source": "cognify", "fact_id": fact_obj.id},
                )
            if verbatim_id:
                linked_memory_ids.append(verbatim_id)
            results.append(
                {
                    "id": fact_obj.id,
                    "memory": fact,
                    "triple": fact_obj.as_triple(),
                    "event": "ADD",
                    "linked_memory_ids": linked_memory_ids.copy(),
                }
            )

        logger.info("cognify_done", user_id=user_id, facts_stored=len(results), linked=len(linked_memory_ids))
        return results

    @staticmethod
    def _parse_facts_response(raw: str) -> list[str]:
        """解析 LLM 输出为 fact 列表 (源码参考 mem0 normalize_facts + extract_json)。"""
        # 去除 markdown 代码块
        cleaned = re.sub(r"^```[a-zA-Z0-9]*\n|\n```$", "", raw.strip())
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("facts_parse_failed", raw=raw[:100])
            return []
        raw_facts = data.get("facts", []) if isinstance(data, dict) else []
        return normalize_facts(raw_facts)

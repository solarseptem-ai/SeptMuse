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

流程:
1. parse_messages → 文本
2. LLM complete(FACT_EXTRACTION_PROMPT) → {"facts": [...]}
3. normalize_facts (处理 str/dict 变体)
4. fact → 三元组 (简单规则解析, 避免二次 LLM)
5. 存为 SemanticFact (subject/predicate/object) + verbatim memory (向量检索)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from septmuse.core.logging import get_logger
from septmuse.embedders.base import Embedder
from septmuse.llms.base import LLM
from septmuse.prompts.extract import ADDITIVE_EXTRACTION_PROMPT, FACT_EXTRACTION_PROMPT
from septmuse.storage.base import MemoryStore
from septmuse.storage.relational_stores.typed_store import TypedMemoryStore

logger = get_logger(__name__)


def parse_messages(messages: Any) -> str:
    """解析 messages 为文本。"""
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
    """归一化 facts。

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


@dataclass
class Decision:
    """LLM 决策抽取的单条结果 (ADD/UPDATE/DELETE/NOOP)."""

    text: str
    event: str  # "ADD" | "UPDATE" | "DELETE" | "NOOP"
    id: str | None = None
    confidence: float = 1.0
    linked_memory_ids: list[str] = None  # 跨记忆链接 ID (对齐 mem0 V3)

    def __post_init__(self):
        if self.linked_memory_ids is None:
            self.linked_memory_ids = []


class FactExtractor:
    """cognify 抽取流水线 (架构文档 §3.2.2)。

    依赖注入 LLM + Embedder + stores, 便于测试 (注入 MockLLM)。
    """

    def __init__(
        self,
        llm: LLM,
        embedder: Embedder,
        typed_store: TypedMemoryStore,
        verbatim_store: MemoryStore | None = None,
        use_additive_prompt: bool = True,
        use_decision: bool = False,
        episodic_store: Any = None,
    ) -> None:
        self.llm = llm
        self.embedder = embedder
        self.typed_store = typed_store
        self.verbatim_store = verbatim_store
        self.use_decision = use_decision
        self.prompt = ADDITIVE_EXTRACTION_PROMPT if use_additive_prompt else FACT_EXTRACTION_PROMPT
        self.episodic_store = episodic_store

    def extract_facts(
        self, messages: Any, existing_memories: list[dict[str, Any]] | None = None
    ) -> list[str]:
        """LLM 抽取 fact 字符串列表。

        Args:
            messages: 消息文本或列表
            existing_memories: 已有记忆列表 (注入 prompt 避免重复抽取, None=纯抽取模式)
        """
        text = parse_messages(messages)
        if not text.strip():
            return []

        from septmuse.prompts.extract import build_extraction_user_prompt

        user_prompt = build_extraction_user_prompt(text, existing_memories)
        raw = self.llm.complete(self.prompt, user_prompt)
        facts = self._parse_facts_response(raw)
        logger.info("facts_extracted", count=len(facts), existing=len(existing_memories or []))
        return facts

    def extract_with_decisions(
        self,
        messages: Any,
        existing_memories: list[dict[str, Any]] | None = None,
        last_k_messages: list[dict] | None = None,
    ) -> list[Decision]:
        """LLM 决策抽取, 返回带 event 的决策列表 (对齐 mem0 ADDITIVE).

        解析失败降级为空列表 (不阻塞业务).
        """
        text = parse_messages(messages)
        if not text.strip():
            return []
        from septmuse.prompts.extract import ADDITIVE_DECISION_PROMPT, build_extraction_user_prompt

        user_prompt = build_extraction_user_prompt(
            text, existing_memories, last_k_messages=last_k_messages or []
        )
        raw = self.llm.complete(ADDITIVE_DECISION_PROMPT, user_prompt)
        return self._parse_decisions_response(raw)

    @staticmethod
    def _parse_decisions_response(raw: str) -> list[Decision]:
        """解析 LLM 决策输出为 Decision 列表, 容错降级.

        向后兼容: 纯字符串 fact (无 event 字段, 如 MockLLM/旧 prompt 输出) 视为 ADD 决策.
        """
        cleaned = re.sub(r"^```[a-zA-Z0-9]*\n|\n```$", "", raw.strip())
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("decisions_parse_failed", raw=raw[:100])
            return []
        raw_facts = data.get("facts", []) if isinstance(data, dict) else []
        decisions: list[Decision] = []
        valid_events = {"ADD", "UPDATE", "DELETE", "NOOP"}
        for f in raw_facts:
            if isinstance(f, str):
                # 向后兼容: 纯字符串 fact 视为 ADD (MockLLM / 旧 prompt 格式)
                text = f.strip()
                if text:
                    decisions.append(Decision(text=text, event="ADD"))
                continue
            if not isinstance(f, dict):
                continue
            text = str(f.get("text", "")).strip()
            event = str(f.get("event", "ADD")).upper()
            if not text or event not in valid_events:
                continue
            raw_linked = f.get("linked_memory_ids", [])
            linked_ids = [str(lid) for lid in raw_linked if lid] if isinstance(raw_linked, list) else []
            decisions.append(
                Decision(
                    text=text,
                    event=event,
                    id=f.get("id"),
                    confidence=float(f.get("confidence", 1.0)),
                    linked_memory_ids=linked_ids,
                )
            )
        return decisions

    def _retrieve_existing(
        self, text: str, user_id: str, top_k: int = 10
    ) -> list[dict[str, Any]]:
        """检索已有记忆 (Phase 1, 避免重复抽取, 对齐 mem0 V3)。

        无 verbatim_store 时返回空列表 (降级为纯抽取模式)。
        """
        if self.verbatim_store is None:
            return []
        try:
            emb = self.embedder.embed(text)
            results = self.verbatim_store.search(
                emb, user_id=user_id, top_k=top_k, threshold=0.0
            )
            return [{"id": r.get("id", ""), "memory": r.get("memory", "")} for r in results]
        except Exception as e:
            logger.warning("retrieve_existing_failed", error=str(e))
            return []

    def _get_last_k_messages(self, user_id: str, limit: int = 5) -> list[dict]:
        """取近期 episodic 事件作为对话上下文 (降级: 无 episodic_store 返回空)."""
        if self.episodic_store is None:
            return []
        try:
            events = self.episodic_store.get_timeline(user_id=user_id, limit=limit)
            return [{"role": "assistant", "content": getattr(e, "content", str(e))} for e in events]
        except Exception:
            return []

    def extract_and_store(
        self,
        messages: Any,
        *,
        user_id: str,
        provenance: str = "inferred",
        extra_metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """完整 cognify 流水线: 检索已有 → 抽取 → 路由.

        use_decision=True 且有 LLM: 走决策路由 (ADD/UPDATE/DELETE/NOOP).
        否则: 旧纯 ADD 路径 (extract_facts → add_fact).

        extra_metadata: 透传给 verbatim store 的额外 metadata (expiration_date/attributed_to 等).
        """
        text = parse_messages(messages)

        # 无决策模式降级: 旧纯 ADD 路径
        if not self.use_decision or self.llm is None:
            return self._legacy_extract_and_store(text, user_id, provenance, extra_metadata=extra_metadata)

        # 决策路由
        existing = self._retrieve_existing(text, user_id)
        last_k = self._get_last_k_messages(user_id)
        decisions = self.extract_with_decisions(messages, existing_memories=existing, last_k_messages=last_k)
        results: list[dict[str, Any]] = []
        linked_memory_ids: list[str] = []

        for d in decisions:
            if d.event == "NOOP":
                results.append({"id": d.id, "memory": d.text, "event": "NOOP", "linked_memory_ids": []})
                continue
            # 置信度守卫: DELETE/UPDATE < 0.7 降级 NOOP
            if d.event in ("DELETE", "UPDATE") and d.confidence < 0.7:
                logger.info("decision_low_confidence_downgrade", decision=d.event, confidence=d.confidence)
                results.append({"id": d.id, "memory": d.text, "event": "NOOP", "linked_memory_ids": []})
                continue
            if d.event == "ADD":
                fact = self._store_add_fact(d.text, user_id, provenance)
                vid = self._store_verbatim_add(
                    d.text, user_id, fact.id,
                    linked_memory_ids=d.linked_memory_ids,
                    extra_metadata=extra_metadata,
                )
                if vid:
                    linked_memory_ids.append(vid)
                results.append(
                    {
                        "id": fact.id,
                        "memory": d.text,
                        "triple": fact.as_triple(),
                        "event": "ADD",
                        "linked_memory_ids": d.linked_memory_ids + ([vid] if vid else []),
                    }
                )
            elif d.event == "UPDATE" and d.id:
                fact = self._store_update_fact(d.id, d.text, user_id)
                if fact:
                    self._store_verbatim_update(d.id, d.text, user_id)
                    results.append(
                        {
                            "id": fact.id,
                            "memory": d.text,
                            "triple": fact.as_triple(),
                            "event": "UPDATE",
                            "linked_memory_ids": [d.id],
                        }
                    )
                else:
                    results.append({"id": d.id, "memory": d.text, "event": "NOOP", "linked_memory_ids": []})
            elif d.event == "DELETE" and d.id:
                self._store_delete_fact(d.id)
                self._store_verbatim_delete(d.id)
                results.append({"id": d.id, "memory": d.text, "event": "DELETE", "linked_memory_ids": []})

        logger.info("cognify_done", user_id=user_id, decisions=len(results), linked=len(linked_memory_ids))
        return results

    def _legacy_extract_and_store(
        self, text: str, user_id: str, provenance: str, *, extra_metadata: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """旧纯 ADD 路径 (无决策模式降级用)."""
        existing = self._retrieve_existing(text, user_id)
        facts = self.extract_facts(text, existing_memories=existing)
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
                vm: dict[str, Any] = {"source": "cognify", "fact_id": fact_obj.id}
                if extra_metadata:
                    vm.update(extra_metadata)
                verbatim_id = self.verbatim_store.add(
                    fact,
                    self.embedder.embed(fact),
                    user_id=user_id,
                    metadata=vm,
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
        return results

    def _store_add_fact(self, fact_text: str, user_id: str, provenance: str) -> Any:
        """ADD 决策: 三元组解析 + 存 SemanticFact."""
        subject, predicate, object_ = fact_to_triple(fact_text, user_id)
        return self.typed_store.add_fact(
            subject,
            predicate,
            object_,
            user_id=user_id,
            confidence=0.7,
            provenance=provenance,
            tags=[],
            embedding=self.embedder.embed(f"{subject} {predicate} {object_}"),
        )

    def _store_verbatim_add(
        self, fact_text: str, user_id: str, fact_id: str,
        *,
        linked_memory_ids: list[str] | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> str | None:
        """ADD 决策: verbatim 存原文 (跨记忆链接)."""
        if self.verbatim_store is None:
            return None
        metadata: dict[str, Any] = {"source": "cognify", "fact_id": fact_id}
        if linked_memory_ids:
            metadata["linked_memory_ids"] = linked_memory_ids
        if extra_metadata:
            metadata.update(extra_metadata)
        return self.verbatim_store.add(
            fact_text,
            self.embedder.embed(fact_text),
            user_id=user_id,
            metadata=metadata,
        )

    def _store_update_fact(self, fact_id: str, fact_text: str, user_id: str) -> Any:
        """UPDATE 决策: 三元组解析 + 更新 SemanticFact."""
        subject, predicate, object_ = fact_to_triple(fact_text, user_id)
        return self.typed_store.update_fact(
            fact_id,
            subject,
            predicate,
            object_,
            embedding=self.embedder.embed(f"{subject} {predicate} {object_}"),
            confidence=0.85,
        )

    def _store_verbatim_update(self, fact_id: str, fact_text: str, user_id: str) -> None:
        """UPDATE 决策: verbatim 更新原文."""
        if self.verbatim_store is None:
            return
        self.verbatim_store.update(fact_id, fact_text, self.embedder.embed(fact_text))

    def _store_delete_fact(self, fact_id: str) -> None:
        """DELETE 决策: 软删除 SemanticFact."""
        self.typed_store.soft_delete_fact(fact_id)

    def _store_verbatim_delete(self, fact_id: str) -> None:
        """DELETE 决策: verbatim 软删除 (吞错, 不阻塞)."""
        if self.verbatim_store is None:
            return
        try:
            self.verbatim_store.delete(fact_id)
        except Exception as e:
            logger.warning("verbatim_delete_failed", fact_id=fact_id, error=str(e))

    @staticmethod
    def _parse_facts_response(raw: str) -> list[str]:
        """解析 LLM 输出为 fact 列表。"""
        # 去除 markdown 代码块
        cleaned = re.sub(r"^```[a-zA-Z0-9]*\n|\n```$", "", raw.strip())
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("facts_parse_failed", raw=raw[:100])
            return []
        raw_facts = data.get("facts", []) if isinstance(data, dict) else []
        return normalize_facts(raw_facts)

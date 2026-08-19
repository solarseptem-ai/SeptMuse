#  Copyright 2026 The sonhhxg0529 Authors. All Rights Reserved.
"""CaptureModule — 捕获/写入领域模块（add_typed / add_fact / add_episode / add_rule / 工作记忆）。"""

from __future__ import annotations

from typing import Any

from septmuse.embedders.base import Embedder
from septmuse.llms.base import LLM
from septmuse.storage.base import MemoryStore
from septmuse.storage.relational_stores.typed_store import TypedMemoryStore


class CaptureModule:
    """捕获/写入领域模块 — 文本→记忆的完整链路。"""

    def __init__(
        self,
        store: MemoryStore,
        embedder: Embedder,
        typed_store: TypedMemoryStore,
        llm: LLM | None = None,
        entity_extractor: Any | None = None,
        entity_store: Any | None = None,
        episodic: Any | None = None,
        semantic: Any | None = None,
        procedural: Any | None = None,
        working_memory: Any | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.typed_store = typed_store
        self.llm = llm
        self.entity_extractor = entity_extractor
        self.entity_store = entity_store
        self.episodic = episodic
        self.semantic = semantic
        self.procedural = procedural
        self.working_memory = working_memory

    def add_typed(
        self,
        content: str,
        memory_type: str = "verbatim",
        *,
        user_id: str = "default",
        agent_id: str | None = None,
        session_id: str | None = None,
        infer: bool = False,
        valid_at: str | None = None,
    ) -> dict[str, Any]:
        """统一类型派发入口 — verbatim/semantic/episodic/procedural。"""
        if memory_type == "semantic":
            parts = content.split(None, 2)
            subject = parts[0] if len(parts) > 0 else content
            predicate = parts[1] if len(parts) > 1 else "is"
            obj = parts[2] if len(parts) > 2 else ""
            return self.add_fact(subject, predicate, obj, user_id=user_id)
        if memory_type == "episodic":
            return self.add_episode(content, user_id=user_id)
        if memory_type == "procedural":
            return self.add_rule(content, user_id=user_id)
        emb = self.embedder.embed(content)
        mid = self.store.add(content, emb, user_id=user_id, agent_id=agent_id, session_id=session_id, infer=infer, valid_at=valid_at)
        return {"id": mid, "event": "ADD"}

    def add_fact(
        self,
        subject: str,
        predicate: str,
        object: str,
        *,
        user_id: str,
        context: str | None = None,
        confidence: float = 1.0,
        provenance: str = "user",
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """添加语义事实三元组。"""
        fact = self.semantic.add_fact(
            subject, predicate, object,
            user_id=user_id, context=context, confidence=confidence,
            provenance=provenance, tags=tags,
        )
        return {"id": fact.id, "triple": fact.as_triple(), "event": "ADD"}

    def update_fact(self, fact_id: str, *, subject: str, predicate: str, object: str, user_id: str) -> dict[str, Any]:
        """更新语义事实。"""
        fact = self.typed_store.update_fact(fact_id, subject, predicate, object)
        if fact is None:
            return {"id": fact_id, "event": "NOT_FOUND"}
        return {"id": fact.id, "triple": [subject, predicate, object], "event": "UPDATE"}

    def search_facts(self, query: str, *, user_id: str, top_k: int = 5) -> list[dict[str, Any]]:
        """语义检索事实 (置信度加权)。"""
        return self.semantic.search_facts(query, user_id=user_id, top_k=top_k)

    def add_episode(
        self,
        content: str,
        *,
        user_id: str,
        event_type: str = "fact",
        session_id: str | None = None,
        observation: str | None = None,
        thoughts: str | None = None,
        action: str | None = None,
        result: str | None = None,
    ) -> dict[str, Any]:
        """添加情节事件。"""
        from septmuse.models.episodic import EpisodeType

        et = {"fact": EpisodeType.FACT, "reasoning": EpisodeType.REASONING, "raw_log": EpisodeType.RAW_LOG}.get(
            event_type, EpisodeType.FACT
        )
        if et == EpisodeType.REASONING and observation is not None:
            event = self.episodic.add_reasoning_episode(
                observation, thoughts or "", action or "", result or "", user_id=user_id
            )
        elif et == EpisodeType.RAW_LOG:
            event = self.episodic.add_raw_log(content, user_id=user_id, session_id=session_id or "unknown")
        else:
            event = self.episodic.add_temporal_event(content, user_id=user_id)
        return {"id": event.id, "event_type": event.event_type, "reference_time": event.reference_time.isoformat()}

    def update_episode(self, episode_id: str, *, content: str, user_id: str) -> dict[str, Any]:
        """更新情节事件。"""
        event = self.typed_store.update_episode(episode_id, content)
        if event is None:
            return {"id": episode_id, "event": "NOT_FOUND"}
        return {"id": event.id, "content": content, "event": "UPDATE"}

    def get_timeline(self, *, user_id: str, event_type: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        """情节时序查询。"""
        events = self.episodic.get_timeline(user_id=user_id, event_type=event_type, limit=limit)
        return [self.episodic.episode_to_dict(e) for e in events]

    def add_rule(
        self, rule: str, *, user_id: str, namespace: str = "default", source_tracing: str | None = None
    ) -> dict[str, Any]:
        """添加程序规则。"""
        r = self.procedural.add_rule(rule, user_id=user_id, namespace=namespace, source_tracing=source_tracing)
        return {"id": r.id, "rule": r.rule, "event": "ADD"}

    def update_rule(self, rule_id: str, *, rule: str, user_id: str) -> dict[str, Any]:
        """更新程序规则。"""
        r = self.typed_store.update_rule(rule_id, rule)
        if r is None:
            return {"id": rule_id, "event": "NOT_FOUND"}
        return {"id": r.id, "rule": rule, "event": "UPDATE"}

    # ── 工作记忆 ──

    def get_working_memory(self, agent_id: str) -> Any:
        """获取 agent 的工作记忆 (从 WorkingMemoryStore 加载, 自动持久化)。"""
        from septmuse.memory.working_memory import WorkingMemory

        wm_store = getattr(self.working_memory, "store", None) if self.working_memory is not None else None
        return WorkingMemory(store=wm_store, agent_id=agent_id)

    def get_blocks(self, agent_id: str) -> list[dict[str, Any]]:
        """列出 agent 的全部 block。"""
        wm = self.get_working_memory(agent_id)
        return [b.model_dump() for b in wm.blocks]

    def update_block(self, agent_id: str, label: str, value: str) -> dict[str, Any]:
        """更新 block value。"""
        wm = self.get_working_memory(agent_id)
        wm.update_block_value(label, value)
        block = wm.get_block(label)
        return {"id": block.id, "label": block.label, "value": block.value, "event": "UPDATE"}

    def core_memory_append(self, agent_id: str, label: str, content: str) -> dict[str, Any]:
        """追加 block 内容。"""
        wm = self.get_working_memory(agent_id)
        wm.core_memory_append(label, content)
        block = wm.get_block(label)
        return {"id": block.id, "label": block.label, "value": block.value, "event": "APPEND"}

    def core_memory_replace(self, agent_id: str, label: str, old_content: str, new_content: str) -> dict[str, Any]:
        """替换 block 内容片段。"""
        wm = self.get_working_memory(agent_id)
        wm.core_memory_replace(label, old_content, new_content)
        block = wm.get_block(label)
        return {"id": block.id, "label": block.label, "value": block.value, "event": "REPLACE"}

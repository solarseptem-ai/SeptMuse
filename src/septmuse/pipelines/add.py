"""AddPipeline — 记忆添加流水线。

提取自 Memory.add()，负责 typed memory 路由、verbatim/infer 模式、
批量嵌入、实体抽取、缓存失效全链路。
"""

from __future__ import annotations

from typing import Any

from septmuse.core.logging import get_logger
from septmuse.core.messages import normalize_messages as _normalize_messages
from septmuse.core.utils import normalize_expiration_date as _normalize_expiration_date
from septmuse.embedders.base import Embedder
from septmuse.memory.episodic import EpisodicMemory
from septmuse.memory.procedural import ProceduralMemory
from septmuse.memory.semantic import SemanticMemory
from septmuse.models.extract import FactExtractor
from septmuse.storage.base import MemoryStore

logger = get_logger(__name__)


class AddPipeline:
    """记忆添加流水线 (原 Memory.add 内联逻辑)。

    职责:
    - typed memory 路由 (fact/episode/rule)
    - verbatim / infer 模式判断
    - 批量嵌入 + 存储写入
    - 批量实体链接 (全局去重)
    - 返回标准格式 {"results": [...], "relations": []}
    """

    def __init__(
        self,
        *,
        store: MemoryStore,
        embedder: Embedder,
        semantic: SemanticMemory,
        episodic: EpisodicMemory,
        procedural: ProceduralMemory,
        extractor: FactExtractor | None = None,
        entity_extractor: Any | None = None,
        entity_store: Any | None = None,
        config_infer: bool = False,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.semantic = semantic
        self.episodic = episodic
        self.procedural = procedural
        self.extractor = extractor
        self.entity_extractor = entity_extractor
        self.entity_store = entity_store
        self.config_infer = config_infer

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------

    def add(
        self,
        messages: Any,
        *,
        user_id: str,
        agent_id: str | None = None,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        infer: bool | None = None,
        memory_type: str | None = None,
        auto_extract_entities: bool = True,
        valid_at: str | None = None,
        # typed memory kwargs
        subject: str | None = None,
        predicate: str | None = None,
        object: str | None = None,
        event_type: str = "fact",
        namespace: str = "default",
        rule: str | None = None,
        expiration_date: str | None = None,
    ) -> dict[str, Any]:
        """添加记忆 (统一 typed memory 入口)。"""
        # typed memory 路由
        if memory_type == "fact":
            return self._add_fact(subject, predicate, object, user_id=user_id, metadata=metadata)

        if memory_type == "episode":
            return self._add_episode(messages, user_id=user_id, session_id=session_id, event_type=event_type, metadata=metadata)

        if memory_type == "rule":
            return self._add_rule(messages, user_id=user_id, namespace=namespace, rule=rule, metadata=metadata)

        # verbatim / infer 模式
        should_infer = self.config_infer if infer is None else infer

        if should_infer and self.extractor is not None:
            extracted = self.extractor.extract_and_store(messages, user_id=user_id)
            return {"results": extracted, "relations": []}

        texts = _normalize_messages(messages)
        if not texts:
            return {"results": [], "relations": []}

        normalized_exp = _normalize_expiration_date(expiration_date)
        if normalized_exp is not None:
            if metadata is None:
                metadata = {}
            metadata["expiration_date"] = normalized_exp

        embeddings = self.embedder.embed_batch(texts)

        records = list(zip(texts, embeddings, strict=True))
        ids = self.store.add_batch(
            records,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            metadata=metadata,
            valid_at=valid_at,
        )

        valid_pairs = [
            (text, mid) for text, mid in zip(texts, ids, strict=True) if mid is not None
        ]
        if auto_extract_entities and self.entity_store is not None and self.entity_extractor is not None:
            self._batch_extract_and_store_entities(valid_pairs, user_id=user_id, agent_id=agent_id)

        results: list[dict[str, Any]] = []
        for text, mid in zip(texts, ids, strict=True):
            if mid is None:
                continue
            results.append({"id": mid, "memory": text, "event": "ADD"})

        logger.info("memory_add_done", user_id=user_id, count=len(results), infer=should_infer)
        return {"results": results, "relations": []}

    # ------------------------------------------------------------------
    # Typed memory 子路由
    # ------------------------------------------------------------------

    def _add_fact(
        self,
        subject: str | None,
        predicate: str | None,
        object: str | None,
        *,
        user_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        fact = self.semantic.add_fact(
            subject or "",
            predicate or "",
            object or "",
            user_id=user_id,
            context=metadata.get("context") if metadata else None,
            confidence=metadata.get("confidence", 1.0) if metadata else 1.0,
            provenance=metadata.get("provenance", "user") if metadata else "user",
            tags=metadata.get("tags") if metadata else None,
        )
        return {"id": fact.id, "triple": fact.as_triple(), "event": "ADD"}

    def _add_episode(
        self,
        messages: Any,
        *,
        user_id: str,
        session_id: str | None = None,
        event_type: str = "fact",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        from septmuse.models.episodic import EpisodeType

        et = {"fact": EpisodeType.FACT, "reasoning": EpisodeType.REASONING, "raw_log": EpisodeType.RAW_LOG}.get(
            event_type, EpisodeType.FACT
        )
        content = messages if isinstance(messages, str) else str(messages)
        if et == EpisodeType.REASONING and metadata and metadata.get("observation"):
            event = self.episodic.add_reasoning_episode(
                metadata["observation"],
                metadata.get("thoughts", ""),
                metadata.get("action", ""),
                metadata.get("result", ""),
                user_id=user_id,
            )
        elif et == EpisodeType.RAW_LOG:
            event = self.episodic.add_raw_log(content, user_id=user_id, session_id=session_id or "unknown")
        else:
            event = self.episodic.add_temporal_event(content, user_id=user_id)
        return {
            "id": event.id,
            "event_type": event.event_type,
            "reference_time": event.reference_time.isoformat(),
        }

    def _add_rule(
        self,
        messages: Any,
        *,
        user_id: str,
        namespace: str = "default",
        rule: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        rule_text = rule or (messages if isinstance(messages, str) else str(messages))
        r = self.procedural.add_rule(
            rule_text,
            user_id=user_id,
            namespace=namespace,
            source_tracing=metadata.get("source_tracing") if metadata else None,
        )
        return {"id": r.id, "rule": r.rule, "event": "ADD"}

    # ------------------------------------------------------------------
    # 实体抽取
    # ------------------------------------------------------------------

    def _batch_extract_and_store_entities(
        self, pairs: list[tuple[str, str]], *, user_id: str, agent_id: str | None = None
    ) -> None:
        """批量实体链接 (全局去重)。"""
        if not pairs or self.entity_extractor is None or self.entity_store is None:
            return
        try:
            from septmuse.extraction.entity import Entity

            global_entities: dict[str, tuple[str, str, set[str]]] = {}
            for text, memory_id in pairs:
                entities = self.entity_extractor.extract(text)
                for entity in entities:
                    key = entity.text.strip().lower()
                    if key in global_entities:
                        global_entities[key][2].add(memory_id)
                    else:
                        global_entities[key] = (entity.entity_type, entity.text, {memory_id})

            for entity_type, entity_text, memory_ids in global_entities.values():
                entity = Entity(text=entity_text, entity_type=entity_type, start=0, end=0)
                for mid in memory_ids:
                    self.entity_store.upsert(entity, mid, user_id=user_id, agent_id=agent_id)
        except Exception as e:
            logger.warning("batch_extract_entities_failed", error=str(e))

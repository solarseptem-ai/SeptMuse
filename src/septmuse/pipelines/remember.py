"""RememberPipeline — V2 remember 编排（深化模块）。

拥有 remember() 的 10 步链路:
    [0] 价值评估 → [0.5] 类型分类 → [0.6] 语义去重 → [1] 捕获预处理
    → [2] raw_log 写入 → [3] verbatim 存储 → [4] 实体抽取
    → [5] 事实抽取 / catch-up 入队 → [5.5] 冲突检测 → [6] 工作记忆更新

Memory.remember() 委托给本管线, 不再持有 remember 编排逻辑。
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

from septmuse.core.logging import get_logger
from septmuse.embedders.base import Embedder
from septmuse.storage.base import MemoryStore
from septmuse.storage.relational_stores.typed_store import TypedMemoryStore

if TYPE_CHECKING:
    from septmuse.capture.classifier import MemoryClassifier
    from septmuse.capture.dedup import SemanticDeduplicator
    from septmuse.capture.importance import ImportanceAssessor
    from septmuse.evolution.catchup import LLMCatchUpQueue
    from septmuse.governance.approval import DedupWindow
    from septmuse.memory.episodic import EpisodicMemory
    from septmuse.models.extract import FactExtractor
    from septmuse.pipelines.capture import CapturePipeline

logger = get_logger(__name__)


class RememberPipeline:
    """V2 remember 编排管线。

    将 remember() 的 10 步链路从 Memory facade 中抽离,
    使 facade 恢复 thin 属性。

    Args:
        store: verbatim 记忆存储
        embedder: 嵌入器
        episodic: 情节记忆（raw_log）
        typed_store: 类型化存储（可选）
        llm: LLM（可选）
        extractor: 事实抽取器（可选）
        entity_extractor: 实体抽取器（可选）
        entity_store: 实体存储（可选）
        working_memory: 工作记忆（可选）
        capture: 捕获流水线（可选）
        dedup_window: SHA256 去重窗口（可选）
        engine: SQLAlchemy engine（可选，用于 catch-up 队列）
    """

    def __init__(
        self,
        store: MemoryStore,
        embedder: Embedder,
        episodic: EpisodicMemory,
        *,
        typed_store: TypedMemoryStore | None = None,
        llm: Any | None = None,
        extractor: FactExtractor | None = None,
        entity_extractor: Any | None = None,
        entity_store: Any | None = None,
        working_memory: Any | None = None,
        capture: CapturePipeline | None = None,
        dedup_window: DedupWindow | None = None,
        engine: Any | None = None,
        importance_assessor: ImportanceAssessor | None = None,
        classifier: MemoryClassifier | None = None,
        deduplicator: SemanticDeduplicator | None = None,
        catchup_queue: LLMCatchUpQueue | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.episodic = episodic
        self.typed_store = typed_store
        self.llm = llm
        self.extractor = extractor
        self.entity_extractor = entity_extractor
        self.entity_store = entity_store
        self.working_memory = working_memory
        self.capture = capture
        self._dedup_window = dedup_window
        self._engine = engine
        self._importance_injected = importance_assessor
        self._classifier_injected = classifier
        self._deduplicator_injected = deduplicator
        self._catchup_injected = catchup_queue

    # ------------------------------------------------------------------
    # Lazy 组件（注入优先，未注入时延迟构造，失败吞错）
    # ------------------------------------------------------------------

    @property
    def _importance_assessor(self) -> ImportanceAssessor:
        if self._importance_injected is not None:
            return self._importance_injected
        if not hasattr(self, "_importance_cached"):
            from septmuse.capture.importance import ImportanceAssessor

            self._importance_cached = ImportanceAssessor(llm=self.llm)
        return self._importance_cached

    @property
    def _classifier(self) -> MemoryClassifier:
        if self._classifier_injected is not None:
            return self._classifier_injected
        if not hasattr(self, "_clf_cached"):
            from septmuse.capture.classifier import MemoryClassifier

            self._clf_cached = MemoryClassifier(llm=self.llm)
        return self._clf_cached

    @property
    def _deduplicator(self) -> SemanticDeduplicator:
        if self._deduplicator_injected is not None:
            return self._deduplicator_injected
        if not hasattr(self, "_dedup_cached"):
            from septmuse.capture.dedup import SemanticDeduplicator

            self._dedup_cached = SemanticDeduplicator(self.embedder, self.store, llm=self.llm)
        return self._dedup_cached

    @property
    def _catchup_queue(self) -> LLMCatchUpQueue | None:
        if self._catchup_injected is not None:
            return self._catchup_injected
        if not hasattr(self, "_cu_queue_cached"):
            if self._engine is not None:
                try:
                    from septmuse.evolution.catchup import LLMCatchUpQueue

                    self._cu_queue_cached = LLMCatchUpQueue(self._engine)
                except Exception as e:
                    logger.warning("catchup_queue_init_failed", error=str(e))
                    self._cu_queue_cached = None
            else:
                self._cu_queue_cached = None
        return self._cu_queue_cached

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------

    def remember(
        self,
        text: str,
        *,
        user_id: str,
        session_id: str | None = None,
        agent_id: str | None = None,
    ) -> dict[str, Any]:
        """V2 remember: 捕获文本 → 存 raw_log → (可选) 抽事实 → 更新工作记忆。

        Returns:
            {"captured": bool, "raw_id": str|None, "fact_ids": list, "memory_id": str}
        """
        result: dict[str, Any] = {
            "captured": False,
            "raw_id": None,
            "fact_ids": [],
            "memory_ids": [],
            "memory_id": None,
            "importance": 0.0,
            "types": [],
            "duplicate": None,
            "conflicts": [],
        }

        if not text or not text.strip():
            return result

        if self.capture is None:
            logger.warning("remember_capture_unavailable", user_id=user_id)
            return result

        # [0] 价值评估
        importance = self._importance_assessor.assess(text)
        result["importance"] = importance.score

        if importance.score < 0.3:
            raw_event = self.episodic.add_raw_log(
                text,
                user_id=user_id,
                session_id=session_id or "default",
                agent_id=agent_id,
            )
            result["raw_id"] = raw_event.id
            result["captured"] = False
            logger.info("remember_low_importance_skipped", user_id=user_id, score=importance.score)
            return result

        # [0.5] 类型分类
        types = self._classifier.classify(text)
        result["types"] = types

        # [0.6] 语义去重
        dup = self._deduplicator.find_duplicate(text, user_id=user_id)
        result["duplicate"] = None
        if dup is not None:
            result["duplicate"] = {
                "memory_id": dup.memory_id,
                "similarity": dup.similarity,
                "method": dup.method,
            }
            logger.info(
                "remember_semantic_dup_found",
                user_id=user_id,
                dup_id=dup.memory_id,
                similarity=dup.similarity,
            )

        preprocess = self.capture.preprocess(text, user_id=user_id, agent_id=agent_id)
        if not preprocess.allowed:
            logger.info("remember_deduped", user_id=user_id, reason=preprocess.reason)
            return result

        stored_text = preprocess.stored_text
        raw_event = self.episodic.add_raw_log(
            stored_text,
            user_id=user_id,
            session_id=session_id or "default",
            agent_id=agent_id,
        )
        result["raw_id"] = raw_event.id

        verbatim_id = self.store.add(
            stored_text,
            self.embedder.embed(stored_text),
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            metadata={"source": "remember", "raw_id": raw_event.id},
        )
        if verbatim_id is None:
            logger.info("remember_deduped_by_store", user_id=user_id)
            return result
        result["memory_id"] = verbatim_id
        result["memory_ids"] = [verbatim_id]

        # [4] 实体抽取
        if self.entity_store is not None and self.entity_extractor is not None:
            self._extract_and_store_entities(stored_text, verbatim_id, user_id=user_id, agent_id=agent_id)

        # [5] 事实抽取 / catch-up 入队
        if self.llm is not None and self.extractor is not None:
            try:
                extracted = self.extractor.extract_and_store(stored_text, user_id=user_id)
                result["fact_ids"] = [e.get("id", "") for e in extracted if isinstance(e, dict)]
            except Exception as e:
                logger.warning("remember_extract_failed", error=str(e))
        else:
            self._enqueue_catchup(
                "fact_extraction", verbatim_id, user_id=user_id, payload={"text": stored_text}
            )

        # [5.5] 冲突检测
        if self.typed_store is not None and self.llm is not None:
            try:
                from septmuse.evolution.conflict import ConflictResolver

                resolver = ConflictResolver(self.typed_store, store=self.store, llm=self.llm)
                detected = resolver.detect_conflicts(user_id=user_id)
                result["conflicts"] = [
                    {"topic": f"{c['key'][0]} {c['key'][1]}", "facts": c["facts"]} for c in detected
                ]
            except Exception as e:
                logger.warning("remember_conflict_detect_failed", error=str(e))

        # [6] 工作记忆更新
        if agent_id is not None and self.working_memory is not None:
            with contextlib.suppress(Exception):
                self.working_memory.core_memory_append("notes", stored_text)

        result["captured"] = True
        logger.info("remember_done", user_id=user_id, raw_id=result["raw_id"], facts=len(result["fact_ids"]))
        return result

    def _extract_and_store_entities(
        self, text: str, memory_id: str, *, user_id: str, agent_id: str | None = None
    ) -> None:
        """自动抽取实体并存入 entity_store (吞错, 不阻塞 add)."""
        if self.entity_extractor is None or self.entity_store is None:
            return
        try:
            entities = self.entity_extractor.extract(text)
            for entity in entities:
                self.entity_store.upsert(entity, memory_id, user_id=user_id, agent_id=agent_id)
        except Exception as e:
            logger.warning("auto_extract_entities_failed", error=str(e))

    def _enqueue_catchup(
        self, task_type: str, memory_id: str, *, user_id: str, payload: dict | None = None
    ) -> None:
        """LLM 不可用时入 catch-up 队列, 恢复后补跑。吞错不阻塞业务。"""
        queue = self._catchup_queue
        if queue is None:
            return
        try:
            queue.enqueue(task_type, memory_id, user_id=user_id, payload=payload)
        except Exception as e:
            logger.warning("catchup_enqueue_failed", task_type=task_type, error=str(e))

    def reset(self) -> None:
        """重置管线状态（去重窗口、缓存）."""
        if self._dedup_window is not None:
            self._dedup_window = type(self._dedup_window)()

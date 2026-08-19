#  Copyright 2026 The sonhhxg0529 Authors. All Rights Reserved.
"""统一 ComponentRegistry — 组件 + 管线 + 领域模块的懒加载注册表。

所有组件、管线、领域模块通过单一 dict 工厂注册，共享懒加载+缓存+失败隔离。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from septmuse.core.logging import get_logger

logger = get_logger(__name__)


class ComponentRegistry:
    """统一组件注册表 — 懒加载 + 缓存 + 失败隔离。"""

    def __init__(self, memory: Any) -> None:
        self._memory = memory
        self._cache: dict[str, Any] = {}
        self._factories: dict[str, Callable[[ComponentRegistry], Any]] = {}

    # ── 只读属性代理（工厂函数通过 reg.store 等访问）──

    @property
    def store(self) -> Any:
        return self._memory.store

    @property
    def embedder(self) -> Any:
        return self._memory.embedder

    @property
    def typed_store(self) -> Any:
        return self._memory.typed_store

    @property
    def graph_store(self) -> Any:
        return self._memory.graph_store

    @property
    def entity_store(self) -> Any:
        return self._memory.entity_store

    @property
    def entity_extractor(self) -> Any:
        return self._memory.entity_extractor

    @property
    def llm(self) -> Any:
        return self._memory.llm

    @property
    def config(self) -> Any:
        return self._memory.config

    @property
    def episodic(self) -> Any:
        return self._memory.episodic

    @property
    def semantic(self) -> Any:
        return self._memory.semantic

    @property
    def procedural(self) -> Any:
        return self._memory.procedural

    @property
    def extractor(self) -> Any:
        return self._memory.extractor

    @property
    def engine(self) -> Any:
        return getattr(self.store, "engine", None)

    # ── 注册与获取 ──

    def register(self, name: str, factory: Callable[[ComponentRegistry], Any]) -> None:
        """注册一个工厂函数。"""
        self._factories[name] = factory

    def get(self, name: str) -> Any:
        """获取命名组件（懒加载 + 缓存 + 失败隔离）。"""
        if name not in self._cache:
            factory = self._factories.get(name)
            if factory is None:
                self._cache[name] = None
            else:
                try:
                    self._cache[name] = factory(self)
                except Exception as e:
                    logger.warning(f"{name}_lazy_init_failed", error=str(e))
                    self._cache[name] = None
        return self._cache[name]

    def has(self, name: str) -> bool:
        """检查组件是否已在缓存中（不触发初始化）。"""
        return name in self._cache

    def invalidate(self, name: str | None = None) -> None:
        """清除缓存。name=None 时全清。"""
        if name is None:
            self._cache.clear()
        else:
            self._cache.pop(name, None)

    def register_all(self) -> None:
        """注册所有内置组件、管线、领域模块。"""
        # ── 管线（pipelines）──
        from septmuse.pipelines.add import AddPipeline
        from septmuse.pipelines.capture import CapturePipeline
        from septmuse.pipelines.forget import ForgetPipeline
        from septmuse.pipelines.recall import RecallPipeline
        from septmuse.pipelines.remember import RememberPipeline
        from septmuse.pipelines.reset import ResetPipeline
        from septmuse.pipelines.search import SearchPipeline, VectorRetrieverAdapter

        self.register(
            "capture",
            lambda r: CapturePipeline(
                r.store,
                r.embedder,
                typed_store=r.typed_store,
                llm=r.llm,
                dedup_window=getattr(r._memory, "_dedup_window", None),
            ),
        )
        self.register(
            "search",
            lambda r: SearchPipeline(
                hybrid_retriever=r._memory._get_hybrid_retriever(),
                vector_retriever=VectorRetrieverAdapter(r.store, r.embedder),
                reranker_factory=lambda name: _create_reranker(name, embedder=r.embedder, llm=r.llm),
                embedder=r.embedder,
                forgetting=r.get("forgetting"),
                default_top_k=r.config.top_k,
                default_threshold=r.config.threshold,
                llm=r.llm,
                graph_searcher=_create_graph_searcher(r),
                progressive_retriever=_create_progressive_retriever(r),
            ),
        )
        self.register(
            "recall",
            lambda r: RecallPipeline(
                search_pipeline=r.get("search"),
                meta=r.get("meta"),
                token_budget=r.get("token_budget"),
                working_memory=r.get("working_memory"),
                summarizer=r.get("summarizer"),
                typed_store=r.typed_store,
                llm=r.llm,
                recall_summarizer=r.get("recall_summarizer"),
                conflict_annotator=r.get("conflict_annotator"),
            ),
        )
        self.register(
            "remember",
            lambda r: RememberPipeline(
                store=r.store,
                embedder=r.embedder,
                episodic=r.episodic,
                typed_store=r.typed_store,
                llm=r.llm,
                extractor=r.extractor,
                entity_extractor=r.entity_extractor,
                entity_store=r.entity_store,
                working_memory=r.get("working_memory"),
                capture=r.get("capture"),
                dedup_window=getattr(r._memory, "_dedup_window", None),
                engine=r.engine,
                importance_assessor=r.get("importance_assessor"),
                classifier=r.get("classifier"),
                deduplicator=r.get("deduplicator"),
                catchup_queue=r.get("catchup_queue"),
            ),
        )
        self.register(
            "add",
            lambda r: AddPipeline(
                store=r.store,
                embedder=r.embedder,
                semantic=r.semantic,
                episodic=r.episodic,
                procedural=r.procedural,
                extractor=r.extractor,
                entity_extractor=r.entity_extractor,
                entity_store=r.entity_store,
                config_infer=r.config.infer,
            ),
        )
        self.register(
            "forget",
            lambda r: ForgetPipeline(
                store=r.store,
                entity_store=r.entity_store,
            ),
        )
        self.register(
            "reset",
            lambda r: ResetPipeline(
                store=r.store,
                entity_store=r.entity_store,
            ),
        )

        # ── 领域模块（domains）──
        from septmuse.domains.capture import CaptureModule
        from septmuse.domains.entities import EntityModule
        from septmuse.domains.evolution import EvolutionModule
        from septmuse.domains.search import SearchModule
        from septmuse.domains.temporal import TemporalModule

        self.register(
            "search_module",
            lambda r: SearchModule(
                store=r.store,
                embedder=r.embedder,
                typed_store=r.typed_store,
                llm=r.llm,
                entity_store=r.entity_store,
            ),
        )
        self.register(
            "evolution_module",
            lambda r: EvolutionModule(
                store=r.store,
                embedder=r.embedder,
                typed_store=r.typed_store,
                llm=r.llm,
                graph_store=r.graph_store,
            ),
        )
        self.register(
            "capture_module",
            lambda r: CaptureModule(
                store=r.store,
                embedder=r.embedder,
                typed_store=r.typed_store,
                llm=r.llm,
                entity_extractor=r.entity_extractor,
                entity_store=r.entity_store,
                episodic=r.episodic,
                semantic=r.semantic,
                procedural=r.procedural,
                working_memory=r.get("working_memory"),
            ),
        )
        self.register(
            "entity_module",
            lambda r: EntityModule(
                store=r.store,
                entity_store=r.entity_store,
                entity_extractor=r.entity_extractor,
            ),
        )
        self.register(
            "temporal_module",
            lambda r: TemporalModule(
                store=r.store,
                embedder=r.embedder,
                llm=r.llm,
            ),
        )

        # ── 组件（components）──
        from septmuse.evolution.self_loop import SelfEvolutionLoop
        from septmuse.evolution.summarizer import Summarizer
        from septmuse.memory.meta import MetacognitionLayer
        from septmuse.retrieval.conflict_annotator import ConflictAnnotator
        from septmuse.retrieval.forgetting import ForgettingRetriever
        from septmuse.retrieval.recall_summarizer import RecallSummarizer
        from septmuse.retrieval.token_budget import TokenBudget

        self.register(
            "meta",
            lambda r: MetacognitionLayer(
                r.embedder,
                r.store,
                r.typed_store,
            ),
        )
        self.register("recall_summarizer", lambda r: RecallSummarizer(llm=r.llm))
        self.register("conflict_annotator", lambda r: ConflictAnnotator(llm=r.llm))
        self.register("importance_assessor", lambda r: _create_importance_assessor(r))
        self.register("classifier", lambda r: _create_classifier(r))
        self.register("deduplicator", lambda r: _create_deduplicator(r))
        self.register("catchup_queue", lambda r: _create_catchup_queue(r))
        self.register("working_memory", lambda r: _create_working_memory(r))
        self.register(
            "self_loop",
            lambda r: SelfEvolutionLoop(
                r.store,
                r.graph_store,
                r.embedder,
                r.typed_store,
                llm=r.llm,
            ),
        )
        self.register(
            "summarizer",
            lambda r: Summarizer(
                r.store,
                r.typed_store,
                r.llm,
            ),
        )
        self.register("forgetting", lambda r: ForgettingRetriever(r.typed_store))
        self.register(
            "token_budget",
            lambda r: TokenBudget(
                budget=getattr(r.config, "token_budget", 2000),
            ),
        )


def _create_reranker(name: str, *, embedder: Any = None, llm: Any = None) -> Any:
    """创建 reranker（工厂函数）。"""
    from septmuse.rerankers import create_reranker

    return create_reranker(name, embedder=embedder, llm=llm)


def _create_graph_searcher(reg: ComponentRegistry) -> Any:
    """工厂: GraphSearcher。"""
    from septmuse.retrieval.graph_search import GraphSearcher

    graph_store = getattr(reg._memory, "_graph_store", None) or reg.graph_store
    return GraphSearcher(graph_store, reg.store)


def _create_progressive_retriever(reg: ComponentRegistry) -> Any:
    """工厂: ProgressiveRetriever。"""
    from septmuse.retrieval.progressive import ProgressiveRetriever

    return ProgressiveRetriever(reg.store, reg.typed_store, reg.embedder)


def _create_working_memory(reg: ComponentRegistry) -> Any:
    """工厂: WorkingMemory (需 engine 存在)。"""
    store_engine = getattr(reg.store, "engine", None)
    if store_engine is None:
        return None
    from septmuse.memory.working_memory import WorkingMemory
    from septmuse.storage.working_memory_stores.factory import create_working_memory_store

    return WorkingMemory(
        store=create_working_memory_store(reg.config, engine=store_engine),
        agent_id=getattr(reg._memory, "_agent_id", "default"),
    )


def _create_importance_assessor(reg: ComponentRegistry) -> Any:
    """工厂: ImportanceAssessor。"""
    from septmuse.capture.importance import ImportanceAssessor

    return ImportanceAssessor(llm=reg.llm)


def _create_classifier(reg: ComponentRegistry) -> Any:
    """工厂: MemoryClassifier。"""
    from septmuse.capture.classifier import MemoryClassifier

    return MemoryClassifier(llm=reg.llm)


def _create_deduplicator(reg: ComponentRegistry) -> Any:
    """工厂: SemanticDeduplicator。"""
    from septmuse.capture.dedup import SemanticDeduplicator

    return SemanticDeduplicator(reg.embedder, reg.store, llm=reg.llm)


def _create_catchup_queue(reg: ComponentRegistry) -> Any:
    """工厂: LLMCatchUpQueue。"""
    engine = reg.engine
    if engine is None:
        return None
    try:
        from septmuse.evolution.catchup import LLMCatchUpQueue

        return LLMCatchUpQueue(engine)
    except Exception:
        return None

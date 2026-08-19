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
"""Memory 核心组件工厂 — init-time 构造 store/embedder/llm/typed_store 等。

与 ComponentRegistry (registry_v2.py) 的区别：
- 本模块：init-time 立即构造，有依赖顺序，返回 _MemoryInitState dataclass。
- ComponentRegistry：lazy-time 懒加载，管线 + 领域模块，通过 r.store 等访问已构造的核心组件。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from septmuse.configs.defaults import MemoryConfig
from septmuse.core.logging import get_logger
from septmuse.embedders.base import Embedder
from septmuse.embedders.resolver import resolve_embedder
from septmuse.llms.base import LLM
from septmuse.memory.episodic import EpisodicMemory
from septmuse.memory.procedural import ProceduralMemory
from septmuse.memory.semantic import SemanticMemory
from septmuse.models.extract import FactExtractor
from septmuse.storage.base import MemoryStore
from septmuse.storage.graph_stores.base import GraphStore
from septmuse.storage.graph_stores.sqlite import SQLiteGraphStore
from septmuse.storage.relational_stores.typed_store import TypedMemoryStore

logger = get_logger(__name__)


@dataclass
class MemoryInitState:
    """Memory.__init__ 的构造结果 —— 工厂输出，Memory 只负责赋值。"""

    config: MemoryConfig
    embedder: Embedder
    store: MemoryStore
    graph_store: GraphStore | None
    typed_store: TypedMemoryStore
    semantic: SemanticMemory
    episodic: EpisodicMemory
    procedural: ProceduralMemory
    llm: LLM | None
    extractor: FactExtractor | None
    entity_extractor: Any
    entity_store: Any
    reranker: Any
    dedup_window: Any


class MemoryComponentFactory:
    """Memory 核心组件工厂 —— 集中构造逻辑、条件导入和错误回退。"""

    @staticmethod
    def build(
        config: MemoryConfig,
        *,
        embedder: Embedder | None = None,
        store: MemoryStore | None = None,
        graph_store: GraphStore | None = None,
        llm: LLM | None = None,
        entity_extractor: Any | None = None,
    ) -> MemoryInitState:
        """构造全部核心组件，非致命异常回退到安全默认值。"""
        resolved_embedder = embedder or resolve_embedder(config)
        resolved_store = store or MemoryComponentFactory._resolve_store(config)

        store_engine = getattr(resolved_store, "engine", None)
        resolved_graph_store = MemoryComponentFactory._resolve_graph_store(
            graph_store, resolved_store, store_engine
        )

        facts_vs = MemoryComponentFactory._create_facts_vector_store(config, resolved_store)
        typed_store = MemoryComponentFactory._create_typed_store(config, store_engine, facts_vs)

        semantic = SemanticMemory(typed_store, resolved_embedder)
        episodic = EpisodicMemory(typed_store)
        procedural = ProceduralMemory(typed_store)

        resolved_llm = MemoryComponentFactory._resolve_llm(config, llm)
        extractor = MemoryComponentFactory._create_extractor(resolved_llm, resolved_embedder, typed_store, resolved_store)

        resolved_entity_extractor = entity_extractor or MemoryComponentFactory._resolve_entity_extractor(config)
        entity_store = MemoryComponentFactory._create_entity_store(store_engine, resolved_embedder)

        reranker = MemoryComponentFactory._resolve_reranker(config, resolved_embedder, resolved_llm)

        from septmuse.governance.approval import DedupWindow

        dedup_window = DedupWindow()

        return MemoryInitState(
            config=config,
            embedder=resolved_embedder,
            store=resolved_store,
            graph_store=resolved_graph_store,
            typed_store=typed_store,
            semantic=semantic,
            episodic=episodic,
            procedural=procedural,
            llm=resolved_llm,
            extractor=extractor,
            entity_extractor=resolved_entity_extractor,
            entity_store=entity_store,
            reranker=reranker,
            dedup_window=dedup_window,
        )

    @staticmethod
    def _resolve_store(config: MemoryConfig) -> MemoryStore:
        from septmuse.storage.relational_stores.factory import RelationalStoreFactory

        return RelationalStoreFactory.create(config)

    @staticmethod
    def _resolve_graph_store(
        injected: GraphStore | None,
        store: MemoryStore,
        store_engine: Any,
    ) -> GraphStore | None:
        if injected is not None:
            return injected
        if store_engine is not None and store_engine.dialect.name == "sqlite":
            import threading

            raw_conn = store_engine.raw_connection()
            return SQLiteGraphStore(raw_conn, threading.Lock())
        return None

    @staticmethod
    def _create_facts_vector_store(config: MemoryConfig, store: MemoryStore) -> Any:
        vs = getattr(store, "_vector_store", None)
        if vs is None or type(vs).__name__ != "QdrantVectorStore":
            return None
        vs_cfg = config.vector_store
        try:
            from septmuse.storage.vector_stores.qdrant import QdrantVectorStore

            return QdrantVectorStore(
                collection_name="septmuse_facts",
                embedding_model_dims=getattr(vs, "_dim", vs_cfg.embedding_model_dims or 512),
                path=vs_cfg.qdrant_path,
                host=vs_cfg.qdrant_host,
                port=vs_cfg.qdrant_port if vs_cfg.qdrant_host else None,
                url=vs_cfg.qdrant_url,
                api_key=vs_cfg.qdrant_api_key,
                enable_bm25=getattr(vs, "_enable_bm25", True),
            )
        except Exception as e:
            logger.warning("facts_vector_store_create_failed", error=str(e))
            return None

    @staticmethod
    def _create_typed_store(
        config: MemoryConfig,
        store_engine: Any,
        facts_vs: Any,
    ) -> TypedMemoryStore:
        if store_engine is not None:
            return TypedMemoryStore(engine=store_engine, vector_store=facts_vs)
        return TypedMemoryStore(db_path=config.db_path, vector_store=facts_vs)

    @staticmethod
    def _resolve_llm(config: MemoryConfig, injected: LLM | None) -> LLM | None:
        if injected is not None:
            return injected
        if not config.llm_provider:
            return None
        try:
            from septmuse.llms import _resolve_llm

            return _resolve_llm(config)
        except Exception as e:
            logger.warning("llm_resolve_failed", error=str(e))
            return None

    @staticmethod
    def _create_extractor(
        llm: LLM | None,
        embedder: Embedder,
        typed_store: TypedMemoryStore,
        store: MemoryStore,
    ) -> FactExtractor | None:
        if llm is None:
            return None
        return FactExtractor(llm, embedder, typed_store, store)

    @staticmethod
    def _resolve_entity_extractor(config: MemoryConfig) -> Any:
        from septmuse.extraction.entity import _resolve_entity_extractor

        return _resolve_entity_extractor(config)

    @staticmethod
    def _create_entity_store(store_engine: Any, embedder: Embedder) -> Any:
        if store_engine is None:
            return None
        try:
            from septmuse.storage.relational_stores.entity_store import EntityStore

            return EntityStore.from_engine(store_engine, embedder)
        except Exception as e:
            logger.warning("entity_store_init_failed", error=str(e))
            return None

    @staticmethod
    def _resolve_reranker(config: MemoryConfig, embedder: Embedder, llm: LLM | None) -> Any:
        try:
            from septmuse.rerankers import create_reranker

            return create_reranker(config.reranker_backend, embedder=embedder, llm=llm)
        except Exception as e:
            logger.warning("reranker_resolve_failed", error=str(e))
            from septmuse.rerankers.noop import NoopReranker

            return NoopReranker()

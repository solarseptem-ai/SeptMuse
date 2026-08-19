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
"""Memory facade — SeptMuse 零配置入口。

pip install septmuse 后:

    from septmuse import Memory
    m = Memory()
    m.add("我喜欢 Python", user_id="alice")
    m.search("alice 喜欢什么", user_id="alice")

核心 API (8 方法):

    add / search / get / get_all / update / delete / delete_all / history

差异化 (保留):
    invalidate — 双时态失效
    cognify    — 知识图谱构建
    get_active_rules / rules_to_prompt / record_rule_outcome — 规则系统

实验性功能 (working memory / causal / rehearsal / metacognition /
dream / compress / conflict / temporal search / graph search / entity CRUD)
请在 ``from septmuse.experimental import ExperimentalMemory`` 中使用。
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from typing import Any

from septmuse.configs.defaults import MemoryConfig, default_config
from septmuse.core.logging import get_logger
from septmuse.core.messages import normalize_messages as _normalize_messages
from septmuse.embedders.base import Embedder
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


def _resolve_embedder(config: MemoryConfig) -> Embedder:
    """解析 embedder (通过 ServiceProvider 延迟 import + 实例化)。

    默认 HashEmbedder (零模型加载, 离线可用)。
    支持: hash / st / onnx / onnx-zh / auto / openai
    """
    from septmuse.services.providers import embedder_provider

    backend = config.embedder.backend.lower()
    if backend in ("sentence-transformers", "sentence_transformers"):
        backend = "st"
    if backend == "onnx-zh":
        return embedder_provider.resolve(
            backend, config=config.embedder, model_name="Xenova/paraphrase-multilingual-MiniLM-L12-v2"
        )
    return embedder_provider.resolve(backend, config=config.embedder)


def _normalize_expiration_date(expiration_date: Any) -> str | None:
    """归一化 expiration_date 为 YYYY-MM-DD 字符串 (对齐 mem0 expiration_date)。"""
    if expiration_date is None:
        return None
    if not isinstance(expiration_date, str):
        raise ValueError("expiration_date must be a string in YYYY-MM-DD format")
    from datetime import datetime

    try:
        datetime.strptime(expiration_date, "%Y-%m-%d")
    except ValueError:
        raise ValueError(f"expiration_date must be in YYYY-MM-DD format, got: {expiration_date}") from None
    return expiration_date


def _is_expired(metadata: dict[str, Any] | None) -> bool:
    """检查 metadata 是否已过期 (基于 expiration_date 字段)。"""
    if not metadata or not isinstance(metadata, dict):
        return False
    exp = metadata.get("expiration_date")
    if not exp or not isinstance(exp, str):
        return False
    from datetime import date, datetime

    try:
        exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
    except ValueError:
        return False
    return exp_date < date.today()


class Memory:
    """SeptMuse 记忆系统零配置 facade。

    零配置: Memory() 用 SQLite + HashEmbedder。
    升级: Memory(config=MemoryConfig(db_path="...")) 自定义。
    """

    def __init__(
        self,
        config: MemoryConfig | None = None,
        *,
        embedder: Embedder | None = None,
        store: MemoryStore | None = None,
        graph_store: GraphStore | None = None,
        llm: LLM | None = None,
        entity_extractor: Any | None = None,
    ) -> None:
        self.config = config or default_config()
        logger.info(
            "memory_init",
            db_path=str(self.config.db_path),
            embedder=self.config.embedder_model,
            infer=self.config.infer,
            inject_embedder=embedder is not None,
            inject_store=store is not None,
            inject_graph_store=graph_store is not None,
            inject_llm=llm is not None,
        )

        self.embedder: Embedder = embedder or _resolve_embedder(self.config)
        self.store = store or self._resolve_store()

        # duck typing: ORMMemoryStore 有 engine 属性
        store_engine = getattr(self.store, "engine", None)

        if graph_store is not None:
            self.graph_store: GraphStore | None = graph_store
        elif store_engine is not None:
            # ORMMemoryStore 路径: SQLite dialect 从 engine 取 raw connection
            if store_engine.dialect.name == "sqlite":
                import threading

                raw_conn = store_engine.raw_connection()
                self.graph_store = SQLiteGraphStore(raw_conn, threading.Lock())
            else:
                # MySQL/PG 暂不支持原生 GraphStore (AGE/Neo4j 后续)
                self.graph_store = graph_store
        else:
            self.graph_store = graph_store

        # typed_store 共享 engine + facts vector_store
        facts_vs = self._create_facts_vector_store()
        if store_engine is not None:
            self.typed_store = TypedMemoryStore(engine=store_engine, vector_store=facts_vs)
        else:
            self.typed_store = TypedMemoryStore(db_path=self.config.db_path, vector_store=facts_vs)
        self.semantic = SemanticMemory(self.typed_store, self.embedder)
        self.episodic = EpisodicMemory(self.typed_store)
        self.procedural = ProceduralMemory(self.typed_store)

        self.llm: LLM | None = llm
        if self.llm is None and self.config.llm_provider:
            try:
                from septmuse.llms import _resolve_llm

                self.llm = _resolve_llm(self.config)
            except Exception as e:
                logger.warning("llm_resolve_failed", error=str(e))
                self.llm = None

        self.extractor: FactExtractor | None = None
        if self.llm is not None:
            self.extractor = FactExtractor(self.llm, self.embedder, self.typed_store, self.store)

        from septmuse.extraction.entity import _resolve_entity_extractor

        self.entity_extractor = entity_extractor or _resolve_entity_extractor(self.config)

        # entity_store (ORMMemoryStore 路径)
        self.entity_store = None
        if store_engine is not None:
            try:
                from septmuse.storage.relational_stores.entity_store import EntityStore

                self.entity_store = EntityStore.from_engine(store_engine, self.embedder)
            except Exception as e:
                logger.warning("entity_store_init_failed", error=str(e))

        try:
            from septmuse.rerankers import create_reranker
            from septmuse.rerankers.noop import NoopReranker

            self._reranker = create_reranker(
                self.config.reranker_backend,
                embedder=self.embedder,
                llm=self.llm,
            )
        except Exception as e:
            logger.warning("reranker_resolve_failed", error=str(e))
            from septmuse.rerankers.noop import NoopReranker

            self._reranker = NoopReranker()

        from septmuse.governance.approval import DedupWindow

        self._dedup_window = DedupWindow()

        self._hybrid_retriever = None
        self._search_pipeline = None
        self._recall_pipeline = None

        self._init_v2_components()

    def _safe_init(self, attr: str, factory: Callable[[], Any]) -> None:
        """尝试构造组件, 失败设 None + 日志 (集中 try/except, 消除 7× 重复)。"""
        try:
            setattr(self, attr, factory())
        except Exception as e:
            setattr(self, attr, None)
            logger.warning(f"{attr}_init_failed", error=str(e))

    def _init_v2_components(self) -> None:
        """初始化 V2 编排组件 (working_memory / capture / token_budget / meta / evolution / forgetting / summarizer)。"""
        store_engine = getattr(self.store, "engine", None)

        def make_working_memory() -> Any:
            from septmuse.memory.working_memory import WorkingMemory
            from septmuse.storage.working_memory_stores.factory import create_working_memory_store

            return WorkingMemory(
                store=create_working_memory_store(self.config, engine=store_engine), agent_id="default"
            )

        def make_capture() -> Any:
            from septmuse.capture.pipeline import CapturePipeline

            return CapturePipeline(
                self.store, self.embedder,
                typed_store=self.typed_store, llm=self.llm, dedup_window=self._dedup_window,
            )

        def make_token_budget() -> Any:
            from septmuse.retrieval.token_budget import TokenBudget

            return TokenBudget()

        def make_meta() -> Any:
            from septmuse.memory.meta import MetacognitionLayer

            return MetacognitionLayer(self.embedder, self.store, self.typed_store)

        def make_evolution() -> Any:
            from septmuse.memory.evolution import EvolutionEngine

            return EvolutionEngine(
                self.store, self.graph_store, self.embedder, self.typed_store, self.llm
            )

        def make_forgetting() -> Any:
            from septmuse.retrieval.forgetting import ForgettingRetriever

            return ForgettingRetriever(self.typed_store)

        def make_summarizer() -> Any:
            from septmuse.evolution.summarizer import Summarizer

            return Summarizer(self.store, self.typed_store, self.llm)

        if store_engine is not None:
            self._safe_init("working_memory", make_working_memory)
        else:
            self.working_memory = None
        self._safe_init("capture", make_capture)
        self._safe_init("token_budget", make_token_budget)
        self._safe_init("meta", make_meta)
        self._safe_init("evolution", make_evolution)
        self._safe_init("forgetting", make_forgetting)
        self._safe_init("summarizer", make_summarizer)

    def _resolve_store(self) -> MemoryStore:
        """解析 store: 统一走 ORMMemoryStore (DatabaseService 自动回退 SQLite 零配置)。"""
        from septmuse.storage.relational_stores.factory import RelationalStoreFactory

        return RelationalStoreFactory.create(self.config)

    def _create_facts_vector_store(self) -> Any:
        """创建 SemanticFact 专用 vector_store (collection_name='septmuse_facts')。

        从 ORMMemoryStore 的 vector_store 判断后端类型, 仅 QdrantVectorStore 支持多 collection。
        本地模式复用同一 QdrantClient (_local_client_cache), 无文件锁冲突。
        其他后端 (SQLAlchemy/Chroma 等) 返回 None, TypedMemoryStore 降级 numpy 全扫描。
        """
        import os

        vs = getattr(self.store, "_vector_store", None)
        if vs is None:
            return None
        if type(vs).__name__ != "QdrantVectorStore":
            return None
        try:
            from septmuse.storage.vector_stores.qdrant import QdrantVectorStore

            return QdrantVectorStore(
                collection_name="septmuse_facts",
                embedding_model_dims=getattr(
                    vs, "_dim", self.config.vector_store.embedding_model_dims or 512
                ),
                path=os.getenv("SEPTMUSE_QDRANT_PATH"),
                host=os.getenv("SEPTMUSE_QDRANT_HOST"),
                port=int(os.getenv("SEPTMUSE_QDRANT_PORT", "6333"))
                if os.getenv("SEPTMUSE_QDRANT_HOST")
                else None,
                url=os.getenv("SEPTMUSE_QDRANT_URL"),
                api_key=os.getenv("SEPTMUSE_QDRANT_API_KEY"),
                enable_bm25=getattr(vs, "_enable_bm25", True),
            )
        except Exception as e:
            logger.warning("facts_vector_store_create_failed", error=str(e))
            return None

    # ------------------------------------------------------------------
    # 核心 API
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
        # typed memory kwargs (memory_type != None 时使用)
        subject: str | None = None,
        predicate: str | None = None,
        object: str | None = None,
        event_type: str = "fact",
        namespace: str = "default",
        rule: str | None = None,
        expiration_date: str | None = None,
    ) -> dict[str, Any]:
        """添加记忆 (统一 typed memory 入口)。

        Args:
            messages: str 或 List[{"role","content"}]
            user_id: 用户 ID (必填, 跨 agent 共享键)
            agent_id: agent ID (可选)
            session_id: 会话 ID (None=不限制)
            metadata: 元数据
            infer: True=LLM 抽取事实; False=原文存; None=用 config.infer
            memory_type: None=verbatim 记忆; "fact"/"episode"/"rule"=类型化记忆
            auto_extract_entities: True=自动抽取实体 (默认)
            valid_at: 事实开始为真的时间 (双时态建模)

        Returns:
            {"results": [{"id","memory","event":"ADD"}], "relations": []}
        """
        # 类型化记忆路由
        if memory_type == "fact":
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

        if memory_type == "episode":
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

        if memory_type == "rule":
            rule_text = rule or (messages if isinstance(messages, str) else str(messages))
            r = self.procedural.add_rule(
                rule_text,
                user_id=user_id,
                namespace=namespace,
                source_tracing=metadata.get("source_tracing") if metadata else None,
            )
            return {"id": r.id, "rule": r.rule, "event": "ADD"}

        # verbatim / infer 模式
        should_infer = self.config.infer if infer is None else infer

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

        # 批量插入 (对齐 mem0 V3 Phase 6, 单次 commit)
        records = list(zip(texts, embeddings, strict=True))
        ids = self.store.add_batch(
            records,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            metadata=metadata,
            valid_at=valid_at,
        )

        # 批量实体链接 (对齐 mem0 V3 Phase 7, 全局去重)
        valid_pairs: list[tuple[str, str]] = [
            (text, mid) for text, mid in zip(texts, ids, strict=True) if mid is not None
        ]
        if auto_extract_entities and self.entity_store is not None and self.entity_extractor is not None:
            self._batch_extract_and_store_entities(valid_pairs, user_id=user_id, agent_id=agent_id)

        results: list[dict[str, Any]] = []
        for text, mid in zip(texts, ids, strict=True):
            if mid is None:
                continue  # 批次内 hash 去重跳过
            results.append({"id": mid, "memory": text, "event": "ADD"})

        self._invalidate_search_cache(user_id=user_id)
        logger.info("memory_add_done", user_id=user_id, count=len(results), infer=should_infer)
        return {"results": results, "relations": []}

    def _extract_and_store_entities(
        self, text: str, memory_id: str, *, user_id: str, agent_id: str | None = None
    ) -> None:
        """自动抽取实体并存入 entity_store (吞错, 不阻塞 add)。"""
        if self.entity_extractor is None or self.entity_store is None:
            return
        try:
            entities = self.entity_extractor.extract(text)
            for entity in entities:
                self.entity_store.upsert(entity, memory_id, user_id=user_id, agent_id=agent_id)
        except Exception as e:
            logger.warning("auto_extract_entities_failed", error=str(e))

    def _batch_extract_and_store_entities(
        self, pairs: list[tuple[str, str]], *, user_id: str, agent_id: str | None = None
    ) -> None:
        """批量实体链接 (对齐 mem0 V3 Phase 7, 全局去重)。

        全局去重: 同一实体 (归一化名) 在多条记忆中出现, 只 upsert 一次,
        但 linked_memory_ids 会包含所有相关 memory_id。

        Args:
            pairs: [(text, memory_id), ...] 列表
        """
        if not pairs:
            return
        from septmuse.extraction.entity import Entity

        try:
            # 1. 全局去重: normalized_key → (entity_type, entity_text, set of memory_ids)
            global_entities: dict[str, tuple[str, str, set[str]]] = {}
            for text, memory_id in pairs:
                entities = self.entity_extractor.extract(text)  # type: ignore[union-attr]
                for entity in entities:
                    key = entity.text.strip().lower()
                    if key in global_entities:
                        global_entities[key][2].add(memory_id)
                    else:
                        global_entities[key] = (entity.entity_type, entity.text, {memory_id})

            # 2. 逐个 upsert (upsert 内部做精确+语义去重, 全局去重减少重复调用)
            for entity_type, entity_text, memory_ids in global_entities.values():
                entity = Entity(text=entity_text, entity_type=entity_type, start=0, end=0)
                for mid in memory_ids:
                    self.entity_store.upsert(  # type: ignore[union-attr]
                        entity, mid, user_id=user_id, agent_id=agent_id
                    )
        except Exception as e:
            logger.warning("batch_extract_entities_failed", error=str(e))

    def _get_hybrid_retriever(self):
        """获取持久化 HybridRetriever 实例 (缓存跨 search 调用生效)。"""
        if self._hybrid_retriever is None:
            from septmuse.retrieval.hybrid import HybridRetriever

            self._hybrid_retriever = HybridRetriever(
                self.store,
                self.embedder,
                entity_extractor=self.entity_extractor,
                entity_store=self.entity_store,
            )
        return self._hybrid_retriever

    def _get_search_pipeline(self):
        """获取持久化 SearchPipeline 实例 (缓存跨 search 调用生效)。"""
        if self._search_pipeline is None:
            from septmuse.rerankers import create_reranker
            from septmuse.retrieval.search_pipeline import SearchPipeline, VectorRetrieverAdapter

            self._search_pipeline = SearchPipeline(
                hybrid_retriever=self._get_hybrid_retriever(),
                vector_retriever=VectorRetrieverAdapter(self.store, self.embedder),
                reranker_factory=lambda name: create_reranker(name, embedder=self.embedder, llm=self.llm),
                embedder=self.embedder,
                forgetting=self.forgetting,
                default_top_k=self.config.top_k,
                default_threshold=self.config.threshold,
            )
        return self._search_pipeline

    def _get_recall_pipeline(self):
        """获取持久化 RecallPipeline 实例 (缓存跨 recall 调用生效)。"""
        if self._recall_pipeline is None:
            from septmuse.retrieval.recall_pipeline import RecallPipeline

            self._recall_pipeline = RecallPipeline(
                search_pipeline=self._get_search_pipeline(),
                meta=self.meta,
                token_budget=self.token_budget,
                working_memory=self.working_memory,
                summarizer=self.summarizer,
                typed_store=self.typed_store,
            )
        return self._recall_pipeline

    def _invalidate_search_cache(self, user_id: str | None = None) -> None:
        """清除 HybridRetriever 搜索缓存 (add/update/delete 后调用)。"""
        if self._hybrid_retriever is not None:
            self._hybrid_retriever.invalidate_cache(user_id=user_id)

    def search(
        self,
        query: str,
        *,
        user_id: str,
        session_id: str | None = None,
        top_k: int | None = None,
        threshold: float | None = None,
        hybrid: bool = True,
        reranker: str | None = None,
        recipe: str | None = None,
        explain: bool = False,
        filters: dict[str, Any] | None = None,
        search_filter: dict[str, Any] | None = None,
        show_expired: bool = False,
        # B 类高级参数
        forgetting: bool = False,
        token_budget: int | None = None,
        inject_prompt: bool = False,
        hyde: bool = False,
        query_rewrite: bool = False,
    ) -> list[dict[str, Any]] | dict[str, Any]:
        """检索记忆 (默认 hybrid BM25+向量 RRF 融合, 支持后处理重排)。

        委托给 SearchPipeline。管线拥有 recipe 解析 → 检索 → reranker → 过期过滤 → 后处理全链路。
        """
        return self._get_search_pipeline().search(
            query,
            user_id=user_id,
            session_id=session_id,
            top_k=top_k,
            threshold=threshold,
            hybrid=hybrid,
            reranker=reranker,
            recipe=recipe,
            explain=explain,
            filters=filters,
            search_filter=search_filter,
            show_expired=show_expired,
            forgetting=forgetting,
            token_budget=token_budget,
            inject_prompt=inject_prompt,
            hyde=hyde,
            query_rewrite=query_rewrite,
        )

    def get_all(
        self, *, user_id: str, session_id: str | None = None, filters: dict[str, Any] | None = None,
        show_expired: bool = False,
    ) -> dict[str, Any]:
        """列出全部记忆。"""
        results = self.store.get_all(user_id=user_id, session_id=session_id, filters=filters)
        all_results = results.get("results", results) if isinstance(results, dict) else results
        if not show_expired:
            all_results = [r for r in all_results if not _is_expired(r.get("metadata"))]
        return {"results": all_results}

    def get(self, memory_id: str) -> dict[str, Any] | None:
        """取单条。"""
        return self.store.get(memory_id)

    def delete(self, memory_id: str) -> dict[str, str]:
        """软删除 (自动清理实体引用)。"""
        if self.entity_store is not None:
            self.entity_store.remove_memory_from_entities(memory_id)
        self.store.delete(memory_id)
        self._invalidate_search_cache()
        return {"status": "deleted", "memory_id": memory_id}

    def delete_all(self, *, user_id: str, session_id: str | None = None) -> dict[str, Any]:
        """批量删除。

        删除该 user (可选 session_id) 的所有记忆 + 清理实体引用。
        """
        all_mems = self.store.get_all(user_id=user_id, session_id=session_id)
        results = all_mems.get("results", all_mems) if isinstance(all_mems, dict) else all_mems
        count = 0
        for mem in results:
            mid = mem.get("id") if isinstance(mem, dict) else mem
            if mid is None:
                continue
            if self.entity_store is not None:
                self.entity_store.remove_memory_from_entities(mid)
            self.store.delete(mid)
            count += 1
        logger.info("memory_delete_all", user_id=user_id, count=count)
        self._invalidate_search_cache(user_id=user_id)
        return {"status": "deleted", "count": count}

    def update(
        self,
        memory_id: str,
        data: str | None = None,
        *,
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """更新记忆。

        Args:
            memory_id: 记忆 ID
            data: 新内容; None=不改 content, 只改 metadata
            user_id: 用户 ID (仅用于日志)
            metadata: 新 metadata; None=不改

        Returns:
            {"id", "memory", "event": "UPDATE"} 或 {"id", "event": "NOT_FOUND"}
        """
        existing = self.store.get(memory_id)
        if existing is None:
            return {"id": memory_id, "event": "NOT_FOUND"}

        new_content = data if data is not None else existing["memory"]
        new_embedding = self.embedder.embed(new_content)

        if metadata is not None:
            merged_meta = {**existing.get("metadata", {}), **metadata}
        else:
            merged_meta = existing.get("metadata", {})

        ok = self.store.update(memory_id, new_content, new_embedding, metadata=merged_meta)
        if not ok:
            return {"id": memory_id, "event": "NOT_FOUND"}
        self._invalidate_search_cache(user_id=user_id)
        logger.info("memory_update_done", memory_id=memory_id, user_id=user_id)
        return {"id": memory_id, "memory": new_content, "event": "UPDATE"}

    def get_history(self, memory_id: str) -> list[dict[str, Any]]:
        """获取记忆变更历史。"""
        return self.store.get_history(memory_id)

    # ------------------------------------------------------------------
    # 差异化: 双时态 + 知识图谱
    # ------------------------------------------------------------------

    def invalidate(self, memory_id: str, *, invalid_at: str | None = None) -> dict[str, Any]:
        """标记事实不再为真 (设置 invalid_at + expired_at, 不删除记忆)。"""
        return self.store.invalidate(memory_id, invalid_at=invalid_at)

    def cognify(self, text: str, *, user_id: str, agent_id: str | None = None) -> dict[str, Any]:
        """构建知识图谱: 存记忆 → 抽三元组 → 存实体/关系 → 建记忆链接。"""
        from septmuse.extraction.cognify import CognifyPipeline

        pipeline = CognifyPipeline(
            self.store,
            self.graph_store,
            self.embedder,
            entity_store=self.entity_store,
            llm=self.llm,
            entity_extractor=self.entity_extractor,
        )
        return pipeline.cognify(text, user_id=user_id, agent_id=agent_id)

    # ------------------------------------------------------------------
    # 差异化: 规则系统
    # ------------------------------------------------------------------

    def get_active_rules(self, *, user_id: str, namespace: str = "default") -> list[dict[str, Any]]:
        """获取应注入的规则 (废弃规则不返回)。"""
        rules = self.procedural.get_active_rules(user_id=user_id, namespace=namespace)
        return [
            {
                "id": r.id,
                "rule": r.rule,
                "confidence": r.confidence,
                "helpful": r.helpful_count,
                "harmful": r.harmful_count,
            }
            for r in rules
        ]

    def rules_to_prompt(self, *, user_id: str, namespace: str = "default") -> str:
        """编译规则为 prompt 注入文本 (仅 active)。"""
        return self.procedural.rules_to_prompt(user_id=user_id, namespace=namespace)

    def record_rule_outcome(self, rule_id: str, helpful: bool) -> dict[str, Any]:
        """记录规则应用结果 (helpful/harmful 追踪 + 自动退化)。"""
        r = self.procedural.record_outcome(rule_id, helpful)
        if r is None:
            return {"error": f"rule {rule_id} not found"}
        return {
            "id": r.id,
            "helpful_count": r.helpful_count,
            "harmful_count": r.harmful_count,
            "confidence": r.confidence,
            "deprecated": r.deprecated,
        }

    # ------------------------------------------------------------------
    # 共享 (实用)
    # ------------------------------------------------------------------

    def list_agents(self, user_id: str) -> list[str]:
        """列出该用户的所有 agent (跨 agent 共享)。"""
        from septmuse.governance.sharing import SharedMemoryAccessor

        return SharedMemoryAccessor(self.store).list_agents(user_id)

    # ------------------------------------------------------------------
    # V2 编排 API (remember / recall / improve / forget)
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
        result: dict[str, Any] = {"captured": False, "raw_id": None, "fact_ids": [], "memory_ids": [], "memory_id": None}

        if not text or not text.strip():
            return result

        if self.capture is None:
            logger.warning("remember_capture_unavailable", user_id=user_id)
            return result

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
        self._invalidate_search_cache(user_id=user_id)

        if self.entity_store is not None and self.entity_extractor is not None:
            self._extract_and_store_entities(stored_text, verbatim_id, user_id=user_id, agent_id=agent_id)

        if self.llm is not None and self.extractor is not None:
            try:
                extracted = self.extractor.extract_and_store(stored_text, user_id=user_id)
                result["fact_ids"] = [e.get("id", "") for e in extracted if isinstance(e, dict)]
            except Exception as e:
                logger.warning("remember_extract_failed", error=str(e))

        if agent_id is not None and self.working_memory is not None:
            with contextlib.suppress(Exception):
                self.working_memory.core_memory_append("notes", stored_text)

        result["captured"] = True
        logger.info("remember_done", user_id=user_id, raw_id=result["raw_id"], facts=len(result["fact_ids"]))
        return result

    def recall(
        self,
        query: str,
        *,
        user_id: str,
        session_id: str | None = None,
        top_k: int | None = None,
        auto_compress: bool = False,
        recipe: str | None = None,
        inject_profile: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """V2 recall: 检索 → token 裁剪 → 注入 prompt → L0 路由 + L2 策略。

        委托给 RecallPipeline。管线拥有 compress → search → budget → prompt → route → strategy 全链路。
        """
        return self._get_recall_pipeline().recall(
            query,
            user_id=user_id,
            session_id=session_id,
            top_k=top_k,
            auto_compress=auto_compress,
            recipe=recipe,
            inject_profile=inject_profile,
            **kwargs,
        )

    def improve(self, *, user_id: str, limit: int = 20) -> dict[str, Any]:
        """V2 improve: Dream 链接生长 + reflect 蒸馏 + L1 覆盖报告持久化。

        Returns:
            {"dream": {"links_created": int, "processed": int}, "rules": int}
        """
        dream_result: dict[str, Any] = {"links_created": 0, "processed": 0}
        if self.evolution is not None:
            try:
                dream = self.evolution.dream(user_id=user_id)
                dream_result = {"links_created": dream.links_created, "processed": dream.processed}
            except Exception as e:
                logger.warning("improve_dream_failed", error=str(e))

        rules_count = 0
        if self.llm is not None and self.evolution is not None:
            try:
                reflect_result = self.evolution.reflect(user_id=user_id, limit=limit)
                rules_count = len(reflect_result.rule_ids) if hasattr(reflect_result, "rule_ids") else 0
            except Exception as e:
                logger.warning("improve_reflect_failed", error=str(e))

        coverage_info: dict[str, Any] = {}
        if self.meta is not None:
            try:
                report = self.meta.analyze_coverage(user_id=user_id)
                coverage_info = {
                    "overall_score": report.overall_score,
                    "weak_areas": report.weak_areas,
                    "strong_areas": report.strong_areas,
                }
                self.typed_store.add_fact(
                    subject=user_id,
                    predicate="has_coverage_score",
                    object=f"{report.overall_score:.2f}",
                    user_id=user_id,
                    context=report.summary(),
                    confidence=report.overall_score,
                    provenance="meta",
                    tags=["meta", "coverage"],
                )
            except Exception as e:
                logger.warning("improve_coverage_persist_failed", error=str(e))

        conflicts_info: dict[str, Any] = {}
        if self.evolution is not None:
            try:
                conflicts_info = self.evolution.resolve_conflicts(user_id=user_id)
            except Exception as e:
                logger.warning("improve_conflict_failed", error=str(e))

        logger.info("improve_done", user_id=user_id, links=dream_result["links_created"], rules=rules_count)
        return {"dream": dream_result, "rules": rules_count, "coverage": coverage_info, "conflicts": conflicts_info}

    def forget(self, memory_id: str, *, user_id: str) -> dict[str, Any]:
        """V2 forget: 先 invalidate 再 delete (决策 5)。

        Returns:
            {"event": "FORGET", "invalidated_at": str|None, "deleted_at": str|None}
        """
        existing = self.store.get(memory_id)
        if existing is not None:
            owner = existing.get("user_id")
            if owner is not None and owner != user_id:
                logger.warning("forget_ownership_mismatch", memory_id=memory_id, user_id=user_id, owner=owner)
                return {
                    "event": "FORGET",
                    "error": "ownership_mismatch",
                    "invalidated_at": None,
                    "deleted_at": None,
                }

        invalidated_at = None
        try:
            inv = self.invalidate(memory_id)
            if isinstance(inv, dict):
                invalidated_at = inv.get("invalid_at") or inv.get("invalidated_at")
        except Exception as e:
            logger.warning("forget_invalidate_failed", error=str(e))

        deleted_at = None
        try:
            if existing is not None:
                self.delete(memory_id)
                from datetime import datetime, timezone

                deleted_at = datetime.now(timezone.utc).isoformat()
        except Exception as e:
            logger.warning("forget_delete_failed", error=str(e))

        logger.info("forget_done", memory_id=memory_id, user_id=user_id)
        return {"event": "FORGET", "invalidated_at": invalidated_at, "deleted_at": deleted_at}

    def compress(self, *, user_id: str, mode: str = "static", buffer_size: int = 20) -> dict[str, Any]:
        """压缩消息 (委托 Summarizer.compress)。

        Returns:
            {"compressed": bool, "evicted": int, "kept": int, "summary_id": str|None}
        """
        if self.summarizer is None:
            return {"compressed": False, "evicted": 0, "kept": 0, "summary_id": None}
        return self.summarizer.compress(user_id=user_id, mode=mode, buffer_size=buffer_size)

    # ------------------------------------------------------------------
    # 资源释放
    # ------------------------------------------------------------------

    def reset(self) -> dict[str, Any]:
        """重置记忆系统 (清表 + 双写清理 + 缓存失效)。

        清除: memories / history / memory_access_logs / septmuse_facts /
        septmuse_episodic / septmuse_procedural / septmuse_entities /
        entity_relations, 并重置 vector_store + keyword_index。
        """
        self.store.reset()

        engine = getattr(self.store, "engine", None)
        if engine is not None:
            from sqlalchemy import text
            from sqlmodel import Session

            with Session(engine) as session:
                for table_name in [
                    "septmuse_facts",
                    "septmuse_episodic",
                    "septmuse_procedural",
                ]:
                    with contextlib.suppress(Exception):
                        session.exec(text(f"DELETE FROM {table_name}"))
                session.commit()

        if self.entity_store is not None:
            with contextlib.suppress(Exception):
                self.entity_store.reset()

        if hasattr(self, "_dedup_window"):
            self._dedup_window = type(self._dedup_window)()

        if self.working_memory is not None:
            with contextlib.suppress(Exception):
                wm_store = getattr(self.working_memory, "store", None)
                if wm_store is not None and hasattr(wm_store, "reset"):
                    wm_store.reset()

        self._hybrid_retriever = None
        self._search_pipeline = None
        self._recall_pipeline = None

        logger.info("memory_reset_done")
        return {"status": "reset", "message": "all memories cleared"}

    def close(self) -> None:
        """关闭存储。"""
        self.store.close()
        self.typed_store.close()
        if self.entity_store is not None:
            self.entity_store.close()

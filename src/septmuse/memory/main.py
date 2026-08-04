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

from typing import Any

from septmuse.configs.defaults import MemoryConfig, default_config
from septmuse.core.logging import get_logger
from septmuse.embedders.base import Embedder
from septmuse.llms.base import LLM
from septmuse.models.episodic import EpisodicMemory
from septmuse.models.extract import FactExtractor
from septmuse.models.fact import SemanticMemory
from septmuse.models.procedural import ProceduralMemory
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


def _normalize_messages(messages: Any) -> list[str]:
    """归一化 messages 为 text 列表 (接受 str 或 List[Dict])。"""
    if isinstance(messages, str):
        return [messages]
    texts: list[str] = []
    for m in messages:
        if isinstance(m, dict):
            content = m.get("content")
            if content:
                texts.append(content)
        elif isinstance(m, str):
            texts.append(m)
    return texts


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

        # typed_store 共享 engine
        if store_engine is not None:
            self.typed_store = TypedMemoryStore(engine=store_engine)
        else:
            self.typed_store = TypedMemoryStore(db_path=self.config.db_path)
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
            from septmuse.storage.relational_stores.entity_store import EntityStore

            self.entity_store = EntityStore.from_engine(store_engine, self.embedder)

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

    def _resolve_store(self) -> MemoryStore:
        """解析 store: 统一走 ORMMemoryStore (DatabaseService 自动回退 SQLite 零配置)。"""
        from septmuse.storage.relational_stores.factory import RelationalStoreFactory

        return RelationalStoreFactory.create(self.config)

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
    ) -> list[dict[str, Any]]:
        """检索记忆 (默认 hybrid BM25+向量 RRF 融合, 支持后处理重排)。

        Args:
            query: 查询文本
            user_id: 用户 ID (必填)
            session_id: 会话 ID (None=不限)
            top_k: 返回数 (默认 config.top_k)
            threshold: 相似阈值 (默认 config.threshold)
            hybrid: True=BM25+向量 RRF 融合 (默认); False=纯向量
            reranker: 重排器 ("noop"/"mmr"/"cross_encoder"/"llm"/"cohere"/"batch_llm", None=用 config)
            recipe: 预置检索 recipe (覆盖 hybrid/reranker/explain)
            explain: True=返回 score_details
            filters: 字段过滤字典 (hybrid + 纯向量均生效)
            search_filter: reranker boost 权重字典 (匹配 user_id/session_id/tags 的结果 score 加权)

        Returns:
            list[{"id","memory","score","metadata","created_at"}]
        """
        if recipe is not None:
            from septmuse.retrieval.recipes import get_recipe

            r = get_recipe(recipe)
            hybrid = r.hybrid
            if r.reranker != "noop":
                reranker = r.reranker
            explain = r.explain

        tk = top_k or self.config.top_k
        if hybrid:
            th = 0.0 if threshold is None else threshold
        else:
            th = threshold if threshold is not None else self.config.threshold

        if hybrid:
            from septmuse.retrieval.hybrid import HybridRetriever

            retriever = HybridRetriever(
                self.store,
                self.embedder,
                entity_extractor=self.entity_extractor,
                entity_store=self.entity_store,
            )
            hybrid_results = retriever.search(
                query,
                user_id=user_id,
                session_id=session_id,
                top_k=tk,
                threshold=th,
                explain=explain,
                filters=filters,
            )
            # reranker 应用 (try/except 降级容错: 失败回退原始结果)
            if reranker is not None and reranker != "noop":
                from septmuse.rerankers import create_reranker
                from septmuse.rerankers.strategies import RerankerStrategyFactory

                try:
                    reranker_instance = create_reranker(reranker, embedder=self.embedder, llm=self.llm)
                    strategy = RerankerStrategyFactory.create("full_memory")
                    tracker, documents = strategy.prepare(hybrid_results)
                    scored = reranker_instance.rerank(query, documents, top_k=tk)
                    hybrid_results = strategy.reconstruct(scored, tracker, hybrid_results, tk, search_filter)
                except Exception as e:
                    logger.warning("reranker_failed_fallback", reranker=reranker, error=str(e))
            return [
                {
                    "id": r.id,
                    "memory": r.memory,
                    "score": r.score,
                    "vector_score": r.vector_score,
                    "bm25_score": r.bm25_score,
                    "metadata": r.metadata,
                    "created_at": r.created_at,
                }
                for r in hybrid_results
            ]

        # 纯向量分支
        emb = self.embedder.embed(query)
        results = self.store.search(
            emb,
            user_id=user_id,
            session_id=session_id,
            top_k=tk,
            threshold=th,
            filters=filters,
        )

        # 纯向量分支也应用 reranker (try/except 降级容错)
        if reranker is not None and reranker != "noop" and results:
            from septmuse.rerankers import create_reranker
            from septmuse.rerankers.strategies import RerankerStrategyFactory
            from septmuse.retrieval.hybrid import HybridResult

            try:
                reranker_instance = create_reranker(reranker, embedder=self.embedder, llm=self.llm)
                strategy = RerankerStrategyFactory.create("full_memory")
                # dict → HybridResult 转换
                hybrid_results = [
                    HybridResult(
                        id=r["id"],
                        memory=r["memory"],
                        score=r["score"],
                        vector_score=r.get("score", 0.0),
                        bm25_score=0.0,
                        entity_boost=0.0,
                        metadata=r.get("metadata", {}),
                        created_at=r.get("created_at"),
                    )
                    for r in results
                ]
                tracker, documents = strategy.prepare(hybrid_results)
                scored = reranker_instance.rerank(query, documents, top_k=tk)
                reranked = strategy.reconstruct(scored, tracker, hybrid_results, tk, search_filter)
                # HybridResult → dict 转换
                results = [
                    {
                        "id": r.id,
                        "memory": r.memory,
                        "score": r.score,
                        "metadata": r.metadata,
                        "created_at": r.created_at,
                    }
                    for r in reranked
                ]
            except Exception as e:
                logger.warning("reranker_failed_fallback", reranker=reranker, error=str(e))

        logger.info("memory_search_done", user_id=user_id, query=query[:50], hits=len(results))
        return results

    def get_all(
        self, *, user_id: str, session_id: str | None = None, filters: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """列出全部记忆。"""
        return {"results": self.store.get_all(user_id=user_id, session_id=session_id, filters=filters)}

    def get(self, memory_id: str) -> dict[str, Any] | None:
        """取单条。"""
        return self.store.get(memory_id)

    def delete(self, memory_id: str) -> dict[str, str]:
        """软删除 (自动清理实体引用)。"""
        if self.entity_store is not None:
            self.entity_store.remove_memory_from_entities(memory_id)
        self.store.delete(memory_id)
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
    # 资源释放
    # ------------------------------------------------------------------

    def close(self) -> None:
        """关闭存储。"""
        self.store.close()
        self.typed_store.close()
        if self.entity_store is not None:
            self.entity_store.close()

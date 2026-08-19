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
"""Memory facade — SeptMuse 零配置入口。"""

from __future__ import annotations

import contextlib
from typing import Any, ClassVar

from septmuse.configs.defaults import MemoryConfig, default_config
from septmuse.core.logging import get_logger
from septmuse.core.utils import is_expired as _is_expired
from septmuse.embedders.base import Embedder
from septmuse.llms.base import LLM
from septmuse.memory._factory import MemoryComponentFactory
from septmuse.memory.registry_v2 import ComponentRegistry
from septmuse.pipelines.search import PostProcessConfig, SearchConfig
from septmuse.storage.base import MemoryStore, TemporalStore
from septmuse.storage.graph_stores.base import GraphStore

logger = get_logger(__name__)


class Memory:
    """SeptMuse 记忆系统零配置 facade。"""

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

        # 核心组件全部由工厂构造，__init__ 只负责赋值
        state = MemoryComponentFactory.build(
            self.config,
            embedder=embedder,
            store=store,
            graph_store=graph_store,
            llm=llm,
            entity_extractor=entity_extractor,
        )

        self.embedder = state.embedder
        self.store = state.store
        self.graph_store = state.graph_store
        self.typed_store = state.typed_store
        self.semantic = state.semantic
        self.episodic = state.episodic
        self.procedural = state.procedural
        self.llm = state.llm
        self.extractor = state.extractor
        self.entity_extractor = state.entity_extractor
        self.entity_store = state.entity_store
        self._reranker = state.reranker
        self._dedup_window = state.dedup_window

        self._hybrid_retriever = None
        self._registry = ComponentRegistry(self)
        self._registry.register_all()

    # ------------------------------------------------------------------
    # Lazy-loaded domain properties (via __getattr__)
    # ------------------------------------------------------------------

    # 6 个领域模块 + 兼容属性
    _DOMAIN_ATTRS: ClassVar[dict[str, str]] = {
        "evolution": "evolution_module",
        "entities": "entity_module",
        "temporal": "temporal_module",
        "self_loop": "self_loop",
        "working_memory": "working_memory",
        "token_budget": "token_budget",
        "forgetting": "forgetting",
        "summarizer": "summarizer",
        "_capture_pipeline": "capture",
    }

    def __getattr__(self, name: str) -> Any:
        """自动委托: mem.evolution → registry.get("evolution_module")。

        如果 name 在 _DOMAIN_ATTRS 中，从 ComponentRegistry 懒加载。
        否则尝试从 registry 按原名查找。
        都找不到则抛出 AttributeError。
        """
        if name in self._DOMAIN_ATTRS:
            return self._registry.get(self._DOMAIN_ATTRS[name])
        result = self._registry.get(name)
        if result is not None:
            return result
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

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
        expiration_date: str | None = None,
        attributed_to: str | None = None,
        # typed memory kwargs (memory_type != None 时使用)
        subject: str | None = None,
        predicate: str | None = None,
        object: str | None = None,
        event_type: str = "fact",
        namespace: str = "default",
        rule: str | None = None,
    ) -> dict[str, Any]:
        """添加记忆 (统一类型化入口)。"""
        # 多说话人归因: attributed_to / actor_id 放入 metadata
        if attributed_to is not None or isinstance(messages, list):
            if metadata is None:
                metadata = {}
            if attributed_to is not None:
                metadata = {**metadata, "attributed_to": attributed_to}
            if isinstance(messages, list) and messages:
                first_msg = messages[0] if isinstance(messages[0], dict) else {}
                actor_id = first_msg.get("name")
                if actor_id is not None:
                    metadata = {**metadata, "actor_id": actor_id}

        result = self._registry.get("add").add(
            messages,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            metadata=metadata,
            infer=infer,
            memory_type=memory_type,
            auto_extract_entities=auto_extract_entities,
            valid_at=valid_at,
            expiration_date=expiration_date,
            subject=subject,
            predicate=predicate,
            object=object,
            event_type=event_type,
            namespace=namespace,
            rule=rule,
        )
        self._invalidate_search_cache(user_id=user_id)
        return result

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

    @property
    def _retriever(self):
        """向后兼容别名 — 返回 HybridRetriever 实例。"""
        return self._get_hybrid_retriever()

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
        return self._registry.get("search").search(
            query,
            user_id=user_id,
            session_id=session_id,
            config=SearchConfig(
                top_k=top_k,
                threshold=threshold,
                hybrid=hybrid,
                reranker=reranker,
                recipe=recipe,
                explain=explain,
                filters=filters,
                search_filter=search_filter,
                show_expired=show_expired,
                hyde=hyde,
                query_rewrite=query_rewrite,
            ),
            post_process=PostProcessConfig(
                forgetting=forgetting,
                token_budget=token_budget,
                inject_prompt=inject_prompt,
            ),
        )

    def get_all(
        self,
        *,
        user_id: str,
        session_id: str | None = None,
        filters: dict[str, Any] | None = None,
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
        """更新记忆。"""
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
        if not isinstance(self.store, TemporalStore):
            return {"error": f"{type(self.store).__name__} 不支持 invalidate (需 temporal 列)"}
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
        return self.evolution.get_active_rules(user_id=user_id, namespace=namespace)

    def rules_to_prompt(self, *, user_id: str, namespace: str = "default") -> str:
        """编译规则为 prompt 注入文本 (仅 active)。"""
        return self.evolution.rules_to_prompt(user_id=user_id, namespace=namespace)

    def record_rule_outcome(self, rule_id: str, helpful: bool) -> dict[str, Any]:
        """记录规则应用结果 (helpful/harmful 追踪 + 自动退化)。"""
        return self.evolution.record_rule_outcome(rule_id, helpful)

    def get_user_profile(
        self, user_id: str, *, include_temporal: bool = True
    ) -> Any:
        """从 SemanticFact 聚合用户结构化画像。委托给 SemanticMemory。"""
        return self.semantic.get_user_profile(user_id, include_temporal=include_temporal)

    # ------------------------------------------------------------------
    # 共享 (实用)
    # ------------------------------------------------------------------

    def list_agents(self, user_id: str) -> list[str]:
        """列出该用户的所有 agent (跨 agent 共享)。"""
        return self.entities.list_agents(user_id)

    # ------------------------------------------------------------------
    # V2 编排 API (remember / recall / improve / forget)
    # ------------------------------------------------------------------

    def capture(
        self,
        text: str,
        *,
        user_id: str,
        agent_id: str | None = None,
        session_id: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """PostToolUse 捕获流水线 (SHA256去重→脱敏→嵌入→双索引)。"""
        if self._capture_pipeline is None:
            return {"captured": False, "memory_id": None, "deduped": False, "redacted": False}
        result = self._capture_pipeline.capture(
            text, user_id=user_id, agent_id=agent_id, session_id=session_id, metadata=kwargs.get("metadata")
        )
        return {
            "captured": result.captured,
            "memory_id": result.memory_id,
            "deduped": result.deduped,
            "redacted": result.redacted,
        }

    def remember(
        self,
        text: str,
        *,
        user_id: str,
        session_id: str | None = None,
        agent_id: str | None = None,
    ) -> dict[str, Any]:
        """V2 remember: 捕获文本并存储。"""
        result = self._registry.get("remember").remember(
            text, user_id=user_id, session_id=session_id, agent_id=agent_id
        )
        if result.get("memory_id"):
            self._invalidate_search_cache(user_id=user_id)
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
        """V2 recall: 检索并注入 prompt。"""
        return self._registry.get("recall").recall(
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
        """V2 improve: Dream + reflect + 规则精炼。"""
        return self.evolution.improve(user_id=user_id, limit=limit)

    def forget(self, memory_id: str, *, user_id: str) -> dict[str, Any]:
        """V2 forget: 失效后删除记忆。"""
        result = self._registry.get("forget").forget(memory_id, user_id=user_id)
        self._invalidate_search_cache()
        return result

    def compress(self, *, user_id: str, mode: str = "static", buffer_size: int = 20) -> dict[str, Any]:
        """压缩消息。"""
        if self.summarizer is None:
            return {"compressed": False, "evicted": 0, "kept": 0, "summary_id": None}
        return self.summarizer.compress(user_id=user_id, mode=mode, buffer_size=buffer_size)

    # ------------------------------------------------------------------
    # 实体操作
    # ------------------------------------------------------------------

    def extract_entities(self, text: str) -> list[dict[str, Any]]:
        """抽取实体 (不存储, 仅返回抽取结果)。"""
        return self.entities.extract_entities(text)

    def add_entity(self, entity_text: str, entity_type: str, memory_id: str, *, user_id: str) -> dict[str, Any]:
        """手动添加实体。"""
        return self.entities.add_entity(entity_text, entity_type, memory_id, user_id=user_id)

    def search_entities(self, query: str, *, user_id: str, top_k: int = 5) -> list[dict[str, Any]]:
        """搜索实体 (精确匹配 + 向量相似度)。"""
        return self.entities.search_entities(query, user_id=user_id, top_k=top_k)

    def get_entity_neighbors(self, entity_id: str) -> list[str]:
        """获取实体的 linked_memory_ids。"""
        if self.entity_store is None:
            return []
        return self.entity_store.get_linked_memories(entity_id)

    def list_entities(self, *, user_id: str, entity_type: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        """列出用户全部未删除实体。"""
        return self.entities.list_entities(user_id=user_id, limit=limit)

    # ------------------------------------------------------------------
    # 时态查询
    # ------------------------------------------------------------------

    def search_at(
        self,
        reference_time: str,
        query: str,
        *,
        user_id: str,
        session_id: str | None = None,
        top_k: int = 5,
        threshold: float = 0.1,
    ) -> list[dict[str, Any]]:
        """时态查询: 查询某时刻为真的相关记忆。"""
        return self.temporal.search_at(reference_time, query, user_id=user_id, top_k=top_k)

    def search_interval(
        self,
        start: str,
        end: str,
        query: str,
        *,
        user_id: str,
        session_id: str | None = None,
        top_k: int = 5,
        threshold: float = 0.1,
    ) -> list[dict[str, Any]]:
        """时间区间查询: 返回 [start, end) 内为真的相关记忆。"""
        return self.temporal.search_interval(start, end, query, user_id=user_id, top_k=top_k)

    def search_natural(
        self, query: str, *, user_id: str, top_k: int = 5, threshold: float = 0.1
    ) -> list[dict[str, Any]]:
        """自然语言时态查询 (LLM 抽时间区间 → 有则时态过滤, 无则回退普通检索)。"""
        return self.temporal.search_natural(query, user_id=user_id, top_k=top_k)

    # ------------------------------------------------------------------
    # 图检索
    # ------------------------------------------------------------------

    def get_entity_relations(self, entity_name: str, *, user_id: str) -> list[dict[str, Any]]:
        """实体间关系遍历 (双向)。"""
        return self.entities.get_entity_relations(entity_name, user_id=user_id)

    def search_graph(
        self, seed_memory_id: str, *, max_depth: int = 2, relation: str | None = None
    ) -> list[dict[str, Any]]:
        """BFS 图遍历检索。"""
        return self.search_module.search_graph(seed_memory_id, max_depth=max_depth)

    def search_graph_fused(
        self,
        query: str,
        *,
        user_id: str,
        seed_memory_id: str,
        max_depth: int = 2,
        relation: str | None = None,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """BFS + 向量结果 RRF 融合。"""
        return self.search_module.search_graph_fused(
            query, user_id=user_id, seed_memory_id=seed_memory_id, max_depth=max_depth, top_k=top_k or self.config.top_k,
        )

    # ------------------------------------------------------------------
    # 冲突解决
    # ------------------------------------------------------------------

    def resolve_conflicts(self, *, user_id: str) -> dict[str, Any]:
        """解决矛盾事实: 新事实覆盖旧事实。"""
        return self.evolution.resolve_conflicts(user_id=user_id)

    def deduplicate_entities(self, *, user_id: str) -> dict[str, Any]:
        """实体去重三段式。"""
        return self.evolution.deduplicate_entities(user_id=user_id)

    def reflect(self, *, user_id: str, limit: int = 20) -> dict[str, Any]:
        """Session 反思: 提取教训→procedural rules。"""
        return self.evolution.reflect(user_id=user_id, limit=limit)

    # ------------------------------------------------------------------
    # 资源释放
    # ------------------------------------------------------------------

    def reset(self) -> dict[str, Any]:
        """重置记忆系统 (清表 + 清理 + 缓存失效)。"""
        self._registry.get("reset").reset()

        if hasattr(self, "_dedup_window"):
            self._dedup_window = type(self._dedup_window)()

        if self._registry is not None:
            self._registry.invalidate()
            # 清理已创建的 working_memory 组件
            if self._registry.has("working_memory"):
                wm = self._registry.get("working_memory")
                if wm is not None:
                    with contextlib.suppress(Exception):
                        wm_store = getattr(wm, "store", None)
                        if wm_store is not None and hasattr(wm_store, "reset"):
                            wm_store.reset()

        self._hybrid_retriever = None

        logger.info("memory_reset_done")
        return {"status": "reset", "message": "all memories cleared"}

    # ------------------------------------------------------------------
    # 实验性 API（原 ExperimentalMemory，已合并到 Memory）
    # ------------------------------------------------------------------

    def get_working_memory(self, agent_id: str) -> Any:
        """获取 agent 的工作记忆 (从 TypedMemoryStore 加载, 自动持久化)。"""
        return self.capture_module.get_working_memory(agent_id)

    def get_blocks(self, agent_id: str) -> list[dict[str, Any]]:
        """列出 agent 的全部 block。"""
        return self.capture_module.get_blocks(agent_id)

    def update_block(self, agent_id: str, label: str, value: str) -> dict[str, Any]:
        """更新 block value。"""
        return self.capture_module.update_block(agent_id, label, value)

    def core_memory_append(self, agent_id: str, label: str, content: str) -> dict[str, Any]:
        """追加 block 内容。"""
        return self.capture_module.core_memory_append(agent_id, label, content)

    def core_memory_replace(self, agent_id: str, label: str, old_content: str, new_content: str) -> dict[str, Any]:
        """替换 block 内容片段。"""
        return self.capture_module.core_memory_replace(agent_id, label, old_content, new_content)

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
        """统一类型派发入口 — 消除 REST/CLI 重复的 verbatim/semantic/episodic/procedural 分支。"""
        return self.capture_module.add_typed(
            content, memory_type, user_id=user_id, agent_id=agent_id, session_id=session_id, infer=infer, valid_at=valid_at,
        )

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
        return self.capture_module.add_fact(
            subject, predicate, object,
            user_id=user_id, context=context, confidence=confidence, provenance=provenance, tags=tags,
        )

    def update_fact(self, fact_id: str, *, subject: str, predicate: str, object: str, user_id: str) -> dict[str, Any]:
        """更新语义事实。"""
        return self.capture_module.update_fact(fact_id, subject=subject, predicate=predicate, object=object, user_id=user_id)

    def search_facts(self, query: str, *, user_id: str, top_k: int = 5) -> list[dict[str, Any]]:
        """语义检索事实 (置信度加权)。"""
        return self.capture_module.search_facts(query, user_id=user_id, top_k=top_k)

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
        return self.capture_module.add_episode(
            content, user_id=user_id, event_type=event_type, session_id=session_id,
            observation=observation, thoughts=thoughts, action=action, result=result,
        )

    def update_episode(self, episode_id: str, *, content: str, user_id: str) -> dict[str, Any]:
        """更新情节事件。"""
        return self.capture_module.update_episode(episode_id, content=content, user_id=user_id)

    def get_timeline(self, *, user_id: str, event_type: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        """情节时序查询。"""
        return self.capture_module.get_timeline(user_id=user_id, event_type=event_type, limit=limit)

    def add_rule(
        self, rule: str, *, user_id: str, namespace: str = "default", source_tracing: str | None = None
    ) -> dict[str, Any]:
        """添加程序规则。"""
        return self.capture_module.add_rule(rule, user_id=user_id, namespace=namespace, source_tracing=source_tracing)

    def update_rule(self, rule_id: str, *, rule: str, user_id: str) -> dict[str, Any]:
        """更新程序规则。"""
        return self.capture_module.update_rule(rule_id, rule=rule, user_id=user_id)

    # ------------------------------------------------------------------
    # 检索变体
    # ------------------------------------------------------------------

    def search_hybrid(
        self, query: str, *, user_id: str, session_id: str | None = None, top_k: int = 5, threshold: float = 0.1
    ) -> list[dict[str, Any]]:
        """BM25+向量 RRF 融合检索 (核心 search(hybrid=True) 的快捷方式)。"""
        return self.search_module.search_hybrid(query, user_id=user_id, top_k=top_k)

    def search_progressive(
        self, query: str, *, user_id: str, top_k: int = 5, threshold: float = 0.1
    ) -> list[dict[str, Any]]:
        """渐进三层检索 recall→locate→expand。"""
        return self.search_module.search_progressive(query, user_id=user_id, top_k=top_k)

    def search_with_strength(
        self, query: str, *, user_id: str, top_k: int = 5, threshold: float = 0.1
    ) -> list[dict[str, Any]]:
        """遗忘曲线加权检索 final_score=relevance×strength。"""
        return self.search_module.search_with_strength(query, user_id=user_id, top_k=top_k)

    def apply_token_budget(self, texts: list[str], scores: list[float] | None = None, budget: int = 2000) -> list[str]:
        """token 预算裁剪。"""
        return self.search_module.apply_token_budget(texts, scores, budget=budget)

    def redact(self, text: str) -> str:
        """隐私脱敏。"""
        return self.search_module.redact(text)

    # ------------------------------------------------------------------
    # 演化
    # ------------------------------------------------------------------

    def link_on_add(self, memory_id: str, text: str, *, user_id: str) -> list[dict[str, Any]]:
        """Zettelkasten 自动建链接。"""
        return self.evolution.link_on_add(memory_id, text, user_id=user_id)

    def get_related(self, memory_id: str) -> list[dict[str, Any]]:
        """获取记忆的链接邻居 (Zettel 链接生长)。"""
        return self.evolution.get_related(memory_id)

    def dream(self, *, user_id: str) -> dict[str, Any]:
        """Dream 整合: 空闲期批量建链接。"""
        return self.evolution.dream(user_id=user_id)

    # ------------------------------------------------------------------
    # 自进化 (SelfEvolutionLoop)
    # ------------------------------------------------------------------

    def evaluate_recall(
        self,
        query: str,
        retrieved_memories: list[dict[str, Any]],
        *,
        user_id: str,
        agent_id: str | None = None,
        namespace: str = "default",
    ) -> dict[str, Any]:
        """检索后自动评估规则是否帮助了当前任务。"""
        return self.evolution.evaluate_recall(
            query, retrieved_memories, user_id=user_id, agent_id=agent_id, namespace=namespace,
        )

    def get_self_model(self, *, user_id: str, agent_id: str | None = None) -> dict[str, Any] | None:
        """获取 agent 的跨会话持久化能力画像。"""
        return self.evolution.get_self_model(user_id=user_id, agent_id=agent_id)

    def get_rule_evolution_chain(self, rule_id: str) -> list[dict[str, Any]]:
        """获取规则的完整演化链。"""
        return self.evolution.get_rule_evolution_chain(rule_id)

    def record_recall_outcome(self, *, user_id: str, agent_id: str | None = None, success: bool) -> None:
        """记录一次检索结果，更新 recall_success_rate。"""
        self.evolution.record_recall_outcome(user_id=user_id, agent_id=agent_id, success=success)

    def refine_rules(self, *, user_id: str, max_refinements: int = 3) -> list[dict[str, Any]]:
        """精炼 mixed/harmful 规则。"""
        return self.evolution.refine_rules(user_id=user_id, max_refinements=max_refinements)

    # ------------------------------------------------------------------
    # 共享
    # ------------------------------------------------------------------

    def is_cross_agent(self, user_id: str) -> bool:
        """检查该用户记忆是否跨 agent 共享。"""
        return self.entities.is_cross_agent(user_id)

    # ------------------------------------------------------------------
    # 因果推理
    # ------------------------------------------------------------------

    def add_causal_edge(
        self,
        cause_event_id: str,
        effect_event_id: str,
        *,
        user_id: str,
        relation: str = "causes",
        confidence: float = 0.5,
    ) -> dict[str, Any]:
        """添加因果边。"""
        edge = self.typed_store.add_causal_edge(
            cause_event_id, effect_event_id, user_id=user_id, relation=relation, confidence=confidence
        )
        return {"id": edge.id, "relation": edge.relation, "confidence": edge.confidence}

    def find_causes(self, event_id: str, *, user_id: str) -> list[dict[str, Any]]:
        """找事件的因果前因 (图遍历)。"""
        from septmuse.retrieval.causal import CausalRetriever

        retriever = CausalRetriever(self.typed_store, llm=self.llm)
        paths = retriever.find_causes(event_id, user_id=user_id)
        return [{"path": p.event_ids(), "confidence": p.confidence, "length": p.length} for p in paths]

    def find_effects(self, event_id: str, *, user_id: str) -> list[dict[str, Any]]:
        """找事件的因果后果 (图遍历)。"""
        from septmuse.retrieval.causal import CausalRetriever

        retriever = CausalRetriever(self.typed_store, llm=self.llm)
        paths = retriever.find_effects(event_id, user_id=user_id)
        return [{"path": p.event_ids(), "confidence": p.confidence, "length": p.length} for p in paths]

    def counterfactual(self, cause_event_id: str, effect_event_id: str, *, user_id: str) -> dict[str, Any]:
        """反事实因果查询 "若 X 未发生, Y 是否仍发生"。"""
        from septmuse.retrieval.causal import CausalRetriever

        retriever = CausalRetriever(self.typed_store, llm=self.llm)
        result = retriever.counterfactual(cause_event_id, effect_event_id, user_id=user_id)
        return {
            "would_still_occur": result.would_still_occur,
            "confidence": result.confidence,
            "reasoning": result.reasoning,
        }

    # ------------------------------------------------------------------
    # 排练/复述
    # ------------------------------------------------------------------

    def rehearse(self, memory_id: str, *, user_id: str) -> dict[str, Any]:
        """主动复述: 记忆强度回升。"""
        return self.evolution.rehearse(memory_id, user_id=user_id)

    def find_rehearse_candidates(self, *, user_id: str) -> list[dict[str, Any]]:
        """找需要复述的记忆 strength<0.3 且 base_value>0.7。"""
        return self.evolution.find_rehearse_candidates(user_id=user_id)

    def rehearse_batch(self, *, user_id: str) -> dict[str, Any]:
        """批量复述全部候选记忆 — 消除 REST/MCP 重复的遍历循环。"""
        return self.evolution.rehearse_batch(user_id=user_id)

    # ------------------------------------------------------------------
    # 元认知
    # ------------------------------------------------------------------

    def meta_route(self, query: str) -> dict[str, Any]:
        """L0 元认知路由: 决定查哪些命名空间。"""
        result = self.meta.route(query)
        return {"namespaces": result.namespaces, "fallback": result.fallback, "scores": result.scores}

    def coverage_report(self, *, user_id: str) -> dict[str, Any]:
        """L1 覆盖自描述: "我记住了什么/记不住什么"。"""
        report = self.meta.analyze_coverage(user_id=user_id)
        return {
            "overall_score": report.overall_score,
            "weak_areas": report.weak_areas,
            "strong_areas": report.strong_areas,
            "namespaces": [
                {"namespace": ns.namespace, "count": ns.count, "coverage_score": ns.coverage_score}
                for ns in report.namespaces
            ],
            "summary": report.summary(),
        }

    def adapt_strategy(self, *, user_id: str) -> dict[str, Any]:
        """L2 策略自调: 基于覆盖报告自调检索策略。"""
        report = self.meta.analyze_coverage(user_id=user_id)
        result = self.meta.adapt_strategy(report)
        return {
            "overall_action": result.overall_action.value,
            "needs_clarification": result.needs_clarification,
            "needs_deepened_retrieval": result.needs_deepened_retrieval,
            "recommendations": [
                {"action": r.action.value, "namespace": r.namespace, "reason": r.reason} for r in result.recommendations
            ],
        }

    def close(self) -> None:
        """关闭存储。"""
        self.store.close()
        self.typed_store.close()
        if self.entity_store is not None:
            self.entity_store.close()

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
"""Experimental API — 非稳定，可能随时变更或移除。

包含 working memory / typed memory 独立入口 / causal reasoning /
rehearsal / metacognition / dream / compress / conflict resolution /
temporal search / graph search / entity CRUD 等实验性方法。

核心 API 请用 ``from septmuse import Memory``。
"""

from __future__ import annotations

from typing import Any

from septmuse.memory.main import Memory
from septmuse.models.block import WorkingMemory


class ExperimentalMemory(Memory):
    """Memory 的实验性扩展——包含全部非核心方法。

    非稳定 API，仅用于实验和过渡。REST/MCP/CLI 中尚未清理的
    实验性端点可暂时使用本类，后续应逐步迁移或移除。
    """

    # ------------------------------------------------------------------
    # 工作记忆 Block
    # ------------------------------------------------------------------

    def get_working_memory(self, agent_id: str) -> WorkingMemory:
        """获取 agent 的工作记忆 (从 TypedMemoryStore 加载, 自动持久化)。"""
        blocks = self.typed_store.ensure_default_blocks(agent_id)
        return WorkingMemory(agent_id, blocks=blocks, store=self.typed_store)

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

    # ------------------------------------------------------------------
    # 类型化记忆独立入口 (核心 Memory 已统一到 add(memory_type=...))
    # ------------------------------------------------------------------

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
            subject,
            predicate,
            object,
            user_id=user_id,
            context=context,
            confidence=confidence,
            provenance=provenance,
            tags=tags,
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

    # ------------------------------------------------------------------
    # 捕获流水线
    # ------------------------------------------------------------------

    def capture(
        self, text: str, *, user_id: str, agent_id: str | None = None, session_id: str | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        """PostToolUse 捕获流水线 (SHA256去重→脱敏→嵌入→双索引)。"""
        from septmuse.capture.pipeline import CapturePipeline

        pipeline = CapturePipeline(
            self.store, self.embedder, typed_store=self.typed_store, llm=self.llm, dedup_window=self._dedup_window
        )
        result = pipeline.capture(
            text, user_id=user_id, agent_id=agent_id, session_id=session_id, metadata=kwargs.get("metadata")
        )
        return {
            "captured": result.captured,
            "memory_id": result.memory_id,
            "deduped": result.deduped,
            "redacted": result.redacted,
        }

    # ------------------------------------------------------------------
    # 检索变体
    # ------------------------------------------------------------------

    def search_hybrid(
        self, query: str, *, user_id: str, session_id: str | None = None, top_k: int = 5, threshold: float = 0.1
    ) -> list[dict[str, Any]]:
        """BM25+向量 RRF 融合检索 (核心 search(hybrid=True) 的快捷方式)。"""
        from septmuse.retrieval.hybrid import HybridRetriever

        retriever = HybridRetriever(
            self.store,
            self.embedder,
            entity_extractor=self.entity_extractor,
            entity_store=self.entity_store,
        )
        results = retriever.search(query, user_id=user_id, session_id=session_id, top_k=top_k, threshold=threshold)
        return [
            {
                "id": r.id,
                "memory": r.memory,
                "score": r.score,
                "vector_score": r.vector_score,
                "bm25_score": r.bm25_score,
            }
            for r in results
        ]

    def search_progressive(
        self, query: str, *, user_id: str, top_k: int = 5, threshold: float = 0.1
    ) -> list[dict[str, Any]]:
        """渐进三层检索 recall→locate→expand。"""
        from septmuse.retrieval.progressive import ProgressiveRetriever

        retriever = ProgressiveRetriever(self.store, self.typed_store, self.embedder)
        results = retriever.retrieve(query, user_id=user_id, top_k=top_k, threshold=threshold)
        return [{"id": r.id, "memory": r.memory, "score": r.score, "memory_type": r.memory_type} for r in results]

    def search_with_strength(
        self, query: str, *, user_id: str, top_k: int = 5, threshold: float = 0.1
    ) -> list[dict[str, Any]]:
        """遗忘曲线加权检索 final_score=relevance×strength。"""
        from septmuse.retrieval.forgetting import ForgettingRetriever

        results = self.search(query, user_id=user_id, top_k=top_k, threshold=threshold)
        retriever = ForgettingRetriever(self.typed_store)
        weighted = retriever.apply_strength(results, user_id=user_id)
        return [
            {
                "id": w.id,
                "memory": w.memory,
                "relevance": w.relevance,
                "strength": w.strength,
                "final_score": w.final_score,
            }
            for w in weighted
        ]

    def apply_token_budget(self, texts: list[str], scores: list[float] | None = None, budget: int = 2000) -> list[str]:
        """token 预算裁剪。"""
        from septmuse.retrieval.token_budget import TokenBudget

        return TokenBudget(budget=budget).fit_texts(texts, scores)

    def redact(self, text: str) -> str:
        """隐私脱敏。"""
        from septmuse.capture.sanitize import PrivacyFilter

        return PrivacyFilter().redact(text)

    # ------------------------------------------------------------------
    # 演化
    # ------------------------------------------------------------------

    def link_on_add(self, memory_id: str, text: str, *, user_id: str) -> list[dict[str, Any]]:
        """Zettelkasten 自动建链接。"""
        assert self.graph_store is not None, "graph_store required for zettel (SQLite default; AGE optional)"
        from septmuse.evolution.zettel import ZettelLinker

        linker = ZettelLinker(self.store, self.graph_store, self.embedder)
        emb = self.embedder.embed(text)
        links = linker.link_on_add(memory_id, text, emb, user_id=user_id)
        return [
            {"id": link.id, "source_id": link.source_id, "target_id": link.target_id, "score": link.score}
            for link in links
        ]

    def get_related(self, memory_id: str) -> list[dict[str, Any]]:
        """获取记忆的链接邻居 (Zettel 链接生长)。"""
        assert self.graph_store is not None, "graph_store required for zettel (SQLite default; AGE optional)"
        from septmuse.evolution.zettel import ZettelLinker

        linker = ZettelLinker(self.store, self.graph_store, self.embedder)
        return linker.get_related_memories(memory_id)

    def reflect(self, *, user_id: str, limit: int = 20) -> dict[str, Any]:
        """Session 反思: 提取教训→procedural rules。"""
        from septmuse.evolution.reflect import SessionReflector

        reflector = SessionReflector(self.typed_store, llm=self.llm)
        result = reflector.reflect(user_id=user_id, limit=limit)
        return {"proposed": result.lessons_proposed, "accepted": result.lessons_accepted, "rule_ids": result.rule_ids}

    def dream(self, *, user_id: str) -> dict[str, Any]:
        """Dream 整合: 空闲期批量建链接。"""
        assert self.graph_store is not None, "graph_store required for dream (SQLite default; AGE optional)"
        from septmuse.evolution.dream import DreamIntegrator

        dreamer = DreamIntegrator(self.store, self.graph_store, self.embedder)
        result = dreamer.dream(user_id=user_id)
        return {"processed": result.processed, "links_created": result.links_created}

    # ------------------------------------------------------------------
    # 共享
    # ------------------------------------------------------------------

    def is_cross_agent(self, user_id: str) -> bool:
        """检查该用户记忆是否跨 agent 共享。"""
        from septmuse.governance.sharing import SharedMemoryAccessor

        return SharedMemoryAccessor(self.store).is_cross_agent(user_id)

    # ------------------------------------------------------------------
    # 双时态查询
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
        valid_memories = self.store.get_temporal_valid(reference_time, user_id=user_id, session_id=session_id)
        if not valid_memories:
            return []

        valid_ids = {m["id"] for m in valid_memories}
        emb = self.embedder.embed(query)
        search_results = self.store.search(emb, user_id=user_id, top_k=top_k * 2, threshold=threshold)
        filtered = [r for r in search_results if r["id"] in valid_ids]

        valid_map = {m["id"]: m for m in valid_memories}
        for r in filtered:
            vm = valid_map.get(r["id"])
            if vm:
                r["valid_at"] = vm.get("valid_at")
                r["invalid_at"] = vm.get("invalid_at")

        return filtered[:top_k]

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
        valid_memories = self.store.get_temporal_interval(start, end, user_id=user_id, session_id=session_id)
        if not valid_memories:
            return []

        valid_ids = {m["id"] for m in valid_memories}
        emb = self.embedder.embed(query)
        search_results = self.store.search(emb, user_id=user_id, top_k=top_k * 2, threshold=threshold)
        filtered = [r for r in search_results if r["id"] in valid_ids]

        valid_map = {m["id"]: m for m in valid_memories}
        for r in filtered:
            vm = valid_map.get(r["id"])
            if vm:
                r["valid_at"] = vm.get("valid_at")
                r["invalid_at"] = vm.get("invalid_at")

        return filtered[:top_k]

    def search_natural(
        self, query: str, *, user_id: str, top_k: int = 5, threshold: float = 0.1
    ) -> list[dict[str, Any]]:
        """自然语言时态查询 (LLM 抽时间区间 → 有则时态过滤, 无则回退普通检索)。"""
        from septmuse.retrieval.temporal import TemporalRetriever

        retriever = TemporalRetriever(self.store, self.embedder, self.llm)
        return retriever.search_natural(query, user_id=user_id, top_k=top_k, threshold=threshold)

    # ------------------------------------------------------------------
    # 消息压缩
    # ------------------------------------------------------------------

    def compress(self, *, user_id: str, mode: str = "static", buffer_size: int = 20) -> dict[str, Any]:
        """压缩消息。"""
        from septmuse.evolution.summarizer import Summarizer

        summarizer = Summarizer(self.store, self.typed_store, self.llm)
        return summarizer.compress(user_id=user_id, mode=mode, buffer_size=buffer_size)

    # ------------------------------------------------------------------
    # 冲突解决 + 实体去重
    # ------------------------------------------------------------------

    def resolve_conflicts(self, *, user_id: str) -> dict[str, Any]:
        """解决矛盾事实: 新事实覆盖旧事实。"""
        from septmuse.evolution.conflict import ConflictResolver

        resolver = ConflictResolver(self.typed_store, self.store, self.llm)
        return resolver.resolve_conflicts(user_id=user_id)

    def deduplicate_entities(self, *, user_id: str) -> dict[str, Any]:
        """实体去重三段式。"""
        from septmuse.evolution.conflict import ConflictResolver

        resolver = ConflictResolver(self.typed_store, self.store, self.llm)
        return resolver.deduplicate_entities(user_id=user_id)

    # ------------------------------------------------------------------
    # cognify 扩展 + 图检索
    # ------------------------------------------------------------------

    def get_entity_relations(self, entity_name: str, *, user_id: str) -> list[dict[str, Any]]:
        """实体间关系遍历 (双向)。"""
        from septmuse.extraction.cognify import CognifyPipeline

        pipeline = CognifyPipeline(
            self.store,
            self.graph_store,
            self.embedder,
            entity_store=self.entity_store,
            llm=self.llm,
            entity_extractor=self.entity_extractor,
        )
        return pipeline.get_entity_neighbors(entity_name, user_id=user_id)

    def search_graph(
        self, seed_memory_id: str, *, max_depth: int = 2, relation: str | None = None
    ) -> list[dict[str, Any]]:
        """BFS 图遍历检索。"""
        assert self.graph_store is not None, "graph_store required for graph search (SQLite default; AGE optional)"
        from septmuse.retrieval.graph_search import GraphSearcher

        searcher = GraphSearcher(self.graph_store, self.store)
        return searcher.search_graph(seed_memory_id, max_depth=max_depth, relation=relation)

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
        assert self.graph_store is not None, "graph_store required for graph search (SQLite default; AGE optional)"
        from septmuse.retrieval.graph_search import GraphSearcher

        searcher = GraphSearcher(self.graph_store, self.store)
        tk = top_k or self.config.top_k
        emb = self.embedder.embed(query)
        vector_results = self.store.search(emb, user_id=user_id, top_k=tk, threshold=self.config.threshold)
        return searcher.fused_search(
            query,
            user_id=user_id,
            seed_memory_id=seed_memory_id,
            vector_results=vector_results,
            max_depth=max_depth,
            relation=relation,
        )

    # ------------------------------------------------------------------
    # 实体 CRUD
    # ------------------------------------------------------------------

    def extract_entities(self, text: str) -> list[dict[str, Any]]:
        """抽取实体 (不存储, 仅返回抽取结果)。"""
        if self.entity_extractor is None:
            return []
        entities = self.entity_extractor.extract(text)
        return [{"text": e.text, "type": e.entity_type, "start": e.start, "end": e.end} for e in entities]

    def add_entity(self, entity_text: str, entity_type: str, memory_id: str, *, user_id: str) -> dict[str, Any]:
        """手动添加实体。"""
        if self.entity_store is None:
            return {"event": "SKIP", "entity": entity_text, "reason": "no entity_store"}
        from septmuse.extraction.entity import Entity

        entity = Entity(text=entity_text, entity_type=entity_type, start=0, end=len(entity_text))
        eid = self.entity_store.upsert(entity, memory_id, user_id=user_id)
        return {"event": "ADD", "entity": entity_text, "entity_id": eid}

    def search_entities(self, query: str, *, user_id: str, top_k: int = 5) -> list[dict[str, Any]]:
        """搜索实体 (精确匹配 + 向量相似度)。"""
        if self.entity_store is None:
            return []
        return self.entity_store.search(query, user_id=user_id, top_k=top_k)

    def get_entity_neighbors(self, entity_id: str) -> list[str]:
        """获取实体的 linked_memory_ids。"""
        if self.entity_store is None:
            return []
        return self.entity_store.get_linked_memories(entity_id)

    def list_entities(self, *, user_id: str, entity_type: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        """列出用户全部未删除实体。"""
        if self.entity_store is None:
            return []
        return self.entity_store.list(user_id=user_id, entity_type=entity_type, limit=limit)

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
        from septmuse.retrieval.forgetting import ForgettingRetriever

        retriever = ForgettingRetriever(self.typed_store)
        result = retriever.rehearse(memory_id, user_id=user_id)
        if result is None:
            return {"error": "memory not found"}
        return {"memory_id": result.memory_id, "strength": result.strength, "access_count": result.access_count}

    def find_rehearse_candidates(self, *, user_id: str) -> list[dict[str, Any]]:
        """找需要复述的记忆 strength<0.3 且 base_value>0.7。"""
        from septmuse.retrieval.forgetting import ForgettingRetriever

        retriever = ForgettingRetriever(self.typed_store)
        candidates = retriever.find_rehearse_candidates(user_id=user_id)
        return [{"memory_id": c.memory_id, "strength": c.strength, "base_value": c.base_value} for c in candidates]

    # ------------------------------------------------------------------
    # 元认知
    # ------------------------------------------------------------------

    def meta_route(self, query: str) -> dict[str, Any]:
        """L0 元认知路由: 决定查哪些命名空间。"""
        from septmuse.meta.router import MetaRouter

        router = MetaRouter(self.embedder)
        result = router.route(query)
        return {"namespaces": result.namespaces, "fallback": result.fallback, "scores": result.scores}

    def coverage_report(self, *, user_id: str) -> dict[str, Any]:
        """L1 覆盖自描述: "我记住了什么/记不住什么"。"""
        from septmuse.meta.coverage import CoverageAnalyzer

        analyzer = CoverageAnalyzer(self.store, self.typed_store)
        report = analyzer.analyze(user_id=user_id)
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
        from septmuse.meta.coverage import CoverageAnalyzer
        from septmuse.meta.strategy import StrategyAdapter

        report = CoverageAnalyzer(self.store, self.typed_store).analyze(user_id=user_id)
        result = StrategyAdapter().adapt(report)
        return {
            "overall_action": result.overall_action.value,
            "needs_clarification": result.needs_clarification,
            "needs_deepened_retrieval": result.needs_deepened_retrieval,
            "recommendations": [
                {"action": r.action.value, "namespace": r.namespace, "reason": r.reason} for r in result.recommendations
            ],
        }

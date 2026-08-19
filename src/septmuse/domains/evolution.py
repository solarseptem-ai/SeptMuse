#  Copyright 2026 The sonhhxg0529 Authors. All Rights Reserved.
"""EvolutionModule — 演化领域模块（improve / reflect / dream / forget / cognify / 冲突解决 / 自进化）。"""

from __future__ import annotations

from typing import Any

from septmuse.embedders.base import Embedder
from septmuse.llms.base import LLM
from septmuse.storage.base import MemoryStore
from septmuse.storage.graph_stores.base import GraphStore
from septmuse.storage.relational_stores.typed_store import TypedMemoryStore


class EvolutionModule:
    """演化领域模块 — 自进化闭环 + 遗忘 + 冲突解决 + 图谱生长。"""

    def __init__(
        self,
        store: MemoryStore,
        embedder: Embedder,
        typed_store: TypedMemoryStore,
        llm: LLM | None = None,
        graph_store: GraphStore | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.typed_store = typed_store
        self.llm = llm
        self.graph_store = graph_store

    @property
    def self_loop(self) -> Any:
        """自进化闭环（懒加载）。"""
        if not hasattr(self, "_self_loop"):
            from septmuse.evolution.self_loop import SelfEvolutionLoop

            self._self_loop = SelfEvolutionLoop(
                self.store, self.graph_store, self.embedder, self.typed_store, llm=self.llm,
            )
        return self._self_loop

    # ── 演进 ──

    def improve(self, *, user_id: str, limit: int = 20) -> dict[str, Any]:
        """自进化闭环: dream + reflect + refine + self_model + coverage 持久化。"""
        import logging

        _log = logging.getLogger(__name__)
        result = self.self_loop.improve(user_id=user_id, limit=limit)
        # 同时跑 dream + conflicts
        dream_result = self.dream(user_id=user_id)
        conflicts = self.resolve_conflicts(user_id=user_id)

        # 持久化 coverage 为 SemanticFact（对齐旧 ImprovePipeline 行为）
        coverage_info: dict[str, Any] = {}
        if result.coverage:
            cov = result.coverage
            coverage_info = {
                "overall_score": cov.get("overall_score", 0),
                "weak_areas": cov.get("weak_areas", []),
                "strong_areas": cov.get("strong_areas", []),
            }
            try:
                self.typed_store.add_fact(
                    subject=user_id,
                    predicate="has_coverage_score",
                    object=f"{cov.get('overall_score', 0):.2f}",
                    user_id=user_id,
                    context=f"Overall: {cov.get('overall_score', 0):.2f}, "
                    f"Weak: {len(cov.get('weak_areas', []))}, "
                    f"Strong: {len(cov.get('strong_areas', []))}",
                    confidence=cov.get("overall_score", 0),
                    provenance="meta",
                    tags=["meta", "coverage"],
                )
            except Exception as e:
                _log.warning("improve_coverage_persist_failed", error=str(e))

        return {
            "dream": dream_result,
            "rules": result.lessons_accepted,
            "coverage": coverage_info,
            "conflicts": conflicts,
            "lessons_proposed": result.lessons_proposed,
            "lessons_accepted": result.lessons_accepted,
            "rule_ids": result.rule_ids,
            "refined": len([r for r in result.refined_rules if r.refined]),
            "self_model_updated": result.self_model_updated,
            "summary": result.summary,
        }

    def reflect(self, *, user_id: str, limit: int = 20) -> dict[str, Any]:
        """反思升华: 从情节提取教训。"""
        from septmuse.evolution.reflect import SessionReflector

        reflector = SessionReflector(self.typed_store, llm=self.llm)
        r = reflector.reflect(user_id=user_id, limit=limit)
        return {"lessons_proposed": r.lessons_proposed, "lessons_accepted": r.lessons_accepted, "rule_ids": r.rule_ids}

    def dream(self, *, user_id: str) -> dict[str, Any]:
        """Dream 整合: 空闲期批量建链接。"""
        if self.graph_store is None:
            return {"processed": 0, "links_created": 0}
        from septmuse.evolution.dream import DreamIntegrator

        dreamer = DreamIntegrator(self.store, self.graph_store, self.embedder)
        result = dreamer.dream(user_id=user_id)
        return {"processed": result.processed, "links_created": result.links_created}

    def resolve_conflicts(self, *, user_id: str) -> dict[str, Any]:
        """矛盾检测 + 解决。"""
        from septmuse.evolution.conflict import ConflictResolver

        resolver = ConflictResolver(self.typed_store, store=self.store, llm=self.llm)
        return resolver.resolve_conflicts(user_id=user_id)

    def deduplicate_entities(self, *, user_id: str) -> dict[str, Any]:
        """实体去重。"""
        from septmuse.evolution.conflict import ConflictResolver

        resolver = ConflictResolver(self.typed_store, store=self.store, llm=self.llm)
        return resolver.deduplicate_entities(user_id=user_id)

    def link_on_add(self, memory_id: str, text: str, *, user_id: str) -> list[dict[str, Any]]:
        """Zettelkasten 自动建链接。"""
        if self.graph_store is None:
            return []
        from septmuse.evolution.zettel import ZettelLinker

        linker = ZettelLinker(self.store, self.graph_store, self.embedder)
        emb = self.embedder.embed(text)
        links = linker.link_on_add(memory_id, text, emb, user_id=user_id)
        return [
            {"id": link.id, "source_id": link.source_id, "target_id": link.target_id, "score": link.score}
            for link in links
        ]

    def get_related(self, memory_id: str) -> list[dict[str, Any]]:
        """获取记忆的链接邻居。"""
        if self.graph_store is None:
            return []
        from septmuse.evolution.zettel import ZettelLinker

        linker = ZettelLinker(self.store, self.graph_store, self.embedder)
        return linker.get_related_memories(memory_id)

    # ── 规则系统 ──

    def get_active_rules(self, *, user_id: str, namespace: str = "default") -> list[dict[str, Any]]:
        """获取应注入的规则。"""
        rules = self.typed_store.get_active_rules(user_id=user_id, namespace=namespace)
        return [
            {"id": r.id, "rule": r.rule, "confidence": r.confidence,
             "helpful_count": r.helpful_count, "harmful_count": r.harmful_count}
            for r in rules
        ]

    def rules_to_prompt(self, *, user_id: str, namespace: str = "default") -> str:
        """编译规则为 prompt 注入文本。"""
        rules = self.typed_store.get_active_rules(user_id=user_id, namespace=namespace)
        if not rules:
            return ""
        parts = ["<procedural_rules>"]
        for r in rules:
            parts.append(f"- {r.rule} (confidence={r.confidence:.2f})")
        parts.append("</procedural_rules>")
        return "\n".join(parts)

    def record_rule_outcome(self, rule_id: str, helpful: bool) -> dict[str, Any]:
        """记录规则应用结果。"""
        r = self.typed_store.record_rule_outcome(rule_id, helpful)
        if r is None:
            return {"error": f"rule {rule_id} not found"}
        return {"id": r.id, "helpful_count": r.helpful_count, "harmful_count": r.harmful_count,
                "confidence": r.confidence, "deprecated": r.deprecated}

    # ── 自进化 API ──

    def evaluate_recall(
        self, query: str, retrieved_memories: list[dict[str, Any]],
        *, user_id: str, agent_id: str | None = None, namespace: str = "default",
    ) -> dict[str, Any]:
        """检索后自动评估规则相关性。"""
        result = self.self_loop.evaluate_recall(
            query, retrieved_memories, user_id=user_id, agent_id=agent_id, namespace=namespace,
        )
        return {
            "query": result.query, "total_rules_evaluated": result.total_rules_evaluated,
            "helpful_count": result.helpful_count, "harmful_count": result.harmful_count,
            "summary": result.summary,
        }

    def get_self_model(self, *, user_id: str, agent_id: str | None = None) -> dict[str, Any] | None:
        """获取 agent 能力画像。"""
        return self.self_loop.get_self_model(user_id=user_id, agent_id=agent_id)

    def get_rule_evolution_chain(self, rule_id: str) -> list[dict[str, Any]]:
        """获取规则演化链。"""
        return self.self_loop.get_rule_evolution_chain(rule_id)

    def record_recall_outcome(self, *, user_id: str, agent_id: str | None = None, success: bool) -> None:
        """记录检索结果。"""
        self.self_loop.record_recall_outcome(user_id=user_id, agent_id=agent_id, success=success)

    def refine_rules(self, *, user_id: str, max_refinements: int = 3) -> list[dict[str, Any]]:
        """精炼规则。"""
        results = self.self_loop.refiner.refine_all(user_id=user_id, max_refinements=max_refinements)
        return [
            {"original_rule_id": r.original_rule_id, "new_rule_id": r.new_rule_id,
             "refined": r.refined, "reason": r.reason, "change_description": r.change_description}
            for r in results
        ]

    # ── 复述 ──

    def rehearse(self, memory_id: str, *, user_id: str) -> dict[str, Any]:
        """主动复述高价值记忆。"""
        from septmuse.retrieval.forgetting import ForgettingRetriever

        fr = ForgettingRetriever(self.typed_store)
        result = fr.rehearse(memory_id, user_id=user_id)
        if result is None:
            return {"error": "memory not found"}
        return {"memory_id": result.memory_id, "strength": result.strength, "access_count": result.access_count}

    def find_rehearse_candidates(self, *, user_id: str, limit: int = 10) -> list[dict[str, Any]]:
        """查找复述候选。"""
        from septmuse.retrieval.forgetting import ForgettingRetriever

        fr = ForgettingRetriever(self.typed_store)
        candidates = fr.find_rehearse_candidates(user_id=user_id)
        return [{"memory_id": c.memory_id, "strength": c.strength, "base_value": c.base_value} for c in candidates[:limit]]

    def rehearse_batch(self, *, user_id: str, limit: int = 10) -> dict[str, Any]:
        """批量复述。"""
        candidates = self.find_rehearse_candidates(user_id=user_id, limit=limit)
        for c in candidates:
            self.rehearse(c["memory_id"], user_id=user_id)
        return {"rehearsed": len(candidates)}

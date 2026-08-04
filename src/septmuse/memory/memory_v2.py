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
"""V2 记忆编排入口 — remember / recall / improve / forget。

全新类, 不继承 Memory。组合 Memory 实例 + 10 子组件。
零 LLM 可用: 无 SEPTMUSE_LLM 时降级为 verbatim + 向量检索 + 规则演化。

用法 1 (从 config 创建):
    v2 = V2Memory(config=MemoryConfig())
    v2.remember("我喜欢 Python", user_id="alice")

用法 2 (传入已有 Memory):
    mem = Memory(config=config)
    v2 = V2Memory(memory=mem)
    v2.recall("帮我写 API", user_id="alice")

详见 docs/specs/2026-08-04-v2-memory-architecture.md §3 + §7。
"""

from __future__ import annotations

import contextlib
from datetime import datetime, timezone
from typing import Any

from septmuse.core.logging import get_logger
from septmuse.memory.capture import CapturePipeline
from septmuse.memory.causal import CausalGraph
from septmuse.memory.episodic import EpisodicMemory
from septmuse.memory.evolution import EvolutionEngine
from septmuse.memory.forgetting import ForgettingManager
from septmuse.memory.meta import MetacognitionLayer
from septmuse.memory.procedural import ProceduralMemory
from septmuse.memory.retrieval import HybridRetriever, TokenBudget
from septmuse.memory.semantic import SemanticMemory
from septmuse.memory.working_memory import WorkingMemory
from septmuse.retrieval.token_budget import BudgetItem

logger = get_logger(__name__)


def _normalize_messages(messages: Any) -> str:
    """把 messages 归一化为纯文本 (str 或 list[dict] → str)。"""
    if isinstance(messages, str):
        return messages.strip()
    if isinstance(messages, list):
        parts: list[str] = []
        for msg in messages:
            if isinstance(msg, dict):
                parts.append(str(msg.get("content", "")))
            elif isinstance(msg, str):
                parts.append(msg)
        return "\n".join(parts).strip()
    return str(messages).strip()


class V2Memory:
    """V2 编排入口 — remember / recall / improve / forget。

    全新类, 不继承 Memory。组合 Memory 实例 + 10 子组件。
    零 LLM 可用: 无 SEPTMUSE_LLM 时降级为 verbatim + 向量检索 + 规则演化。

    4 个编排方法:
    - remember(): 捕获 + 分流 + 多形态写 (情节 raw_log 恒做, 语义事实仅 LLM 时)
    - recall(): 元认知路由 + 检索 + token 预算
    - improve(): Dream + reflect + 冲突 + L1 报告
    - forget(): 先 invalidate 再 delete + 实体清理
    """

    def __init__(
        self,
        memory: Any | None = None,
        *,
        config: Any | None = None,
    ) -> None:
        from septmuse.memory.main import Memory

        self.mem = memory or Memory(config=config)

        # === 平面 B 后端 (从 Memory 复用) ===
        self.store = self.mem.store
        self.embedder = self.mem.embedder
        self.typed_store = self.mem.typed_store
        self.graph_store = self.mem.graph_store
        self.entity_store = self.mem.entity_store
        self.entity_extractor = self.mem.entity_extractor
        self.llm = self.mem.llm
        self._dedup_window = self.mem._dedup_window

        # === 平面 A 子组件 (4 个, 全新创建, 不复用 self.mem.semantic) ===
        # WorkingMemory 走独立 WorkingMemoryStore (非 typed_store, 决策 3)
        self.working_memory = WorkingMemory(
            store=self._create_working_memory_store(),
            agent_id="default",
        )
        # Semantic/Episodic/Procedural 走 typed_store (共享关系型后端)
        self.semantic = SemanticMemory(self.typed_store, self.embedder)
        self.episodic = EpisodicMemory(self.typed_store)
        self.procedural = ProceduralMemory(self.typed_store)

        # === 平面 C 子组件 (6 个) ===
        self.capture = CapturePipeline(
            self.store,
            self.embedder,
            typed_store=self.typed_store,
            llm=self.llm,
            dedup_window=self._dedup_window,
        )
        self.retrieval = HybridRetriever(
            self.store,
            self.embedder,
            entity_extractor=self.entity_extractor,
            entity_store=self.entity_store,
        )
        self.token_budget = TokenBudget(budget=2000)
        self.meta = MetacognitionLayer(
            self.embedder,
            self.store,
            self.typed_store,
        )
        self.evolution = EvolutionEngine(
            self.store,
            self.graph_store,
            self.embedder,
            self.typed_store,
            self.llm,
        )
        self.causal = CausalGraph(self.typed_store, llm=self.llm)
        self.forgetting = ForgettingManager(self.typed_store)

        logger.info(
            "v2_memory_init",
            has_llm=self.llm is not None,
            has_graph_store=self.graph_store is not None,
            has_entity_store=self.entity_store is not None,
        )

    def _create_working_memory_store(self) -> Any:
        """创建工作记忆独立后端 (WorkingMemoryStore ABC)。"""
        from septmuse.storage.working_memory_stores.factory import create_working_memory_store

        store_engine = getattr(self.mem.store, "engine", None)
        return create_working_memory_store(self.mem.config, engine=store_engine)

    # ------------------------------------------------------------------
    # 4 个编排方法
    # ------------------------------------------------------------------

    def remember(
        self,
        messages: Any,
        *,
        user_id: str,
        agent_id: str | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """捕获 + 分流 + 多形态写。

        流程:
        1. 捕获 (去重 + 脱敏)
        2. 情节 raw_log (恒做, 不需要 LLM)
        3. 语义事实 (仅 LLM 可用时, 零 LLM 降级跳过)
        4. 工作 block (可选, agent_id 存在时)

        关键约束: 不直接产程序规则 (程序规则留给 improve 从情节蒸馏)
        """
        text = _normalize_messages(messages)
        if not text:
            return {"captured": False, "reason": "empty text"}

        # 1. 捕获 (去重 + 脱敏)
        capture_result = self.capture.capture(
            text,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
        )
        if not capture_result.captured:
            return {"captured": False, "reason": capture_result.errors or "duplicate"}

        # 2. 情节 raw_log (恒做, 不需要 LLM)
        raw = self.episodic.add_raw_log(
            capture_result.stored_text,
            user_id=user_id,
            session_id=session_id or "unknown",
            agent_id=agent_id,
        )

        # 3. 语义事实 (仅 LLM 可用时, 零 LLM 降级跳过)
        fact_ids: list[str] = []
        if self.llm is not None and self.mem.extractor is not None:
            try:
                extracted = self.mem.extractor.extract_and_store(capture_result.stored_text, user_id=user_id)
                results = extracted.get("results", [])
                fact_ids = [r.get("id", "") for r in results if r.get("id")]
            except Exception as e:
                logger.warning("v2_remember_extract_failed", error=str(e))

        # 4. 工作 block (可选, agent_id 存在时)
        if agent_id:
            with contextlib.suppress(KeyError, ValueError):
                self.working_memory.core_memory_append("persona", text[:200])

        logger.info(
            "v2_remember_done",
            user_id=user_id,
            raw_id=raw.id,
            fact_count=len(fact_ids),
            captured=True,
        )
        return {
            "raw_id": raw.id,
            "fact_ids": fact_ids,
            "memory_id": capture_result.memory_id,
            "captured": True,
        }

    def recall(
        self,
        query: str,
        *,
        user_id: str,
        top_k: int = 5,
        recipe: str | None = None,
    ) -> dict[str, Any]:
        """元认知路由 + 检索 + token 预算。

        流程:
        1. L0 路由
        2. 三信号检索 (over-fetch)
        3. 图扩展 (recipe 可选, 暂简化)
        4. 遗忘曲线加权
        5. token 预算裁剪
        6. L2 策略自调 (仅 L1 报告存在时, 决策 6)
        7. block + 规则注入
        """
        # 1. L0 路由
        route = self.meta.route(query)

        # 2. 三信号检索 (over-fetch)
        results = self.retrieval.search(query, user_id=user_id, top_k=top_k * 4)

        # 3. 图扩展 (recipe 可选, 暂简化)
        # TODO: recipe == "GRAPH_BFS" 时用 causal.bfs + rrf_fuse

        # 4. 遗忘曲线加权
        result_dicts = [
            {
                "id": r.id,
                "memory": r.memory,
                "score": r.score,
                "metadata": r.metadata,
            }
            for r in results
        ]
        weighted = self.forgetting.apply_strength(result_dicts, user_id=user_id)

        # 5. token 预算裁剪
        items = [BudgetItem(text=w.memory, score=w.final_score) for w in weighted]
        budgeted = self.token_budget.fit(items)

        # 6. L2 策略自调 (仅 L1 报告存在时, 决策 6)
        l1_report = self._load_coverage_report(user_id)
        strategy = None
        if l1_report is not None:
            strategy = self.meta.adapt_strategy(l1_report)

        # 7. block + 规则注入
        prompt_parts: list[str] = []
        wm_prompt = self.working_memory.compile_to_prompt()
        if wm_prompt:
            prompt_parts.append(wm_prompt)
        rule_prompt = self.procedural.rules_to_prompt(user_id=user_id)
        if rule_prompt:
            prompt_parts.append(rule_prompt)
        injected_prompt = "\n".join(prompt_parts)

        memories = [
            {"id": item.text[:50], "memory": item.text, "score": item.score}
            for item in budgeted.items
        ]

        logger.info(
            "v2_recall_done",
            user_id=user_id,
            candidates=len(results),
            returned=len(memories),
            route=route.namespaces,
            has_strategy=strategy is not None,
        )
        return {
            "memories": memories,
            "injected_prompt": injected_prompt,
            "route": {"namespaces": route.namespaces, "fallback": route.fallback},
            "strategy": strategy,
            "used_tokens": budgeted.used_tokens,
        }

    def improve(self, *, user_id: str, limit: int = 50) -> dict[str, Any]:
        """Dream + reflect + 冲突 + L1 报告。

        流程:
        1. Dream 链接生长 (embedding 相似度, 不需要 LLM)
        2. 情节 → 程序规则蒸馏 (仅 LLM 可用时, 无 LLM 跳过)
        3. 冲突解决 (精确归一化 + difflib, 不需要 LLM)
        4. L1 报告生成 + 持久化 (纯统计, 不需要 LLM)
        """
        # 1. Dream 链接生长
        dream_result = self.evolution.dream(user_id=user_id)

        # 2. 情节 → 程序规则蒸馏 (仅 LLM 可用时)
        rules_accepted = 0
        if self.llm is not None:
            try:
                reflect_result = self.evolution.reflect(user_id=user_id, limit=limit)
                rules_accepted = reflect_result.lessons_accepted
            except Exception as e:
                logger.warning("v2_improve_reflect_failed", error=str(e))

        # 3. 冲突解决
        try:
            conflicts = self.evolution.resolve_conflicts(user_id=user_id)
        except Exception as e:
            logger.warning("v2_improve_conflict_failed", error=str(e))
            conflicts = {"conflicts_found": 0, "resolved": 0, "invalidated_ids": []}

        # 4. L1 报告生成 + 持久化
        report = self.meta.analyze_coverage(user_id=user_id)
        self._persist_coverage(user_id=user_id, overall_score=report.overall_score)

        logger.info(
            "v2_improve_done",
            user_id=user_id,
            dream_links=dream_result.links_created,
            rules_accepted=rules_accepted,
            conflicts_resolved=conflicts.get("resolved", 0),
            coverage_score=report.overall_score,
        )
        return {
            "dream": {
                "links_created": dream_result.links_created,
                "processed": dream_result.processed,
            },
            "rules": rules_accepted,
            "conflicts": conflicts,
            "coverage": {
                "overall_score": report.overall_score,
                "weak_areas": report.weak_areas,
                "strong_areas": report.strong_areas,
            },
        }

    def forget(self, memory_id: str, *, user_id: str) -> dict[str, Any]:
        """先 invalidate 再 delete + 实体清理 (决策 5)。

        流程:
        1. 双时态失效 (标记不再为真, 保留历史)
        2. 软删除 (不参与检索)
        3. 实体清理 (如有 entity_store)
        4. 图边清理 (如有 graph_store)
        """
        now = datetime.now(timezone.utc).isoformat()

        # 1. 双时态失效
        try:
            self.store.invalidate(memory_id, invalid_at=now)
        except Exception as e:
            logger.warning("v2_forget_invalidate_failed", memory_id=memory_id, error=str(e))

        # 2. 软删除
        try:
            self.store.delete(memory_id)
        except Exception as e:
            logger.warning("v2_forget_delete_failed", memory_id=memory_id, error=str(e))

        # 3. 实体清理
        if self.entity_store is not None:
            try:
                # EntityStore 无统一 remove_memory_reference, 用 duck typing 尝试
                if hasattr(self.entity_store, "remove_memory_reference"):
                    self.entity_store.remove_memory_reference(memory_id)
            except Exception as e:
                logger.warning("v2_forget_entity_cleanup_failed", error=str(e))

        # 4. 图边清理
        if self.graph_store is not None:
            try:
                self._delete_graph_edges(memory_id)
            except Exception as e:
                logger.warning("v2_forget_graph_cleanup_failed", error=str(e))

        logger.info("v2_forget_done", memory_id=memory_id, user_id=user_id)
        return {
            "memory_id": memory_id,
            "event": "FORGET",
            "invalidated_at": now,
            "deleted_at": now,
        }

    # ------------------------------------------------------------------
    # 内部辅助方法
    # ------------------------------------------------------------------

    def _load_coverage_report(self, user_id: str) -> Any:
        """加载 L1 覆盖报告 (improve 阶段生成 + 持久化)。

        首次使用时 improve 还没跑过, 报告不存在 → 返回 None (决策 6)。
        improve 跑过后报告存在 → 重新调 analyze_coverage 获取最新报告。
        """
        facts = self.semantic.get_all_facts(user_id=user_id)
        for fact in facts:
            tags = getattr(fact, "tags", []) or []
            if "meta" in tags and "coverage" in tags:
                # 标记存在, 重新生成最新报告
                return self.meta.analyze_coverage(user_id=user_id)
        return None

    def _persist_coverage(self, *, user_id: str, overall_score: float) -> None:
        """持久化 L1 覆盖报告标记 (存为 SemanticFact, tags=["meta","coverage"])。"""
        self.semantic.add_fact(
            subject="coverage",
            predicate="overall_score",
            object=f"{overall_score:.4f}",
            user_id=user_id,
            tags=["meta", "coverage"],
            confidence=1.0,
            provenance="meta",
            embed=False,
        )

    def _delete_graph_edges(self, memory_id: str) -> None:
        """清理图边 (如有 graph_store)。"""
        # 用 duck typing 尝试 graph_store 的清理方法
        if hasattr(self.graph_store, "delete_edges_for_memory"):
            self.graph_store.delete_edges_for_memory(memory_id)
        elif hasattr(self.graph_store, "delete_memory"):
            self.graph_store.delete_memory(memory_id)

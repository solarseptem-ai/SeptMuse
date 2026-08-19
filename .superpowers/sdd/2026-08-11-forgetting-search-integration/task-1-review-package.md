## Commits

54a991f feat(search): add forgetting/token_budget/inject_prompt/hyde/query_rewrite params + no-op for hyde/query_rewrite

## Diff Stats

 src/septmuse/memory/main.py        | 494 +++++++++++++++++++++++++++++++++++--
 tests/unit/test_search_enhanced.py |  16 ++
 2 files changed, 486 insertions(+), 24 deletions(-)

## Full Diff

diff --git a/src/septmuse/memory/main.py b/src/septmuse/memory/main.py
index c12b658..9673c8b 100644
--- a/src/septmuse/memory/main.py
+++ b/src/septmuse/memory/main.py
@@ -29,30 +29,31 @@ pip install septmuse 后:
     cognify    — 知识图谱构建
     get_active_rules / rules_to_prompt / record_rule_outcome — 规则系统
 
 实验性功能 (working memory / causal / rehearsal / metacognition /
 dream / compress / conflict / temporal search / graph search / entity CRUD)
 请在 ``from septmuse.experimental import ExperimentalMemory`` 中使用。
 """
 
 from __future__ import annotations
 
+import contextlib
 from typing import Any
 
 from septmuse.configs.defaults import MemoryConfig, default_config
 from septmuse.core.logging import get_logger
 from septmuse.embedders.base import Embedder
 from septmuse.llms.base import LLM
-from septmuse.models.episodic import EpisodicMemory
+from septmuse.memory.episodic import EpisodicMemory
+from septmuse.memory.procedural import ProceduralMemory
+from septmuse.memory.semantic import SemanticMemory
 from septmuse.models.extract import FactExtractor
-from septmuse.models.fact import SemanticMemory
-from septmuse.models.procedural import ProceduralMemory
 from septmuse.storage.base import MemoryStore
 from septmuse.storage.graph_stores.base import GraphStore
 from septmuse.storage.graph_stores.sqlite import SQLiteGraphStore
 from septmuse.storage.relational_stores.typed_store import TypedMemoryStore
 
 logger = get_logger(__name__)
 
 
 def _resolve_embedder(config: MemoryConfig) -> Embedder:
     """解析 embedder (通过 ServiceProvider 延迟 import + 实例化)。
@@ -80,20 +81,51 @@ def _normalize_messages(messages: Any) -> list[str]:
     for m in messages:
         if isinstance(m, dict):
             content = m.get("content")
             if content:
                 texts.append(content)
         elif isinstance(m, str):
             texts.append(m)
     return texts
 
 
+def _normalize_expiration_date(expiration_date: Any) -> str | None:
+    """归一化 expiration_date 为 YYYY-MM-DD 字符串 (对齐 mem0 expiration_date)。"""
+    if expiration_date is None:
+        return None
+    if not isinstance(expiration_date, str):
+        raise ValueError("expiration_date must be a string in YYYY-MM-DD format")
+    from datetime import datetime
+
+    try:
+        datetime.strptime(expiration_date, "%Y-%m-%d")
+    except ValueError:
+        raise ValueError(f"expiration_date must be in YYYY-MM-DD format, got: {expiration_date}") from None
+    return expiration_date
+
+
+def _is_expired(metadata: dict[str, Any] | None) -> bool:
+    """检查 metadata 是否已过期 (基于 expiration_date 字段)。"""
+    if not metadata or not isinstance(metadata, dict):
+        return False
+    exp = metadata.get("expiration_date")
+    if not exp or not isinstance(exp, str):
+        return False
+    from datetime import date, datetime
+
+    try:
+        exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
+    except ValueError:
+        return False
+    return exp_date < date.today()
+
+
 class Memory:
     """SeptMuse 记忆系统零配置 facade。
 
     零配置: Memory() 用 SQLite + HashEmbedder。
     升级: Memory(config=MemoryConfig(db_path="...")) 自定义。
     """
 
     def __init__(
         self,
         config: MemoryConfig | None = None,
@@ -161,43 +193,121 @@ class Memory:
         if self.llm is not None:
             self.extractor = FactExtractor(self.llm, self.embedder, self.typed_store, self.store)
 
         from septmuse.extraction.entity import _resolve_entity_extractor
 
         self.entity_extractor = entity_extractor or _resolve_entity_extractor(self.config)
 
         # entity_store (ORMMemoryStore 路径)
         self.entity_store = None
         if store_engine is not None:
-            from septmuse.storage.relational_stores.entity_store import EntityStore
+            try:
+                from septmuse.storage.relational_stores.entity_store import EntityStore
 
-            self.entity_store = EntityStore.from_engine(store_engine, self.embedder)
+                self.entity_store = EntityStore.from_engine(store_engine, self.embedder)
+            except Exception as e:
+                logger.warning("entity_store_init_failed", error=str(e))
 
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
 
+        self._hybrid_retriever = None
+
+        self._init_v2_components()
+
+    def _init_v2_components(self) -> None:
+        """初始化 V2 编排组件 (working_memory / capture / token_budget / meta / evolution / forgetting)。"""
+        store_engine = getattr(self.store, "engine", None)
+
+        self.working_memory = None
+        if store_engine is not None:
+            try:
+                from septmuse.memory.working_memory import WorkingMemory
+                from septmuse.storage.working_memory_stores.factory import create_working_memory_store
+
+                wm_store = create_working_memory_store(self.config, engine=store_engine)
+                self.working_memory = WorkingMemory(store=wm_store, agent_id="default")
+            except Exception as e:
+                logger.warning("working_memory_init_failed", error=str(e))
+
+        self.capture = None
+        try:
+            from septmuse.capture.pipeline import CapturePipeline
+
+            self.capture = CapturePipeline(
+                self.store,
+                self.embedder,
+                typed_store=self.typed_store,
+                llm=self.llm,
+                dedup_window=self._dedup_window,
+            )
+        except Exception as e:
+            logger.warning("capture_init_failed", error=str(e))
+
+        self.token_budget = None
+        try:
+            from septmuse.retrieval.token_budget import TokenBudget
+
+            self.token_budget = TokenBudget()
+        except Exception as e:
+            logger.warning("token_budget_init_failed", error=str(e))
+
+        self.meta = None
+        try:
+            from septmuse.memory.meta import MetacognitionLayer
+
+            self.meta = MetacognitionLayer(self.embedder, self.store, self.typed_store)
+        except Exception as e:
+            logger.warning("meta_init_failed", error=str(e))
+
+        self.evolution = None
+        try:
+            from septmuse.memory.evolution import EvolutionEngine
+
+            self.evolution = EvolutionEngine(
+                self.store, self.graph_store, self.embedder, self.typed_store, self.llm
+            )
+        except Exception as e:
+            logger.warning("evolution_init_failed", error=str(e))
+
+        self.forgetting = None
+        try:
+            from septmuse.retrieval.forgetting import ForgettingRetriever
+
+            self.forgetting = ForgettingRetriever(self.typed_store)
+        except Exception as e:
+            logger.warning("forgetting_init_failed", error=str(e))
+
+        self.summarizer = None
+        try:
+            from septmuse.evolution.summarizer import Summarizer
+
+            self.summarizer = Summarizer(self.store, self.typed_store, self.llm)
+        except Exception as e:
+            logger.warning("summarizer_init_failed", error=str(e))
+
     def _resolve_store(self) -> MemoryStore:
         """解析 store: 统一走 ORMMemoryStore (DatabaseService 自动回退 SQLite 零配置)。"""
         from septmuse.storage.relational_stores.factory import RelationalStoreFactory
 
         return RelationalStoreFactory.create(self.config)
 
     def _create_facts_vector_store(self) -> Any:
         """创建 SemanticFact 专用 vector_store (collection_name='septmuse_facts')。
 
         从 ORMMemoryStore 的 vector_store 判断后端类型, 仅 QdrantVectorStore 支持多 collection。
@@ -248,20 +358,21 @@ class Memory:
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
+        expiration_date: str | None = None,
     ) -> dict[str, Any]:
         """添加记忆 (统一 typed memory 入口)。
 
         Args:
             messages: str 或 List[{"role","content"}]
             user_id: 用户 ID (必填, 跨 agent 共享键)
             agent_id: agent ID (可选)
             session_id: 会话 ID (None=不限制)
             metadata: 元数据
             infer: True=LLM 抽取事实; False=原文存; None=用 config.infer
@@ -325,20 +436,26 @@ class Memory:
         should_infer = self.config.infer if infer is None else infer
 
         if should_infer and self.extractor is not None:
             extracted = self.extractor.extract_and_store(messages, user_id=user_id)
             return {"results": extracted, "relations": []}
 
         texts = _normalize_messages(messages)
         if not texts:
             return {"results": [], "relations": []}
 
+        normalized_exp = _normalize_expiration_date(expiration_date)
+        if normalized_exp is not None:
+            if metadata is None:
+                metadata = {}
+            metadata["expiration_date"] = normalized_exp
+
         embeddings = self.embedder.embed_batch(texts)
 
         # 批量插入 (对齐 mem0 V3 Phase 6, 单次 commit)
         records = list(zip(texts, embeddings, strict=True))
         ids = self.store.add_batch(
             records,
             user_id=user_id,
             agent_id=agent_id,
             session_id=session_id,
             metadata=metadata,
@@ -351,20 +468,21 @@ class Memory:
         ]
         if auto_extract_entities and self.entity_store is not None and self.entity_extractor is not None:
             self._batch_extract_and_store_entities(valid_pairs, user_id=user_id, agent_id=agent_id)
 
         results: list[dict[str, Any]] = []
         for text, mid in zip(texts, ids, strict=True):
             if mid is None:
                 continue  # 批次内 hash 去重跳过
             results.append({"id": mid, "memory": text, "event": "ADD"})
 
+        self._invalidate_search_cache(user_id=user_id)
         logger.info("memory_add_done", user_id=user_id, count=len(results), infer=should_infer)
         return {"results": results, "relations": []}
 
     def _extract_and_store_entities(
         self, text: str, memory_id: str, *, user_id: str, agent_id: str | None = None
     ) -> None:
         """自动抽取实体并存入 entity_store (吞错, 不阻塞 add)。"""
         if self.entity_extractor is None or self.entity_store is None:
             return
         try:
@@ -404,35 +522,60 @@ class Memory:
             # 2. 逐个 upsert (upsert 内部做精确+语义去重, 全局去重减少重复调用)
             for entity_type, entity_text, memory_ids in global_entities.values():
                 entity = Entity(text=entity_text, entity_type=entity_type, start=0, end=0)
                 for mid in memory_ids:
                     self.entity_store.upsert(  # type: ignore[union-attr]
                         entity, mid, user_id=user_id, agent_id=agent_id
                     )
         except Exception as e:
             logger.warning("batch_extract_entities_failed", error=str(e))
 
+    def _get_hybrid_retriever(self):
+        """获取持久化 HybridRetriever 实例 (缓存跨 search 调用生效)。"""
+        if self._hybrid_retriever is None:
+            from septmuse.retrieval.hybrid import HybridRetriever
+
+            self._hybrid_retriever = HybridRetriever(
+                self.store,
+                self.embedder,
+                entity_extractor=self.entity_extractor,
+                entity_store=self.entity_store,
+            )
+        return self._hybrid_retriever
+
+    def _invalidate_search_cache(self, user_id: str | None = None) -> None:
+        """清除 HybridRetriever 搜索缓存 (add/update/delete 后调用)。"""
+        if self._hybrid_retriever is not None:
+            self._hybrid_retriever.invalidate_cache(user_id=user_id)
+
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
-    ) -> list[dict[str, Any]]:
+        show_expired: bool = False,
+        # B 类高级参数
+        forgetting: bool = False,
+        token_budget: int | None = None,
+        inject_prompt: bool = False,
+        hyde: bool = False,
+        query_rewrite: bool = False,
+    ) -> list[dict[str, Any]] | dict[str, Any]:
         """检索记忆 (默认 hybrid BM25+向量 RRF 融合, 支持后处理重排)。
 
         Args:
             query: 查询文本
             user_id: 用户 ID (必填)
             session_id: 会话 ID (None=不限)
             top_k: 返回数 (默认 config.top_k)
             threshold: 相似阈值 (默认 config.threshold)
             hybrid: True=BM25+向量 RRF 融合 (默认); False=纯向量
             reranker: 重排器 ("noop"/"mmr"/"cross_encoder"/"llm"/"cohere"/"batch_llm", None=用 config)
@@ -446,77 +589,79 @@ class Memory:
         """
         if recipe is not None:
             from septmuse.retrieval.recipes import get_recipe
 
             r = get_recipe(recipe)
             hybrid = r.hybrid
             if r.reranker != "noop":
                 reranker = r.reranker
             explain = r.explain
 
+        if hyde:
+            logger.info("hyde_not_enabled", hint="requires LLM + prompt engineering (future)")
+        if query_rewrite:
+            logger.info("query_rewrite_not_enabled", hint="requires LLM (future)")
+
         tk = top_k or self.config.top_k
+        fetch_k = tk * 2 if not show_expired else tk
         if hybrid:
             th = 0.0 if threshold is None else threshold
         else:
             th = threshold if threshold is not None else self.config.threshold
 
         if hybrid:
-            from septmuse.retrieval.hybrid import HybridRetriever
-
-            retriever = HybridRetriever(
-                self.store,
-                self.embedder,
-                entity_extractor=self.entity_extractor,
-                entity_store=self.entity_store,
-            )
+            retriever = self._get_hybrid_retriever()
             hybrid_results = retriever.search(
                 query,
                 user_id=user_id,
                 session_id=session_id,
-                top_k=tk,
+                top_k=fetch_k,
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
-                    scored = reranker_instance.rerank(query, documents, top_k=tk)
-                    hybrid_results = strategy.reconstruct(scored, tracker, hybrid_results, tk, search_filter)
+                    scored = reranker_instance.rerank(query, documents, top_k=fetch_k)
+                    hybrid_results = strategy.reconstruct(scored, tracker, hybrid_results, fetch_k, search_filter)
                 except Exception as e:
                     logger.warning("reranker_failed_fallback", reranker=reranker, error=str(e))
-            return [
+            hybrid_return = [
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
+            if not show_expired:
+                hybrid_return = [r for r in hybrid_return if not _is_expired(r.get("metadata"))]
+            return hybrid_return[:tk]
 
         # 纯向量分支
         emb = self.embedder.embed(query)
         results = self.store.search(
             emb,
             user_id=user_id,
             session_id=session_id,
-            top_k=tk,
+            top_k=fetch_k,
             threshold=th,
             filters=filters,
         )
 
         # 纯向量分支也应用 reranker (try/except 降级容错)
         if reranker is not None and reranker != "noop" and results:
             from septmuse.rerankers import create_reranker
             from septmuse.rerankers.strategies import RerankerStrategyFactory
             from septmuse.retrieval.hybrid import HybridResult
 
@@ -531,73 +676,82 @@ class Memory:
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
-                scored = reranker_instance.rerank(query, documents, top_k=tk)
-                reranked = strategy.reconstruct(scored, tracker, hybrid_results, tk, search_filter)
+                scored = reranker_instance.rerank(query, documents, top_k=fetch_k)
+                reranked = strategy.reconstruct(scored, tracker, hybrid_results, fetch_k, search_filter)
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
-        return results
+        if not show_expired:
+            results = [r for r in results if not _is_expired(r.get("metadata"))]
+        return results[:tk]
 
     def get_all(
-        self, *, user_id: str, session_id: str | None = None, filters: dict[str, Any] | None = None
+        self, *, user_id: str, session_id: str | None = None, filters: dict[str, Any] | None = None,
+        show_expired: bool = False,
     ) -> dict[str, Any]:
         """列出全部记忆。"""
-        return {"results": self.store.get_all(user_id=user_id, session_id=session_id, filters=filters)}
+        results = self.store.get_all(user_id=user_id, session_id=session_id, filters=filters)
+        all_results = results.get("results", results) if isinstance(results, dict) else results
+        if not show_expired:
+            all_results = [r for r in all_results if not _is_expired(r.get("metadata"))]
+        return {"results": all_results}
 
     def get(self, memory_id: str) -> dict[str, Any] | None:
         """取单条。"""
         return self.store.get(memory_id)
 
     def delete(self, memory_id: str) -> dict[str, str]:
         """软删除 (自动清理实体引用)。"""
         if self.entity_store is not None:
             self.entity_store.remove_memory_from_entities(memory_id)
         self.store.delete(memory_id)
+        self._invalidate_search_cache()
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
+        self._invalidate_search_cache(user_id=user_id)
         return {"status": "deleted", "count": count}
 
     def update(
         self,
         memory_id: str,
         data: str | None = None,
         *,
         user_id: str | None = None,
         metadata: dict[str, Any] | None = None,
     ) -> dict[str, Any]:
@@ -620,20 +774,21 @@ class Memory:
         new_embedding = self.embedder.embed(new_content)
 
         if metadata is not None:
             merged_meta = {**existing.get("metadata", {}), **metadata}
         else:
             merged_meta = existing.get("metadata", {})
 
         ok = self.store.update(memory_id, new_content, new_embedding, metadata=merged_meta)
         if not ok:
             return {"id": memory_id, "event": "NOT_FOUND"}
+        self._invalidate_search_cache(user_id=user_id)
         logger.info("memory_update_done", memory_id=memory_id, user_id=user_id)
         return {"id": memory_id, "memory": new_content, "event": "UPDATE"}
 
     def get_history(self, memory_id: str) -> list[dict[str, Any]]:
         """获取记忆变更历史。"""
         return self.store.get_history(memory_id)
 
     # ------------------------------------------------------------------
     # 差异化: 双时态 + 知识图谱
     # ------------------------------------------------------------------
@@ -694,20 +849,311 @@ class Memory:
     # ------------------------------------------------------------------
     # 共享 (实用)
     # ------------------------------------------------------------------
 
     def list_agents(self, user_id: str) -> list[str]:
         """列出该用户的所有 agent (跨 agent 共享)。"""
         from septmuse.governance.sharing import SharedMemoryAccessor
 
         return SharedMemoryAccessor(self.store).list_agents(user_id)
 
+    # ------------------------------------------------------------------
+    # V2 编排 API (remember / recall / improve / forget)
+    # ------------------------------------------------------------------
+
+    def remember(
+        self,
+        text: str,
+        *,
+        user_id: str,
+        session_id: str | None = None,
+        agent_id: str | None = None,
+    ) -> dict[str, Any]:
+        """V2 remember: 捕获文本 → 存 raw_log → (可选) 抽事实 → 更新工作记忆。
+
+        Returns:
+            {"captured": bool, "raw_id": str|None, "fact_ids": list, "memory_id": str}
+        """
+        result: dict[str, Any] = {"captured": False, "raw_id": None, "fact_ids": [], "memory_ids": [], "memory_id": None}
+
+        if not text or not text.strip():
+            return result
+
+        if self.capture is None:
+            logger.warning("remember_capture_unavailable", user_id=user_id)
+            return result
+
+        preprocess = self.capture.preprocess(text, user_id=user_id, agent_id=agent_id)
+        if not preprocess.allowed:
+            logger.info("remember_deduped", user_id=user_id, reason=preprocess.reason)
+            return result
+
+        stored_text = preprocess.stored_text
+        raw_event = self.episodic.add_raw_log(
+            stored_text,
+            user_id=user_id,
+            session_id=session_id or "default",
+            agent_id=agent_id,
+        )
+        result["raw_id"] = raw_event.id
+
+        verbatim_id = self.store.add(
+            stored_text,
+            self.embedder.embed(stored_text),
+            user_id=user_id,
+            agent_id=agent_id,
+            session_id=session_id,
+            metadata={"source": "remember", "raw_id": raw_event.id},
+        )
+        if verbatim_id is None:
+            logger.info("remember_deduped_by_store", user_id=user_id)
+            return result
+        result["memory_id"] = verbatim_id
+        result["memory_ids"] = [verbatim_id]
+        self._invalidate_search_cache(user_id=user_id)
+
+        if self.entity_store is not None and self.entity_extractor is not None:
+            self._extract_and_store_entities(stored_text, verbatim_id, user_id=user_id, agent_id=agent_id)
+
+        if self.llm is not None and self.extractor is not None:
+            try:
+                extracted = self.extractor.extract_and_store(stored_text, user_id=user_id)
+                result["fact_ids"] = [e.get("id", "") for e in extracted if isinstance(e, dict)]
+            except Exception as e:
+                logger.warning("remember_extract_failed", error=str(e))
+
+        if agent_id is not None and self.working_memory is not None:
+            with contextlib.suppress(Exception):
+                self.working_memory.core_memory_append("notes", stored_text)
+
+        result["captured"] = True
+        logger.info("remember_done", user_id=user_id, raw_id=result["raw_id"], facts=len(result["fact_ids"]))
+        return result
+
+    def recall(
+        self,
+        query: str,
+        *,
+        user_id: str,
+        session_id: str | None = None,
+        top_k: int | None = None,
+        auto_compress: bool = False,
+        recipe: str | None = None,
+        inject_profile: bool = False,
+        **kwargs: Any,
+    ) -> dict[str, Any]:
+        """V2 recall: 检索 → token 裁剪 → 注入 prompt → L0 路由 + L2 策略。
+
+        Returns:
+            {"memories": list, "injected_prompt": str, "route": dict, "strategy": StrategyResult|None}
+        """
+        if auto_compress and self.summarizer is not None:
+            with contextlib.suppress(Exception):
+                self.summarizer.compress(user_id=user_id)
+
+        memories = self.search(
+            query, user_id=user_id, session_id=session_id, top_k=top_k, recipe=recipe, **kwargs
+        )
+
+        from types import SimpleNamespace
+
+        from septmuse.retrieval.token_budget import BudgetItem
+
+        items = [
+            BudgetItem(text=m.get("memory", ""), score=m.get("score", 0.0), id=m.get("id"))
+            for m in memories
+        ]
+        if self.token_budget is not None:
+            budget_result = self.token_budget.fit(items)
+        else:
+            budget_result = SimpleNamespace(texts=[item.text for item in items], used_tokens=0)
+
+        wm_prompt = ""
+        if self.working_memory is not None:
+            wm_prompt = self.working_memory.compile_to_prompt() + "\n"
+
+        memory_texts = "\n".join(f"- {t}" for t in budget_result.texts)
+        injected_prompt = f"{wm_prompt}<memory>\n{memory_texts}\n</memory>"
+
+        route_result = None
+        if self.meta is not None:
+            try:
+                route_result = self.meta.route(query)
+            except Exception as e:
+                logger.warning("recall_route_failed", error=str(e))
+
+        strategy = None
+        if self.meta is not None:
+            coverage_facts = self.typed_store.get_all_facts(user_id=user_id)
+            has_coverage = any(
+                "meta" in (f.tags or []) and "coverage" in (f.tags or []) for f in coverage_facts
+            )
+            if has_coverage:
+                try:
+                    report = self.meta.analyze_coverage(user_id=user_id)
+                    strategy = self.meta.adapt_strategy(report)
+                except Exception as e:
+                    logger.warning("recall_strategy_failed", error=str(e))
+
+        logger.info("recall_done", user_id=user_id, hits=len(memories), tokens=budget_result.used_tokens)
+        return {
+            "memories": memories,
+            "injected_prompt": injected_prompt,
+            "route": route_result,
+            "strategy": strategy,
+        }
+
+    def improve(self, *, user_id: str, limit: int = 20) -> dict[str, Any]:
+        """V2 improve: Dream 链接生长 + reflect 蒸馏 + L1 覆盖报告持久化。
+
+        Returns:
+            {"dream": {"links_created": int, "processed": int}, "rules": int}
+        """
+        dream_result: dict[str, Any] = {"links_created": 0, "processed": 0}
+        if self.evolution is not None:
+            try:
+                dream = self.evolution.dream(user_id=user_id)
+                dream_result = {"links_created": dream.links_created, "processed": dream.processed}
+            except Exception as e:
+                logger.warning("improve_dream_failed", error=str(e))
+
+        rules_count = 0
+        if self.llm is not None and self.evolution is not None:
+            try:
+                reflect_result = self.evolution.reflect(user_id=user_id, limit=limit)
+                rules_count = len(reflect_result.rule_ids) if hasattr(reflect_result, "rule_ids") else 0
+            except Exception as e:
+                logger.warning("improve_reflect_failed", error=str(e))
+
+        coverage_info: dict[str, Any] = {}
+        if self.meta is not None:
+            try:
+                report = self.meta.analyze_coverage(user_id=user_id)
+                coverage_info = {
+                    "overall_score": report.overall_score,
+                    "weak_areas": report.weak_areas,
+                    "strong_areas": report.strong_areas,
+                }
+                self.typed_store.add_fact(
+                    subject=user_id,
+                    predicate="has_coverage_score",
+                    object=f"{report.overall_score:.2f}",
+                    user_id=user_id,
+                    context=report.summary(),
+                    confidence=report.overall_score,
+                    provenance="meta",
+                    tags=["meta", "coverage"],
+                )
+            except Exception as e:
+                logger.warning("improve_coverage_persist_failed", error=str(e))
+
+        conflicts_info: dict[str, Any] = {}
+        if self.evolution is not None:
+            try:
+                conflicts_info = self.evolution.resolve_conflicts(user_id=user_id)
+            except Exception as e:
+                logger.warning("improve_conflict_failed", error=str(e))
+
+        logger.info("improve_done", user_id=user_id, links=dream_result["links_created"], rules=rules_count)
+        return {"dream": dream_result, "rules": rules_count, "coverage": coverage_info, "conflicts": conflicts_info}
+
+    def forget(self, memory_id: str, *, user_id: str) -> dict[str, Any]:
+        """V2 forget: 先 invalidate 再 delete (决策 5)。
+
+        Returns:
+            {"event": "FORGET", "invalidated_at": str|None, "deleted_at": str|None}
+        """
+        existing = self.store.get(memory_id)
+        if existing is not None:
+            owner = existing.get("user_id")
+            if owner is not None and owner != user_id:
+                logger.warning("forget_ownership_mismatch", memory_id=memory_id, user_id=user_id, owner=owner)
+                return {
+                    "event": "FORGET",
+                    "error": "ownership_mismatch",
+                    "invalidated_at": None,
+                    "deleted_at": None,
+                }
+
+        invalidated_at = None
+        try:
+            inv = self.invalidate(memory_id)
+            if isinstance(inv, dict):
+                invalidated_at = inv.get("invalid_at") or inv.get("invalidated_at")
+        except Exception as e:
+            logger.warning("forget_invalidate_failed", error=str(e))
+
+        deleted_at = None
+        try:
+            if existing is not None:
+                self.delete(memory_id)
+                from datetime import datetime, timezone
+
+                deleted_at = datetime.now(timezone.utc).isoformat()
+        except Exception as e:
+            logger.warning("forget_delete_failed", error=str(e))
+
+        logger.info("forget_done", memory_id=memory_id, user_id=user_id)
+        return {"event": "FORGET", "invalidated_at": invalidated_at, "deleted_at": deleted_at}
+
+    def compress(self, *, user_id: str, mode: str = "static", buffer_size: int = 20) -> dict[str, Any]:
+        """压缩消息 (委托 Summarizer.compress)。
+
+        Returns:
+            {"compressed": bool, "evicted": int, "kept": int, "summary_id": str|None}
+        """
+        if self.summarizer is None:
+            return {"compressed": False, "evicted": 0, "kept": 0, "summary_id": None}
+        return self.summarizer.compress(user_id=user_id, mode=mode, buffer_size=buffer_size)
+
     # ------------------------------------------------------------------
     # 资源释放
     # ------------------------------------------------------------------
 
+    def reset(self) -> dict[str, Any]:
+        """重置记忆系统 (清表 + 双写清理 + 缓存失效)。
+
+        清除: memories / history / memory_access_logs / septmuse_facts /
+        septmuse_episodic / septmuse_procedural / septmuse_entities /
+        entity_relations, 并重置 vector_store + keyword_index。
+        """
+        self.store.reset()
+
+        engine = getattr(self.store, "engine", None)
+        if engine is not None:
+            from sqlalchemy import text
+            from sqlmodel import Session
+
+            with Session(engine) as session:
+                for table_name in [
+                    "septmuse_facts",
+                    "septmuse_episodic",
+                    "septmuse_procedural",
+                ]:
+                    with contextlib.suppress(Exception):
+                        session.exec(text(f"DELETE FROM {table_name}"))
+                session.commit()
+
+        if self.entity_store is not None:
+            with contextlib.suppress(Exception):
+                self.entity_store.reset()
+
+        if hasattr(self, "_dedup_window"):
+            self._dedup_window = type(self._dedup_window)()
+
+        if self.working_memory is not None:
+            with contextlib.suppress(Exception):
+                wm_store = getattr(self.working_memory, "store", None)
+                if wm_store is not None and hasattr(wm_store, "reset"):
+                    wm_store.reset()
+
+        self._hybrid_retriever = None
+
+        logger.info("memory_reset_done")
+        return {"status": "reset", "message": "all memories cleared"}
+
     def close(self) -> None:
         """关闭存储。"""
         self.store.close()
         self.typed_store.close()
         if self.entity_store is not None:
             self.entity_store.close()
diff --git a/tests/unit/test_search_enhanced.py b/tests/unit/test_search_enhanced.py
index 5f07bbe..9056e26 100644
--- a/tests/unit/test_search_enhanced.py
+++ b/tests/unit/test_search_enhanced.py
@@ -48,10 +48,26 @@ def test_search_default_returns_list(memory):
     memory.add("test fact", user_id="alice", infer=False)
     results = memory.search("test", user_id="alice")
     assert isinstance(results, list)
 
 
 def test_search_no_llm_forgetting_works(memory):
     """无 LLM 时 forgetting 仍可用 (不依赖 LLM)."""
     memory.add("some fact", user_id="alice", infer=False)
     results = memory.search("some", user_id="alice", forgetting=True)
     assert len(results) > 0
+
+
+def test_search_hyde_noop(memory):
+    """search(hyde=True) 不报错, 返回与默认一致。"""
+    memory.add("test fact", user_id="alice", infer=False)
+    results = memory.search("test", user_id="alice", hyde=True)
+    assert isinstance(results, list)
+    assert len(results) > 0
+
+
+def test_search_query_rewrite_noop(memory):
+    """search(query_rewrite=True) 不报错, 返回与默认一致。"""
+    memory.add("test fact", user_id="alice", infer=False)
+    results = memory.search("test", user_id="alice", query_rewrite=True)
+    assert isinstance(results, list)
+    assert len(results) > 0

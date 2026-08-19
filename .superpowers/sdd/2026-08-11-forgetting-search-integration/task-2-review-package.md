## Commits

84cfc18 feat(search): wire forgetting/token_budget/inject_prompt post-processing into search()

## Diff Stats

 src/septmuse/memory/main.py | 74 +++++++++++++++++++++++++++++++++++++++++++--
 1 file changed, 72 insertions(+), 2 deletions(-)

## Full Diff

diff --git a/src/septmuse/memory/main.py b/src/septmuse/memory/main.py
index 9673c8b..e27682a 100644
--- a/src/septmuse/memory/main.py
+++ b/src/septmuse/memory/main.py
@@ -540,20 +540,78 @@ class Memory:
                 entity_extractor=self.entity_extractor,
                 entity_store=self.entity_store,
             )
         return self._hybrid_retriever
 
     def _invalidate_search_cache(self, user_id: str | None = None) -> None:
         """清除 HybridRetriever 搜索缓存 (add/update/delete 后调用)。"""
         if self._hybrid_retriever is not None:
             self._hybrid_retriever.invalidate_cache(user_id=user_id)
 
+    def _apply_search_post_processing(
+        self,
+        results: list[dict[str, Any]],
+        *,
+        user_id: str,
+        forgetting: bool = False,
+        token_budget: int | None = None,
+        inject_prompt: bool = False,
+    ) -> list[dict[str, Any]] | dict[str, Any]:
+        """search 后处理链: forgetting → token_budget → inject_prompt。
+
+        后处理失败不阻塞检索——吞错降级, 返回原始结果。
+        """
+        # [1] forgetting: 强度加权 + 测试效应
+        if forgetting and self.forgetting is not None:
+            try:
+                weighted = self.forgetting.apply_strength(results, user_id=user_id)
+                original_map = {r.get("id"): r for r in results}
+                results = [
+                    {
+                        "id": w.id,
+                        "memory": w.memory,
+                        "score": w.final_score,
+                        "final_score": w.final_score,
+                        "strength": w.strength,
+                        "metadata": w.metadata,
+                        "created_at": original_map.get(w.id, {}).get("created_at"),
+                    }
+                    for w in weighted
+                ]
+            except Exception as e:
+                logger.warning("forgetting_apply_failed", error=str(e))
+        elif forgetting and self.forgetting is None:
+            logger.warning("forgetting_component_unavailable")
+
+        # [2] token_budget: 按加权后 score 贪心裁剪
+        if token_budget is not None and results:
+            try:
+                from septmuse.retrieval.token_budget import BudgetItem, TokenBudget
+
+                items = [
+                    BudgetItem(text=r.get("memory", ""), score=r.get("score", 0.0), id=r.get("id")) for r in results
+                ]
+                budget = TokenBudget(budget=token_budget)
+                fitted = budget.fit(items)
+                chosen_ids = {item.id for item in fitted.items}
+                results = [r for r in results if r.get("id") in chosen_ids]
+            except Exception as e:
+                logger.warning("token_budget_failed", error=str(e))
+
+        # [3] inject_prompt: 返回 dict (提前 return)
+        if inject_prompt:
+            memory_texts = "\n".join(f"- {r.get('memory', '')}" for r in results)
+            injected_prompt = f"<memory>\n{memory_texts}\n</memory>"
+            return {"results": results, "injected_prompt": injected_prompt}
+
+        return results
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
@@ -639,21 +697,27 @@ class Memory:
                     "score": r.score,
                     "vector_score": r.vector_score,
                     "bm25_score": r.bm25_score,
                     "metadata": r.metadata,
                     "created_at": r.created_at,
                 }
                 for r in hybrid_results
             ]
             if not show_expired:
                 hybrid_return = [r for r in hybrid_return if not _is_expired(r.get("metadata"))]
-            return hybrid_return[:tk]
+            return self._apply_search_post_processing(
+                hybrid_return[:tk],
+                user_id=user_id,
+                forgetting=forgetting,
+                token_budget=token_budget,
+                inject_prompt=inject_prompt,
+            )
 
         # 纯向量分支
         emb = self.embedder.embed(query)
         results = self.store.search(
             emb,
             user_id=user_id,
             session_id=session_id,
             top_k=fetch_k,
             threshold=th,
             filters=filters,
@@ -695,21 +759,27 @@ class Memory:
                         "created_at": r.created_at,
                     }
                     for r in reranked
                 ]
             except Exception as e:
                 logger.warning("reranker_failed_fallback", reranker=reranker, error=str(e))
 
         logger.info("memory_search_done", user_id=user_id, query=query[:50], hits=len(results))
         if not show_expired:
             results = [r for r in results if not _is_expired(r.get("metadata"))]
-        return results[:tk]
+        return self._apply_search_post_processing(
+            results[:tk],
+            user_id=user_id,
+            forgetting=forgetting,
+            token_budget=token_budget,
+            inject_prompt=inject_prompt,
+        )
 
     def get_all(
         self, *, user_id: str, session_id: str | None = None, filters: dict[str, Any] | None = None,
         show_expired: bool = False,
     ) -> dict[str, Any]:
         """列出全部记忆。"""
         results = self.store.get_all(user_id=user_id, session_id=session_id, filters=filters)
         all_results = results.get("results", results) if isinstance(results, dict) else results
         if not show_expired:
             all_results = [r for r in all_results if not _is_expired(r.get("metadata"))]

## Commits

8eb3850 feat(recall): pass forgetting=True to search() for V2 orchestration

## Diff Stats

 src/septmuse/memory/main.py                  |  3 ++-
 tests/unit/test_search_forget_integration.py | 26 ++++++++++++++++++++++++++
 2 files changed, 28 insertions(+), 1 deletion(-)

## Full Diff

diff --git a/src/septmuse/memory/main.py b/src/septmuse/memory/main.py
index e27682a..39c7748 100644
--- a/src/septmuse/memory/main.py
+++ b/src/septmuse/memory/main.py
@@ -1013,22 +1013,23 @@ class Memory:
     ) -> dict[str, Any]:
         """V2 recall: 检索 → token 裁剪 → 注入 prompt → L0 路由 + L2 策略。
 
         Returns:
             {"memories": list, "injected_prompt": str, "route": dict, "strategy": StrategyResult|None}
         """
         if auto_compress and self.summarizer is not None:
             with contextlib.suppress(Exception):
                 self.summarizer.compress(user_id=user_id)
 
+        forgetting = kwargs.pop("forgetting", True)
         memories = self.search(
-            query, user_id=user_id, session_id=session_id, top_k=top_k, recipe=recipe, **kwargs
+            query, user_id=user_id, session_id=session_id, top_k=top_k, recipe=recipe, forgetting=forgetting, **kwargs
         )
 
         from types import SimpleNamespace
 
         from septmuse.retrieval.token_budget import BudgetItem
 
         items = [
             BudgetItem(text=m.get("memory", ""), score=m.get("score", 0.0), id=m.get("id"))
             for m in memories
         ]
diff --git a/tests/unit/test_search_forget_integration.py b/tests/unit/test_search_forget_integration.py
new file mode 100644
index 0000000..cdd3f8a
--- /dev/null
+++ b/tests/unit/test_search_forget_integration.py
@@ -0,0 +1,26 @@
+"""forgetting + search 组合集成测试。"""
+
+from __future__ import annotations
+
+import os
+
+import pytest
+
+
+@pytest.fixture
+def memory(tmp_path):
+    os.environ["SEPTMUSE_EMBEDDER"] = "hash"
+    os.environ["SEPTMUSE_TOKENIZER"] = "space"
+    from septmuse.configs import MemoryConfig
+    from septmuse.memory.main import Memory
+
+    return Memory(config=MemoryConfig(db_path=str(tmp_path / "test.db")))
+
+
+def test_recall_passes_forgetting(memory):
+    """recall() 内部传 forgetting=True, 返回结果有 final_score。"""
+    memory.add("alice likes python", user_id="alice", infer=False)
+    result = memory.recall("python", user_id="alice")
+    assert "memories" in result
+    if len(result["memories"]) > 0:
+        assert "final_score" in result["memories"][0]

## Commits

2a6ee07 fix(recall): pop inject_prompt/token_budget from kwargs to prevent TypeError

## Diff Stats

 src/septmuse/memory/main.py | 2 ++
 1 file changed, 2 insertions(+)

## Full Diff

diff --git a/src/septmuse/memory/main.py b/src/septmuse/memory/main.py
index 39c7748..a395fc4 100644
--- a/src/septmuse/memory/main.py
+++ b/src/septmuse/memory/main.py
@@ -1014,20 +1014,22 @@ class Memory:
         """V2 recall: 检索 → token 裁剪 → 注入 prompt → L0 路由 + L2 策略。
 
         Returns:
             {"memories": list, "injected_prompt": str, "route": dict, "strategy": StrategyResult|None}
         """
         if auto_compress and self.summarizer is not None:
             with contextlib.suppress(Exception):
                 self.summarizer.compress(user_id=user_id)
 
         forgetting = kwargs.pop("forgetting", True)
+        kwargs.pop("inject_prompt", None)  # recall() has its own prompt assembly; search() inject_prompt not supported
+        kwargs.pop("token_budget", None)   # recall() has its own token budget logic; search() token_budget not supported
         memories = self.search(
             query, user_id=user_id, session_id=session_id, top_k=top_k, recipe=recipe, forgetting=forgetting, **kwargs
         )
 
         from types import SimpleNamespace
 
         from septmuse.retrieval.token_budget import BudgetItem
 
         items = [
             BudgetItem(text=m.get("memory", ""), score=m.get("score", 0.0), id=m.get("id"))

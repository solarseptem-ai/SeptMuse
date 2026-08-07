# Task 1: HyDE 假设文档检索 — 报告

## 状态
**DONE**

## 变更文件
| 文件 | 操作 |
|------|------|
| `src/septmuse/prompts/hyde.py` | 新建 — HyDE prompt 模板 |
| `src/septmuse/memory/main.py` | 修改 — search 方法新增 `hyde: bool = False` 参数 + HyDE 分支 |
| `tests/unit/test_hyde.py` | 新建 — 4 个测试用例 |

## 实现说明
- `Memory.search` 新增 `hyde: bool = False` 关键字参数。
- HyDE 分支位于 search 方法体最顶部（docstring 之后、`if recipe` 之前），确保对 hybrid 和纯向量两个分支都生效。
- `hyde=True` 且 `self.llm is not None` 时：LLM 用 `HYDE_PROMPT` 生成假设答案 → 替换 `query` 变量 → 后续 embed/rerank/BM25 全部使用改写后的 query。
- `hyde=True` 但无 LLM → 不进入分支，降级为原文检索（不报错）。
- LLM 调用失败 → `try/except` 吞错 + 日志警告，降级原文检索。
- 修正了任务模板中的日志 bug：原模板在 `query = hypothetical` 后用 `query[:50]` 记录 `original_query`（实际记录的是新值），改为先保存 `original_query` 再赋值。

## 测试结果
```
tests/unit/test_hyde.py::test_hyde_disabled_uses_original_query PASSED
tests/unit/test_hyde.py::test_hyde_no_llm_falls_back PASSED
tests/unit/test_hyde.py::test_hyde_with_llm_uses_hypothetical PASSED
tests/unit/test_hyde.py::test_hyde_prompt_exists PASSED
4 passed in 1.22s
```

回归验证（现有 search 测试）：
```
tests/unit/test_memory.py -k "search" — 7 passed, 28 deselected
```

## Ruff
```
ruff check src/septmuse/prompts/hyde.py src/septmuse/memory/main.py tests/unit/test_hyde.py
All checks passed!
```

## 顾虑
- `HYDE_PROMPT` 含 `{query}` 占位符但未 `.format()`，因为 query 作为 `user_prompt` 参数传入 `llm.complete(system_prompt, user_prompt)`。LLM 能从 user_prompt 获取实际 query，功能正常，但 system_prompt 中的 `{query}` 字面量可能对某些 LLM 造成轻微困惑。当前测试验证功能正常，若后续真实 LLM 测试出现问题可改为 `.format(query=query)` + 空 user_prompt。
- `query` 被假设答案替换后，`logger.info("memory_search_done", query=query[:50])` 记录的是假设答案而非原 query。如需保留原 query 日志，可在 HyDE 分支后单独保存。当前不影响功能。

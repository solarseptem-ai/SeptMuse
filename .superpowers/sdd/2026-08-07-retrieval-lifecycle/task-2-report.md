# Task 2: 上下文感知查询改写 — 报告

## 状态
**DONE**

## 摘要
为 `Memory.search` 新增 `query_rewrite: bool = False` 参数；当 `query_rewrite=True` + LLM + `session_id` 时，从 episodic store 取近期 5 条对话上下文，LLM 将 query 改写为自包含形式后再检索；无 LLM 或无 `session_id` 时降级原文。改写分支置于 HyDE 之前，二者可叠加。

## 变更文件
- **新增** `src/septmuse/prompts/rewrite.py` — `QUERY_REWRITE_PROMPT`（LLM 角色/指令，无占位符，数据经 user_prompt 传入）
- **修改** `src/septmuse/memory/main.py` — `search` 签名加 `query_rewrite: bool = False` + docstring；HyDE 分支前插入改写分支
- **新增** `tests/unit/test_query_rewrite.py` — 5 测试（禁用/无LLM降级/无session降级/有上下文改写/prompt存在）

## 实现要点
- 任务 hint 假设 `self.episodic.get_episodes(user_id=, limit=)` 存在，实际 `EpisodicMemory` 暴露的是 `get_timeline(*, user_id, limit=50)`（`get_episodes` 在 `TypedMemoryStore` 上）。已用 `get_timeline`。
- `session_id` 在 `HybridRetriever.search` 是**硬过滤**（"仅搜该会话的记忆"），非 boost。原测试用例在 `add` 时未带 `session_id` 却用 `session_id="s1"` 检索，导致 0 结果——属测试 setup 缺陷，已修正测试用例在 `add` 时传入匹配的 `session_id`（非业务逻辑规避）。
- LLM 调用遵循 `complete(system_prompt, user_prompt)` ABC：system_prompt=QUERY_REWRITE_PROMPT（指令），user_prompt=格式化的 context+query。

## 验证证据
- `ruff check src/septmuse/prompts/rewrite.py src/septmuse/memory/main.py tests/unit/test_query_rewrite.py` → All checks passed!
- `pytest tests/unit/test_query_rewrite.py` → 5 passed
- `pytest tests/unit/test_hyde.py tests/unit/test_search_enhanced.py` → 9 passed（无回归）
- `pytest tests/unit/test_memory.py` → 6 failed 均为 pre-existing OpenAI embedder/LLM 测试（需 API key），与本次改动无关

## 顾虑
- `get_timeline` 不按 `session_id` 过滤，取的是该 user 全部近期 episodic 事件。多会话并发场景下可能混入其他会话上下文；当前 P1 范围内可接受，后续可扩展 `TypedMemoryStore.get_episodes` 支持 session_id 过滤。
- 无 `session_id` 时降级原文——若调用方希望改写但未传 session_id，改写不会触发（符合任务规格，但需调用方知晓）。

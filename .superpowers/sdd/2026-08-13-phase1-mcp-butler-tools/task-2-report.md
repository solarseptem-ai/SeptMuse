# Task 2 Report: recall MCP 工具

## Status

**DONE**

## Commits

- `75d6069` — feat(mcp): add recall tool — butler read entry point

## Changes

### src/septmuse/api/mcp/tools.py
- 在 `remember` 工具之后、`logger.info` 之前插入 `recall` 工具
- `recall(query, user_id, agent_id, session_id, top_k=5, token_budget=2000)` — 调用 `ExperimentalMemory.recall(query, *, user_id, session_id, top_k, inject_prompt=True, token_budget)`
- `count=19` → `count=20`

### tests/unit/test_mcp_butler_tools.py
- 追加 3 个测试函数：`test_recall_tool_basic`、`test_recall_tool_with_top_k`、`test_recall_tool_no_user_id`

### tests/unit/test_mcp_tools.py
- `test_tools_registered_count`: `== 19` → `== 20`
- `test_required_tool_names`: expected set 添加 `"recall"`

## Test Results

- recall 测试：3 passed / 0 failed
- butler 工具全量（remember + recall）：6 passed / 0 failed
- test_mcp_tools.py 全量回归：26 passed / 0 failed
- ruff check：All checks passed

## TDD 循环

1. **Red**: `ImportError: cannot import name 'recall'` — 预期失败
2. **Green**: 3 recall 测试全部通过

## Concerns

无。

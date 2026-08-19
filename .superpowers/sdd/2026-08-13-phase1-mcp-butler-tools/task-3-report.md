# Task 3 报告：improve MCP 工具

## 状态

DONE_WITH_CONCERNS

## 提交

- `429373f` — feat(mcp): add improve tool — butler evolution entry point

## 测试结果

- improve 工具测试：3 passed / 0 failed
- 全量 butler 工具测试（Task 1-3 回归）：9 passed / 0 failed
- test_mcp_tools.py 注册测试（count=21 + "improve" 名称）：2 passed / 0 failed
- ruff check：All checks passed

## 修改文件

1. `src/septmuse/api/mcp/tools.py` — 新增 `improve` 工具（`@mcp.tool` 装饰，薄层委托 `mem.improve(user_id=uid, limit=20)`），`count=20` → `count=21`
2. `tests/unit/test_mcp_butler_tools.py` — 追加 3 个测试函数（test_improve_tool_basic / test_improve_tool_no_user_id / test_improve_tool_no_memory）
3. `tests/unit/test_mcp_tools.py` — `== 20` → `== 21`，expected set 加 `"improve"`

## 顾虑

**计划文档中测试代码与实现代码存在矛盾**：测试断言 `mock_memory.improve.assert_called_once_with(user_id="alice", limit=20)`，要求工具显式传递 `limit=20`；但计划给出的实现代码是 `mem.improve(user_id=uid)`，不传 `limit`。MagicMock 只记录显式传入的参数，不记录默认值，因此用计划原文的实现代码测试会失败。

按 TDD 原则（测试是 spec），实现改为 `mem.improve(user_id=uid, limit=20)` —— 显式传递 `limit=20`（与 `ExperimentalMemory.improve(*, user_id, limit=20)` 的默认值一致，无行为差异）。这是对计划实现代码的最小偏离，仅为通过测试。

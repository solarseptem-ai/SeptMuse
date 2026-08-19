# Task 3 Report: recall() pass forgetting=True

## Status
DONE

## Commit
- `8eb3850` feat(recall): pass forgetting=True to search() for V2 orchestration
- Parent: `84cfc18` (Task 2: wire forgetting/token_budget/inject_prompt post-processing)

## Files Changed
- `src/septmuse/memory/main.py` — recall() 方法: 在 search() 调用前 `kwargs.pop("forgetting", True)`, 显式传 `forgetting=forgetting` 给 search()
- `tests/unit/test_search_forget_integration.py` — 新建集成测试: 验证 recall() 返回结果含 `final_score`

## Implementation
recall() 内部调用 search() 时, 默认 `forgetting=True` (V2 编排默认启用遗忘加权). 调用方可通过 `recall(query, forgetting=False)` 覆盖. `kwargs.pop("forgetting", True)` 防止重复 key 传给 search().

```python
forgetting = kwargs.pop("forgetting", True)
memories = self.search(
    query, user_id=user_id, session_id=session_id, top_k=top_k, recipe=recipe, forgetting=forgetting, **kwargs
)
```

## TDD Verification
1. **Red**: 测试先写, 运行后 FAIL — `assert 'final_score' in {'bm25_score': 1.0, 'created_at': ..., 'id': ..., 'memory': 'alice likes python', ...}` (recall() 未传 forgetting, 结果无 final_score)
2. **Green**: 修改 recall() 后, 测试 PASS

## Test Summary
- `tests/unit/test_search_forget_integration.py::test_recall_passes_forgetting` — PASSED
- 回归 (recall/orchestration/forgetting/compress/v2/async_v2): 68 passed, 0 failed
- 广泛回归 (memory/search_cache/search_enhanced/async_memory/async_add_search): 88 passed, 2 pre-existing failures (与本次改动无关, baseline 同样失败)

## Lint
- `ruff check src/septmuse/memory/main.py tests/unit/test_search_forget_integration.py` — All checks passed
- `ruff format --check tests/unit/test_search_forget_integration.py` — 1 file already formatted (新文件干净)
- `ruff format --check src/septmuse/memory/main.py` — 仍报 would reformat, 但为 **pre-existing** (Task 2 commit `84cfc18` 已存在, stash 验证确认), 本次改动未引入新违规

## Concerns
1. **main.py 存在 pre-existing 格式化问题**: `ruff format --check` 在 Task 2 commit (`84cfc18`) 后即已失败 (多处: lines 279/326/775/943/1056 等), 非本次改动引入. 受 Windows ruff format 清空文件 bug 限制, 无法用 `ruff format` 修复, 需手动调整或换平台. 本次仅确保自身改动行 (line ~1024) 符合 ruff 期望格式.
2. **2 个 pre-existing 测试失败**: `test_search_cache.py::test_cache_invalidate_all` 和 `test_async_add_search.py::test_add_memory_type_rule` — baseline 同样失败, 与 recall() 改动无关.
3. **QdrantClient `__del__` 关闭异常**: 测试 stderr 出现 `ImportError: sys.meta_path is None, Python is likely shutting down` — Python 解释器关闭时的无害 teardown 噪声, 非测试失败.

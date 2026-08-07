# Task 3 Report — BudgetItem.id + CapturePipeline.preprocess

## Status: DONE

## Files changed
1. `src/septmuse/retrieval/token_budget.py` — `BudgetItem` 加 `id: str | None = None` 字段 (line 55)
2. `src/septmuse/capture/pipeline.py` — 新增 `PreprocessResult` dataclass (line 57-64) + `CapturePipeline.preprocess` 方法 (去重+脱敏, 不写 store)
3. `tests/unit/test_token_budget_id.py` — 新增 3 测试 (id 默认 None / 设值 / fit 保留)
4. `tests/unit/test_capture_preprocess.py` — 新增 7 测试 (允许 / 去重 / 不写 store / 空文本 / 空格 / text_hash / 跨用户不去重)

## Test results
- **新测试 (RED→GREEN):** `test_token_budget_id.py` + `test_capture_preprocess.py` → 10 passed
- **回归:** `test_capture.py` + `test_governance.py` + 2 新文件 → **82 passed, 0 failed** (无回归)
- 覆盖 blast radius: `BudgetItem` (8 callers)、`TokenBudget` (12 callers)、`WriteValidator` (10 callers)、`PrivacyFilter` (21 callers) — 全部通过

## Ruff
```
ruff check src/septmuse/retrieval/token_budget.py src/septmuse/capture/pipeline.py
→ All checks passed!
```

## Implementation notes
- `BudgetItem.id` 默认 `None`, 向后兼容 (现有 caller 只传 text/score/metadata 不受影响)
- `preprocess` 复用 `WriteValidator.validate` (含 SHA256 去重 + per-user scope, validate 内部会 add 到 DedupWindow, 二次同文本被拒) 和 `PrivacyFilter.redact`
- `preprocess` 不调 `store.add` / `embedder.embed` — 纯预处理, 测试 `test_preprocess_no_write_to_store` 用 MagicMock 验证 `mock_store.add.call_count == 0`
- `PreprocessResult` dataclass: `allowed`/`stored_text`/`text_hash`/`redacted`/`reason`, 与 `PipelineResult` 分离 (capture 写 store, preprocess 不写)
- `preprocess` 空/纯空格文本先拦截 (`reason="empty text"`), 不进 validate
- `text_hash` 字段类型 `str | None` (与 ValidationResult 的 `str` 默认 `""` 不同 — PreprocessResult 初始 None, validate 通过后赋实际 hash)

## Concerns
- **DedupWindow 副作用:** `preprocess` 调 `validate` 会把文本 hash 加入 DedupWindow 时间窗。Task 4 中若 `V2.remember` 先 `preprocess` 再 delegate 到 `Memory.add`, 需确保 `Memory.add` 不重复走 dedup (或共享同一 DedupWindow 实例避免双检/漏检)。此为 Task 4 设计决策, Task 3 行为符合测试预期。
- 无其他顾虑。向后兼容, 测试通过, lint 通过。

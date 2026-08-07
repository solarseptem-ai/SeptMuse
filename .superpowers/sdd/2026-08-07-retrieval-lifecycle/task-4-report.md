# Task 4 Report: 压缩归档 facade

## Status: DONE

## Summary
`Memory.compress()` facade 委托 Summarizer；`recall(auto_compress=True)` 记忆超阈值自动触发压缩。6 新测试全过，10 编排回归全过，ruff clean。

## Changes

### Modified
- `src/septmuse/memory/main.py`:
  - `__init__`: 新增 `self.summarizer` (try/except 容错, 无 Summarizer 时 None)
  - 新增 `compress(*, user_id, mode="static", buffer_size=20)` facade 方法 — 委托 `Summarizer.compress`，压缩后调 `_invalidate_search_cache`
  - `recall` 签名新增 `auto_compress: bool = False` — 记忆超 20 阈值时自动调 `compress`

### Created
- `tests/unit/test_compress_facade.py` — 6 测试覆盖 summarizer 初始化、compress 委托、低于阈值不压缩、无 LLM 降级、auto_compress 触发/不触发

## TDD Evidence
- RED: 6/6 failed (summarizer attr missing, compress method missing, auto_compress param missing)
- GREEN: 6/6 passed after implementation
- 1 test data fix: `test_compress_facade` 原文本 "fact number {i}" 触发 0.95 语义去重 (5th add event=SKIP)，改为差异化文本 (断言不变)

## Regression
- `test_memory_orchestration.py`: 10/10 passed (recall 签名变更无破坏)
- `ruff check`: All checks passed

## Concerns
- auto_compress 阈值 (20) 和 buffer_size (20) 硬编码；后续可加 config 字段
- Summarizer 未修改 (符合要求)

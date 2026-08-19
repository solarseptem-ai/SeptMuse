# Task 2 Report: forgetting/token_budget/inject_prompt 后处理

## Status: DONE

## Commit
- Hash: `84cfc1851935ddd8f54f46b10b94b37be092b346`
- Message: `feat(search): wire forgetting/token_budget/inject_prompt post-processing into search()`
- File changed: `src/septmuse/memory/main.py` (+72, -2)

## Changes
1. **Added `_apply_search_post_processing` method** (between `_invalidate_search_cache` and `search`):
   - [1] forgetting: calls `ForgettingRetriever.apply_strength()`, maps `StrengthWeightedResult` back to dict (preserves `created_at` via `original_map` lookup since the dataclass lacks that field), swallows errors with `logger.warning`
   - [2] token_budget: builds `BudgetItem` list, calls `TokenBudget(budget=N).fit()`, filters results to `chosen_ids` set, swallows errors
   - [3] inject_prompt: returns `{"results": [...], "injected_prompt": "<memory>...</memory>"}` dict (early return)
   - Graceful degradation: `self.forgetting is None` → logs warning and skips; all post-processing failures → returns original results
2. **Wired hybrid branch return** (`hybrid_return[:tk]` → `self._apply_search_post_processing(hybrid_return[:tk], ...)`)
3. **Wired pure vector branch return** (`results[:tk]` → `self._apply_search_post_processing(results[:tk], ...)`)

## Test Summary
- `tests/unit/test_search_enhanced.py`: **7 passed** (including previously-failing `test_search_token_budget` and `test_search_inject_prompt`)
- Regression: `tests/unit/test_memory.py` + `tests/e2e/` → 66 passed, 1 failed (pre-existing `CapturePipeline not callable`, confirmed failing without my changes via `git stash`)
- Regression: `test_forgetting.py` + `test_forgetting_params.py` + `test_token_budget_id.py` + `test_governance.py` + `test_retrieval.py` → all pass
- `ruff check src/septmuse/memory/main.py` → All checks passed
- My additions are format-clean under `ruff format --diff` (pre-existing format issues in other methods remain, but are out of scope)

## Concerns
1. **Pre-existing format issues**: `ruff format --check` fails on `main.py` due to pre-existing diffs in `_init_v2_components` (EvolutionEngine call), `_create_facts_vector_store` (QdrantVectorStore args), and `recall()` (BudgetItem comprehension). These are from other uncommitted work, not my additions. Per brief, I did not run `ruff format <file>` on Windows (file-clearing bug).
2. **Pre-existing test failure**: `test_session_id.py::test_capture_with_session_id` and `test_cross_agent_capture.py::test_capture_persists` fail with `TypeError: 'CapturePipeline' object is not callable` — confirmed pre-existing via `git stash` (fails identically without my changes). Unrelated to search post-processing.
3. **token_budget test semantics**: `test_search_token_budget` adds 20 memories of ~70 tokens each with `token_budget=50`. Since each item exceeds the budget, all items are dropped → results empty → `total=0 <= 50` passes. This is the correct greedy behavior (skip items that individually exceed budget), though the test assertion is lenient (empty list trivially satisfies `<= 50`).

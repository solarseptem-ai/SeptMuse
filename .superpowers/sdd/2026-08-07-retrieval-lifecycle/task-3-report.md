# Task 3: 遗忘曲线参数化 — 完成报告

## 状态: DONE

## 摘要

为 `MemoryStrength.decay()` 和 `ForgettingRetriever` 添加 `half_life_days` 参数化半衰期，支持 7 天 (默认) / 1 天 (会话级) / inf (永久) 三种场景，向后完全兼容。

## 变更清单

| 文件 | 变更 |
|------|------|
| `src/septmuse/models/strength.py:93` | `decay()` 新增 `half_life_days: float \| None = None` 参数; None=原公式, float=半衰期公式 `stability = half_life_days * 86400 / ln(2)`, inf=永不衰减 |
| `src/septmuse/retrieval/forgetting.py:58` | `ForgettingRetriever.__init__` 新增 keyword-only `half_life_days: float = 7.0`; `apply_strength` 传递给 `decay()` |
| `src/septmuse/configs/base.py:78` | `MemoryConfig` 新增 `forgetting_half_life_days: float = Field(default=7.0, ...)` |
| `src/septmuse/memory/main.py:203` | `ForgettingManager` 创建时传入 `half_life_days=self.config.forgetting_half_life_days` |
| `tests/unit/test_forgetting_params.py` | 新建 7 个测试 (默认/快/慢/永久/retriever/config/Memory 集成) |

## 验证证据

- **新测试**: 7 passed (`pytest tests/unit/test_forgetting_params.py -v`)
- **现有遗忘测试**: 21 passed (`test_forgetting.py` 无回归)
- **编排测试**: 10 passed (`test_memory_orchestration.py` 无回归)
- **ruff check**: All checks passed (5 文件)
- **pre-existing 失败**: `test_qdrant_insert_and_search` (Qdrant 远程连接) + `test_search_cache::test_cache_invalidate_on_update` (ExperimentalMemory 搜索缓存) — 与本次变更无关

## 数学验证

- `half_life_days=1.0`, elapsed=1 天 → `exp(-86400 / (86400/ln2)) = exp(-ln2) = 0.5` ✓
- `half_life_days=inf` → `stability=inf` → `exp(-t/inf) = exp(0) = 1.0` (永不衰减) ✓
- `half_life_days=None` → 原公式 `effective_base * S_FACTOR` (向后兼容) ✓

## 顾虑

无。`half_life_days` 是 `decay()` 的运行时参数 (非 SQLModel 存储字段), `ForgettingRetriever.__init__` 用 keyword-only + 默认值 7.0, 现有调用方 `ForgettingRetriever(typed_store)` 无需改动。

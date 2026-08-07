# Task 6 报告 — UserProfile 模型 + get_user_profile 聚合

## 状态

**DONE** — 有验证证据

## 一句话总结

从 SemanticFact 聚合结构化用户画像 (attributes/preferences/plans/relationships/raw_facts)，支持时态去重 (最新 active = current) 与 include_temporal 开关，9 个新测试全部通过。

## 文件变更

### 新增
- `src/septmuse/models/profile.py` — `UserProfile` + `UserProfileValue` dataclass + `_classify()` predicate 分类函数
- `tests/unit/test_user_profile.py` — 9 个测试覆盖聚合/矛盾/软删除/temporal/raw/空用户/关系

### 修改
- `src/septmuse/memory/main.py` — 在 `_profile_to_prompt` 占位后新增 `get_user_profile(user_id, *, include_temporal=True)` 方法
- `src/septmuse/models/__init__.py` — 导出 `UserProfile` / `UserProfileValue` (API 完整性)

## 验证证据

### 测试 (TDD Red → Green)
- **RED** (实现前): 9 个测试全部失败 — `ModuleNotFoundError: No module named 'septmuse.models.profile'` + `AttributeError: 'Memory' object has no attribute 'get_user_profile'`
- **GREEN** (实现后): `9 passed in 1.86s`

### Lint
```
ruff check src/septmuse/models/profile.py src/septmuse/memory/main.py tests/unit/test_user_profile.py
→ All checks passed!
```
(修复 1 个 SIM114: 合并 `if` 分支为 `or` 逻辑)

### 回归检查
- `test_cognify.py + test_fact_decision.py + test_memory.py`: `57 passed, 6 failed`
- 6 failed 全部为 pre-existing LLM 测试 (`TestResolveEmbedderOpenAI` × 3 + `TestResolveLLMBaseUrl` × 3)，需真实 openai 包/API key，与本次改动无关 (AGENTS.md 基线已记录)

## 设计说明

- **时态模型**: SemanticFact 无 valid_at/invalid_at 双时态列，画像的"当前有效" = `is_deleted=False`，"最新" = `updated_at` 最新 (touch() 更新)
- **聚合逻辑**: 按 predicate 分组 → 按 updated_at 降序排 → 最新非删除 fact 标 `is_current=True`
- **dict 桶** (attributes/preferences/relationships): 同 predicate 只留一个值，current 优先覆盖非 current
- **list 桶** (plans): 可多值，include_temporal=False 时只留 current
- **raw_facts**: 未分类 predicate 的兜底 dict 列表
- **include_temporal=False**: 丢弃所有非 current 值 (历史 + 软删除)

## 顾虑

- `raw_facts` 未对 include_temporal 做过滤 (无测试约束，保持简单兜底)
- `_classify` 的 predicate 词表是静态硬编码，未覆盖的 predicate 全部落入 raw_facts (后续可扩展或 LLM 辅助分类)

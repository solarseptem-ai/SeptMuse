# Task 2 报告: 增强 update_fact + extract_and_store 决策路由 + Memory.add 接决策

## 状态

**DONE** — 全部 Task 2 测试通过, ruff 干净, 现有测试无回归。

## 修改文件

| 文件 | 改动 |
|------|------|
| `src/septmuse/storage/relational_stores/typed_store.py` | 增强 `update_fact`: 新增 `embedding`/`confidence` keyword 参数 (可选更新向量与置信度) |
| `src/septmuse/models/extract.py` | `FactExtractor.__init__` 新增 `use_decision: bool = False` 参数; 重写 `extract_and_store` 为决策路由 (ADD/UPDATE/DELETE/NOOP) + 置信度守卫 (DELETE/UPDATE <0.7 降级 NOOP) + `_legacy_extract_and_store` 旧路径降级; 新增 7 个 `_store_*` 辅助方法; `_parse_decisions_response` 向后兼容纯字符串 fact (视为 ADD) |
| `src/septmuse/memory/main.py` | `Memory.__init__` 创建 `FactExtractor` 时传 `use_decision=True` (Memory.add 的 infer=True 路径自动走决策路由) |
| `tests/unit/test_fact_decision.py` | 追加 6 个 `extract_and_store` 决策路由测试 (ADD/UPDATE/DELETE/低置信度降级/NOOP/legacy 降级) |
| `tests/unit/test_add_decision.py` | 新建 3 个 `Memory.add(infer=True)` 决策路由测试 (ADD/无 LLM 降级/infer=False verbatim) |

## 关键实现细节

1. **决策路由** (`extract_and_store`): `use_decision=True` 且有 LLM 时走 `extract_with_decisions` → ADD/UPDATE/DELETE/NOOP 路由; 否则降级 `_legacy_extract_and_store` (旧纯 ADD 路径, 保持旧返回结构)。
2. **置信度守卫**: DELETE/UPDATE 决策 `confidence < 0.7` 时降级为 NOOP (避免低置信度破坏性操作)。
3. **向后兼容修复**: `_parse_decisions_response` 原本只接受 dict fact, 会跳过 MockLLM/旧 prompt 的纯字符串 fact (`{"facts": ["Likes Python"]}`)。改为将纯字符串 fact 视为 ADD 决策 (默认置信度 1.0)。这是修复 `ExperimentalMemory(Memory)` + `MockLLM` 回归的关键。
4. **structlog kwarg 冲突修复**: `logger.info(..., event=d.event)` 与 structlog 内部 `event` 参数冲突, 改为 `decision=d.event`。

## 测试结果

```
# Task 1 + Task 2 + 回归 (fact_extraction + incremental_extraction)
PYTHONPATH=src pytest tests/unit/test_fact_decision.py tests/unit/test_add_decision.py tests/unit/test_fact_extraction.py tests/unit/test_incremental_extraction.py -q
=> 43 passed in 16.94s

# test_memory.py (仅 pre-existing 失败)
PYTHONPATH=src pytest tests/unit/test_memory.py -q
=> 6 failed, 29 passed in 9.20s
   (6 failed 全为 TestResolveEmbedderOpenAI x3 + TestResolveLLMBaseUrl x3 — AGENTS.md 记录的 pre-existing 基线, 需 openai 包/API key, 与本次改动无关)
```

- Task 2 新增测试: 9 个全部通过 (6 in test_fact_decision.py + 3 in test_add_decision.py)
- Task 1 已有测试: 6 个全部通过 (无回归)
- fact_extraction 回归: 修复 2 个原本因 use_decision 切换而失败的测试 (test_add_infer_true_extracts_facts, test_inferred_facts_searchable), 现全通过
- incremental_extraction: 全通过
- test_memory.py: 6 个 pre-existing 失败 (OpenAI/LLM API key 基线, 非本次改动引入)

## Ruff 结果

```
ruff check src/septmuse/models/extract.py src/septmuse/storage/relational_stores/typed_store.py src/septmuse/memory/main.py
=> All checks passed!
```

## 顾虑

- **verbatim_store.update/delete 用 fact_id 而非 verbatim memory_id**: UPDATE/DELETE 决策路由时, `_store_verbatim_update`/`_store_verbatim_delete` 用 SemanticFact id 调 `verbatim_store.update/delete`。若该 fact 未经 cognify 流水线创建 (无对应 verbatim memory), 则 ORMMemoryStore.update 返回 False / delete 无操作 (不报错)。这与测试预期一致 (测试只验证 SemanticFact 更新/删除), 但生产环境若需 verbatim 同步更新, 应在 cognify 完整流水线 (fact+verbatim 同时创建) 下运行。当前行为安全 (吞错/无操作)。
- **未提交 git**: 按要求未执行 git commit。

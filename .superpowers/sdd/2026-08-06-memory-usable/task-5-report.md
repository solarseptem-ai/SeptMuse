# Task 5 报告: add 语义去重 + search 参数化

## 状态: DONE

## 一句话总结
在 `Memory.add` verbatim 路径增加 0.95 阈值语义去重 (单条文本), 在 `Memory.search` 增加 `forgetting`/`token_budget`/`inject_prompt` 三个可选参数, 全部新测试通过且无回归。

## 修改文件

| 文件 | 变更 |
|------|------|
| `src/septmuse/memory/main.py` | `add`: 单条 verbatim 语义去重 (embed_batch 后, add_batch 前, threshold=0.95); `search`: 签名加 3 参数 + 返回类型 `list \| dict`, 两分支重构为 if/else 收敛到统一后处理块 (遗忘加权 → token 预算裁剪 → prompt 注入) |
| `tests/unit/test_add_semantic_dedup.py` | 新增 3 测试 (同文本跳过 / 不同文本不去重 / 不同用户不去重) |
| `tests/unit/test_search_enhanced.py` | 新增 5 测试 (forgetting / token_budget / inject_prompt / 默认返回 list / 无 LLM forgetting) |

## 实现细节 (与原计划的偏差)

### 偏差 1: apply_strength 返回 dataclass, 非 dict
- **任务描述**: "`ForgettingManager.apply_strength` takes `list[dict]` and returns `list[dict]` with `final_score` added"
- **实际代码** (`src/septmuse/retrieval/forgetting.py:61`): 返回 `list[StrengthWeightedResult]` (dataclass)
- **问题**: 测试 `assert "score" in results[0]` 对 dataclass 会 `TypeError` (dataclass 无 `__contains__`)
- **修正**: forgetting 分支把 `StrengthWeightedResult` 转回 dict (含 `id`/`memory`/`score`/`final_score`/`relevance`/`strength`/`metadata`), 与 `recall` 的属性访问模式 (`w.id`/`w.memory`/`w.final_score`, main.py:805-811) 对齐

### 偏差 2: token_budget 用用户传入的预算, 非 self.token_budget 内置预算
- **任务描述代码**: `budgeted = self.token_budget.fit(items)` — 但 `self.token_budget` 在 `__init__` 初始化为 `TokenBudget(budget=2000)` (main.py:198)
- **问题**: 用户传 `token_budget=50`, 但 `self.token_budget.fit` 用内置 2000 预算, 不裁剪 → 测试失败
- **修正**: `TokenBudget(budget=token_budget).fit(items)` — 用用户指定预算新建 TokenBudget 实例

## 测试结果

### 新测试 (8/8 通过)
```
tests/unit/test_add_semantic_dedup.py::test_add_verbatim_semantic_dedup_identical PASSED
tests/unit/test_add_semantic_dedup.py::test_add_verbatim_different_not_deduped PASSED
tests/unit/test_add_semantic_dedup.py::test_add_verbatim_different_users_not_deduped PASSED
tests/unit/test_search_enhanced.py::test_search_forgetting_param PASSED
tests/unit/test_search_enhanced.py::test_search_token_budget PASSED
tests/unit/test_search_enhanced.py::test_search_inject_prompt PASSED
tests/unit/test_search_enhanced.py::test_search_default_returns_list PASSED
tests/unit/test_search_enhanced.py::test_search_no_llm_forgetting_works PASSED
8 passed in 2.05s
```

### 回归测试 (test_memory.py + test_memory_orchestration.py)
```
6 failed, 39 passed, 3 warnings in 12.60s
```
- **6 failed = 全部 pre-existing OpenAI 基线** (与任务预期一致):
  - `TestResolveEmbedderOpenAI` × 3: OpenAI embedder 回退到 OnnxEmbedder (无 API key)
  - `TestResolveLLMBaseUrl` × 3: OpenAI LLM 未解析 (无 key, 返回 None)
- **test_memory_orchestration.py 全部通过** (无 FAILED 行) — recall/remember/forget 委托链未受影响
- 这 6 个失败测试 embedder/LLM *解析* 路径, 不触及我修改的 `search`/`add` 方法

## Lint
```
ruff check src/septmuse/memory/main.py → All checks passed!
```

## 顾虑

1. **search 返回类型变为 `list | dict`**: 当 `inject_prompt=True` 返回 dict, 否则返回 list (向后兼容)。`recall()` (main.py:802) 调用 `self.search(query, user_id=, top_k=, recipe=)` 不传新参数 (默认 False/None), 仍返回 list, recall 自己做 forgetting/budget/injection, 不受影响。

2. **语义去重仅 verbatim + 单条**: 多条文本仍走 batch MD5 去重 (store 层)。threshold=0.95 极严格 (几乎相同才跳过), HashEmbedder 下同文本 cosine=1.0 必触发。不同文本 (如 "I like Python" vs "I live in Tokyo") HashEmbedder cosine ~0.29, 不误杀。

3. **Windows qdrant 关闭噪声**: 测试输出尾部有大量 `ImportError: sys.meta_path is None` (QdrantClient.__del__ 在解释器关闭时触发), 与测试结果无关, 不影响 pass/fail 判定。

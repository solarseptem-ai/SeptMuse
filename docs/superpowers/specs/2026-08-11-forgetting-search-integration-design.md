# Forgetting + Search 高级参数接入设计

> 日期: 2026-08-11
> 范围: A+B（ForgettingRetriever/MemoryStrength 空转修复 + search 高级参数）
> 状态: 已确认，待实施

## 1. 问题

### A 类：ForgettingRetriever + MemoryStrength 空转

`ForgettingRetriever`（`retrieval/forgetting.py:48`）有完整的 `apply_strength()` 方法（`final_score = 0.7*relevance + 0.3*strength`）、`rehearse()`、`archive_stale()`，在 `_init_v2_components()`（`main.py:288-294`）中初始化为 `self.forgetting`。

但 `search()` 和 `recall()` **从不调用** `self.forgetting`。遗忘曲线（SeptMuse 核心差异化特性）完全空转。

`MemoryStrength` 模型（`models/strength.py`）的 `decay()`/`rehearse()`/`should_rehearse()`/`should_archive()` 完整实现，但检索评分只用 vector + BM25 + entity boost，**strength 加权从未接入**。

### B 类：search() 缺少高级参数

`Memory.search()`（`main.py:550`）只支持 `hybrid`/`reranker`/`recipe`/`explain`/`filters`/`search_filter`/`show_expired`。

缺失参数（测试已期望，实现为空）：
- `forgetting=True` — 遗忘加权（`test_search_enhanced.py:18`）
- `token_budget=N` — 预算裁剪（`test_search_enhanced.py:27`）
- `inject_prompt=True` — 返回含 `injected_prompt` 的 dict（`test_search_enhanced.py:37`）
- `hyde` / `query_rewrite` — 查询改写（async search 声明了但 sync 不支持）

async `search()`（`async_main.py:219`）已声明这些参数，但委托 sync `Memory.search()` 时会 TypeError（sync 不接受）。

## 2. 方案

后处理式扩展 `search()`：在现有检索 + reranker 逻辑之后、return 之前，插入后处理链。所有新参数默认 False/None，现有调用方零影响。

## 3. 设计

### 3.1 search() 新签名

```python
def search(
    self,
    query: str,
    *,
    user_id: str,
    session_id: str | None = None,
    top_k: int | None = None,
    threshold: float | None = None,
    hybrid: bool = True,
    reranker: str | None = None,
    recipe: str | None = None,
    explain: bool = False,
    filters: dict[str, Any] | None = None,
    search_filter: dict[str, Any] | None = None,
    show_expired: bool = False,
    # 新增：B 类高级参数
    forgetting: bool = False,
    token_budget: int | None = None,
    inject_prompt: bool = False,
    hyde: bool = False,
    query_rewrite: bool = False,
) -> list[dict[str, Any]] | dict[str, Any]:
```

### 3.2 后处理管道流

参数执行顺序：

```
[0] hyde/query_rewrite → log（不改 query）
[1] 现有检索 (hybrid/纯向量 + reranker + expired 过滤)
[2] forgetting=True → apply_strength() → 加权 + 测试效应
[3] token_budget=N → TokenBudget.fit() → 裁剪
[4] inject_prompt=True → 返回 dict（提前 return）
[5] 默认 → return list[:tk]
```

后处理不影响 HybridRetriever 内部逻辑和缓存 key。所有新参数默认 False/None，现有调用方零影响。

`forgetting` 在 `token_budget` 之前执行：先加权评分，再按新分数裁剪。

### 3.3 forgetting 接入

调用链：

```
search(forgetting=True)
    ↓
results = [...现有检索结果 dict 列表...]
    ↓
weighted = self.forgetting.apply_strength(results, user_id=user_id)
    ↓ 返回 list[StrengthWeightedResult]
results = [_strength_to_dict(w) for w in weighted]
    ↓ 转回 dict，保留 "final_score" 字段
```

#### StrengthWeightedResult → dict 转换

```python
def _strength_to_dict(w: StrengthWeightedResult) -> dict[str, Any]:
    return {
        "id": w.id,
        "memory": w.memory,
        "score": w.final_score,      # final_score 覆盖原 score（加权后）
        "final_score": w.final_score,  # 额外保留 final_score 字段
        "strength": w.strength,
        "metadata": w.metadata,
        "created_at": w.created_at,
    }
```

`score` 字段更新为 `final_score` 值（而非保留原 relevance score），这样下游 `token_budget` 裁剪按加权后分数排序。同时保留 `final_score` 字段供测试断言。

#### apply_strength 副作用

`apply_strength()` 内部做两件事：
1. 计算：`final_score = 0.7 * relevance + 0.3 * decayed_strength`
2. 测试效应：更新 `last_accessed = now` + `access_count += 1` + 写回 `update_strength()`

`search(forgetting=True)` 有写副作用（更新 MemoryStrength 表）。这是正确行为——检索到的记忆应该延缓衰减（测试效应），符合架构文档 §6.2。

#### self.forgetting is None 降级

当 `_init_v2_components()` 中 ForgettingRetriever 初始化失败时 `self.forgetting = None`。此时 `forgetting=True` 不抛异常，结果不变，log warning。遵循项目"吞错降级"模式（与 `record_access`、`entity_store` 一致）。

#### 归档记忆过滤

`apply_strength()` 内部已处理：`if strength.archived: continue`（跳过归档记忆）。`forgetting=True` 时归档记忆自动被过滤。

### 3.4 token_budget 接入

```python
if token_budget is not None and results:
    from septmuse.retrieval.token_budget import BudgetItem, TokenBudget

    items = [
        BudgetItem(text=r.get("memory", ""), score=r.get("score", 0.0), id=r.get("id"))
        for r in results
    ]
    budget = TokenBudget(budget=token_budget)
    fitted = budget.fit(items)
    chosen_ids = {item.id for item in fitted.items}
    results = [r for r in results if r.get("id") in chosen_ids]
```

- 不复用 `self.token_budget`（默认 2000 的单例），每次创建临时实例
- `estimate_tokens = len(text) // 4`（chars/4），与测试断言 `sum(len(r["memory"]) // 4)` 一致
- 裁剪在 `forgetting` 之后执行，按加权后分数排序裁剪
- 不依赖 `self.token_budget` 是否初始化成功

### 3.5 inject_prompt 接入

```python
if inject_prompt:
    memory_texts = "\n".join(f"- {r.get('memory', '')}" for r in results)
    injected_prompt = f"<memory>\n{memory_texts}\n</memory>"
    return {"results": results, "injected_prompt": injected_prompt}

return results  # 默认返回 list
```

- `injected_prompt` 格式与 `recall()` 中的 `<memory>` 块一致（不含 `working_memory` 前缀，因 `search()` 不涉及工作记忆）
- 在 `forgetting` + `token_budget` 之后执行，注入的文本是已加权已裁剪的结果
- 返回 dict 后提前 return，不再走 `return results[:tk]`

### 3.6 hyde / query_rewrite no-op

```python
if hyde:
    logger.info("hyde_not_enabled", hint="requires LLM + prompt engineering (future)")
if query_rewrite:
    logger.info("query_rewrite_not_enabled", hint="requires LLM (future)")
```

在 search() 开头（检索之前）执行，但不改变 query。只 log，不抛异常，不改变任何行为。为未来 LLM 实现预留接口。

### 3.7 recall() 变更

`recall()`（`main.py:942`）传 `forgetting=True` 给 search()：

```python
forgetting = kwargs.pop("forgetting", True)  # 默认 True，调用方可覆盖
memories = self.search(
    query, user_id=user_id, session_id=session_id,
    top_k=top_k, recipe=recipe, forgetting=forgetting, **kwargs
)
```

- recall() 只传 `forgetting=True`，不传 `token_budget`/`inject_prompt`——recall() 有自己的 token_budget 逻辑和 prompt 组装逻辑
- 用 `kwargs.pop("forgetting", True)` 防止重复 key

### 3.8 async search() 自动修复

async `search()`（`async_main.py:219`）已声明 `forgetting`/`token_budget`/`inject_prompt`/`hyde`/`query_rewrite` 参数。sync `search()` 加这些参数后，async 委托自动通——`has_advanced` 为 True 时走 `to_thread(self._sync.search, ...)` 传全部参数，sync 现在接受它们，不再 TypeError。

### 3.9 错误处理

全部新后处理逻辑用 `try/except` 包裹：

```python
if forgetting and self.forgetting is not None:
    try:
        weighted = self.forgetting.apply_strength(results, user_id=user_id)
        results = [_strength_to_dict(w) for w in weighted]
    except Exception as e:
        logger.warning("forgetting_apply_failed", error=str(e))

if token_budget is not None and results:
    try:
        budget = TokenBudget(budget=token_budget)
        fitted = budget.fit(items)
        ...
    except Exception as e:
        logger.warning("token_budget_failed", error=str(e))
```

后处理失败不影响检索本身——用户至少拿到原始检索结果。

## 4. 测试

### 4.1 现有 failing 测试 → 将变为 passing

| 测试 | 文件 | 预期 |
|------|------|------|
| `test_search_forgetting_param` | `test_search_enhanced.py:18` | `search(forgetting=True)` 返回有 `score` 或 `final_score` 的结果 |
| `test_search_token_budget` | `test_search_enhanced.py:27` | `search(token_budget=50)` 总 chars/4 ≤ 50 |
| `test_search_inject_prompt` | `test_search_enhanced.py:37` | `search(inject_prompt=True)` 返回 dict 含 `results` + `injected_prompt` |
| `test_search_default_returns_list` | `test_search_enhanced.py:46` | 默认返回 list（已 passing，不受影响） |
| `test_search_no_llm_forgetting_works` | `test_search_enhanced.py:53` | 无 LLM 时 `search(forgetting=True)` 正常工作 |

5 个 pre-existing failing 测试将变为 passing。

### 4.2 新增测试

- `test_search_forget_then_budget` — forgetting 加权后按 final_score 裁剪
- `test_search_inject_with_forgetting` — inject_prompt + forgetting 组合
- `test_search_hyde_noop` — hyde=True 不报错不改变结果
- `test_search_query_rewrite_noop` — query_rewrite=True 不报错不改变结果
- `test_recall_passes_forgetting` — recall() 内部传 forgetting=True

## 5. 改动文件

| 文件 | 改动 |
|------|------|
| `src/septmuse/memory/main.py` | `search()` 加 5 参数 + 后处理逻辑；`recall()` 传 `forgetting=True` |
| `src/septmuse/memory/async_main.py` | 无需改动（async search 已声明参数，sync 修复后自动通） |
| `tests/unit/test_search_enhanced.py` | 无改动（现有测试将 pass） |
| `tests/unit/test_search_forget_integration.py` | 新增（组合场景测试） |

## 6. 不在范围内

以下能力缺口留给后续迭代：

- **C 类**：pipeline cognify LLM 抽取 placeholder、synchronizer reconcile 空壳、conflict LLM 判定 TODO
- **D 类**：MCP 4 工具 stub、CLI mcp serve、deps.py P1/P2 服务、pgvector filters、社区检测
- **E 类**：治理层（权限 + 审计）接入 Memory facade
- **hyde/query_rewrite 实际实现**：本次只接受参数 + no-op，未来需 LLM prompt 工程

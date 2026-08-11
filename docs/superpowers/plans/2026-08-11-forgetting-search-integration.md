# Forgetting + Search 高级参数接入 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 ForgettingRetriever + MemoryStrength 接入 search/recall 管道，并为 search() 添加 forgetting/token_budget/inject_prompt/hyde/query_rewrite 参数。

**Architecture:** 后处理式扩展——在现有 search() 检索+reranker 之后、return 之前，插入 forgetting→token_budget→inject_prompt 后处理链。所有新参数默认 False/None，现有调用方零影响。

**Tech Stack:** Python 3.10+, SQLModel, pytest, ruff (line-length 120)

## Global Constraints

- **PYTHONPATH=src** — 测试必须设此环境变量，否则 import septmuse 失败
- **测试环境** — `SEPTMUSE_EMBEDDER=hash`, `SEPTMUSE_TOKENIZER=space`（conftest 强制）
- **ruff line-length 120** — `select = ["E","F","I","W","UP","B","SIM","RUF"]`
- **禁止改现有测试代码** — 仅可新增测试用例
- **Windows ruff format bug** — 不用 `ruff format <file>`，用 `ruff format --check`（只读）或 `ruff check --fix`
- **score 统一为相似度 [0,1]** — forgetting 的 final_score 也遵守此约定

## 关键事实（实施前必读）

- `StrengthWeightedResult` dataclass 字段: `id: str`, `memory: str`, `relevance: float`, `strength: float`, `final_score: float`, `metadata: dict` — **无 `created_at`**，需从原始 results dict 查找
- `apply_strength()` 输入 `list[dict]`，输出 `list[StrengthWeightedResult]`（已按 final_score 降序排序，归档记忆已过滤）
- `TypedMemoryStore` 有 `get_or_create_strength()`(:550)、`update_strength()`(:567)、`get_all_strengths()`(:596) — forgetting 可正常工作
- `TokenBudget.estimate_tokens = len(text) // 4`（chars/4）— 与测试断言一致
- async `search()` (`async_main.py:219`) 已声明全部 5 参数并传给 sync(:265-269) — sync 修复后 async 自动通
- `search()` 有两个 return 点：hybrid 分支(`main.py:638`) 和纯向量分支(`main.py:694`)

---

### Task 1: search() 新增参数 + hyde/query_rewrite no-op

**Files:**
- Modify: `src/septmuse/memory/main.py:550-565`（search 签名）
- Modify: `src/septmuse/memory/main.py:584`（recipe 处理后，插入 no-op）
- Test: `tests/unit/test_search_enhanced.py`（新增 2 个测试）

**Interfaces:**
- Produces: `Memory.search()` 新增 5 个 keyword-only 参数: `forgetting: bool = False`, `token_budget: int | None = None`, `inject_prompt: bool = False`, `hyde: bool = False`, `query_rewrite: bool = False`；返回类型 `list[dict] | dict`

- [ ] **Step 1: 写 hyde/query_rewrite no-op 测试**

在 `tests/unit/test_search_enhanced.py` 末尾追加:

```python
def test_search_hyde_noop(memory):
    """search(hyde=True) 不报错, 返回与默认一致。"""
    memory.add("test fact", user_id="alice", infer=False)
    results = memory.search("test", user_id="alice", hyde=True)
    assert isinstance(results, list)
    assert len(results) > 0


def test_search_query_rewrite_noop(memory):
    """search(query_rewrite=True) 不报错, 返回与默认一致。"""
    memory.add("test fact", user_id="alice", infer=False)
    results = memory.search("test", user_id="alice", query_rewrite=True)
    assert isinstance(results, list)
    assert len(results) > 0
```

- [ ] **Step 2: 运行测试验证失败**

```
$env:PYTHONPATH = "src"; $env:SEPTMUSE_EMBEDDER = "hash"; $env:SEPTMUSE_TOKENIZER = "space"; pytest tests/unit/test_search_enhanced.py::test_search_hyde_noop tests/unit/test_search_enhanced.py::test_search_query_rewrite_noop -v
```

Expected: FAIL with `TypeError: search() got an unexpected keyword argument 'hyde'`

- [ ] **Step 3: 修改 search() 签名 + 加 no-op**

在 `src/septmuse/memory/main.py` 的 `search()` 方法签名中，在 `show_expired: bool = False,` 之后添加 5 个新参数:

```python
        show_expired: bool = False,
        # B 类高级参数
        forgetting: bool = False,
        token_budget: int | None = None,
        inject_prompt: bool = False,
        hyde: bool = False,
        query_rewrite: bool = False,
    ) -> list[dict[str, Any]] | dict[str, Any]:
```

在 recipe 处理之后（约 `main.py:591` 之后，`tk = top_k...` 之前）插入 no-op:

```python
        if hyde:
            logger.info("hyde_not_enabled", hint="requires LLM + prompt engineering (future)")
        if query_rewrite:
            logger.info("query_rewrite_not_enabled", hint="requires LLM (future)")
```

- [ ] **Step 4: 运行测试验证通过**

```
$env:PYTHONPATH = "src"; $env:SEPTMUSE_EMBEDDER = "hash"; $env:SEPTMUSE_TOKENIZER = "space"; pytest tests/unit/test_search_enhanced.py::test_search_hyde_noop tests/unit/test_search_enhanced.py::test_search_query_rewrite_noop -v
```

Expected: PASS

- [ ] **Step 5: 验证默认行为不变**

```
$env:PYTHONPATH = "src"; $env:SEPTMUSE_EMBEDDER = "hash"; $env:SEPTMUSE_TOKENIZER = "space"; pytest tests/unit/test_search_enhanced.py::test_search_default_returns_list -v
```

Expected: PASS（默认返回 list）

- [ ] **Step 6: ruff 检查**

```
ruff check src/septmuse/memory/main.py tests/unit/test_search_enhanced.py
```

Expected: No errors

- [ ] **Step 7: 提交**

```bash
git add src/septmuse/memory/main.py tests/unit/test_search_enhanced.py
git commit -m "feat(search): add forgetting/token_budget/inject_prompt/hyde/query_rewrite params + no-op for hyde/query_rewrite"
```

---

### Task 2: forgetting 后处理

**Files:**
- Modify: `src/septmuse/memory/main.py:638`（hybrid return 点）
- Modify: `src/septmuse/memory/main.py:694`（纯向量 return 点）
- Add: `src/septmuse/memory/main.py`（新增 `_apply_search_post_processing` 方法）
- Test: `tests/unit/test_search_enhanced.py::test_search_forgetting_param`（已存在，当前 failing）
- Test: `tests/unit/test_search_enhanced.py::test_search_no_llm_forgetting_works`（已存在，当前 failing）

**Interfaces:**
- Consumes: `ForgettingRetriever.apply_strength(results: list[dict], *, user_id: str) -> list[StrengthWeightedResult]`（`retrieval/forgetting.py:62`）
- Consumes: `StrengthWeightedResult` dataclass 字段: `id`, `memory`, `relevance`, `strength`, `final_score`, `metadata`（无 `created_at`）
- Produces: `_apply_search_post_processing()` 方法，签名: `(results, *, user_id, forgetting, token_budget, inject_prompt) -> list[dict] | dict`

- [ ] **Step 1: 运行现有失败测试确认状态**

```
$env:PYTHONPATH = "src"; $env:SEPTMUSE_EMBEDDER = "hash"; $env:SEPTMUSE_TOKENIZER = "space"; pytest tests/unit/test_search_enhanced.py::test_search_forgetting_param tests/unit/test_search_enhanced.py::test_search_no_llm_forgetting_works -v
```

Expected: FAIL（forgetting 参数被接受但无后处理，结果有 `score` 但无 `final_score` — 实际可能 pass 因为原 results 有 `score`。确认行为后继续）

- [ ] **Step 2: 添加 `_apply_search_post_processing` 方法**

在 `Memory` 类中，`_invalidate_search_cache` 方法之后、`search` 方法之前，添加:

```python
    def _apply_search_post_processing(
        self,
        results: list[dict[str, Any]],
        *,
        user_id: str,
        forgetting: bool = False,
        token_budget: int | None = None,
        inject_prompt: bool = False,
    ) -> list[dict[str, Any]] | dict[str, Any]:
        """search 后处理链: forgetting → token_budget → inject_prompt。

        后处理失败不阻塞检索——吞错降级, 返回原始结果。
        """
        # [1] forgetting: 强度加权 + 测试效应
        if forgetting and self.forgetting is not None:
            try:
                weighted = self.forgetting.apply_strength(results, user_id=user_id)
                original_map = {r.get("id"): r for r in results}
                results = [
                    {
                        "id": w.id,
                        "memory": w.memory,
                        "score": w.final_score,
                        "final_score": w.final_score,
                        "strength": w.strength,
                        "metadata": w.metadata,
                        "created_at": original_map.get(w.id, {}).get("created_at"),
                    }
                    for w in weighted
                ]
            except Exception as e:
                logger.warning("forgetting_apply_failed", error=str(e))
        elif forgetting and self.forgetting is None:
            logger.warning("forgetting_component_unavailable")

        # [2] token_budget: 按加权后 score 贪心裁剪
        if token_budget is not None and results:
            try:
                from septmuse.retrieval.token_budget import BudgetItem, TokenBudget

                items = [
                    BudgetItem(text=r.get("memory", ""), score=r.get("score", 0.0), id=r.get("id"))
                    for r in results
                ]
                budget = TokenBudget(budget=token_budget)
                fitted = budget.fit(items)
                chosen_ids = {item.id for item in fitted.items}
                results = [r for r in results if r.get("id") in chosen_ids]
            except Exception as e:
                logger.warning("token_budget_failed", error=str(e))

        # [3] inject_prompt: 返回 dict (提前 return)
        if inject_prompt:
            memory_texts = "\n".join(f"- {r.get('memory', '')}" for r in results)
            injected_prompt = f"<memory>\n{memory_texts}\n</memory>"
            return {"results": results, "injected_prompt": injected_prompt}

        return results
```

- [ ] **Step 3: 接入 hybrid 分支 return 点**

在 `main.py` 的 hybrid 分支中（约 `main.py:636-638`），将:

```python
            if not show_expired:
                hybrid_return = [r for r in hybrid_return if not _is_expired(r.get("metadata"))]
            return hybrid_return[:tk]
```

改为:

```python
            if not show_expired:
                hybrid_return = [r for r in hybrid_return if not _is_expired(r.get("metadata"))]
            return self._apply_search_post_processing(
                hybrid_return[:tk],
                user_id=user_id,
                forgetting=forgetting,
                token_budget=token_budget,
                inject_prompt=inject_prompt,
            )
```

- [ ] **Step 4: 接入纯向量分支 return 点**

在纯向量分支中（约 `main.py:692-694`），将:

```python
        logger.info("memory_search_done", user_id=user_id, query=query[:50], hits=len(results))
        if not show_expired:
            results = [r for r in results if not _is_expired(r.get("metadata"))]
        return results[:tk]
```

改为:

```python
        logger.info("memory_search_done", user_id=user_id, query=query[:50], hits=len(results))
        if not show_expired:
            results = [r for r in results if not _is_expired(r.get("metadata"))]
        return self._apply_search_post_processing(
            results[:tk],
            user_id=user_id,
            forgetting=forgetting,
            token_budget=token_budget,
            inject_prompt=inject_prompt,
        )
```

- [ ] **Step 5: 运行 forgetting 测试验证通过**

```
$env:PYTHONPATH = "src"; $env:SEPTMUSE_EMBEDDER = "hash"; $env:SEPTMUSE_TOKENIZER = "space"; pytest tests/unit/test_search_enhanced.py::test_search_forgetting_param tests/unit/test_search_enhanced.py::test_search_no_llm_forgetting_works -v
```

Expected: PASS

- [ ] **Step 6: 运行 token_budget 测试验证通过**

```
$env:PYTHONPATH = "src"; $env:SEPTMUSE_EMBEDDER = "hash"; $env:SEPTMUSE_TOKENIZER = "space"; pytest tests/unit/test_search_enhanced.py::test_search_token_budget -v
```

Expected: PASS

- [ ] **Step 7: 运行 inject_prompt 测试验证通过**

```
$env:PYTHONPATH = "src"; $env:SEPTMUSE_EMBEDDER = "hash"; $env:SEPTMUSE_TOKENIZER = "space"; pytest tests/unit/test_search_enhanced.py::test_search_inject_prompt -v
```

Expected: PASS

- [ ] **Step 8: 运行全部 search_enhanced 测试**

```
$env:PYTHONPATH = "src"; $env:SEPTMUSE_EMBEDDER = "hash"; $env:SEPTMUSE_TOKENIZER = "space"; pytest tests/unit/test_search_enhanced.py -v
```

Expected: 全部 PASS（7 个测试）

- [ ] **Step 9: ruff 检查**

```
ruff check src/septmuse/memory/main.py
```

Expected: No errors

- [ ] **Step 10: 提交**

```bash
git add src/septmuse/memory/main.py
git commit -m "feat(search): wire forgetting/token_budget/inject_prompt post-processing into search()"
```

---

### Task 3: recall() 传 forgetting=True

**Files:**
- Modify: `src/septmuse/memory/main.py:942`（recall 中的 search 调用）
- Test: `tests/unit/test_search_forget_integration.py`（新建）

**Interfaces:**
- Consumes: Task 2 的 `_apply_search_post_processing()` + search() 新参数
- Produces: `recall()` 默认传 `forgetting=True`，调用方可通过 `forgetting=False` 覆盖

- [ ] **Step 1: 写 recall forgetting 测试**

创建 `tests/unit/test_search_forget_integration.py`:

```python
"""forgetting + search 组合集成测试。"""
from __future__ import annotations

import os

import pytest


@pytest.fixture
def memory(tmp_path):
    os.environ["SEPTMUSE_EMBEDDER"] = "hash"
    os.environ["SEPTMUSE_TOKENIZER"] = "space"
    from septmuse.configs import MemoryConfig
    from septmuse.memory.main import Memory
    return Memory(config=MemoryConfig(db_path=str(tmp_path / "test.db")))


def test_recall_passes_forgetting(memory):
    """recall() 内部传 forgetting=True, 返回结果有 final_score。"""
    memory.add("alice likes python", user_id="alice", infer=False)
    result = memory.recall("python", user_id="alice")
    assert "memories" in result
    if len(result["memories"]) > 0:
        assert "final_score" in result["memories"][0] or "score" in result["memories"][0]
```

- [ ] **Step 2: 运行测试验证失败**

```
$env:PYTHONPATH = "src"; $env:SEPTMUSE_EMBEDDER = "hash"; $env:SEPTMUSE_TOKENIZER = "space"; pytest tests/unit/test_search_forget_integration.py::test_recall_passes_forgetting -v
```

Expected: 可能 PASS 或 FAIL — recall() 目前不传 forgetting，但原结果有 `score`，断言可能通过。如果 PASS，调整断言改为更严格:

```python
        assert "final_score" in result["memories"][0]
```

再运行确认 FAIL（recall 不传 forgetting → 无 final_score）。

- [ ] **Step 3: 修改 recall() 传 forgetting=True**

在 `src/septmuse/memory/main.py` 的 `recall()` 方法中（约 `main.py:942`），将:

```python
        memories = self.search(
            query, user_id=user_id, session_id=session_id, top_k=top_k, recipe=recipe, **kwargs
        )
```

改为:

```python
        forgetting = kwargs.pop("forgetting", True)
        memories = self.search(
            query, user_id=user_id, session_id=session_id,
            top_k=top_k, recipe=recipe, forgetting=forgetting, **kwargs
        )
```

- [ ] **Step 4: 运行测试验证通过**

```
$env:PYTHONPATH = "src"; $env:SEPTMUSE_EMBEDDER = "hash"; $env:SEPTMUSE_TOKENIZER = "space"; pytest tests/unit/test_search_forget_integration.py::test_recall_passes_forgetting -v
```

Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add src/septmuse/memory/main.py tests/unit/test_search_forget_integration.py
git commit -m "feat(recall): pass forgetting=True to search() for V2 orchestration"
```

---

### Task 4: 组合场景集成测试

**Files:**
- Test: `tests/unit/test_search_forget_integration.py`（追加）

**Interfaces:**
- Consumes: Task 2 的 post-processing 链 + Task 3 的 recall forgetting

- [ ] **Step 1: 写组合场景测试**

在 `tests/unit/test_search_forget_integration.py` 追加:

```python
def test_search_forget_then_budget(memory):
    """forgetting 加权后按 final_score 裁剪到 token 预算内。"""
    for i in range(10):
        memory.add(f"fact number {i} about python programming", user_id="alice", infer=False)
    results = memory.search(
        "python", user_id="alice", forgetting=True, token_budget=20
    )
    total = sum(len(r["memory"]) // 4 for r in results)
    assert total <= 20
    if results:
        assert "final_score" in results[0]


def test_search_inject_with_forgetting(memory):
    """inject_prompt + forgetting 组合返回 dict 含 final_score。"""
    memory.add("alice knows python", user_id="alice", infer=False)
    result = memory.search(
        "python", user_id="alice", forgetting=True, inject_prompt=True
    )
    assert isinstance(result, dict)
    assert "results" in result
    assert "injected_prompt" in result
    assert "<memory>" in result["injected_prompt"]
    if result["results"]:
        assert "final_score" in result["results"][0]


def test_search_forgetting_degrades_gracefully(memory):
    """forgetting=True 时 self.forgetting 可用, 结果有 final_score。"""
    memory.add("some fact about coding", user_id="alice", infer=False)
    results = memory.search("coding", user_id="alice", forgetting=True)
    assert len(results) > 0
    assert "final_score" in results[0]
```

- [ ] **Step 2: 运行测试验证通过**

```
$env:PYTHONPATH = "src"; $env:SEPTMUSE_EMBEDDER = "hash"; $env:SEPTMUSE_TOKENIZER = "space"; pytest tests/unit/test_search_forget_integration.py -v
```

Expected: 全部 PASS

- [ ] **Step 3: 提交**

```bash
git add tests/unit/test_search_forget_integration.py
git commit -m "test: add forgetting+budget/inject combo integration tests"
```

---

### Task 5: 全量验证 + ruff

**Files:**
- 无代码改动，仅验证

- [ ] **Step 1: 运行 search 相关全部测试**

```
$env:PYTHONPATH = "src"; $env:SEPTMUSE_EMBEDDER = "hash"; $env:SEPTMUSE_TOKENIZER = "space"; pytest tests/unit/test_search_enhanced.py tests/unit/test_search_forget_integration.py -v
```

Expected: 全部 PASS

- [ ] **Step 2: 运行 recall/v2 相关测试（确认不回归）**

```
$env:PYTHONPATH = "src"; $env:SEPTMUSE_EMBEDDER = "hash"; $env:SEPTMUSE_TOKENIZER = "space"; pytest tests/unit/test_memory_orchestration.py tests/unit/test_v2_memory.py tests/unit/test_async_v2_orchestration.py -v
```

Expected: 无新增失败（pre-existing failures 除外）

- [ ] **Step 3: 运行 async search 测试**

```
$env:PYTHONPATH = "src"; $env:SEPTMUSE_EMBEDDER = "hash"; $env:SEPTMUSE_TOKENIZER = "space"; pytest tests/unit/test_async_memory.py tests/unit/test_async_add_search.py -v
```

Expected: 无新增失败

- [ ] **Step 4: ruff check 全量**

```
ruff check src/septmuse/memory/main.py tests/unit/test_search_enhanced.py tests/unit/test_search_forget_integration.py
```

Expected: No errors

- [ ] **Step 5: ruff format check（只读，不触发 Windows bug）**

```
ruff format --check src/septmuse/memory/main.py tests/unit/test_search_enhanced.py tests/unit/test_search_forget_integration.py
```

Expected: No changes needed

- [ ] **Step 6: 运行 smoke test 确认不回归**

```
$env:PYTHONPATH = "src"; $env:SEPTMUSE_EMBEDDER = "hash"; $env:SEPTMUSE_TOKENIZER = "space"; pytest tests/unit/test_memory.py -q --tb=no
```

Expected: 与基线一致（1319 passed, 16 failed pre-existing, 23 skipped），或新增 pass 数 ≥ 5（原 5 个 search_enhanced 失败转 pass）

- [ ] **Step 7: 确认 pre-existing failures 减少**

```
$env:PYTHONPATH = "src"; $env:SEPTMUSE_EMBEDDER = "hash"; $env:SEPTMUSE_TOKENIZER = "space"; pytest tests/unit/test_search_enhanced.py -q --tb=no
```

Expected: 7 passed（原 5 failed + 2 passed → 7 passed）

---

## Self-Review

### Spec coverage

| Spec 节 | 对应 Task |
|---------|-----------|
| §3.1 search() 新签名 | Task 1 Step 3 |
| §3.2 后处理管道流 [0] hyde/query_rewrite | Task 1 Step 3 |
| §3.2 后处理管道流 [2] forgetting | Task 2 Step 2-4 |
| §3.2 后处理管道流 [3] token_budget | Task 2 Step 2（在 helper 中） |
| §3.2 后处理管道流 [4] inject_prompt | Task 2 Step 2（在 helper 中） |
| §3.3 forgetting 接入 + StrengthWeightedResult→dict | Task 2 Step 2 |
| §3.4 token_budget 接入 | Task 2 Step 2（在 helper 中） |
| §3.5 inject_prompt 接入 | Task 2 Step 2（在 helper 中） |
| §3.6 hyde/query_rewrite no-op | Task 1 Step 3 |
| §3.7 recall() 传 forgetting=True | Task 3 Step 3 |
| §3.8 async search() 自动修复 | Task 5 Step 3（验证不回归） |
| §3.9 错误处理 try/except | Task 2 Step 2（在 helper 中） |
| §4.1 现有 failing 测试 → passing | Task 2 Step 5-8, Task 5 Step 7 |
| §4.2 新增测试 | Task 1 Step 1, Task 3 Step 1, Task 4 Step 1 |

### Placeholder scan

无 TBD/TODO/"implement later"。全部代码步骤包含实际代码。✅

### Type consistency

- `_apply_search_post_processing` 签名在 Task 2 定义，在 Task 2 Step 3-4 调用 — 参数名一致 ✅
- `forgetting`/`token_budget`/`inject_prompt` 参数名在 search() 签名、helper、recall() 中一致 ✅
- `StrengthWeightedResult` 字段引用 `w.id`/`w.memory`/`w.final_score`/`w.strength`/`w.metadata` — 与 dataclass 定义一致 ✅

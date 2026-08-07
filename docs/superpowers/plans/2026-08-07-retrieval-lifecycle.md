# 检索质量 + 生命周期 4 子项实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development.

**Goal:** HyDE 假设文档检索 + 上下文感知查询改写 + 遗忘曲线参数化 + 压缩归档 facade

**Spec:** `docs/superpowers/specs/2026-08-07-retrieval-lifecycle-design.md`

## Global Constraints
- 包名 `septmuse`, src/ layout, `PYTHONPATH=src` 运行测试
- ruff line-length 120, **禁用 `ruff format`**, 只用 `ruff check`
- 代码注释中文, **不 git commit**
- 无 LLM 时全部降级, 现有测试不改断言

## 已确认接口

```python
# MemoryStrength.decay (models/strength.py:93)
def decay(self, now: datetime | None = None) -> float
# 公式: strength * exp(-elapsed / stability), stability = effective_base * S_FACTOR
# 参数化: 加 half_life_days: float | None = None, 非 None 时 stability = half_life_days * 86400 / ln(2)

# ForgettingRetriever (retrieval/forgetting.py:48)
def __init__(self, typed_store: TypedMemoryStore) -> None
def apply_strength(self, results, *, user_id, now=None) -> list[StrengthWeightedResult]

# Memory.search (memory/main.py) — 已有 forgetting/token_budget/inject_prompt 参数
# Memory.recall (memory/main.py) — 委托 search + 遗忘 + 预算 + block

# Summarizer (evolution/summarizer.py:45)
def __init__(self, store, typed_store, llm=None)
def compress(self, *, user_id, mode="static", buffer_size=20) -> dict

# EpisodicMemory (memory/episodic.py)
# self.episodic.get_episodes(user_id=, limit=) -> list[EpisodicEvent]
```

---

## Task 1: HyDE 假设文档检索

**Files:** Create `src/septmuse/prompts/hyde.py`, Modify `src/septmuse/memory/main.py` (search), Create `tests/unit/test_hyde.py`

- [ ] Step 1: 写失败测试 `tests/unit/test_hyde.py`
- [ ] Step 2: 运行验证失败
- [ ] Step 3: 实现 `prompts/hyde.py` + Memory.search hyde 分支
- [ ] Step 4: 运行验证通过
- [ ] Step 5: ruff check

## Task 2: 上下文感知查询改写

**Files:** Create `src/septmuse/prompts/rewrite.py`, Modify `src/septmuse/memory/main.py` (search), Create `tests/unit/test_query_rewrite.py`

- [ ] Step 1-5: 同 TDD 流程

## Task 3: 遗忘曲线参数化

**Files:** Modify `src/septmuse/models/strength.py`, `src/septmuse/retrieval/forgetting.py`, `src/septmuse/configs/defaults.py`, `src/septmuse/memory/main.py`, Create `tests/unit/test_forgetting_params.py`

- [ ] Step 1-5: 同 TDD 流程

## Task 4: 压缩归档 facade

**Files:** Modify `src/septmuse/memory/main.py` (__init__ + compress + recall), Create `tests/unit/test_compress_facade.py`

- [ ] Step 1-5: 同 TDD 流程

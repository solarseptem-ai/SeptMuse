# P2: 质量增强设计

> 日期: 2026-08-07
> 状态: 待审阅

## 1. 背景

P0+P1 完成后,Memory facade 已对齐 mem0 V3 核心管线。P2 补 3 项质量增强:

| # | 差距 | mem0 对应 |
|---|------|-----------|
| 1 | BM25 自适应归一化 | `get_bm25_params` + `normalize_bm25` sigmoid |
| 2 | `attributed_to` / `actor_id` | 多说话人归因 |
| 3 | procedural memory 自动生成 | `_create_procedural_memory` LLM 生成 |

## 2. 改造

### Task 1: BM25 自适应归一化

**现状**: SeptMuse BM25 归一化 = `score / max_score` (简单除法)。长查询 BM25 原始分偏高,短查询偏低,简单除法导致 BM25 分数与向量分数不可比。

**改造** (对齐 mem0 `normalize_bm25` sigmoid):
- 新增 `src/septmuse/retrieval/scoring.py`:
  - `get_bm25_params(query)` — 按查询词数返回 `(midpoint, steepness)`
  - `normalize_bm25(raw_score, midpoint, steepness)` — sigmoid 归一化到 [0,1]
  - `score_and_rank(semantic_results, bm25_scores, entity_boosts, threshold, top_k, explain)` — 加性融合

- `SQLiteBM25Index.retrieve` 改用 sigmoid 归一化替代 `score / max_score`
- `BM25Scorer.score` 返回原始分 (不变)

**参数表** (对齐 mem0):
| 词数 | midpoint | steepness |
|------|----------|-----------|
| ≤3 | 5.0 | 0.7 |
| ≤6 | 7.0 | 0.6 |
| ≤9 | 9.0 | 0.5 |
| ≤15 | 10.0 | 0.5 |
| >15 | 12.0 | 0.5 |

### Task 2: attributed_to / actor_id

**现状**: SeptMuse 不区分消息来源 (user/assistant),不支持多说话人。

**改造** (对齐 mem0 `attributed_to` + `actor_id`):
- `Memory.add` 加 `attributed_to: str | None = None` (user/assistant)
- `Memory.add` 从 `message["name"]` 提取 `actor_id`
- 存入 metadata
- `Memory.search` / `get_all` 返回结果含 `attributed_to` + `actor_id` (从 metadata 提取)

### Task 3: procedural memory 自动生成

**现状**: `Memory.add(memory_type="rule")` 手动添加规则文本。mem0 `_create_procedural_memory` 用 LLM 从对话生成。

**改造**:
- `src/septmuse/prompts/extract.py` 新增 `PROCEDURAL_MEMORY_SYSTEM_PROMPT`
- `Memory.add(memory_type="procedural")` 路由到 LLM 生成:
  1. 拼接 messages + system prompt
  2. LLM 生成规则文本
  3. `procedural.add_rule(rule_text, user_id=)`
- 无 LLM 降级: `messages` 原文作为 rule_text

## 3. 不破坏的承诺

- sigmoid 归一化替代 max_score,阈值不变 (BM25 分数范围 [0,1] 不变)
- `attributed_to` 默认 None (向后兼容)
- `memory_type="procedural"` 新增值,不影响 `memory_type="rule"`
- 现有测试不改

## 4. 测试

| 文件 | 覆盖 |
|------|------|
| `test_bm25_adaptive.py` | get_bm25_params + normalize_bm25 sigmoid + score_and_rank |
| `test_attributed_to.py` | add attributed_to + actor_id 存储 + search 返回 |
| `test_procedural_auto.py` | LLM 生成规则 + 无 LLM 降级 |

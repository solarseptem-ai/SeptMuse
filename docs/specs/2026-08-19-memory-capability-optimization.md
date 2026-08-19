# 记忆能力优化：LOCOMO 评估 + 死 Recipe 激活

> 日期：2026-08-19
> 状态：设计已确认，待实施
> 动机：全面提升 SeptMuse 记忆能力——检索召回质量 + 功能补全

## 问题陈述

SeptMuse 当前记忆能力存在两个层面的缺失：

1. **无评估基线**：项目无 benchmark/eval harness。318 个单元测试验证逻辑正确性，但无法度量检索召回质量。无法量化"好"或"更好"。
2. **4 个死 Recipe 字段**：`SearchRecipe` 定义了 8 个配置字段，`SearchPipeline` 只路由 4 个（`hybrid`/`reranker`/`explain`/`forgetting`），其余 4 个（`graph_bfs`/`progressive`/`hyde`/`query_rewrite`）log 后丢弃。对应的检索增强技术已有部分实现，但 SearchPipeline 不路由它们。

## 方案概述

**A+B 组合**：

- **A（评估驱动）**：用 LOCOMO 基准（Snap Research, ACL 2024）建立评估 harness，量化当前检索能力，为后续优化提供度量基线。
- **B（功能补全）**：激活 4 个死 recipe 字段，补齐已知缺失的检索增强能力。

先建 harness（跑 baseline），再激活死字段（跑改进），用 harness 证明改善。

## Section 1: LOCOMO 评估 Harness

### 背景

LOCOMO-10 是 10 个多轮对话、~1540 道题的长对话记忆基准。4 个评估类别：

| 类别 | 题数 | 测试能力 |
|------|------|----------|
| multi-hop | 282 | 跨记忆推理 |
| temporal | 321 | 时间推理 |
| open-domain | 96 | 开放域 |
| single-hop | 841 | 单跳事实 |

流程：Ingest（对话→记忆系统）→ Search（查询→检索记忆）→ Answer（LLM 从检索结果生成答案）→ Judge（LLM 判定 CORRECT/WRONG）。

### 设计

单文件 `scripts/eval_locomo.py`，自包含。`opensource/` 禁止 import（AGENTS.md），所需部分在 `scripts/` 内重写。

**组件**：

1. **数据集**：从 GitHub 下载 `locomo10.json`，缓存到 `datasets/locomo/`
2. **SeptMuseClient**：async wrapper 包装同步 `Memory`
   - `add(messages, user_id, timestamp)` → `Memory.remember()` 摄入对话
   - `search(query, user_id, top_k)` → `Memory.search()` 检索记忆
   - `delete_user(user_id)` → 清除用户记忆
3. **LLM Client**：复用 SeptMuse 的 Ollama provider（`SEPTMUSE_LLM=ollama`）
   - answerer：从检索记忆生成答案
   - judge：判定答案 CORRECT/WRONG（JSON 结构化输出）
4. **Prompt 模板**：从 `opensource/memory-benchmarks/benchmarks/locomo/prompts.py` 重写（不 import）
5. **指标**：按类别 × cutoff(10/20/50/200) 统计 accuracy

**两种模式**：

- `--predict-only`：只跑 ingest+search，不耗 LLM。用 embedding 相似度对比 ground truth 做代理指标
- `--full`：ingest+search+answer+judge，算端到端 accuracy。需要 Ollama 运行

**运行环境**：

用户 Ollama 环境：
- LLM：`qwen3.5:latest`（6.6 GB）
- Embedder：`bge-m3:latest`（1.2 GB，1024 维）
- Ollama 基址：`http://localhost:7521`

```powershell
$env:PYTHONPATH = "src"
$env:SEPTMUSE_RATE_LIMIT = "0"

# LLM（OllamaLLM 已有 OLLAMA_HOST env 回退）
$env:SEPTMUSE_LLM = "ollama"
$env:SEPTMUSE_LLM_MODEL = "qwen3.5:latest"
$env:OLLAMA_HOST = "http://localhost:7521"

# Embedder（需先修 OllamaEmbedder 参数名，见下方 bugfix）
$env:SEPTMUSE_EMBEDDER = "ollama"
$env:SEPTMUSE_EMBEDDER_MODEL = "bge-m3:latest"
$env:SEPTMUSE_EMBEDDER_BASE_URL = "http://localhost:7521"
$env:SEPTMUSE_EMBEDDER_DIMS = 1024
$env:SEPTMUSE_EMBEDDING_DIMS = 1024

# 向量后端（LOCOMO 评估用 SQLite，不走 Qdrant）
$env:SEPTMUSE_VECTOR_BACKEND = "sqlite"

# predict-only（无需 Ollama LLM，但需 embedder）
python scripts/eval_locomo.py --project-name baseline --conversations 0 --predict-only

# full（需要 Ollama LLM + embedder 运行）
python scripts/eval_locomo.py --project-name baseline --conversations 0 --full
```

### Section 0: OllamaEmbedder 参数名 bugfix

OllamaEmbedder 构造参数叫 `ollama_base_url`，但 config 字段叫 `base_url`。`ServiceProvider._config_to_kwargs()` 按构造参数名过滤，`base_url` 不匹配 `ollama_base_url`，导致 `SEPTMUSE_EMBEDDER_BASE_URL` 传不进去。

**修复**：`src/septmuse/embedders/ollama.py` — 将 `ollama_base_url` 参数重命名为 `base_url`，与 config 字段一致。同时加 `OLLAMA_HOST` env 回退（与 OllamaLLM 一致）。

**输出**：`results/locomo/` 下 per-question JSON + 汇总 JSON

### 不做的事

- 不写 web UI（opensource/ 的 Next.js 前端不需要）
- 不写 checkpoint/resume 逻辑（先跑通，后续加）
- 不写 docker（SeptMuse 自己有 Docker）

## Section 2: 激活 4 个死 Recipe 字段

### 现状

`SearchPipeline.search()` 在 `src/septmuse/pipelines/search.py:186-211` 处理 recipe：

```python
if r.graph_bfs:
    logger.info("recipe_graph_bfs_not_routed", hint="GraphSearcher not wired into pipeline (future)")
if r.progressive:
    logger.info("recipe_progressive_not_routed", hint="ProgressiveRetriever not wired into pipeline (future)")
if r.hyde:
    logger.info("recipe_hyde_not_enabled", hint="requires LLM + prompt engineering (future)")
if r.query_rewrite:
    logger.info("recipe_query_rewrite_not_enabled", hint="requires LLM (future)")
```

4 个字段 log 后丢弃，对应的检索器/技术已存在但未接入。

### 2a. HyDE（假设文档嵌入）

**原理**：用户问"张三负责什么"→ LLM 生成假设答案"张三是公司CTO，负责技术架构决策"→ 用假设答案的向量（而非原始查询）做向量检索。假设答案在向量空间更接近已存储的记忆。

**实现**：

- 新文件 `src/septmuse/retrieval/hyde.py`
- 函数 `generate_hypothetical_document(query: str, llm) -> str`
- prompt: `"Given this question, write a 1-2 sentence hypothetical answer that directly addresses it: {query}"`
- SearchPipeline: `if cfg.hyde and self._llm:` → 生成假设文档 → 用假设文档的向量做向量检索，BM25 仍用原始查询（保留关键词匹配）
- 无 LLM → 跳过（保持 no-op）

### 2b. Query Rewrite（查询重写）

**原理**：用户问"张三和李四聊过什么"→ LLM 生成变体"张三李四的对话"、"张三李四讨论的话题"→ 多路检索后 RRF 融合。

**实现**：

- 新文件 `src/septmuse/retrieval/query_rewrite.py`
- 函数 `rewrite_query(query: str, llm) -> list[str]`（返回原始 + 2 个变体）
- prompt: `"Rewrite this question into 2 different phrasings that ask for the same information: {query}"`
- SearchPipeline: `if cfg.query_rewrite and self._llm:` → 对每个变体跑检索 → RRF 融合
- 无 LLM → 跳过

### 2c. Graph BFS 路由

**原理**：hybrid 检索后，取 top-1 结果的 id 作为 seed，跑 GraphSearcher.search_graph() 做 BFS 图遍历，找到关联记忆，RRF 融合。

**实现**：

- 不需要新文件——GraphSearcher 已存在（`src/septmuse/retrieval/graph_search.py`），有 `search_graph()` 和 `rrf_fuse()`
- SearchPipeline: `if cfg.graph_bfs and self._graph_searcher:` → hybrid 结果 top-1 做 seed → BFS → RRF 融合
- 无 graph_searcher → 跳过

### 2d. Progressive 路由

**原理**：ProgressiveRetriever（`src/septmuse/retrieval/progressive.py`）已存在，做三层渐进检索（向量 → BM25 → 图遍历）。

**实现**：

- 不需要新文件——ProgressiveRetriever 已存在
- SearchPipeline: `if cfg.progressive and self._progressive_retriever:` → 用 progressive_retriever 替代 hybrid_retriever
- 无 progressive_retriever → 跳过

### 2e. 依赖注入变更

`SearchPipeline.__init__` 新增 3 个可选参数：

```python
def __init__(
    self,
    hybrid_retriever: Retriever,
    vector_retriever: Retriever,
    reranker_factory: Callable[[str], Any],
    embedder: Any,
    forgetting: Any | None = None,
    default_top_k: int = 5,
    default_threshold: float = 0.1,
    llm: Any | None = None,              # 新增
    graph_searcher: Any | None = None,    # 新增
    progressive_retriever: Any | None = None,  # 新增
) -> None:
```

Memory 构造 SearchPipeline 时传入已有组件（llm、graph_searcher 已在 Memory 上构造）。

### 不改 entity boost

重新计算后：entity boost = `0.5/(60+1)` ≈ 0.008，单实体匹配约为 rank-0 RRF 分数（1/61 ≈ 0.016）的一半。3 实体匹配 = 0.024，足以把 rank-5 结果推到 rank-0 之上。不像之前判断的那么弱。让 LOCOMO eval 告诉我们是否需要调。

## 测试策略

- **HyDE / query_rewrite**：新模块各加单元测试（mock LLM，验证生成逻辑 + 无 LLM 降级）
- **Graph BFS / Progressive 路由**：在 `test_search_pipeline.py` 加测试（验证 recipe 触发正确检索器）
- **LOCOMO harness**：跑 `--predict-only` 模式验证端到端流程通（至少 1 个对话）
- **回归**：现有 318 个测试全过

## 实施顺序

0. **OllamaEmbedder 参数名 bugfix** — 改 `ollama_base_url` → `base_url` + 加 `OLLAMA_HOST` env 回退
1. **LOCOMO harness（predict-only）** — 先建评估基线，用 bge-m3 embedder 跑 baseline
2. **Graph BFS + Progressive 路由** — 不需要新文件，只改 SearchPipeline + 注入，最简单
3. **HyDE + Query Rewrite** — 需要新文件 + LLM 集成（qwen3.5 做 HyDE/查询重写）
4. **LOCOMO harness（full）** — 用 Ollama LLM 跑完整评估，对比 baseline

## 验证

```powershell
$env:PYTHONPATH = "src"
$env:SEPTMUSE_EMBEDDER = "hash"
$env:SEPTMUSE_TOKENIZER = "space"
$env:SEPTMUSE_VECTOR_BACKEND = "sqlite"
$env:SEPTMUSE_RATE_LIMIT = "0"

# 单元测试
python -m pytest tests/unit/ -q

# Lint
ruff check src/ tests/ scripts/
ruff format --check src/ tests/ scripts/

# LOCOMO baseline
python scripts/eval_locomo.py --project-name baseline --conversations 0 --predict-only

# LOCOMO after optimization
python scripts/eval_locomo.py --project-name optimized --conversations 0 --full
```

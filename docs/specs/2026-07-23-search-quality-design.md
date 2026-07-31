# 检索质量提升设计文档（P1-Task 1 + P1-Task 2）

> 日期：2026-07-23
> 前置文档：`docs/plans/development-roadmap.md`（P1 Phase）
> 范围：P1-Task 1（Reranker 框架）+ P1-Task 2（Entity Boost 三信号融合）
> 不包含：P1-Task 3（BFS 图遍历，依赖 P0-Task 3 cognify，被 LLM Provider 阻塞）、P1-Task 4（检索 Recipes，依赖 Task 1+2+3）
> 借鉴来源：mem0 `BaseReranker` / `LLMReranker` / TS `CrossEncoderReranker` + graphiti `BGERerankerClient` / `maximal_marginal_relevance` + mem0 `_search_vector_store` 三信号融合

---

## 1. 架构总览

```
┌─────────────────────────────────────────────────────────────┐
│                    Memory.search(query, ...)                 │
│                         orchestrates                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────────────────────┐    ┌─────────────────┐ │
│  │     HybridRetriever.search()     │    │   Reranker ABC │ │
│  │                                   │    │                 │ │
│  │  1. 向量召回 (store.search)       │    │  NoopReranker  │ │
│  │  2. BM25 召回 (BM25Scorer)        │───▶│  MMRReranker   │ │
│  │  3. Entity boost (EntityStore)   │    │  CrossEncoder  │ │
│  │  4. 三信号融合 (RRF + boost)     │    │  LLMReranker   │ │
│  │  5. 返回 HybridResult[]          │    │                 │ │
│  └──────────────────────────────────┘    └─────────────────┘ │
│           │                                      │           │
│           ▼                                      ▼           │
│     (可选 explain=True)              (可选 reranker="...")    │
│     score_details 返回               重排后 HybridResult[]     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 关键决策

- **后处理 Reranker 模式**（借鉴 mem0 `search → rerank`）：Reranker 作为独立 concern，在 `HybridRetriever.search()` 返回结果后做后处理重排。关注分离清晰，可独立测试，不改变现有检索流程。
- **Reranker ABC** 放 `concerns/retrieval/reranker.py`，操作 `list[HybridResult]`（非 dict），签名 `rerank(query, results, *, top_k) -> list[HybridResult]`。
- **Entity boost** 直接集成进 `HybridRetriever`（第三信号），不新建独立类。`entity_extractor` 和 `entity_store` 为 `HybridRetriever.__init__` 可选参数，都为 None 时退化为双信号行为（向后兼容）。
- **Memory.search()** 先调 `HybridRetriever.search()` 得到三信号融合结果，再可选调 `Reranker.rerank()` 重排。
- **MemoryConfig** 新增 `reranker_backend` 字段（`noop`/`mmr`/`cross_encoder`/`llm`）。

---

## 2. Reranker ABC + 4 种实现

### 2.1 ABC 定义

```python
# concerns/retrieval/reranker.py

class Reranker(ABC):
    """重排器抽象基类 (借鉴 mem0 BaseReranker + graphiti CrossEncoderClient)。"""

    @abstractmethod
    def rerank(
        self,
        query: str,
        results: list[HybridResult],
        *,
        top_k: int | None = None,
    ) -> list[HybridResult]:
        """对检索结果重排，返回按相关性降序排列的 HybridResult 列表。

        实现方应:
        - 保留原始 HybridResult 的其他字段 (id, memory, metadata, created_at)
        - 更新 score 字段为重排后的分数
        - 可选追加 rerank_score 字段到 metadata
        """
        ...
```

### 2.2 NoopReranker

透传 reranker，不改变顺序和 score。

- 直接返回 `results[:top_k]`
- 无外部依赖，无延迟
- 默认 reranker

### 2.3 MMRReranker

最大边际相关性 reranker（借鉴 graphiti `maximal_marginal_relevance`），去冗余。

- 用 embedder 对 query 和每条 result 的 memory 计算向量
- 贪心迭代选择：每轮从未选集合中选 MMR 分数最高的候选加入 selected，直到选满 top_k
- MMR 公式（每轮对每个未选候选 doc 计算）：`mmr = lambda * sim(query, doc) - (1-lambda) * max(sim(doc, selected))`
- 参数 `lambda_param: float = 0.7`（0=最大化多样性，1=最大化相关性）
- 去冗余：相似度 >0.9 的结果只保留排名靠前的一个
- 纯数学计算，无外部依赖，<1ms

### 2.4 CrossEncoderReranker

ONNX cross-encoder reranker（借鉴 graphiti `BGERerankerClient` + mem0 TS `CrossEncoderReranker`）。

- 延迟 import onnxruntime（仅使用时加载，对齐 embedder 策略）
- 模型 `BAAI/bge-reranker-v2-m3` ONNX 量化版，ModelScope 下载到 `~/.septmuse/models/`
- `sigmoid(logit)` 归一化到 [0,1]，更新 `result.score`
- onnxruntime 不可用时降级为 NoopReranker + 日志警告（不崩溃）
- 可选 extra：`pip install septmuse[reranker]`（`onnxruntime>=1.16`）
- CPU <50ms

### 2.5 LLMReranker

LLM 打分 reranker（借鉴 mem0 `LLMReranker`）。

- 构造时传入 `LLM` 实例（P3-Task 1 完成后可用，当前为可选）
- system prompt："给 query 和 document 打分 0.0-1.0，仅返回数字"
- `LLM.complete(system_prompt, user_prompt)` 逐条打分
- `_extract_score` 正则提取数字，clamp [0,1]，无数字返回 0.5
- 输入截断 `_MAX_INPUT_LEN = 4000` 防止 prompt flooding
- 无 LLM 实例时抛 `ValueError("LLMReranker requires an LLM instance")`
- 延迟到 `search()` 调用时才抛错（构造时不报错，对齐 `SEPTMUSE_LLM` 未设时的行为）

### 2.6 `_resolve_reranker` 工厂函数

```python
def _resolve_reranker(
    backend: str = "noop",
    *,
    embedder: Embedder | None = None,
    llm: LLM | None = None,
    model_cache_dir: str | None = None,
) -> Reranker:
    match backend:
        case "noop":          return NoopReranker()
        case "mmr":           return MMRReranker(embedder=embedder, lambda_param=0.7)
        case "cross_encoder": return CrossEncoderReranker(model_cache_dir=...)
        case "llm":           return LLMReranker(llm=llm)
        case _:               raise ValueError(f"Unknown reranker: {backend}")
```

---

## 3. Entity Boost 三信号融合

### 3.1 当前状态

`HybridRetriever.search()` 目前是双信号 RRF：
```
fused = vector_weight/(k+v_rank) + keyword_weight/(k+k_rank)
```

### 3.2 新设计：三信号融合（借鉴 mem0 `_search_vector_store` scoring）

```
fused = vector_weight/(k+v_rank) + keyword_weight/(k+k_rank) + entity_boost
```

### 3.3 Entity boost 计算

1. 从 query 用 `EntityExtractor.extract(query)` 抽取实体
2. 对每个实体在 `EntityStore.search(entity_text, user_id)` 中搜索匹配实体
3. 收集匹配实体的 `linked_memory_ids`
4. 对每个候选记忆，如果它在 `linked_memory_ids` 中，获得 boost 分数。如果一个记忆被多个匹配实体关联，boost 分数累加（sum）。

**Boost 公式**（借鉴 mem0 `similarity × 0.5 × 1/(1+0.001×(n-1)²)`）：
```python
# n = 该实体关联的记忆总数（越多说明越泛化，权重越低）
# 每个匹配实体独立计算 boost，多实体 boost 累加
entity_boost_per_entity = 0.5 * 1.0 / (1.0 + 0.001 * (n - 1) ** 2)
# 记忆的最终 entity_boost = sum(所有匹配实体的 boost_per_entity)
```

- n=1 时 boost=0.5（最大）
- n=10 时 boost≈0.335
- n=100 时 boost≈0.005（几乎无 boost，高度泛化实体不提升）

### 3.4 集成方式

`HybridRetriever.__init__` 新增可选参数：
```python
def __init__(
    self,
    store: MemoryStore,
    embedder: Embedder,
    vector_weight: float = 0.7,
    keyword_weight: float = 0.3,
    entity_extractor: EntityExtractor | None = None,  # 新增
    entity_store: EntityStore | None = None,           # 新增
) -> None:
```

- `entity_extractor` 和 `entity_store` 都为 None 时，退化为当前双信号行为（向后兼容）
- 都存在时启用三信号融合
- 只有其中一个时：日志警告 + 退化为双信号

### 3.5 `HybridResult` 扩展

```python
@dataclass
class HybridResult:
    id: str
    memory: str
    score: float
    vector_score: float = 0.0
    bm25_score: float = 0.0
    entity_boost: float = 0.0          # 新增
    metadata: dict[str, Any] | None = None
    created_at: str | None = None
```

### 3.6 `explain=True` 支持

`HybridRetriever.search()` 新增 `explain: bool = False` 参数：
- `explain=False`（默认）：`HybridResult.score` = 融合总分
- `explain=True`：额外在 `metadata["score_details"]` 返回明细：
  ```json
  {"vector": 0.012, "bm25": 0.008, "entity_boost": 0.3, "combined": 0.32}
  ```

### 3.7 检索流程

```
1. 向量召回 (store.search)           → vector_rank, vector_scores
2. BM25 召回 (BM25Scorer)            → keyword_rank
3. Entity boost (entity_extractor + entity_store)  → entity_boosts: dict[mid, float]
4. 三信号融合:
   fused = vw/(k+v_rank) + kw/(k+k_rank) + entity_boost
   (仅当 entity_boost > 0 时加入)
5. 排序 → 返回 HybridResult[]
```

---

## 4. Memory Facade 集成

### 4.1 `Memory.search()` 签名扩展

```python
def search(
    self,
    query: str,
    *,
    user_id: str,
    top_k: int = 5,
    threshold: float = 0.1,
    reranker: str | None = None,      # 新增: "noop"/"mmr"/"cross_encoder"/"llm"
    explain: bool = False,             # 新增: 返回 score_details
    **kwargs,
) -> list[HybridResult]:
```

- `reranker=None` 时用 `MemoryConfig.reranker_backend`（默认 `"noop"`）
- `reranker="mmr"` 时覆盖配置，本次用 MMR 重排

### 4.2 `Memory.__init__` 扩展

新增 `reranker` 实例属性，由 `_resolve_reranker` 工厂创建：
```python
self._reranker = _resolve_reranker(
    config.reranker_backend,
    embedder=self._embedder,
    llm=self._llm,              # 可能为 None（P3-Task 1 完成前）
    model_cache_dir=config.model_cache_dir,
)
```

- `LLMReranker` 在 `config.reranker_backend == "llm"` 但 `self._llm is None` 时：延迟到 `search()` 调用时才抛 `ValueError`（构造时不报错，对齐 `SEPTMUSE_LLM` 未设时的行为）

### 4.3 `MemoryConfig` 扩展

```python
class MemoryConfig(BaseModel):
    # ... 已有字段 ...
    reranker_backend: str = "noop"   # 新增
```

### 4.4 环境变量

| 变量 | 默认 | 作用 |
|------|------|------|
| `SEPTMUSE_RERANKER` | `noop` | `noop`/`mmr`/`cross_encoder`/`llm` |

### 4.5 `search()` 流程

```
1. HybridRetriever.search(query, user_id, top_k, threshold, explain)
   → 三信号融合结果 list[HybridResult]
2. self._reranker.rerank(query, results, top_k=top_k)
   → 重排后的 list[HybridResult]
3. return results
```

- `NoopReranker` 直接透传，无开销
- 其他 reranker 在结果集上做后处理重排

### 4.6 CLI / REST / MCP 集成

**CLI**：`search` 命令新增 `--reranker` 可选参数
```bash
python -m septmuse.cli.main search "Python" --reranker mmr
```

**REST**：`POST /memories/search` 新增 `reranker` query/body 参数
```json
{"query": "Python", "reranker": "mmr", "top_k": 5}
```

**MCP**：`search_memory` 工具新增 `reranker` 参数

### 4.7 pyproject.toml

新增可选 extra：
```toml
[project.optional-dependencies]
reranker = ["onnxruntime>=1.16"]
```

---

## 5. 测试策略

### 5.1 测试文件布局

| 文件 | 内容 | 预计测试数 |
|------|------|-----------|
| `tests/unit/test_reranker.py` | Reranker ABC + 4 实现 + `_resolve_reranker` | ~20 |
| `tests/unit/test_hybrid_entity_boost.py` | Entity boost 三信号融合 + explain | ~12 |
| `tests/unit/test_retrieval.py` | 已有，新增 reranker 集成测试 | +5 |
| `tests/e2e/test_reranker_e2e.py` | e2e 跨会话持久化 + reranker | 3 |

### 5.2 测试要点

**Reranker 单元测试**（`test_reranker.py`）：
- `NoopReranker`：透传不变序，空输入返回空，top_k 截断
- `MMRReranker`：去冗余（相似 >0.9 只留一个），lambda 参数效果，需要 embedder mock
- `CrossEncoderReranker`：onnxruntime 不可用时降级为 Noop + 警告，模型加载 mock
- `LLMReranker`：LLM mock 返回 "0.8" → score=0.8，无 LLM 实例抛 ValueError，`_extract_score` 边界（负数 clamp 0，超 1 clamp 1，无数字返回 0.5）
- `_resolve_reranker`：4 种 backend 正确分派，未知 backend 抛 ValueError

**Entity boost 单元测试**（`test_hybrid_entity_boost.py`）：
- 无 entity_extractor/entity_store → 退化为双信号（向后兼容）
- query 抽取到实体 → 匹配实体的 `linked_memory_ids` 获得 boost
- 实体关联记忆多（n 大）→ boost 衰减
- `explain=True` → `metadata["score_details"]` 含 vector/bm25/entity_boost/combined
- entity_store 为空 → boost 全为 0，退化为双信号

**集成测试**（`test_retrieval.py` 新增）：
- `Memory.search(reranker="mmr")` 端到端可用
- `Memory.search(reranker="noop")` 与无 reranker 结果一致
- `MemoryConfig.reranker_backend` 配置生效

**e2e 测试**（`test_reranker_e2e.py`）：
- 跨会话：写入记忆 → 新 Memory 实例 → search with reranker
- MMR 去冗余在真实 SQLite 上的效果
- explain=True 返回完整 score_details

### 5.3 验收标准

- 现有 757 passed 测试零回归
- 新增 ~40 测试 → 总计 ~797 passed
- `ruff check` + `ruff format --check` clean
- CrossEncoderReranker onnxruntime 不可用时 skip（非 fail），对齐 embedder skip 模式

---

## 6. 文件变更清单

### 新增文件

| 文件 | 内容 |
|------|------|
| `src/septmuse/concerns/retrieval/reranker.py` | Reranker ABC + NoopReranker + MMRReranker + CrossEncoderReranker + LLMReranker + `_resolve_reranker` |
| `tests/unit/test_reranker.py` | ~20 单元测试 |
| `tests/unit/test_hybrid_entity_boost.py` | ~12 单元测试 |
| `tests/e2e/test_reranker_e2e.py` | 3 e2e 测试 |

### 修改文件

| 文件 | 变更 |
|------|------|
| `src/septmuse/concerns/retrieval/hybrid.py` | `HybridResult` +`entity_boost` 字段；`HybridRetriever.__init__` +`entity_extractor`/`entity_store` 参数；`search()` +entity boost 三信号融合 +`explain` 参数 |
| `src/septmuse/orchestration/memory.py` | `__init__` +`_resolve_reranker` 初始化；`search()` +`reranker`/`explain` 参数；`HybridRetriever` 传入 entity_extractor/entity_store |
| `src/septmuse/configs/defaults.py` | `MemoryConfig` +`reranker_backend` 字段 |
| `src/septmuse/cli/main.py` | `search` 命令 +`--reranker` 参数 |
| `src/septmuse/api/rest/__init__.py` | `POST /memories/search` +`reranker` 参数 |
| `src/septmuse/api/mcp/tools.py` | `search_memory` 工具 +`reranker` 参数 |
| `pyproject.toml` | +`reranker` extra (`onnxruntime>=1.16`) |
| `CHANGELOG.md` | Added — Reranker 框架 + Entity boost 三信号融合 |
| `AGENTS.md` | +`SEPTMUSE_RERANKER` 环境变量 + Reranker 章节 + skip 数更新 |

---

## 7. 借鉴来源映射

| 设计要素 | 借鉴来源 | 具体文件/类 |
|----------|----------|------------|
| Reranker ABC 签名 | mem0 `BaseReranker` | `opensource/mem0/mem0/reranker/base.py` |
| LLMReranker system prompt + `_extract_score` | mem0 `LLMReranker` | `opensource/mem0/mem0/reranker/llm_reranker.py` |
| CrossEncoderReranker sigmoid 归一化 | mem0 TS `CrossEncoderReranker` | `opensource/mem0/mem0-ts/src/oss/src/rerankers/cross_encoder.ts` |
| CrossEncoderReranker BGE 模型 | graphiti `BGERerankerClient` | `opensource/graphiti/graphiti_core/cross_encoder/bge_reranker_client.py` |
| MMRReranker MMR 公式 | graphiti `maximal_marginal_relevance` | `opensource/graphiti/graphiti_core/search/search_utils.py` |
| NoopReranker 透传 | MemOS `NoopReranker` | `opensource/MemOS/src/memos/reranker/noop.py` |
| Entity boost 公式 | mem0 `_search_vector_store` scoring | `opensource/mem0/mem0/memory/main.py:1584-1769` |
| 三信号融合 | mem0 scoring `similarity × 0.5 × 1/(1+0.001×(n-1)²)` | `opensource/mem0/mem0/utils/scoring.py` |
| `explain=True` score_details | mem0 `search(explain=True)` | `opensource/mem0/mem0/memory/main.py:1335` |

---

## 8. 不包含（Out of Scope）

- **P1-Task 3 BFS 图遍历**：依赖 P0-Task 3 cognify 流水线，被 LLM Provider 阻塞。待 P3-Task 1 完成后补。
- **P1-Task 4 检索 Recipes**：依赖 P1-Task 1+2+3，Task 3 未完成。待 Task 3 完成后做。
- **CrossEncoderReranker ONNX 模型实际下载逻辑**：设计阶段确定用 `BAAI/bge-reranker-v2-m3` ONNX 量化版 + ModelScope，具体实现细节在实施计划中确定。
- **LLM Provider 实现**：P3-Task 1 范围。LLMReranker 框架先实现，LLM 实例可选，P3-Task 1 完成后即插即用。

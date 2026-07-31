# P1 存储抽象层设计 — SeptMuse 记忆基础设施补齐

- **状态**: Draft — 待用户 review
- **日期**: 2026-07-21
- **范围**: P1（4 阶段中的第 1 阶段）
- **依赖决策**: 分阶段四步走 / 全可选 extras_require / 接口对齐重新实现

## 1. 背景

SeptMuse v0.1.0 Alpha 已具备三维正交架构 + 因果链/遗忘曲线/元认知等认知增强能力，但记忆基础设施存在 8 项缺口（基于 codegraph 实证对比 mem0/letta/cognee/graphiti/ReMe/MemOS）。用户决策分 4 阶段补齐：

- **P1 存储抽象层**（本文档）：VectorStoreBase + KeywordIndexBase + GraphStore 扩展
- **P2 权限层**：ACL + MemoryAccessLog（借鉴 mem0 openmemory）
- **P3 时态层**：双时态字段 valid_at/invalid_at/expired_at（借鉴 graphiti EntityEdge）
- **P4 编排+扩展**：Pipeline DAG（借鉴 cognee）+ vision（借鉴 mem0）+ auto_dream（借鉴 ReMe）

## 2. 目标与非目标

### 目标

1. 向量库后端可插拔：SQLite（默认）/ pgvector / Chroma / Qdrant
2. 关键词索引可插拔：SQLite BM25（默认，纯 Python）/ rank-bm25（extras）
3. 图存储后端可插拔：SQLite（默认）/ AGE / Neo4j
4. 混合检索（向量 + 关键词 RRF 融合）开箱即用
5. 零配置默认不变（SQLite + HashEmbedder）
6. 现有 614 单元 + 23 e2e 测试零退化

### 非目标

- 不实现 mem0 30 种向量库后端（YAGNI，只做 SQLite/pgvector/Chroma/Qdrant 4 种）
- 不实现 graphiti 12 ops 接口 + 事务 + session（YAGNI，只扩展现有 GraphStore）
- 不实现 mem0 ACL/双时态/vision（P2/P3/P4 范围）
- 不改 Memory facade 和 TypedMemoryStore（concerns 层不变）

## 3. 架构

### 3.1 重构后的分层

```
┌─────────────────────────────────────────────────────────┐
│ Memory facade (orchestration/memory.py)                 │ 不变
│ TypedMemoryStore (storage/typed_store.py)               │ 不变
├─────────────────────────────────────────────────────────┤
│ MemoryStore ABC (storage/base.py)                       │ 已有，微调
│  add/search/get_all/get/delete/update/get_history       │ 现有 8 方法不变
│  list_agents/list_users/get_shared_memories              │ 现有 3 关系查询不变
│  + keyword_search(新增,默认返回[])                       │ 新增
│  + hybrid_search(新增,默认 RRF 融合实现)                 │ 新增
├─────────────────────────────────────────────────────────┤
│ 组合器层                                                 │ P1 重构
│  SQLiteCompositeStore  = VectorStore + KeywordIndex     │ 重构 SQLiteMemoryStore
│  PGVectorCompositeStore = VectorStore + KeywordIndex    │ 重构 PGVectorStore
├─────────────────────────────────────────────────────────┤
│ 存储后端 ABC                                             │ P1 新增
│  VectorStoreBase    KeywordIndexBase   GraphStore(扩展)  │
│  (借鉴 mem0, 5 方法) (借鉴 ReMe, 4 方法) (+delete_edge)  │
├─────────────────────────────────────────────────────────┤
│ 具体实现                                                 │
│  SQLiteVectorStore(默认)    SQLiteBM25Index(默认)         │
│  ChromaVectorStore[chroma]  RankBM25Index[bm25]          │
│  QdrantVectorStore[qdrant]                              │
│  SQLiteGraphStore(已有)     AGEGraphStore(已有)           │
│  Neo4jGraphStore[neo4j]                                 │
└─────────────────────────────────────────────────────────┘
```

### 3.2 关键设计决策

1. **MemoryStore 不拆分，只扩展**：现有 8 方法签名不变，新增 `keyword_search`（默认返回 `[]`）和 `hybrid_search`（默认 RRF 融合实现）。

2. **SQLiteMemoryStore 重构为组合器**：内部委托给 SQLiteVectorStore + SQLiteBM25Index，旧 `add/search` 签名不变，保证 614 测试零退化。

3. **VectorStoreBase 只管向量 CRUD**（5 方法）：`insert_vectors`/`search_vectors`/`delete_vector`/`get_vector`/`list_vectors`。不照搬 mem0 `create_col`/`list_cols`/`col_info` 等集合管理（YAGNI）。

4. **KeywordIndexBase 只管关键词 CRUD**（4 方法，对齐 ReMe）：`add_docs`/`retrieve`/`delete_docs`/`clear`。改同步接口匹配 SeptMuse 风格。

5. **GraphStore 扩展，不新建 GraphDriverBase**：现有 5 方法不变，新增 `delete_edge`。graphiti 的 12 ops 接口 + 事务 + session 为双时态图/community/saga 设计，YAGNI。

6. **混合检索在 MemoryStore 层**：`hybrid_search` 默认实现 = 向量 `search` + 关键词 `keyword_search` + RRF（Reciprocal Rank Fusion，k=60）融合排序。

## 4. 接口定义

### 4.1 VectorStoreBase（新增 `storage/vector/base.py`）

借鉴 mem0 `vector_stores/base.py`（100 行 12 方法），精简为 5 方法。

```python
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

@dataclass
class VectorSearchResult:
    """向量检索结果 (对齐 mem0 OutputData id/score/payload 三字段)。"""
    id: str
    score: float          # 相似度 [0,1], 越高越相似
    payload: dict[str, Any] | None = None

@dataclass
class VectorEntry:
    """向量条目。"""
    id: str
    vector: list[float]
    payload: dict[str, Any] | None = None

class VectorStoreBase(ABC):
    """向量存储后端抽象 (借鉴 mem0 vector_stores/base.py, 精简为 5 方法)。

    只管向量 CRUD, 不管 memories 表/history 表 (那是 MemoryStore 的职责)。
    score 语义: 相似度 [0,1], 越高越相似。
    实现方需保证 user_id 隔离 (通过 filters 参数)。
    """

    @abstractmethod
    def insert_vectors(
        self,
        vectors: list[list[float]],
        ids: list[str],
        payloads: list[dict[str, Any]] | None = None,
    ) -> None:
        """批量插入向量。id 与 vector 一一对应。"""
        ...

    @abstractmethod
    def search_vectors(
        self,
        query_vector: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        """向量检索, 返回 top_k 结果 (按 score 降序)。

        filters: payload 字段过滤, 如 {"user_id": "alice"}。
        """
        ...

    @abstractmethod
    def delete_vector(self, vector_id: str) -> bool:
        """删除向量。True=删除成功, False=不存在。"""
        ...

    @abstractmethod
    def get_vector(self, vector_id: str) -> VectorEntry | None:
        """取单条向量。不存在返回 None。"""
        ...

    @abstractmethod
    def list_vectors(
        self,
        filters: dict[str, Any] | None = None,
        limit: int | None = None,
    ) -> list[VectorEntry]:
        """列向量。filters 按 payload 字段过滤。"""
        ...
```

### 4.2 KeywordIndexBase（新增 `storage/keyword/base.py`）

借鉴 ReMe `keyword_index/base_keyword_index.py`（63 行 4 抽象方法），改同步接口。

```python
from __future__ import annotations
from abc import ABC, abstractmethod

class KeywordIndexBase(ABC):
    """关键词索引后端抽象 (借鉴 ReMe keyword_index/base_keyword_index.py, 改同步)。

    score 语义: 归一化 BM25 分数 [0,1], 越高越相关 (与 VectorStoreBase 对齐)。
    实现方需保证:
    - add_docs 幂等 (同 id 覆盖)
    - delete_docs 静默跳过不存在的 id
    - retrieve 返回 score 已归一化
    """

    @abstractmethod
    def add_docs(self, docs: dict[str, str]) -> None:
        """添加或替换文档 (id->text)。已存在的 id 覆盖。"""
        ...

    @abstractmethod
    def retrieve(self, query: str, limit: int = 5) -> dict[str, float]:
        """检索, 返回 {doc_id: score} (按 score 降序, top_k=limit)。"""
        ...

    @abstractmethod
    def delete_docs(self, doc_ids: list[str]) -> None:
        """删除文档。不存在的 id 静默跳过。"""
        ...

    @abstractmethod
    def clear(self) -> None:
        """清空索引。"""
        ...
```

### 4.3 GraphStore 扩展（现有 `storage/graph/base.py`）

现有 5 方法（add_edge/get_edges/get_neighbors/has_edge/close）不变，新增 1 方法：

```python
class GraphStore(ABC):
    # ... 现有 5 方法不变 ...

    @abstractmethod
    def delete_edge(self, edge_id: str) -> bool:
        """删除边。True=删除成功, False=不存在。"""
        ...
```

### 4.4 MemoryStore 扩展（现有 `storage/base.py`）

现有 8 方法 + 3 关系查询签名完全不变，新增 2 方法（默认实现，子类可选覆盖）：

```python
class MemoryStore(ABC):
    # ... 现有 8 方法 + 3 关系查询不变 ...

    def keyword_search(
        self, query: str, *, user_id: str, top_k: int = 5
    ) -> list[dict[str, Any]]:
        """关键词检索 (BM25)。默认返回空 (子类有 KeywordIndex 时覆盖)。

        返回格式同 search: [{"id", "memory", "score", "metadata", "created_at"}]
        """
        return []

    def hybrid_search(
        self, query: str, query_embedding: list[float], *, user_id: str,
        top_k: int = 5, alpha: float = 0.5,
    ) -> list[dict[str, Any]]:
        """混合检索 (向量 + 关键词 RRF 融合)。

        alpha: 向量权重 [0,1]。0=纯关键词, 1=纯向量, 0.5=均衡。
        默认实现: 向量 search + 关键词 keyword_search, RRF 融合排序。
        """
        vec_results = self.search(query_embedding, user_id=user_id, top_k=top_k * 2)
        kw_results = self.keyword_search(query, user_id=user_id, top_k=top_k * 2)
        return _rrf_fuse(vec_results, kw_results, alpha=alpha)[:top_k]
```

### 4.5 RRF 融合函数（模块级）

```python
def _rrf_fuse(
    vec_results: list[dict], kw_results: list[dict], *, alpha: float = 0.5, k: int = 60
) -> list[dict]:
    """RRF 融合排序 (借鉴 Cormack 2009, k=60 标准参数)。

    score = alpha * 1/(k+rank_vec) + (1-alpha) * 1/(k+rank_kw)
    """
    scores: dict[str, float] = {}
    meta: dict[str, dict] = {}
    for rank, r in enumerate(vec_results):
        scores[r["id"]] = scores.get(r["id"], 0.0) + alpha / (k + rank + 1)
        meta.setdefault(r["id"], r)
    for rank, r in enumerate(kw_results):
        scores[r["id"]] = scores.get(r["id"], 0.0) + (1 - alpha) / (k + rank + 1)
        meta.setdefault(r["id"], r)
    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [{**meta[mid], "score": sc} for mid, sc in ordered]
```

## 5. 数据流

### 5.1 SQLiteCompositeStore（重构 SQLiteMemoryStore）

重构前——SQLiteMemoryStore 直接管 memories + history 两表 + numpy 余弦。

重构后——SQLiteCompositeStore 组合 VectorStore + KeywordIndex + 直管 memories/history。

**add 流程**:
```
SQLiteCompositeStore.add(content, embedding, user_id, ...)
  1. memory_id = f"mem-{uuid4()}"
  2. INSERT memories(id, user_id, content, ...) [去 embedding 列]
  3. SQLiteVectorStore.insert_vectors([embedding], [memory_id], [{"user_id":...}])
     └→ INSERT vector_entries(id, vector, payload)
  4. SQLiteBM25Index.add_docs({memory_id: content})
     └→ 内存倒排索引更新
  5. 失败回滚: 任何步骤失败 → 反向删除已写入数据
  6. 返回 memory_id
```

**search 流程**:
```
SQLiteCompositeStore.search(query_embedding, user_id, top_k)
  1. SQLiteVectorStore.search_vectors(query_embedding, filters={"user_id":...}, top_k)
     └→ SELECT id, vector FROM vector_entries WHERE payload->user_id=...
     └→ numpy 余弦 → top_k → [{id, score, payload}]
  2. JOIN memories 表补全 content/metadata
     └→ SELECT content, metadata, created_at FROM memories WHERE id IN (...)
  3. 返回 [{"id", "memory", "score", "metadata", "created_at"}]
```

**hybrid_search 流程**:
```
SQLiteCompositeStore.hybrid_search(query, query_embedding, user_id, top_k, alpha)
  1. vec = SQLiteVectorStore.search_vectors(...) → JOIN memories → vec_results
  2. kw = SQLiteBM25Index.retrieve(query, limit=top_k*2) → JOIN memories → kw_results
  3. _rrf_fuse(vec_results, kw_results, alpha) → top_k
```

### 5.2 表结构变更（向后兼容）

**新表 `vector_entries`**（向量数据移出 memories 表）:
```sql
CREATE TABLE IF NOT EXISTS vector_entries (
    id       TEXT PRIMARY KEY,           -- 同 memories.id
    vector   TEXT NOT NULL,              -- JSON list[float]
    payload  TEXT DEFAULT '{}'           -- JSON {"user_id","agent_id"}
);
```

**memories 表**：保留 embedding 列（双写迁移，保证旧库兼容）。

- 阶段1（本 PR）：memories 表保留 embedding 列，add 时双写（memories.embedding + vector_entries.vector），search 时优先查 vector_entries，回退查 memories.embedding。
- 阶段2（下个版本）：移除 embedding 列。

## 6. 实现清单

| # | 文件 | 类型 | 行数估 | 借鉴源 |
|---|------|------|--------|--------|
| 1 | `storage/vector/base.py` | 新增 ABC | ~60 | mem0 `vector_stores/base.py`（12→5 精简） |
| 2 | `storage/vector/sqlite_vec.py` | 新增默认实现 | ~150 | mem0 `vector_stores/faiss.py`（numpy 余弦） |
| 3 | `storage/vector/chroma.py` | 新增 extras=[chroma] | ~120 | mem0 `vector_stores/chroma.py` |
| 4 | `storage/vector/qdrant.py` | 新增 extras=[qdrant] | ~140 | mem0 `vector_stores/qdrant.py` |
| 5 | `storage/keyword/__init__.py` | 新增模块 | - | - |
| 6 | `storage/keyword/base.py` | 新增 ABC | ~40 | ReMe `keyword_index/base_keyword_index.py`（async→sync） |
| 7 | `storage/keyword/sqlite_bm25.py` | 新增默认实现 | ~180 | ReMe `bm25_index.py`（纯 Python BM25） |
| 8 | `storage/keyword/rank_bm25.py` | 新增 extras=[bm25] | ~80 | rank-bm25 库适配 |
| 9 | `storage/graph/base.py` | 扩展 +1 方法 | +10 | graphiti `delete_edge` |
| 10 | `storage/sqlite/store.py` | 重构为组合器 | ~250（原 376 精简） | mem0 组合模式 |
| 11 | `storage/vector/pgvector.py` | 重构为组合器 | ~300（原 487 精简） | - |
| 12 | `storage/graph/neo4j.py` | 新增 extras=[neo4j] | ~150 | graphiti `driver/ne Driver` |

**配置/打包**:
| # | 文件 | 改动 |
|---|------|------|
| 13 | `configs/defaults.py` | +`vector_backend`/`keyword_backend`/`graph_backend` 字段（默认 sqlite） |
| 14 | `pyproject.toml` | +`[chroma]`/`[qdrant]`/`[neo4j]`/`[bm25]` extras |

**测试**:
| # | 文件 | 测试数 |
|---|------|--------|
| 15 | `tests/unit/test_vector_store_base.py` | ~15（ABC 契约 + SQLiteVectorStore CRUD） |
| 16 | `tests/unit/test_keyword_index.py` | ~12（SQLiteBM25 + 契约） |
| 17 | `tests/unit/test_hybrid_search.py` | ~8（RRF 融合 + alpha 边界） |
| 18 | `tests/unit/test_composite_store.py` | ~10（组合器 add/search/hybrid 集成） |
| 19 | 现有 614 测试 | 零退化（回归基线） |

## 7. 错误处理

| 场景 | 策略 |
|------|------|
| chroma/qdrant 未安装 | `ImportError` → 友好提示 `pip install septmuse[chroma]` |
| 向量维度不匹配 | `insert_vectors` 抛 `ValueError`，不静默截断 |
| BM25 索引空查询 | `retrieve("")` 返回空 dict，不报错 |
| vector_entries 缺旧库数据 | SQLiteVectorStore 回退查 memories.embedding（兼容） |
| 组合器部分后端失败 | add 失败回滚（向量+关键词+memories 原子性，try/except + 反向删除） |
| neo4j 连接失败 | 构造时抛 `ConnectionError`，不静默降级（显式失败优于隐藏 bug） |

## 8. 配置扩展

`MemoryConfig` 新增字段（`configs/defaults.py`）:

```python
class MemoryConfig(BaseModel):
    # ... 现有字段不变 ...

    vector_backend: str = Field(
        default="sqlite",
        description="向量后端: sqlite(默认)/pgvector/chroma/qdrant",
    )
    keyword_backend: str = Field(
        default="sqlite_bm25",
        description="关键词后端: sqlite_bm25(默认)/rank_bm25/none",
    )
    graph_backend: str = Field(
        default="sqlite",
        description="图后端: sqlite(默认)/age/neo4j",
    )
```

环境变量:
- `SEPTMUSE_VECTOR_BACKEND`: 覆盖向量后端
- `SEPTMUSE_KEYWORD_BACKEND`: 覆盖关键词后端
- `SEPTMUSE_GRAPH_BACKEND`: 覆盖图后端

## 9. pyproject.toml extras

```toml
[project.optional-dependencies]
chroma = ["chromadb>=0.5.0"]
qdrant = ["qdrant-client>=1.9.0"]
neo4j = ["neo4j>=5.0.0"]
bm25 = ["rank-bm25>=0.2.2"]
all-backends = ["chromadb>=0.5.0", "qdrant-client>=1.9.0", "neo4j>=5.0.0", "rank-bm25>=0.2.2"]
```

## 10. 执行顺序（TDD + 增量）

```
Step 1:  VectorStoreBase ABC + SQLiteVectorStore         (test_vector_store_base.py 先写测试)
Step 2:  KeywordIndexBase + SQLiteBM25Index              (test_keyword_index.py 先写测试)
Step 3:  GraphStore.delete_edge 扩展                     (test_graph_store.py 加测试)
Step 4:  MemoryStore.keyword_search + hybrid_search      (test_hybrid_search.py 先写测试)
Step 5:  SQLiteCompositeStore 重构 SQLiteMemoryStore      (跑 614 回归)
Step 6:  PGVectorCompositeStore 重构 PGVectorStore        (跑 9 skipped 逻辑)
Step 7:  ChromaVectorStore + QdrantVectorStore           (extras, @pytest.mark.integration)
Step 8:  RankBM25Index                                    (extras)
Step 9:  configs/defaults.py + pyproject.toml extras
Step 10: 全量回归 ruff + pytest 614+45 新测试全绿
```

## 11. 验证标准

- [ ] `ruff check src/ tests/` All checks passed
- [ ] `ruff format --check` 133 files unchanged
- [ ] `pytest tests/unit -q` 614 + 45 新测试 passed（零退化）
- [ ] `pytest tests/e2e -q` 23 e2e passed
- [ ] `pip install -e .` 成功（默认依赖不变）
- [ ] `pip install -e ".[chroma]"` 成功（extras 可选）
- [ ] `SEPTMUSE_VECTOR_BACKEND=chroma` 启动成功（后端切换可用）
- [ ] `python -m septmuse.api.mcp.server` MCP stdio 启动 0.5s 内（零配置默认不变）

## 12. 风险与缓解

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| SQLiteMemoryStore 重构破坏 614 测试 | 中 | 高 | 双写迁移 + 旧签名不变 + 逐步重构 Step 5 验证 |
| chroma/qdrant 在 Windows 测试不稳定 | 中 | 中 | 标记 @pytest.mark.integration，CI 默认 skip |
| BM25 纯 Python 性能差 | 低 | 低 | 默认 SQLite BM25 够用；大批量用 rank-bm25 extras |
| 向量维度不匹配（HashEmbedder 384 vs Chroma 默认） | 低 | 中 | insert_vectors 校验 + ValueError |
| 双写迁移导致数据不一致 | 低 | 中 | add 失败回滚 + search 回退查 memories.embedding |

## 13. 后续阶段预告

- **P2 权限层**：ACL（RBAC 扩展）+ MemoryAccessLog（借鉴 mem0 openmemory）
- **P3 时态层**：schemas 加 valid_at/invalid_at/expired_at（借鉴 graphiti EntityEdge）
- **P4 编排+扩展**：Pipeline DAG（借鉴 cognee）+ vision（借鉴 mem0）+ auto_dream（借鉴 ReMe）

每个阶段独立 spec → plan → implementation 循环。

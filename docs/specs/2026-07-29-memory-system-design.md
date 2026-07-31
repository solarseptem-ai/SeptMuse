# SeptMuse 记忆系统完整设计文档

> **项目**: SeptMuse — solarseptem-ai 平台第 4 子系统（Agent 记忆系统）
> **版本**: v2.0
> **日期**: 2026-07-29
> **状态**: Draft
> **前置文档**: [架构分析](2026-07-29-memory-architecture-analysis.md)
> **参考**: SeptOntDilig 设计文档风格（分层 + 数据流跳转 + 代码示例 + Phase 范围）

---

## 一、项目概述

### 1.1 背景

Agent 需要跨会话记忆能力。现有开源实现（mem0/Letta/ReMe/cognee/graphiti/MemOS）各有侧重但均不完整。SeptMuse 采用**三维正交 + 数据流管道**双视角架构，既保留认知科学的类型分层，又提供 mem0 式可追踪的写入/检索路径。

### 1.2 核心目标

| 目标 | 衡量标准 |
|------|---------|
| 全形态覆盖 | block/向量/图/文件/激活/参数化 六种存储形态可插拔 |
| 认知分层清晰 | 工作/情节/语义/程序 四类内容严格分离，不混轴 |
| 数据流可追踪 | 从 API 入口到存储到检索，每步可跳转追踪 |
| 零配置可用 | `pip install septmuse` 即用，SQLite + HashEmbedder |
| 三项创新增量 | 因果链 / 遗忘曲线 / 元认知自描述 |

### 1.3 输出方式

| 入口 | 传输 | 文件 |
|------|------|------|
| Python API | 进程内调用 | `from septmuse import Memory` |
| REST API | FastAPI (:8000) | `septmuse serve --with-rest` |
| MCP Tools | stdio / SSE / Streamable HTTP | `septmuse mcp` |
| CLI | argparse | `septmuse <command>` |

### 1.4 演进方式

- **Phase 1-3**: 已完成（MVP + 认知分层 + 横切关注点 + 创新增量 P3）
- **Phase 4**: 进行中（Dream 升级 + 参数化记忆 + 激活记忆）
- **Phase 5**: 计划中（多租户 RBAC + 生产部署）

---

## 二、整体架构

### 2.1 分层架构图（含数据流跳转路径）

```
┌──────────────────────────────────────────────────────────────┐
│                    接入层（四种入口）                          │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│   │ Python   │  │ REST API │  │ MCP Tools│  │   CLI    │    │
│   │ Memory() │  │ FastAPI  │  │ FastMCP  │  │ argparse │    │
│   └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘    │
└────────┼─────────────┼─────────────┼─────────────┼──────────┘
         │             │             │             │
         └─────────────┴─────────────┴─────────────┘
                              │
                              ▼  ← 跳转点①: 统一入口
┌──────────────────────────────────────────────────────────────┐
│                 Memory facade（编排层）                        │
│            orchestration/memory.py                            │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  add()         search()        cognify()     reflect() │  │
│  │  → 写入路由    → 检索路由      → 图谱构建    → 蒸馏     │  │
│  └────────────────────────────────────────────────────────┘  │
│         │           │              │              │            │
└─────────┼───────────┼──────────────┼──────────────┼───────────┘
          │           │              │              │
          ▼           ▼              ▼              ▼  ← 跳转点②: 内容类型路由
┌──────────────────────────────────────────────────────────────┐
│              平面A：内容类型层                                 │
│              content_types/                                    │
│                                                               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐     │
│  │ Working  │  │ Episodic │  │ Semantic │  │Procedural│     │
│  │ (block)  │  │ (episode)│  │ (fact)   │  │ (rule)   │     │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘     │
└───────┼─────────────┼─────────────┼─────────────┼───────────┘
        │             │             │             │
        ▼             ▼             ▼             ▼  ← 跳转点③: 存储形态路由
┌──────────────────────────────────────────────────────────────┐
│              平面B：存储形态层                                 │
│              storage/                                          │
│                                                               │
│  ┌───────┐  ┌───────┐  ┌───────┐  ┌───────┐  ┌───────┐     │
│  │ block │  │ 向量  │  │  图   │  │ 文件  │  │ 激活  │     │
│  │SQLite │  │SQLite │  │SQLite │  │(计划) │  │(计划) │     │
│  └───┬───┘  └───┬───┘  └───┬───┘  └───┬───┘  └───┬───┘     │
│      │          │          │          │          │           │
│      └──────────┴──────────┴──────────┘          │           │
│                 │                                 │           │
│           ┌─────▼──────┐                          │           │
│           │ 源同步器   │  ← 多形态一致性           │           │
│           │ (计划)     │                          │           │
│           └────────────┘                          │           │
└─────────────────────────────────────────────────────┴──────────┘
                              │
                              ▼  ← 跳转点④: 横切关注点
┌──────────────────────────────────────────────────────────────┐
│              平面C：横切关注点层                                │
│              concerns/                                         │
│                                                               │
│  ┌───────┐  ┌───────┐  ┌───────┐  ┌───────┐  ┌───────┐     │
│  │ 捕获  │  │ 检索  │  │ 治理  │  │ 演化  │  │ 共享  │     │
│  │capture│  │retriev│  │govern │  │evoltn │  │sharin │     │
│  └───────┘  └───────┘  └───────┘  └───────┘  └───────┘     │
│              ┌───────────────────┐                           │
│              │  元认知（L0/L1/L2）│                           │
│              └───────────────────┘                           │
└──────────────────────────────────────────────────────────────┘
```

**跳转点说明**：

| 跳转点 | 位置 | 含义 |
|--------|------|------|
| ① | 接入层 → facade | 所有入口统一到 Memory facade，不直接操作存储 |
| ② | facade → 内容类型 | `add()` 根据 `infer` / `memory_type` 路由到四种内容类型 |
| ③ | 内容类型 → 存储形态 | 每种内容类型可写入多种存储形态（如语义=向量+图+文件） |
| ④ | facade → 横切关注点 | 横切关注点是**显式调用**（如 `m.capture()`、`m.link_on_add()`、REST 层权限检查），不是自动 AOP 拦截。未来计划引入 hook 自动化 |

### 2.2 核心设计原则

1. **三维正交 + 数据流追踪**：三维矩阵管"怎么扩展"，管道图管"怎么用"
2. **facade 即边界**：外部只跟 Memory facade 交互，不直接碰存储
3. **零配置优先**：SQLite + HashEmbedder 默认，无 API key 无外部服务
4. **渐进落地**：先跑通工作+语义最小闭环，再逐层叠加
5. **能力边界明确**：不管 context window 管理（上层 Agent 框架负责）

### 2.3 目录映射

| 平面 | 目录 | 说明 |
|------|------|------|
| 接入层 | `cli/` `api/rest/` `api/mcp/` | 三种 API 入口 |
| 编排层 | `orchestration/memory.py` | Memory facade |
| 平面A | `content_types/` | work/episodic/semantic/procedural |
| 平面B | `storage/` | base/sqlite/typed_store/entity_store/graph |
| 平面C | `concerns/` | capture/retrieval/governance/evolution/sharing/metacognition |
| 基础设施 | `providers/` `configs/` `observability/` | embedder/llm/reranker + 配置 + 日志 |

---

## 三、记忆类型设计（平面A）

### 3.1 类型定义（修复后）

```
工作记忆 (context 内, 零检索)
  └─ block (文本, LLM 自编辑)

长时记忆 (跨会话, 需召回)
  ├─ 情节 (带时间锚点的事件)
  │   ├─ 时序事件 (fact)
  │   ├─ 推理经验 (reasoning)
  │   └─ 原始日志 (raw_log)
  ├─ 语义 (事实/偏好/关系)
  │   └─ 身份 (特例) ← tags=["identity"] 的语义事实, 与 persona block 双形态同步
  └─ 程序 (how-to/skill/规则)

(感觉记忆: 瞬时态, <1s, 不持久化, 不参与平面B/C)
```

**修复点**：

| 原设计 | 问题 | 修复 |
|--------|------|------|
| 激活记忆放在工作记忆下 | 操作语义不同(文本vs张量) | 激活移到平面B作为存储形态 |
| 身份归入语义子类 | 横跨工作+语义, 混轴 | 标注 `[sync]` 双形态同步 |
| 感觉记忆在平面A | 不参与B/C组合 | 移出平面A, 标注瞬时态 |

### 3.2 工作记忆（Working Memory）

```python
class WorkingMemory:
    """LLM context window 内, 零检索可见。Agent 可用工具自编辑。"""
    agent_id: str
    blocks: list[Block]  # label: persona / human / task

    def core_memory_append(self, label: str, content: str) -> None: ...
    def core_memory_replace(self, label: str, old: str, new: str) -> None: ...
    def compile_to_xml(self) -> str: ...  # 注入 system prompt
```

**编译输出**（注入 system prompt）：
```xml
<memory>
  <block label="persona">I am a self-improving agent...</block>
  <block label="human">Name: Timber. Occupation: ...</block>
</memory>
```

### 3.3 情节记忆（Episodic）

```python
# schemas/episodic.py — table="septmuse_episodic"
class EpisodicEvent(SQLModel, table=True):
    id: str                          # "epi-{uuid}"
    event_type: str                  # EpisodeType: fact | reasoning | raw_log
    content: str                     # 事件内容
    reference_time: datetime         # 时间锚点 (Zep reference_time)
    user_id: str                     # 用户 ID
    agent_id: str | None             # agent ID
    session_id: str | None           # 会话 ID (raw_log 关联)
    # reasoning 专用字段 (event_type=reasoning, 对齐 LangMem Episode)
    observation: str | None
    thoughts: str | None
    action: str | None
    result: str | None
    created_at: datetime
    is_deleted: bool = False
```

### 3.4 语义记忆（Semantic）

```python
# schemas/semantic.py — table="septmuse_facts"
class SemanticFact(SQLModel, table=True):
    id: str                          # "fact-{uuid}"
    subject: str                     # 三元组主语
    predicate: str                   # 三元组谓语
    object: str                      # 三元组宾语
    context: str | None              # 上下文限定
    org_id: str = "default"          # 多租户 (LangMem namespace)
    user_id: str                     # 用户 ID (跨 agent 共享键)
    confidence: float = 1.0          # 置信度 [0,1], 区分事实/推断
    provenance: str = "user"         # user | inferred | tool | observed
    tags: list[str] = []             # identity / role / preference / ...
    embedding: bytes | None = None   # 向量共存 (平面B)
    created_at: datetime
    updated_at: datetime
    is_deleted: bool = False
```

> **注意**: `valid_at` / `invalid_at` 不在 SemanticFact 上，而在 `memories` 表（verbatim 存储层）。双时态是 verbatim 记忆的属性，不是语义事实的属性。`Memory.add(valid_at=...)` 写入 verbatim 表，`Memory.invalidate()` 也操作 verbatim 表。

**身份同步机制**：身份记忆是特例——`tags=["identity"]` 的语义事实与 persona block（工作记忆）理论上应通过源同步器保持一致。当前源同步器尚未实现，身份记忆需手动同步。普通语义事实不涉及 block。

### 3.5 程序记忆（Procedural）

```python
# schemas/procedural.py — table="septmuse_procedural"
class ProceduralRule(SQLModel, table=True):
    id: str                          # "rule-{uuid}"
    rule: str                        # 规则内容 (how-to/skill/heuristic)
    namespace: str = "default"       # 命名空间
    user_id: str                     # 用户 ID
    helpful_count: int = 0           # 带来正面结果次数
    harmful_count: int = 0           # 带来负面结果次数
    source_tracing: str | None       # 溯源到 episodic session
    deprecated: bool = False         # 规则退化标记 (harmful > helpful 且 >= 3)
    tags: list[str] = []             # 分类标签
    created_at: datetime
    updated_at: datetime
    is_deleted: bool = False

    @property
    def confidence(self) -> float:   # = helpful / (helpful + harmful), 无记录时 0.5
        ...
```

---

## 四、存储形态设计（平面B）

### 4.1 形态定义

| 形态 | 实现 | 说明 |
|------|------|------|
| block | SQLite | 工作记忆文本块（TypedMemoryStore 管理） |
| 向量 | SQLite FTS5 + 向量 | 嵌入存储 + BM25 全文索引（MemoryStore/SQLiteMemoryStore） |
| 图（记忆间） | SQLite memory_links 表 | 记忆间边，由 GraphStore ABC 管理（add_edge/get_neighbors） |
| 图（实体间） | SQLite entity_relations 表 | 实体间关系边，由 CognifyPipeline 直接操作（不走 GraphStore ABC） |
| 文件 | （计划） | Markdown 双向同步（源同步器管理） |
| 激活 | （计划） | KV-Cache 张量存储 |
| 参数化 | （计划） | LoRA 权重 |

### 4.2 存储抽象

```python
# storage/base.py
class MemoryStore(ABC):
    """记忆存储后端抽象 (平面B 向量形态)"""
    def add(self, content: str, embedding: list[float], *,
            user_id: str, agent_id: str | None = None,
            metadata: dict | None = None,
            valid_at: str | None = None) -> str: ...  # valid_at 写入 memories 表
    def search(self, query_embedding: list[float], *,
               user_id: str, top_k: int = 5, threshold: float = 0.1) -> list[dict]: ...
    def get(self, memory_id: str) -> dict | None: ...
    def delete(self, memory_id: str) -> None: ...  # 软删除
    def update(self, memory_id: str, content: str, embedding: list[float], *,
               metadata: dict | None = None) -> bool: ...
    def invalidate(self, memory_id: str, *, invalid_at: str | None = None) -> dict: ...
    def get_temporal_valid(self, reference_time: str, *, user_id: str) -> list[dict]: ...

# storage/graph/base.py
class GraphStore(ABC):
    """图存储抽象 (平面B 图形态, 6 方法)"""
    def add_edge(self, source_id: str, target_id: str,
                 relation: str = "related_to", score: float = 0.0) -> str: ...
    def get_edges(self, node_id: str) -> list[GraphEdge]: ...
    def get_neighbors(self, node_id: str, relation: str | None = None) -> list[str]: ...
    def has_edge(self, source_id: str, target_id: str, relation: str) -> bool: ...
    def delete_edge(self, edge_id: str) -> bool: ...
    def close(self) -> None: ...

# storage/entity_store.py
class EntityStore:
    """实体向量库 (平面B, 借鉴 mem0 V3 去图化)"""
    def upsert(self, entity: Entity, memory_id: str, *,
               user_id: str, agent_id: str | None = None) -> str: ...
    def search(self, query: str, *, user_id: str, top_k: int = 5) -> list[dict]: ...
    def list(self, *, user_id: str, entity_type: str | None = None, limit: int = 100) -> list[dict]: ...
    def get_linked_memories(self, entity_id: str) -> list[str]: ...
```

### 4.3 源同步器（平面B 内部，不是平面C）

源同步器保证同一份记忆在多形态间的一致性。它是**形态间协调机制**，与平面C 的生命周期关注点不在同一概念层级。

```python
class SourceSynchronizer:
    """多形态共存一致性 (平面B 内部)"""
    def sync_on_add(self, memory_id, text, embedding, *, user_id):
        # 向量库写 + 图库建链接 + 文件写 Markdown
        ...
    def sync_on_update(self, memory_id, new_text, *, user_id):
        # 所有形态同步更新
        ...
```

### 4.4 默认后端

三个后端独立配置（`SEPTMUSE_VECTOR_BACKEND` / `SEPTMUSE_KEYWORD_BACKEND` / `SEPTMUSE_GRAPH_BACKEND`）：

| 后端 | 向量 (`VECTOR`) | 关键词 (`KEYWORD`) | 图 (`GRAPH`) | 条件 |
|------|------|--------|-----|------|
| SQLite (默认) | sqlite | sqlite_bm25 | sqlite | 零配置 |
| Postgres | pgvector | sqlite_bm25 / none | age (Apache AGE) | `SEPTMUSE_TEST_PG_DSN` |
| Chroma | chroma | — | — | `pip install septmuse[chroma]` |
| Qdrant | qdrant | — | — | `pip install septmuse[qdrant]` |
| Neo4j | — | — | neo4j | `pip install septmuse[neo4j]` |

---

## 五、横切关注点设计（平面C）

| 关注点 | 目录 | 职责 |
|--------|------|------|
| 捕获 | `concerns/capture/` | PostToolUse hook → SHA256 去重 → 脱敏 → 嵌入 → 双索引 |
| 检索 | `concerns/retrieval/` | 混合检索 + RRF 融合 + reranker + 图遍历 + 时态 + recipe |
| 治理 | `concerns/governance/` | 权限 + 审计 + token 预算 + 隐私脱敏 + 写审批 |
| 演化 | `concerns/evolution/` | Zettel链接 + reflect蒸馏 + Dream + 冲突解决 + 实体去重 |
| 共享 | `concerns/sharing/` | user_id 跨 agent 共享 |
| 元认知 | `concerns/metacognition/` | L0 路由 + L1 覆盖自描述 + L2 策略自调 |

**修复点**：源同步器从平面C 移到平面B 内部。

---

## 六、写入管道（数据流图）

> 借鉴 mem0 管道图风格。每步可跳转到对应代码。

```
Messages In (str | list[dict])
    │
    ▼
┌─────────────────────┐
│  1. 路由             │  infer=True → LLM 事实抽取 (FactExtractor → §3.4 SemanticFact)
│  Memory.add()       │  infer=False → verbatim 原文存 (默认)
│  type=verbatim/     │  type=semantic → 三元组 (add_fact, 独立入口)
│       semantic/     │  type=episodic → 情节事件 (add_episode, 独立入口)
│       episodic/      │  type=procedural → 程序规则 (add_rule, 独立入口)
│       procedural     │
└─────────┬───────────┘
          │  (仅 infer=True 路径)
          ▼
┌─────────────────────┐
│  2. LLM 抽取 (可选)  │  FactExtractor.extract_and_store()
│  content_types/      │  ADDITIVE_EXTRACTION_PROMPT (9 few-shot)
│  semantic/extract.py │  输出 linked_memory_ids (跨记忆链接)
│                     │  无 LLM → 跳过此步, 直接 verbatim
└─────────┬───────────┘
          │  (infer=False 或抽取后)
          ▼
┌─────────────────────┐
│  3. 嵌入             │  HashEmbedder (默认, 离线, 0.5s)
│  embedder.embed()   │  ONNX (ModelScope, 无 torch, CPU <50ms)
│                     │  SentenceTransformer (延迟 import, ~30s)
│                     │  嵌入失败 → 降级 HashEmbedder (零配置兜底)
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  4. 存储             │  store.add() → memories 表 (向量 + BM25 索引)
│  storage/sqlite/     │  valid_at/invalid_at 写入 memories 表 (双时态)
│  store.py            │
└─────────┬───────────┘
          │  (仅 auto_extract_entities=True 且 verbatim 模式)
          ▼
┌─────────────────────┐
│  5. 实体抽取         │  entity_extractor.extract(text) → Entity 列表
│  EntityStore.upsert  │  每个 Entity 关联 memory_id, 写入 septmuse_entities 表
│                     │  失败 → logger.warning, 不阻塞主流程
└─────────┬───────────┘
          │
          ▼
    Memory Object
    {"results": [{"id","memory","event":"ADD"}], "relations": []}
```

> **注意**: 图链接（`graph_store.add_edge`）**不在 `add()` 内自动执行**。图链接由独立方法创建：
> - `m.link_on_add(memory_id, text, ...)` — Zettelkasten 自动建链接（需显式调用）
> - `m.cognify(text, ...)` — 完整知识图谱流水线（存记忆→抽三元组→存实体/关系→建链接）
>
> 权限检查和审计日志在 **REST/MCP 层**，不在 `add()` 内。`record_access` 在 `search`/`get`/`delete` 中调用。

---

## 七、检索管道（数据流图）

> 借鉴 mem0 检索管道。支持 recipe 一键切换策略。

```
Query In
    │
    ▼
┌─────────────────────┐
│  1. 混合检索 (默认)   │  向量相似度 (cosine) — store.search()
│  Memory.search()    │  BM25 关键词匹配 — store.keyword_search()
│  HybridRetriever    │  实体提升 (entity_boost) — EntityStore.search()
│                     │  三信号在 HybridRetriever 内并行评分
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  2. 融合              │  RRF (k=60) 向量+BM25+entity 三信号融合
│  rrf_fuse()         │  score 统一为相似度 [0,1]
│                     │  hybrid=False 时跳过, 纯向量检索
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  3. 重排 (可选)       │  noop (默认, 透传, 零开销)
│  Reranker            │  mmr (最大边际相关性, 去冗余, 相似度>0.9只留一个)
│  SEPTMUSE_RERANKER   │  cross_encoder (ONNX bge-reranker-v2-m3)
│  或 --reranker 参数   │  llm (LLM 逐条打分 0-1)
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  4. 审计 + 返回       │  record_access (审计日志, 吞错不阻塞)
│  access_log.py      │  返回 list[dict]
└─────────────────────┘

Results: [{"id","memory","score","vector_score","bm25_score","entity_boost",...}]
```

> **以下步骤不在 `search()` 默认流程中，是独立入口**：
>
> | 独立方法 | 功能 | 何时用 |
> |---------|------|--------|
> | `meta_route(query)` | L0 元认知路由：决定查哪些命名空间 | 需要跨命名空间智能路由时显式调用 |
> | `search_at(reference_time, query)` | 时态过滤：valid_at <= time AND invalid_at > time | 查某时刻为真的事实 |
> | `search_interval(start, end, query)` | 时态区间查询 | 查时间段内为真的记忆 |
> | `search_natural(query)` | LLM 自然语言时间抽取 → 时态查询 | "上周"、"三天前"等自然语言 |
> | `search_graph(seed_id)` | BFS 图遍历（graph_score=1/2^depth） | 关联追溯 |
> | `search_graph_fused(query, seed_id)` | BFS + 向量 RRF 融合 | 关联 + 语义混合 |
> | `search_progressive(query)` | 渐进三层 recall→locate→expand | 低信噪比场景 |
> | `search_with_strength(query)` | 遗忘曲线加权 final_score=relevance×strength | 时间衰减排序 |

**Recipe 一键切换**：

| Recipe | 混合 | 重排 | 图遍历 | 适用场景 |
|--------|------|------|--------|----------|
| HYBRID_RRF (默认) | ✅ | noop | — | 通用 |
| HYBRID_RRF_ENTITY | ✅ | noop | — | +explain 评分详情 |
| HYBRID_RRF_CROSS_ENCODER | ✅ | cross_encoder | — | 高精度 |
| HYBRID_RRF_MMR | ✅ | mmr | — | 去冗余 |
| GRAPH_BFS | — | — | ✅ | 关联追溯 |
| PROGRESSIVE | 渐进三层 | — | — | recall→locate→expand |
| FORGETTING | ✅ | — | — | 遗忘曲线加权 |

---

## 八、三个创新增量（自研）

### 8.1 因果链记忆（Causal Memory Graph）

14 家开源均无因果边 + 反事实查询。

```python
m.add_causal_edge(cause_event_id, effect_event_id, *, relation="causes", confidence=0.5)
m.find_causes(event_id, *, user_id)  # 前因路径
m.find_effects(event_id, *, user_id)  # 后果路径
m.counterfactual(cause_event_id, effect_event_id, *, user_id)  # 反事实
# → {"would_still_occur": bool, "confidence": float, "reasoning": str}
```

### 8.2 Ebbinghaus 遗忘曲线

14 家均无强度衰减 + 主动复述。

```python
# 遗忘曲线加权
m.search_with_strength(query, *, user_id)
# → final_score = relevance × strength

# 主动复述强化
m.rehearse(memory_id, *, user_id)
# → strength 回升, access_count++

# 找需要复述的候选
m.find_rehearse_candidates(*, user_id)
# → strength < 0.3 且 base_value > 0.7
```

### 8.3 元认知自描述

ReMe 仅 L0 路由，SeptMuse 增加 L1 覆盖自描述 + L2 策略自调。

```python
m.meta_route(query)  # L0: 路由 → 决定查哪些命名空间
m.coverage_report(*, user_id)  # L1: "我记住了什么/记不住什么"
m.adapt_strategy(*, user_id)  # L2: 基于覆盖报告自调检索策略
```

---

## 九、组合矩阵（修复后）

|              | block | 向量 |  图  | 文件 | 激活 | 参数化 |
|--------------|:-----:|:----:|:----:|:----:|:----:|:-----:|
| 工作记忆      |  ✅   |  N   |  N   |  —   |  —   |  N    |
| 情节（长时）  |  —    |  ✅  |  ✅  |  —   |  —   |  N    |
| 语义（长时）  |  —    |  ✅  |  ✅  |  —   |  —   |  N    |
| 程序（长时）  |  ✅   |  ✅  |  —   |  —   |  —   |  —    |

**图例**：
- `✅` = 该组合成立且已实现
- `—` = 概念成立但当前不实现
- `N` = 概念不成立（如工作记忆×向量=N，因为工作记忆在 context window 内不需嵌入检索）

> **身份记忆特例**：`语义×block` 标 `—` 而非 `✅`，因为只有 `tags=["identity"]` 的语义事实会与 persona block 双形态同步。普通语义事实不涉及 block。双形态同步由源同步器（计划中）管理，当前需手动同步。

**修复点**：
- 身份从 `[sync]` 改为 `—`（普通语义事实不涉及 block；身份是特例，注释说明）
- 激活从工作记忆行移除（激活现在是平面B 存储形态，不是平面A 内容类型）
- 每个空白格标注 `N` 或 `—`，区分"概念不成立"和"当前不实现"

---

## 十、技术选型

| 层 | 选型 | 理由 |
|----|------|------|
| 后端框架 | FastAPI | 异步原生、类型安全、自动 OpenAPI |
| 数据模型 | Pydantic v2 | 强类型、与 FastAPI 一致 |
| 默认存储 | SQLite | 零配置、嵌入式 |
| 向量库 (可选) | pgvector | 与 Postgres 同实例，减少依赖 |
| 图库 (可选) | Apache AGE | Postgres 图扩展 |
| 全文检索 | SQLite FTS5 / BM25 | 混合检索一路 |
| 嵌入器 | HashEmbedder (默认) / ONNX | 离线可用 / ModelScope 下载 |
| LLM | OpenAI / Ollama / Anthropic / DashScope | 可插拔 |
| MCP 集成 | FastMCP | @mcp.tool 注册工具 |
| 日志 | structlog | 结构化日志 |

---

## 十一、演进路线

### Phase 1：最小闭环（MVP）✅ 已完成
- **范围**: 工作 Block + 语义记忆（向量）+ verbatim 模式 + 混合检索
- **目标**: agent 能记住用户偏好并在下一会话召回
- **验证**: 跨会话偏好召回率 ≥ 80%

### Phase 2：认知分层完整 ✅ 已完成
- **范围**: 情节（时序+推理+日志）+ 程序（规则退化）+ 双时态
- **目标**: 四类内容类型齐备，人可读审计

### Phase 3：横切关注点 + LLM 深度集成 进行中（4/5 Task 完成）
- **范围**: hook 捕获 + 渐进检索 + 治理 + 演化 + 共享 + LLM 抽取 + 冲突解决 + 蒸馏
- **目标**: 生产可用
- **当前**: P3-Task 5 LLM 自编辑记忆 待完成

### Phase 4：创新增量 + 激活/参数化 计划中
- **范围**: Dream 升级（四阶段）+ 激活记忆 (KV-Cache) + 参数化 (LoRA) + 源同步器
- **前提**: Phase 3 完成后启动；激活/参数化需自托管模型后端

### Phase 5：生产部署
- **范围**: 多租户 RBAC + Docker 生产 + pgvector + AGE
- **目标**: 企业级部署

---

## 十二、降级策略

零配置优先意味着所有可选依赖不可用时，系统必须优雅降级而非崩溃。

| 组件 | 降级条件 | 降级行为 | 影响范围 |
|------|---------|---------|---------|
| LLM | `SEPTMUSE_LLM` 未设 / provider 不可用 | `infer=False` verbatim 模式（原文存，不 LLM 抽取） | 无事实抽取、无 LLM 蒸馏、无自然语言时态查询 |
| Embedder | ONNX/SentenceTransformer 不可用 | 自动降级 HashEmbedder（离线，0.5s，零模型加载） | 检索质量降低（hash 是确定性 hash 非语义），但功能不中断 |
| Reranker | `SEPTMUSE_RERANKER=cross_encoder` 但 ONNX 不可用 | 降级 noop（透传，不改变顺序） | 无重排，但检索结果不受影响 |
| EntityExtractor | spaCy 不可用 | 降级 regex（纯 Python regex + 词表，零配置） | 实体类型仅 4 类（PROPER/QUOTED/TOPIC/IDENTIFIER），无 noun_chunks |
| GraphStore | `graph_store is None`（PGVectorStore 无图后端） | Zettel/Dream/cognify/search_graph 不可用，`assert` 崩溃 | 图相关功能全部不可用，但 CRUD 正常 |
| SQLite `:memory:` | FastAPI TestClient 跨线程 | 连到新空库（数据不可见） | **REST/e2e 测试必须用文件 DB**：`tmp_path / "test.db"` |
| 审计日志 | `record_access` 失败 | 吞错（`logger.warning`），不阻塞业务 | 审计日志缺失但不影响主流程 |
| EntityStore | `entity_store is None`（非 SQLiteMemoryStore） | `extract_entities`/`search_entities`/`list_entities` 返回空 | 实体功能不可用，但 CRUD 正常 |

**设计原则**：降级链是 HashEmbedder → 功能降级 → 崩溃。优先保证 CRUD 不中断，其次保证检索可用，最后才追求检索质量。

---

## 十三、并发与存储约束

| 约束 | 原因 | 应对 |
|------|------|------|
| SQLite `:memory:` 跨线程失效 | 每个 `:memory:` 库是 per-connection 的，FastAPI 跨线程会连到新空库 | REST/e2e 测试一律用 `tmp_path / "test.db"` 文件路径 |
| SQLite 写锁 | SQLite 默认单写者，`threading.Lock` 串行化写操作 | 高并发写场景需切 pgvector（Postgres MVCC） |
| SQLite `ALTER TABLE` 运行时迁移 | 无 alembic，`_migrate_add_state_columns` 等在代码内 `ALTER TABLE` | 首次启动自动迁移，旧库兼容（`hasattr` 检查列是否存在） |
| `graph_store is None` assert | PGVectorStore 无图后端，图相关方法直接 `assert` 崩溃 | 调用前检查 `self.graph_store is not None`，或用 SQLite 默认后端 |
| FastMCP 返回注解限制 | `from __future__ import annotations` 导致 FastMCP `func_metadata` 把返回注解当字符串解析会炸 | MCP tools.py 禁用 future annotations，用具体类型 |
| `record_access` 向后兼容 | 旧 store 可能没有 `get_access_logs` 方法 | `hasattr` 检查，吞错不阻塞 |

**多线程安全**：SQLiteMemoryStore 用 `threading.Lock` 保护所有读写。TypedMemoryStore 和 EntityStore 复用同一 lock。REST API（FastAPI）是 async，但 SQLite 操作是同步的（`run_in_executor` 或直接调用）。

---

## 十四、验证标准

- [ ] 三维正交骨架在代码中表现为独立模块（`content_types/` / `storage/` / `concerns/`）
- [ ] 写入管道 5 步可追踪（路由→抽取→嵌入→存储→横切hook）
- [ ] 检索管道 6 步可追踪（路由→评分→时态→融合→重排→审计）
- [ ] 身份记忆标注 `[sync]` 双形态同步，不归入单一类型
- [ ] 激活记忆在平面B（存储形态），不在平面A（内容类型）
- [ ] 组合矩阵每个空白格标注 `N`（概念不成立）或 `—`（当前不实现）
- [ ] 源同步器在平面B 内部，不在平面C
- [ ] 三个创新增量各有独立测试集
- [ ] `pip install septmuse` 零配置可用
- [ ] 1008 passed (unit + e2e 合计), 36 skipped

---

## 十五、附录

### 15.1 术语对照

| 术语 | 说明 |
|------|------|
| Memory facade | SeptMuse 统一入口，所有 API 通过它操作 |
| 跳转点 | 架构图中数据流可追踪的关键路由节点 |
| 三维正交 | 内容类型 × 存储形态 × 横切关注点 三个平面分离 |
| `[sync]` | 双形态同步标记，表示记忆跨类型存在 |
| Recipe | 预置检索配置，一键切换混合/重排/图遍历策略 |
| 双时态 | valid_at（开始为真）+ invalid_at（不再为真） |
| RRF | Reciprocal Rank Fusion，多信号融合（k=60） |

### 15.2 环境变量

| 变量 | 默认 | 作用 |
|------|------|------|
| `SEPTMUSE_DB_PATH` | `~/.septmuse/septmuse.db` | SQLite 路径 |
| `SEPTMUSE_EMBEDDER` | `hash` | `hash`/`onnx`/`onnx-zh`/`auto`/`st` |
| `SEPTMUSE_LLM` | 未设 | `openai`/`ollama`/`anthropic`/`dashscope` |
| `SEPTMUSE_INFER` | `false` | `true` 启用 LLM 抽取事实 |
| `SEPTMUSE_RERANKER` | `noop` | `noop`/`mmr`/`cross_encoder`/`llm` |
| `SEPTMUSE_API_KEY` | 未设 | 未设=开发模式；已设=生产认证 |

### 15.3 参考文档

- [架构分析](2026-07-29-memory-architecture-analysis.md) — 6 家开源对比 + 问题诊断
- [接口文档](../api/README.md) — 四种 API 入口完整参考
- [开发路线图](../plans/development-roadmap.md) — 28 Task 逐项跟踪
- [mem0 architecture](https://github.com/mem0ai/mem0) — 写入/检索管道图参考
- [ReMe framework](https://github.com/AgentScope/ReMe) — 调用链 + 能力边界参考
- [MemOS ARCHITECTURE](https://github.com/MemTensor/MemOS) — 分层 + L1/L2/L3 参考
- [graphiti](https://github.com/getzep/Graphiti) — temporal context graph 参考

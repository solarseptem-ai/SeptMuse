# SeptMuse V2 记忆架构设计

> 日期：2026-08-04
> 前置文档：`docs/specs/agent-memory-architecture.md`（三维正交架构总设计）
> 目标：在不动 V1 `Memory`（main.py）的前提下，新增 `memory/memory_v2.py` + `memory/base.py`，实现 V2 编排入口 + 记忆 ABC 分层。
> 原则：组合不继承、零 LLM 降级、子组件不直接耦合、编排逻辑在 facade。

---

## Brainstorming 决策记录

| # | 问题 | 决策 | 理由 |
|---|---|---|---|
| 1 | 两套 SemanticMemory 实例并存 | V2 自己创建新的子组件实例 | 不复用 Memory 内部实例，V2 完全独立 |
| 2 | ABC 方法签名不匹配已有实现 | 理解 A：ABC 只做类型标记 + 各层特有方法 | 不强制统一 add/search，各子类保持原名（add_fact/add_raw_log/add_rule） |
| 3 | working_memory_store 自相矛盾 | 路径 B：独立 WorkingMemoryStore | Block 表迁移到独立后端，彻底分库 |
| 4 | memory/ 子组件 vs models/ 关系 | B：全新定义操作类 | memory/ 下从头实现，不 import models/ 的操作类 |
| 5 | forget 语义 | 语义 3：先 invalidate 再 delete | 彻底遗忘但保留双时态历史轨迹 |
| 6 | L1 报告首次不存在 | 策略 A：跳过 L2 正常检索 | L1 是离线生成，recall 不被阻塞 |
| 边界 | 数据模型是否也全新 | A：数据模型共享 | SQLModel.metadata 单例，重复定义同名 table 会冲突 |

---

## 目录

1. [架构总览](#1-架构总览)
2. [记忆 ABC 分层（base.py）](#2-记忆-abc-分层basepy)
3. [V2Memory 编排入口](#3-v2memory-编排入口)
4. [10 个子组件清单](#4-10-个子组件清单)
5. [存储划分](#5-存储划分)
6. [目录结构](#6-目录结构)
7. [4 个编排方法数据流转](#7-4-个编排方法数据流转)
8. [零 LLM 降级路径](#8-零-llm-降级路径)
9. [V1 与 V2 关系 + 迁移路径](#9-v1-与-v2-关系--迁移路径)
10. [环境变量配置](#10-环境变量配置)
11. [验收标准](#11-验收标准)

---

## 1. 架构总览

```
┌──────────────────────────────────────────────────────────────────┐
│  V2Memory (memory/memory_v2.py)                                  │
│  编排入口：remember / recall / improve / forget                  │
│  组合 Memory 实例 + 10 子组件（不继承 Memory）                    │
└──────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  平面 A          │  │  平面 B          │  │  平面 C          │
│  内容类型操作    │  │  存储形态后端    │  │  横切关注点      │
│  (4 子组件)     │  │  (8 后端)        │  │  (6 子组件)      │
│                  │  │                  │  │                  │
│  working_memory  │  │  working_memory  │  │  capture         │
│  semantic        │  │  _store          │  │  retrieval       │
│  episodic        │  │  store (关系型)  │  │  meta             │
│  procedural      │  │  vector_store   │  │  evolution        │
│                  │  │  keyword_index   │  │  causal           │
│                  │  │  graph_store     │  │  forgetting       │
│                  │  │  file_store      │  │                  │
│                  │  │  activation      │  │                  │
│                  │  │  parametric      │  │                  │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

**核心约束**：
- V2Memory **不继承** Memory，组合 Memory 实例
- 子组件之间**不直接耦合**，编排逻辑在 V2Memory 里
- 平面 B 存储后端通过子组件**间接使用**
- 零 LLM 时系统降级为 verbatim + 向量检索 + 规则演化，不崩溃

---

## 2. 记忆 ABC 分层（base.py）

新建 `memory/base.py`，定义记忆的 ABC 基类，区分短期记忆 vs 长期记忆。

### 2.1 类继承关系

```
MemoryABC (ABC)                        ← 所有记忆的根抽象
  ├─ ShortTermMemory (ABC)             ← 短期记忆（context 内，零检索）
  │   └─ WorkingMemory                 ← 工作记忆（Block，已有实现）
  │
  └─ LongTermMemory (ABC)               ← 长期记忆（跨会话，需召回）
      ├─ SemanticMemory                 ← 语义记忆（已有实现）
      ├─ EpisodicMemory                 ← 情节记忆（已有实现）
      └─ ProceduralMemory               ← 程序记忆（已有实现）
```

### 2.2 ABC 方法定义

```python
# memory/base.py

from abc import ABC, abstractmethod


class MemoryABC(ABC):
    """记忆根抽象 — 类型标记，不强制统一 add/search 方法。

    设计决策（brainstorming 确认）：
    - 各子类 add 方法保持原名（add_fact / add_raw_log / add_rule / core_memory_append）
    - 不强制统一为 add()，因为参数签名不同（add_fact(subject,predicate,object) vs add_raw_log(transcript)）
    - ABC 价值：类型标记（isinstance 判断短期 vs 长期）+ 各层特有方法约束
    """


class ShortTermMemory(MemoryABC):
    """短期记忆 — context window 内，零检索即可见。

    特征：
    - 编译注入 system prompt（compile_to_prompt）
    - 容量有限（token/char limit）
    - 超限驱逐到长期记忆
    - 跨会话持久化但每次会话加载到 context

    借鉴：Atkinson-Shiffrin 工作记忆 + Letta Block。

    各子类 add 方法保持原名（core_memory_append / core_memory_replace），
    不强制统一为 add()。
    """

    @abstractmethod
    def compile_to_prompt(self) -> str:
        """编译为可注入 system prompt 的文本。"""
        ...

    @abstractmethod
    def get_limit(self) -> int:
        """获取容量上限（字符数或 token 数）。"""
        ...

    @abstractmethod
    def evict_overflow(self) -> list[dict]:
        """驱逐超限内容到长期记忆，返回被驱逐的内容列表。"""
        ...


class LongTermMemory(MemoryABC):
    """长期记忆 — 跨会话持久，需检索召回。

    特征：
    - 持久化到 DB（SQLite/PG/MySQL）
    - 需要向量/BM25/图检索才能召回
    - 双时态（valid_at / invalid_at）
    - 软删除 + 历史保留

    借鉴：Atkinson-Shiffrin 长时记忆 + Tulving 分层（情节/语义/程序）。

    各子类 add 方法保持原名（add_fact / add_raw_log / add_rule），
    不强制统一为 add()。
    """

    @abstractmethod
    def invalidate(self, memory_id: str, *, invalid_at: str | None = None) -> bool:
        """标记事实不再为真（双时态：设 invalid_at，不删除）。"""
        ...

    @abstractmethod
    def get_history(self, memory_id: str) -> list[dict]:
        """获取记忆变更历史（审计用）。"""
        ...

    @abstractmethod
    def get_all(self, *, user_id: str, limit: int = 100) -> list[dict]:
        """列出用户全部记忆（分页）。"""
        ...
```

### 2.3 与现有实现的关系

| ABC 基类 | V2 子组件（memory/ 下全新定义） | 数据模型（models/ 共享） |
|---|---|---|
| `ShortTermMemory` | `memory/working_memory.py` → `WorkingMemory` | `models/block.py` → `Block`（SQLModel table） |
| `LongTermMemory` | `memory/semantic.py` → `SemanticMemory` | `models/semantic.py` → `SemanticFact`（SQLModel table） |
| `LongTermMemory` | `memory/episodic.py` → `EpisodicMemory` | `models/episodic.py` → `EpisodicEvent`（SQLModel table） |
| `LongTermMemory` | `memory/procedural.py` → `ProceduralMemory` | `models/procedural.py` → `ProceduralRule`（SQLModel table） |

**设计决策（brainstorming 确认）**：
- **数据模型共享**：V2 import models/ 的 SQLModel table 定义（Block/SemanticFact/EpisodicEvent/ProceduralRule），避免 SQLModel.metadata 单例冲突
- **操作类全新**：V2 在 memory/ 下重新定义 WorkingMemory/SemanticMemory 等操作类，不 import models/ 的操作类
- **不注册现有实现为 ABC 子类**：models/ 下的操作类不动、不继承 ABC，只有 memory/ 下的 V2 操作类继承 ABC

---

## 3. V2Memory 编排入口

### 3.1 类定义

```python
# memory/memory_v2.py

class V2Memory:
    """V2 编排入口 — remember / recall / improve / forget。

    全新类，不继承 Memory。组合 Memory 实例 + 10 子组件。
    零 LLM 可用：无 SEPTMUSE_LLM 时降级为 verbatim + 向量检索 + 规则演化。

    用法 1（从 config 创建）:
        v2 = V2Memory(config=MemoryConfig())
        v2.remember("我喜欢 Python", user_id="alice")

    用法 2（传入已有 Memory）:
        mem = Memory(config=config)
        v2 = V2Memory(memory=mem)
        v2.recall("帮我写 API", user_id="alice")
    """

    def __init__(
        self,
        memory: Memory | None = None,
        *,
        config: MemoryConfig | None = None,
    ) -> None:
        # === 持有 Memory 实例（组合，不继承）===
        self.mem = memory or Memory(config=config)

        # === 平面 B 后端（从 Memory 复用）===
        self.store = self.mem.store               # ORMMemoryStore
        self.embedder = self.mem.embedder         # HashEmbedder
        self.typed_store = self.mem.typed_store    # TypedMemoryStore
        self.graph_store = self.mem.graph_store   # GraphStore
        self.entity_store = self.mem.entity_store # EntityStore
        self.entity_extractor = self.mem.entity_extractor
        self.llm = self.mem.llm                   # LLM | None
        self._dedup_window = self.mem._dedup_window

        # === 10 个子组件（V2 全新创建，不复用 Memory 内部实例）===
        # 平面 A（4 个）— memory/ 下全新定义，数据模型共享 models/
        self.working_memory = WorkingMemory(
            store=...,  # WorkingMemoryStore（独立后端，非 typed_store）
            agent_id="default",
        )
        self.semantic = SemanticMemory(self.typed_store, self.embedder)
        self.episodic = EpisodicMemory(self.typed_store)
        self.procedural = ProceduralMemory(self.typed_store)
        # 平面 C（6 个）
        self.capture = CapturePipeline(...)
        self.retrieval = HybridRetriever(...)
        self.token_budget = TokenBudget(budget=2000)
        self.meta = MetacognitionLayer(...)
        self.evolution = EvolutionEngine(...)
        self.causal = CausalGraph(...)
        self.forgetting = ForgettingManager(...)
```

### 3.2 为什么组合不继承

| 维度 | 继承 Memory | 组合 Memory（选这个） |
|---|---|---|
| Memory 类污染 | V2 方法塞进 Memory，违反"不改 main.py" | Memory 不动 |
| 方法暴露 | 用户看到 47+4 方法 | 用户只看到 4 方法 |
| 可替换性 | 绑死 Memory 实现 | 可换 Memory 实现 |
| 测试隔离 | V2 测试要 mock Memory 继承链 | 只 mock 子组件 |

---

## 4. 10 个子组件清单

| # | 子组件 | 平面 | 类型 | 职责 | V2 文件（memory/ 下） | import 来源 |
|---|---|:-:|---|---|---|---|
| 1 | `working_memory` | A | ShortTermMemory | Block CRUD + compile_to_xml + 超限驱逐 | memory/working_memory.py | models/block.py |
| 2 | `semantic` | A | LongTermMemory | SemanticFact CRUD + 向量检索 + 置信度加权 | memory/semantic.py | models/fact.py |
| 3 | `episodic` | A | LongTermMemory | EpisodicEvent CRUD + 时序查询 + 三子类 | memory/episodic.py | models/episodic.py |
| 4 | `procedural` | A | LongTermMemory | ProceduralRule CRUD + helpful/harmful 退化 | memory/procedural.py | models/procedural.py |
| 5 | `capture` | C | Pipeline | SHA-256 去重 + 隐私脱敏 + 隐式 hook | memory/capture.py | capture/pipeline.py |
| 6 | `retrieval` | C | Retriever | 三信号融合（向量+BM25+entity boost） | memory/retrieval.py | retrieval/hybrid.py |
| 7 | `meta` | C | Metacognition | L0 路由 + L1 覆盖报告 + L2 策略自调 | memory/meta.py | meta/router.py + meta/coverage.py + meta/strategy.py |
| 8 | `evolution` | C | Engine | Dream 链接生长 + reflect 蒸馏 + 冲突解决 | memory/evolution.py | evolution/dream.py + evolution/reflect.py + evolution/conflict.py |
| 9 | `causal` | C | Graph | CausalEdge CRUD + BFS + 反事实查询 | memory/causal.py | retrieval/causal.py |
| 10 | `forgetting` | C | Manager | MemoryStrength decay + rehearse + 归档 | memory/forgetting.py | retrieval/forgetting.py |

### 4.1 子组件初始化

V2Memory 创建全新的子组件实例，不复用 Memory 内部的 semantic/episodic/procedural。
数据模型（SQLModel table）共享 models/，操作类全新定义在 memory/。

```python
# memory/memory_v2.py

from septmuse.memory.working_memory import WorkingMemory
from septmuse.memory.semantic import SemanticMemory
from septmuse.memory.episodic import EpisodicMemory
from septmuse.memory.procedural import ProceduralMemory
from septmuse.memory.capture import CapturePipeline
from septmuse.memory.retrieval import HybridRetriever, TokenBudget
from septmuse.memory.meta import MetacognitionLayer
from septmuse.memory.evolution import EvolutionEngine
from septmuse.memory.causal import CausalGraph
from septmuse.memory.forgetting import ForgettingManager


class V2Memory:
    def __init__(self, memory=None, *, config=None):
        self.mem = memory or Memory(config=config)

        # === 平面 B 后端（从 Memory 复用）===
        self.store = self.mem.store
        self.embedder = self.mem.embedder
        self.typed_store = self.mem.typed_store
        self.graph_store = self.mem.graph_store
        self.entity_store = self.mem.entity_store
        self.entity_extractor = self.mem.entity_extractor
        self.llm = self.mem.llm
        self._dedup_window = self.mem._dedup_window

        # === 平面 A 子组件（4 个，全新创建，不复用 self.mem.semantic）===
        # WorkingMemory 走独立 WorkingMemoryStore（非 typed_store）
        self.working_memory = WorkingMemory(
            store=self._create_working_memory_store(),
            agent_id="default",
        )
        # Semantic/Episodic/Procedural 走 typed_store（共享关系型后端）
        self.semantic = SemanticMemory(self.typed_store, self.embedder)
        self.episodic = EpisodicMemory(self.typed_store)
        self.procedural = ProceduralMemory(self.typed_store)

        # === 平面 C 子组件（6 个）===
        self.capture = CapturePipeline(
            self.store, self.embedder,
            typed_store=self.typed_store,
            llm=self.llm,
            dedup_window=self._dedup_window,
        )
        self.retrieval = HybridRetriever(
            self.store, self.embedder,
            entity_extractor=self.entity_extractor,
            entity_store=self.entity_store,
        )
        self.token_budget = TokenBudget(budget=2000)
        self.meta = MetacognitionLayer(
            self.embedder, self.store, self.typed_store,
        )
        self.evolution = EvolutionEngine(
            self.store, self.graph_store, self.embedder,
            self.typed_store, self.llm,
        )
        self.causal = CausalGraph(self.typed_store)
        self.forgetting = ForgettingManager(self.typed_store, self.embedder)

    def _create_working_memory_store(self):
        """创建工作记忆独立后端（WorkingMemoryStore ABC）。"""
        from septmuse.storage.working_memory_stores.factory import create_working_memory_store
        return create_working_memory_store(self.mem.config)
```

**关键设计决策**：
- **V2 创建全新子组件实例**：不复用 `self.mem.semantic` 等 Memory 内部实例（决策 1）
- **数据模型共享**：V2 子组件 import models/ 的 SQLModel table（Block/SemanticFact 等），操作类全新（决策 4+边界 A）
- **WorkingMemory 独立后端**：走 WorkingMemoryStore，非 typed_store（决策 3）
- **Semantic/Episodic/Procedural 走 typed_store**：长时记忆共享关系型后端

---

## 5. 存储划分

### 5.1 子组件存储映射

| 子组件 | 持久化？ | 存什么 | 默认存储 | 可选存储 | ABC 类型 |
|---|:-:|---|---|---|---|
| `working_memory` | ✅ | Block（persona/human） | SQLite + 内存缓存 | Redis | ShortTermMemory |
| `semantic` | ✅ | SemanticFact 三元组 | RelationalStore (SQLite) | PG/MySQL | LongTermMemory |
| `episodic` | ✅ | EpisodicEvent 事件 | RelationalStore (SQLite) | PG/MySQL | LongTermMemory |
| `procedural` | ✅ | ProceduralRule 规则 | RelationalStore (SQLite) | PG/MySQL | LongTermMemory |
| `capture` | ❌ | 流水线中间态 | 内存 | — | — |
| `retrieval` | ❌ | 只读（向量+BM25+图） | 读 VectorStore/KeywordIndex/GraphStore | — | — |
| `meta` | ✅ | L1 报告存为 SemanticFact | RelationalStore（复用 semantic） | — | — |
| `evolution` | ✅ | 产出存为 Rule/Fact/EntityRelation | RelationalStore（复用） | — | — |
| `causal` | ✅ | CausalEdge 因果边 | RelationalStore (SQLite) | PG/MySQL | — |
| `forgetting` | ✅ | MemoryStrength 强度 | RelationalStore (SQLite) | PG/MySQL | — |

### 5.2 平面 B 存储后端

| 后端 | 默认 | 可选 | 存什么 | 子组件使用方 |
|---|---|---|---|---|
| working_memory_store | SQLite + 内存缓存 | Redis | Block | working_memory |
| store (关系型) | SQLite | PG/MySQL | 长时记忆 6 类表 | semantic/episodic/procedural/causal/forgetting |
| vector_store | SQLite + numpy 全量 | pgvector/chroma/qdrant | embedding | retrieval |
| keyword_index | SQLite BM25 | PG FTS/MySQL FULLTEXT | BM25 索引 | retrieval |
| graph_store | SQLite 关系表 | AGE/Neo4j | 实体节点+边 | retrieval/evolution/causal |
| file_store | 文件系统 | — | Markdown | 独立（FileMemoryStore） |
| activation | 内存 | — | KV-Cache | 独立（ActivationMemory） |
| parametric | 文件系统 | — | LoRA 权重 | 独立（LoRAMemory） |

### 5.3 零配置默认

```
工作记忆 → SQLite + 内存缓存（与长时记忆同文件）
长时记忆 → SQLite（一个 .db 文件含关系表 + 向量表 + BM25 索引 + 图）
文件记忆 → ~/.septmuse/files/
激活记忆 → 内存
参数化记忆 → 不启用
```

**一个 SQLite 文件搞定一切**，无 Redis、无外挂服务、无 LLM。

---

## 6. 目录结构

```
src/septmuse/
│
├─ memory/                              ← V2 编排 + V1 facade + ABC + 10 子组件
│   │
│   │   === 记忆 ABC 分层 + V1/V2 入口 ===
│   ├─ base.py                          ← 新增（MemoryABC + ShortTermMemory + LongTermMemory）
│   ├─ main.py                          (Memory V1，核心 CRUD，不动)
│   ├─ memory_v2.py                     ← 新增（V2Memory 编排入口）
│   ├─ async_main.py                    (AsyncMemory，不动)
│   ├─ cube.py                          (MemCube，不动)
│   ├─ os.py                            (MemOS，不动)
│   ├─ registry.py                      (MemoryRegistry，不动)
│   │
│   │   === 平面 A 内容类型子组件（4 个，直接扁平放）===
│   ├─ working_memory.py                ← 新增 WorkingMemory（import models.block + ABC 注册 + 超限驱逐）
│   ├─ semantic.py                      ← 新增 SemanticMemory（import models.fact + ABC 注册）
│   ├─ episodic.py                      ← 新增 EpisodicMemory（import models.episodic + ABC 注册）
│   ├─ procedural.py                    ← 新增 ProceduralMemory（import models.procedural + ABC 注册）
│   │
│   │   === 平面 C 横切关注点子组件（6 个，直接扁平放）===
│   ├─ capture.py                        ← 新增 CapturePipeline（import capture.pipeline）
│   ├─ retrieval.py                     ← 新增 HybridRetriever（import retrieval.hybrid + TokenBudget）
│   ├─ meta.py                          ← 新增 MetacognitionLayer（聚合 meta/ 三子模块 L0/L1/L2）
│   ├─ evolution.py                     ← 新增 EvolutionEngine（聚合 evolution/ 三子模块 Dream/reflect/冲突）
│   ├─ causal.py                        ← 新增 CausalGraph（import retrieval.causal）
│   ├─ forgetting.py                    ← 新增 ForgettingManager（import retrieval.forgetting）
│   │
│   └─ __init__.py                      (导出 V2Memory + ABC + 10 子组件)
│
├─ models/                              ← 平面 A 数据模型
│   ├─ block.py                         (Block + WorkingMemory → 继承 ShortTermMemory)
│   ├─ episodic.py                      (EpisodicEvent + EpisodicMemory → 继承 LongTermMemory)
│   ├─ semantic.py                      (SemanticFact)
│   ├─ fact.py                          (SemanticMemory → 继承 LongTermMemory)
│   ├─ procedural.py                    (ProceduralRule + ProceduralMemory → 继承 LongTermMemory)
│   ├─ causal.py                        (CausalEdge)
│   ├─ strength.py                      (MemoryStrength)
│   └─ __init__.py
│
├─ storage/                             ← 平面 B 存储形态
│   │
│   ├─ working_memory_stores/          ← 新增（工作记忆独立后端）
│   │   ├─ base.py                      (WorkingMemoryStore ABC)
│   │   ├─ sqlite_store.py              (SQLite + 内存缓存，默认)
│   │   ├─ redis_store.py               (Redis，可选)
│   │   └─ factory.py
│   │
│   ├─ relational_stores/               (长时记忆关系型)
│   │   ├─ orm_store.py                 (ORMMemoryStore)
│   │   ├─ async_orm_store.py
│   │   ├─ typed_store.py               (TypedMemoryStore)
│   │   ├─ entity_store.py
│   │   └─ factory.py
│   │
│   ├─ vector_stores/                   (向量)
│   │   ├─ sqlite_vec.py                (numpy 全量，默认)
│   │   ├─ sqlalchemy_vec.py
│   │   ├─ pgvector_store.py
│   │   ├─ chroma.py
│   │   ├─ qdrant.py
│   │   ├─ base.py
│   │   └─ factory.py
│   │
│   ├─ keyword_stores/                  (BM25)
│   │   ├─ sqlalchemy_keyword.py
│   │   ├─ postgres_fts.py
│   │   ├─ mysql_fulltext.py
│   │   ├─ base.py
│   │   └─ factory.py
│   │
│   ├─ graph_stores/                    (图)
│   │   ├─ sqlite.py
│   │   ├─ age.py
│   │   ├─ neo4j.py
│   │   ├─ base.py
│   │   └─ factory.py
│   │
│   ├─ file_stores/                     (文件记忆，只留 markdown)
│   │   ├─ markdown.py
│   │   └─ __init__.py
│   │
│   ├─ activation/                      ← 新增（从 storage/activation.py 移入）
│   │   └─ __init__.py
│   │
│   ├─ parametric/                      ← 新增（从 file_stores/lora*.py 移入）
│   │   ├─ lora.py
│   │   ├─ lora_base.py
│   │   └─ __init__.py
│   │
│   ├─ base.py
│   ├─ filters.py
│   └─ migrations/
│
├─ capture/                             ← 平面 C 捕获
│   ├─ hooks.py                         (PostToolUseHook)
│   ├─ pipeline.py                      (CapturePipeline)
│   ├─ sanitize.py                      (隐私脱敏)
│   └─ __init__.py
│
├─ retrieval/                           ← 平面 C 检索
│   ├─ hybrid.py                        (HybridRetriever 三信号)
│   ├─ graph_search.py                  (GraphSearcher BFS)
│   ├─ temporal.py                      (TemporalRetriever 时态)
│   ├─ causal.py                        (CausalGraph 因果)
│   ├─ forgetting.py                    (ForgettingManager 遗忘)
│   ├─ token_budget.py                  (TokenBudget 裁剪)
│   ├─ recipes.py                       (SearchRecipe 7 预置)
│   └─ __init__.py
│
├─ meta/                                ← 平面 C 元认知
│   ├─ router.py                        (MetaRouter L0 路由)
│   ├─ coverage.py                      (CoverageAnalyzer L1 覆盖)
│   ├─ strategy.py                      (StrategyAdapter L2 策略)
│   └─ __init__.py
│
├─ evolution/                           ← 平面 C 演化
│   ├─ dream.py                         (DreamIntegrator)
│   ├─ reflect.py                       (SessionReflector)
│   ├─ conflict.py                     (ConflictResolver)
│   ├─ cognify.py                       (CognifyPipeline)
│   ├─ summarizer.py                   (Summarizer)
│   ├─ zettel.py                        (ZettelLinker)
│   ├─ degradation.py
│   └─ __init__.py
│
├─ governance/                          ← 平面 C 治理 + 共享
│   ├─ approval.py                      (写校验 + 去重)
│   ├─ access.py                        (权限检查)
│   ├─ audit.py                         (访问日志)
│   ├─ rbac.py                          (RBAC 角色)
│   ├─ sharing.py                       (跨 agent 共享)
│   └─ __init__.py
│
├─ sync/                                ← 平面 C 源同步
│   ├─ synchronizer.py                  (SourceSynchronizer)
│   └─ __init__.py
│
├─ extraction/                          ← 平面 C 抽取（cognify 支撑）
│   ├─ entity.py                        (EntityExtractor)
│   ├─ triplet.py                      (TripletExtractor)
│   ├─ cognify.py                       (CognifyPipeline)
│   └─ __init__.py
│
├─ embedders/                           ← 支撑层
├─ llms/                                ← 支撑层
├─ rerankers/                           ← 支撑层
├─ services/                            ← 支撑层
├─ configs/                             ← 支撑层
├─ prompts/                             ← 支撑层
├─ core/                                ← 支撑层
├─ api/                                 ← 入口层
├─ cli/                                 ← 入口层
│
└─ experimental.py                      ← V2 后废弃（能力迁移到 memory_v2.py）
```

---

## 7. 4 个编排方法数据流转

### 7.1 `remember()` — 捕获 + 分流 + 多形态写

```
入口: messages + user_id + agent_id + session_id
  │
  ├─ 1. 平面 C 捕获（去重 + 脱敏）
  │   text = _normalize(messages)
  │   if capture.is_duplicate(text, window=300s): → return {captured: False}
  │   text = capture.redact(text)
  │
  ├─ 2. 平面 A 分流
  │   ├─ 情节 raw_log（恒做，不需要 LLM）
  │   │   raw = episodic.add_raw_log(text, session_id, user_id)
  │   ├─ 语义事实（仅 LLM 可用时）
  │   │   if llm is not None:
  │   │       facts = mem.extractor.extract_and_store(text, user_id)
  │   │   else:
  │   │       facts = []  # 零 LLM 降级，只存 raw_log
  │   └─ 工作 block（可选，agent_id 存在时）
  │       if agent_id: _update_persona(agent_id, text)
  │
  ├─ 3. 平面 B 多形态写（实体链接，复用 Memory 已有）
  │   for fact in facts:
  │       mem._batch_extract_and_store_entities(...)
  │
  └─ return {raw_id, fact_ids, captured: True}
```

**关键约束**：
- **不直接产程序规则** — 程序规则留给 `improve()` 从情节蒸馏
- 无 LLM 时只存 raw_log + block，不抽事实（零 LLM 降级）

### 7.2 `recall()` — 元认知路由 + 检索 + token 预算

```
入口: query + user_id + top_k + recipe
  │
  ├─ 1. 平面 C 元认知 L0 路由
  │   route = meta_router.route(query)
  │
  ├─ 2. 平面 C 检索（三信号融合，over-fetch）
  │   results = retrieval.search(query, user_id, top_k*4)
  │
  ├─ 3. 平面 B 图扩展（recipe 可选）
  │   if recipe == "GRAPH_BFS" and results:
  │       graph = causal.bfs(results[0].id, depth=2)
  │       results = rrf_fuse(results, graph)
  │
  ├─ 4. 平面 C 遗忘曲线加权
  │   results = forgetting.apply_strength(results, user_id)
  │
  ├─ 5. 平面 C 治理：token 预算裁剪
  │   budgeted = token_budget.fit(results, 2000)
  │
  ├─ 6. 平面 C 元认知 L2 策略自调（仅 L1 报告存在时）
  │   l1_report = self._load_coverage_report(user_id)
  │   if l1_report is not None:
  │       strategy = meta_strategy.adapt(l1_report)
  │       # strategy 可能触发"加深检索"或"主动澄清提问"
  │   else:
  │       skip L2  # 首次使用无报告，正常检索（决策 6）
  │
  ├─ 7. 平面 A 多类型合并 + block 注入
  │   prompt = working_memory.compile_to_prompt() + procedural.rules_to_prompt()
  │
  └─ return {memories: budgeted, injected_prompt: prompt, route}
```

**L1 报告 fallback**（决策 6）：首次使用时 `improve()` 还没跑过，L1 报告不存在。recall 跳过 L2 策略自调，正常检索。improve 跑过后 L1 报告存在，recall 才用 L2 策略。

### 7.3 `improve()` — Dream + reflect + 冲突 + L1 报告

```
入口: user_id + limit
  │
  ├─ 1. 平面 C 演化：Dream 链接生长
  │   dream_result = dreamer.dream(user_id)
  │
  ├─ 2. 平面 A 升华：情节 → 程序规则
  │   if llm is not None:
  │       rules = reflector.reflect(user_id, limit)
  │   else:
  │       rules = []  # 无 LLM 跳过蒸馏
  │
  ├─ 3. 平面 C 演化：冲突解决
  │   conflicts = conflict_resolver.resolve_conflicts(user_id)
  │
  ├─ 4. 平面 C 元认知 L1 报告生成 + 持久化
  │   report = meta_coverage.analyze(user_id)
  │   _persist_coverage(report, user_id)
  │   # 存为 SemanticFact(tags=["meta","coverage"])
  │
  └─ return {dream: dream_result, rules: len(rules), conflicts}
```

### 7.4 `forget()` — 先 invalidate 再 delete + 实体清理

```
入口: memory_id + user_id
  │
  ├─ 1. 平面 A 双时态失效（标记不再为真，保留历史）
  │   store.invalidate(memory_id, invalid_at=now)
  │   # 设 invalid_at + expired_at，保留历史可追溯
  │
  ├─ 2. 平面 A 软删除（不参与检索）
  │   store.delete(memory_id)  # is_deleted=1, state=deleted
  │
  ├─ 3. 平面 C 共享清理
  │   if entity_store:
  │       entity_store.remove_memory_reference(memory_id)
  │
  ├─ 4. 平面 B 图边清理
  │   if graph_store:
  │       _delete_graph_edges(memory_id)
  │
  └─ return {memory_id, event: "FORGET", invalidated_at, deleted_at}
```

**forget 语义决策 5**：先 invalidate（标记不再为真，保留双时态历史）再 delete（软删除不参与检索）。彻底遗忘但仍保留历史轨迹。`invalidate()` 作为独立方法保留给"只标记不再为真但不删除"的场景（如冲突解决中旧事实被新事实推翻 = invalidate，不 delete）。

---

## 8. 零 LLM 降级路径

### 8.1 子组件 LLM 依赖度

| 子组件 | LLM 依赖 | 无 LLM 时行为 |
|---|:-:|---|
| `working_memory` | 不需要 | 正常工作 |
| `semantic` | 可降级 | 用户显式 add_fact（不自动抽三元组） |
| `episodic` | 不需要 | 正常工作（存 raw_log） |
| `procedural` | 必需（蒸馏） | 无 LLM 跳过 reflect，程序库为空 |
| `capture` | 不需要 | 去重 + 脱敏正常 |
| `retrieval` | 不需要 | HashEmbedder + regex 实体 |
| `meta` L0 路由 | 不需要 | embedding 相似度路由 |
| `meta` L1 覆盖 | 可降级 | 纯统计报告（无自然语言自述） |
| `meta` L2 策略 | 可降级 | 规则驱动（覆盖度 < 阈值 → 澄清） |
| `evolution` Dream | 可降级 | embedding 相似度找关联 |
| `evolution` reflect | 必需 | 无 LLM 跳过蒸馏 |
| `evolution` 冲突 | 可降级 | 精确归一化 + difflib 模糊 |
| `causal` 抽取 | 必需 | 无 LLM 跳过自动抽取 |
| `causal` 反事实 | 必需 | 纯图遍历返回路径 |
| `forgetting` | 不需要 | 纯数学 exp(-t/S) |

### 8.2 零 LLM 模式下系统形态

```
remember():
  ├─ 捕获去重 + 脱敏 ✅
  ├─ 情节 raw_log 存原文 ✅
  ├─ 语义事实：跳过（不抽三元组） ⚠️
  └─ 工作 block：正常 ✅

recall():
  ├─ L0 路由（embedding 相似度）✅
  ├─ 三信号检索（HashEmbedder + regex）✅
  ├─ 图 BFS（纯图遍历）✅
  ├─ 遗忘曲线加权（纯数学）✅
  ├─ token 预算裁剪 ✅
  └─ block + 规则注入 ✅（程序库为空则不注入）

improve():
  ├─ Dream 链接生长（embedding 相似度）✅
  ├─ reflect 蒸馏：跳过 ⚠️
  ├─ 冲突解决（精确归一化 + difflib）✅
  └─ L1 报告（纯统计）✅

forget():
  └─ 全部正常 ✅
```

**零 LLM 时系统退化为**：高保真日志 + 向量检索 + 规则链接生长 + 纯统计元认知 + 遗忘曲线衰减。不崩溃。

---

## 9. V1 与 V2 关系 + 迁移路径

### 9.1 当前状态（阶段 1）

```
V1 Memory (memory/main.py)         ← 核心 CRUD，21 方法，不动
V1 ExperimentalMemory (experimental.py) ← 上帝类 47 方法，REST/MCP 用
V2 V2Memory (memory/memory_v2.py)   ← 新增，4 编排方法，独立可用
V2 MemoryABC (memory/base.py)       ← 新增，记忆 ABC 分层
```

### 9.2 迁移路径

```
阶段 1（当前）:
  V2Memory 独立可用，不改 V1，不改 REST/MCP

阶段 2（REST/MCP 接入 V2）:
  REST 新增 V2 端点:
    POST /memories/remember     → v2.remember()
    POST /memories/recall       → v2.recall()
    POST /memories/improve      → v2.improve()
    POST /memories/forget/{id}  → v2.forget()
  V1 端点保留（向后兼容）

阶段 3（废弃 ExperimentalMemory）:
  ExperimentalMemory 的 47 方法迁移:
    工作记忆 Block → V2 working_memory 子组件
    元认知 L0/L1/L2 → V2 meta 子组件
    因果/遗忘 → V2 causal/forgetting 子组件
    演化 → V2 evolution 子组件
  experimental.py 删除

阶段 4（最终）:
  V2Memory 是唯一 facade
  Memory (V1) 作为 V2Memory 的内部组件保留
  REST/MCP 只暴露 V2 端点 + 分类端点
```

---

## 10. 环境变量配置

```bash
# === 工作记忆（独立选型）===
SEPTMUSE_WORKING_MEMORY_BACKEND=sqlite  # sqlite（默认）/ redis
SEPTMUSE_REDIS_URL=                     # backend=redis 时必填

# === 长时记忆-关系型（统一选型，6 类表共享）===
SEPTMUSE_DB_PATH=~/.septmuse/septmuse.db  # SQLite 路径（默认零配置）
SEPTMUSE_DB_DSN=                          # PG/MySQL DSN（可选，覆盖 DB_PATH）

# === 长时记忆-向量（独立选型，可跨库）===
SEPTMUSE_VECTOR_BACKEND=sqlite           # sqlite（默认）/ pgvector / chroma / qdrant
SEPTMUSE_VECTOR_PATH=                    # chroma/qdrant 目录或 URL

# === 长时记忆-关键词（独立选型）===
SEPTMUSE_KEYWORD_BACKEND=sqlite_bm25     # sqlite_bm25（默认）/ rank_bm25 / none

# === 长时记忆-图（独立选型）===
SEPTMUSE_GRAPH_BACKEND=sqlite            # sqlite（默认）/ age / neo4j

# === 嵌入/LLM/实体 ===
SEPTMUSE_EMBEDDER=hash                   # hash（默认离线）/ onnx / onnx-zh / auto / st
SEPTMUSE_LLM=                             # 未设 = 零 LLM 降级
SEPTMUSE_LLM_MODEL=                       # 覆盖 provider 默认模型
SEPTMUSE_ENTITY_EXTRACTOR=regex          # regex（默认）/ spacy / none

# === 重排 ===
SEPTMUSE_RERANKER=noop                   # noop（默认）/ mmr / cross_encoder / llm

# === 文件/模型缓存 ===
SEPTMUSE_FILE_WORKSPACE=~/.septmuse/files/
SEPTMUSE_MODEL_CACHE=~/.septmuse/models/
```

**零配置默认**：全 SQLite + HashEmbedder + regex + verbatim + noop reranker，`pip install septmuse` 即用。

---

## 11. 验收标准

- [ ] `memory/base.py` 新增 MemoryABC + ShortTermMemory + LongTermMemory 三个 ABC
- [ ] `memory/working_memory.py` 新增，import models/block.py + 注册 ShortTermMemory + 超限驱逐
- [ ] `memory/semantic.py` 新增，import models/fact.py + 注册 LongTermMemory
- [ ] `memory/episodic.py` 新增，import models/episodic.py + 注册 LongTermMemory
- [ ] `memory/procedural.py` 新增，import models/procedural.py + 注册 LongTermMemory
- [ ] `memory/capture.py` 新增，import capture/pipeline.py
- [ ] `memory/retrieval.py` 新增，import retrieval/hybrid.py + TokenBudget
- [ ] `memory/meta.py` 新增 MetacognitionLayer，聚合 meta/ 三子模块（L0+L1+L2）
- [ ] `memory/evolution.py` 新增 EvolutionEngine，聚合 evolution/ 三子模块
- [ ] `memory/causal.py` 新增，import retrieval/causal.py
- [ ] `memory/forgetting.py` 新增，import retrieval/forgetting.py
- [ ] `memory/memory_v2.py` 新增 V2Memory，不继承 Memory，组合 Memory 实例
- [ ] V2Memory 从 `memory/` 下 10 个子组件文件 import（不直接 import 各功能目录）
- [ ] V2Memory 持有 10 个子组件，4 个编排方法（remember/recall/improve/forget）
- [ ] 零 LLM 模式端到端可用：无 `SEPTMUSE_LLM` 时 4 个编排方法不崩溃且降级路径可观测
- [ ] remember 不直接产程序规则（程序规则只从 improve 蒸馏）
- [ ] recall 的 L1 读预生成报告（improve 阶段生成 + 持久化）
- [ ] improve 产出 L1 报告持久化为 SemanticFact(tags=["meta","coverage"])
- [ ] 子组件之间不直接耦合（编排逻辑在 V2Memory 里）
- [ ] 不改 main.py、不改 REST/MCP（V2Memory 独立可用）
- [ ] `storage/working_memory_stores/` 新增（WorkingMemoryStore ABC + SQLite + Redis 可选）
- [ ] `storage/activation/` 从 `storage/activation.py` 移入
- [ ] `storage/parametric/` 从 `file_stores/lora*.py` 移入
- [ ] `experimental.py` 标记为 deprecated（阶段 3 删除）
- [ ] 新增测试 `tests/unit/test_v2_memory.py` ≥ 10 个用例
- [ ] 新增测试 `tests/unit/test_memory_abc.py` 验证 ABC 契约
- [ ] ruff check + pytest 全绿，零退化

---

## 附录：子组件构造参数（待 codegraph 确认）

| 子组件 | 已知构造参数 | 确认状态 |
|---|---|---|
| CapturePipeline | store, embedder, typed_store, llm, dedup_window | ✅ |
| HybridRetriever | store, embedder, vector_weight, keyword_weight, entity_extractor, entity_store | ✅ |
| TokenBudget | budget | ✅ |
| MetaRouter | embedder, namespaces, threshold | ✅ |
| CoverageAnalyzer | store, typed_store | ✅ |
| StrategyAdapter | （无参数） | ✅ |
| DreamIntegrator | store, graph_store, embedder, batch_size, threshold | ✅ |
| ConflictResolver | typed_store, store, llm | ✅ |
| SessionReflector | ? | 待确认 |
| CausalGraph | ? | 待确认 |
| ForgettingManager | ? | 待确认 |

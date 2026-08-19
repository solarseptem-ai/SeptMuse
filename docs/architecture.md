# SeptMuse 架构图

English/中文

读本文档后再改 `src/septmuse/`。术语见 [CONTEXT.md](../CONTEXT.md)，决策见 [Agent Notes](../adr/)。

## 架构总览

```mermaid
graph TD
    subgraph Entry["入口层"]
        CLI["CLI<br/>cli/main.py<br/>全同步"]
        MCP["MCP tools.py<br/>23 工具<br/>to_thread → Memory"]
        REST["REST<br/>api/rest/__init__.py<br/>23 端点"]
    end

    subgraph Facade["Facade 层"]
        MEM["Memory<br/>memory/main.py (~786 行)<br/>60+ 方法"]
        AMEM["AsyncMemory<br/>memory/async_main.py (619 行)<br/>42 方法, 74% to_thread 桥接"]
    end

    REST -->|"业务路由"| AMEM
    CLI --> MEM
    MCP --> MEM
    AMEM -->|"内部持有同步 Memory<br/>共享 vector_store, 新建 sync engine"| MEM

    subgraph Factory["工厂层"]
        MCF["MemoryComponentFactory<br/>memory/_factory.py<br/>构造 12 核心组件"]
        REG["ComponentRegistry<br/>memory/registry_v2.py<br/>27 注册项 (7 管线 + 6 领域 + 14 组件)<br/>懒加载 + 缓存 + 失败隔离"]
    end

    MEM -->|"__init__ 构造组件"| MCF
    MEM -->|"__init__ 注册"| REG

    subgraph Pipelines["pipelines/ — 多步编排"]
        SP["SearchPipeline<br/>持久 HybridRetriever<br/>recipe → 检索 → reranker → 后处理"]
        RP["RememberPipeline<br/>价值 → 分类 → 去重 → 存储 → 抽取"]
        RCP["RecallPipeline<br/>compress → search → budget → inject"]
        CP2["CapturePipeline<br/>SHA256 去重 → 脱敏 → 嵌入"]
        AP["AddPipeline<br/>verbatim/infer 路由 → 类型分派"]
        FP["ForgetPipeline<br/>invalidate → delete → 图清理"]
        RTP["ResetPipeline"]
    end

    subgraph Domains["domains/ — 按域分组（浅透传）"]
        CM["CaptureModule<br/>14 方法"]
        SM["SearchModule<br/>8 方法<br/>⚠ new retriever/call"]
        EM2["EntityModule<br/>8 方法"]
        TM2["TemporalModule<br/>3 方法"]
        EVM["EvolutionModule<br/>23 方法（有深度）"]
        ML["MetacognitionLayer"]
    end

    REG --> Pipelines
    REG --> Domains
    MEM -.->|"7 方法委托管线"| REG
    MEM -.->|"49 方法委托领域<br/>经 __getattr__"| REG

    subgraph Storage["存储层"]
        ORM["ORMMemoryStore<br/>shallow facade"]
        REPO["MemoryRepository<br/>deep impl (508 行)"]
        TS["TypedMemoryStore<br/>668 行, 6 职责"]
        ES["EntityStore<br/>顶点 + 关系边"]
        GS2["GraphStore<br/>SQLite/Neo4j/AGE"]
        WMS["WorkingMemoryStore<br/>SQLite/Redis"]
        VS["VectorStoreBase<br/>4 实现"]
        KI["KeywordIndexBase<br/>5 实现"]
        AORM["AsyncORMMemoryStore<br/>543 行, 重复 MemoryRepository"]
    end

    MEM ==>|"12 方法内联直碰<br/>get/update/delete/invalidate<br/>cognify/causal×4<br/>get_user_profile/get_entity_neighbors"| Storage
    AMEM -->|"原生 async (16%)"| AORM
    AORM -->|"to_thread"| VS
    AORM -->|"to_thread"| KI

    ORM --> REPO
    ORM --> VS
    ORM --> KI
    MEM --> ORM
    MEM --> TS
    MEM --> ES
    MEM --> GS2

    subgraph Dead["死代码 / 已废弃"]
        PR["PipelineRegistry<br/>pipelines/__init__.py<br/>~105 行, 零实例化"]
        SH["4 re-export shim<br/>memory/causal.py<br/>memory/forgetting.py<br/>memory/retrieval.py<br/>memory/capture.py"]
        LRA["LoRA + Activation<br/>+ Sync + Markdown<br/>~1058 行, 零生产调用"]
    end

    classDef deadNode fill:#fee2e2,stroke:#dc2626,stroke-dasharray:4 4,color:#991b1b
    classDef leakNode fill:#fef3c7,stroke:#d97706
    classDef deepNode fill:#0f172a,stroke:#059669,color:#fff
    class PR,SH,LRA deadNode
    class SM leakNode
    class MEM,REPO deepNode
```

## Memory facade 委派路径

facade 有三条委派路径 + 一组内联方法。内联方法绕过管线和领域模块，直接碰底层存储组件。

```mermaid
graph LR
    subgraph Facade["Memory facade — 60+ 方法"]
        direction TB
        G1["7 方法 → 管线<br/>add/search/capture/<br/>remember/recall/forget/reset"]
        G2["49 方法 → 领域模块<br/>add_fact/search_hybrid/<br/>extract_entities/reflect/<br/>search_at/improve/..."]
        G3["~12 方法 → 内联<br/>get/get_all/update/delete/<br/>invalidate/cognify/<br/>causal×4/<br/>get_user_profile/<br/>get_entity_neighbors"]
    end

    subgraph Target["委派目标"]
        direction TB
        T1["ComponentRegistry<br/>→ pipelines/"]
        T2["ComponentRegistry<br/>→ domains/"]
        T3["直接碰底层组件<br/>store / typed_store /<br/>entity_store / semantic"]
    end

    G1 --> T1
    G2 --> T2
    G3 -.->|"⚠ 绕过抽象"| T3

    classDef leak fill:#fef3c7,stroke:#d97706
    class G3,T3 leak
```

**内联方法的 leak 点**（facade 有对应领域模块却直碰底层）：

| facade 方法 | 直碰 | 应走的领域模块 |
|---|---|---|
| `get_user_profile` | `self.semantic` | CaptureModule |
| `get_entity_neighbors` | `self.entity_store` | EntityModule |
| `cognify` | 内联构造 `CognifyPipeline` | 无（管线未注册） |
| `add_causal_edge` | `self.typed_store` | 无领域模块 |
| `find_causes`/`find_effects`/`counterfactual` | 内联构造 `CausalRetriever` | 无领域模块 |
| `get`/`get_all`/`update`/`delete`/`invalidate` | `self.store` | 无（CRUD 不属领域模块） |

## 存储层

```mermaid
graph TD
    subgraph ABC["ABC 层"]
        MS["MemoryStore<br/>8 抽象 + 7 默认<br/>+ 4 Protocol (Temporal/Keyword/AccessLog/SharedQuery)"]
        VSB["VectorStoreBase<br/>7 抽象 + 4 默认"]
        KIB["KeywordIndexBase<br/>4 抽象"]
        GSB["GraphStore<br/>9 抽象 + 2 默认"]
        WMSB["WorkingMemoryStore<br/>5 抽象"]
    end

    subgraph Relational["关系存储"]
        ORM["ORMMemoryStore<br/>shallow: repo + vector + keyword 三写协调"]
        REPO["MemoryRepository<br/>deep: 19 方法, 10 filter 操作符"]
        AORM["AsyncORMMemoryStore<br/>重写 MemoryRepository 12/13 方法<br/>⚠ search 过采样 3x vs 5x<br/>⚠ filters 丢 10 操作符"]
    end

    subgraph Typed["类型化存储"]
        TS["TypedMemoryStore (668 行)<br/>6 聚合根共享 engine"]
        SF["SemanticFact CRUD + search"]
        EE["EpisodicEvent CRUD"]
        PR2["ProceduralRule CRUD"]
        CE["CausalEdge"]
        MS2["MemoryStrength"]
        BLK["Block CRUD<br/>⚠ split-brain with WorkingMemoryStore"]
    end

    TS --> SF
    TS --> EE
    TS --> PR2
    TS --> CE
    TS --> MS2
    TS --> BLK

    subgraph Entity["实体存储"]
        ES["EntityStore (527 行)<br/>顶点 + 关系边<br/>⚠ 嵌入序列化 3 套不兼容<br/>⚠ get_neighbors 与 GraphStore 同名不同表"]
    end

    subgraph Graph["图存储"]
        GS["SQLiteGraphStore<br/>(生产唯一)"]
        NEO["Neo4jGraphStore<br/>(registry-only)"]
        AGE["AGEGraphStore<br/>(registry-only)"]
    end

    subgraph Working["工作记忆"]
        WMS["WorkingMemoryStore"]
        SWM["SQLiteWorkingMemoryStore"]
        RWM["RedisWorkingMemoryStore"]
        WM2["memory/working_memory.py<br/>WorkingMemory (路径 B, 类型安全)"]
        WM1["models/block.py<br/>WorkingMemory (路径 A, duck typing)"]
    end

    WMS --> SWM
    WMS --> RWM
    WMS --> WM2
    BLK -.->|"路径 A<br/>TypedMemoryStore 内联"| WM1

    subgraph Vector["向量存储"]
        QVS["QdrantVectorStore<br/>(默认, HNSW)"]
        SVS["SQLAlchemyVectorStore<br/>(降级, numpy 余弦)"]
        CVS["ChromaVectorStore"]
        PVS["PgvectorVectorStore"]
    end

    MS --> ORM
    MS --> AORM
    ORM --> REPO
    VSB --> QVS
    VSB --> SVS
    VSB --> CVS
    VSB --> PVS
    GSB --> GS
    GSB --> NEO
    GSB --> AGE
    WMSB --> WMS

    ORM -.->|"共享 engine"| TS
    ES -.->|"边模型重叠"| GS

    classDef leak fill:#fef3c7,stroke:#d97706
    classDef dead fill:#fee2e2,stroke:#dc2626,stroke-dasharray:4 4
    class BLK,AORM,ES leak
    class NEO,AGE dead
```

**存储层摩擦点**：

| 摩擦 | 位置 | 说明 |
|---|---|---|
| WorkingMemory Block 双写 | `typed_store.py:607-663` vs `working_memory_stores/` | 两条活跃路径, 两个同名 `WorkingMemory` 类 |
| 嵌入序列化 3 套 | `entity_store.py:382` (struct.pack) / `typed_store.py:109` (json) / `sqlalchemy_vec.py:95` (json TEXT) | 无一走 VectorStoreBase seam |
| EntityStore 边 vs GraphStore 边 | `entity_store.py:392-449` vs `graph_stores/base.py:42-64` | 两个 `get_neighbors` 同名同语义不同表 |
| AsyncORMMemoryStore 行为分叉 | `async_orm_store.py:193` (3x) vs `orm_store.py:255` (5x) | search 过采样倍数不一致 |
| filters 能力丢失 | `async_orm_store.py:210-216` vs `repositories/memory.py:36-89` | async 侧只支持 `==`, 同步侧 10 操作符 |

## 核心数据流

```
remember: 文本 → RememberPipeline → 价值评估 → 类型分类 → 语义去重 → 捕获 → raw_log → verbatim 存储 → 实体抽取 → 事实抽取 → 冲突检测 → 工作记忆更新

recall: 查询 → RecallPipeline → compress → search → budget → prompt → route → strategy

search: 查询 → SearchPipeline → recipe 解析 → 检索 → reranker → 过期过滤 → 后处理
```

**注意**：`search_hybrid`/`search_progressive`/`search_graph` 不走 SearchPipeline，走 SearchModule——每次新建 HybridRetriever，无 entity boost、无缓存、无 reranker。

## 架构入口

- **facade**：`src/septmuse/memory/main.py` `Memory` 类（零配置入口）— `MemoryComponentFactory.build()` 构造 12 核心组件，`ComponentRegistry.register_all()` 注册 27 项（7 管线 + 6 领域 + 14 组件）
- **AsyncMemory**：`src/septmuse/memory/async_main.py` — REST 专用。42 方法中 31 个纯 `to_thread` 桥接到内部同步 Memory，6 个"原生 async"方法底层仍走 sync vector_store。内部持有同步 Memory + 同步 ORMMemoryStore（新建 sync engine，共享 vector_store/keyword_index）
- **三 API 入口**：
  - CLI：`src/septmuse/cli/main.py`（全同步，走 Memory）
  - REST：`src/septmuse/api/rest/__init__.py` `create_app()` → FastAPI（走 AsyncMemory）
  - MCP：`src/septmuse/api/mcp/tools.py`（23 工具，`to_thread` 到同步 Memory，不经 AsyncMemory）
- **组件注册**：`src/septmuse/memory/registry_v2.py` `ComponentRegistry` — 懒加载 + 缓存 + 失败隔离。`register_all()` 注册 7 管线 + 6 领域模块 + 14 组件。facade 经 `__getattr__` + `_DOMAIN_ATTRS` 映射访问领域模块
- **存储抽象**：`storage/base.py` `MemoryStore` ABC (8 抽象 + 4 Protocol) + `vector_stores/base.py` `VectorStoreBase` (7 抽象, 4 实现) + `keyword_stores/base.py` `KeywordIndexBase` (4 抽象, 5 实现) + `graph_stores/base.py` `GraphStore` (9 抽象, 3 实现) + `working_memory_stores/base.py` `WorkingMemoryStore` (5 抽象, 2 实现)
- **默认后端**：`ORMMemoryStore`（shallow facade）+ `MemoryRepository`（deep impl）+ `TypedMemoryStore`（6 聚合根）+ `EntityStore`（顶点+边）+ `SQLiteGraphStore`。Vector store 默认 `QdrantVectorStore`，降级 `SQLAlchemyVectorStore`
- **配置**：`src/septmuse/configs/base.py` `MemoryConfig`（pydantic）+ `default_config()` 读环境变量
- **治理**：`src/septmuse/governance/access.py` `MemoryState` enum + `check_memory_access_permissions`（4 层）；401 认证（`api/auth.py`）vs 403 授权（`governance/access.py`）
- **共享工具**：`src/septmuse/core/utils.py` — `rrf_fuse()` + `utcnow_iso()`

## 关键设计决策

| 决策 | 说明 |
|------|------|
| **三维正交架构** | 内容类型（工作/情节/语义/程序）× 存储形态（block/向量/图）× 横切关注点（捕获/检索/治理/演化/共享/元认知）。注：文件/激活/参数化三种存储形态零生产调用，见死代码 |
| **零配置** | SQLite + HashEmbedder 默认。`pip install septmuse[onnx]` 升级到 bge-zh (512 dim) |
| **ComponentRegistry** | `registry_v2.py` 取代 `PipelineRegistry`（已死）。统一管管线 + 领域模块 + 组件的懒加载 |
| **facade 三路委派** | 7 方法→管线, 49 方法→领域模块, ~12 方法内联直碰底层。内联方法是抽象 leak |
| **score 统一为相似度 [0,1]** | 向量 cosine、BM25 归一化、RRF 融合 (k=60) 都遵守 |
| **Entity boost** | `0.5/(RRF_K+1)` ≈ 0.008，不劫持排序。仅 SearchPipeline 的持久 HybridRetriever 有此信号 |
| **Bitemporal** | `memories` 表 `valid_at`/`invalid_at`/`expired_at` 三列 |
| **治理分层** | 401 认证 vs 403 授权；state 状态机 (active/paused/archived/deleted)；access-logs 审计 |

## 仓库边界

- **`opensource/`** 是只读参考库（mem0/letta/ReMe/cognee/graphiti/MemOS 源码）— 禁止修改、禁止 import
- **迁移机制**：`storage/migrations/runner.py` `MigrationRunner`，m001-m007 + `schema_version` 表追踪。无 alembic 目录
- **`docs/specs/`** 设计规格、`docs/plans/` 实施计划、`.sdd/` subagent-driven ledger
- **`.codegraph/`** 已索引 — 优先用 `codegraph_explore` 查代码

## 死代码清单

| 文件 | 行数 | 状态 |
|---|---|---|
| `pipelines/__init__.py` PipelineRegistry | ~105 | 零实例化，被 ComponentRegistry 取代 |
| `memory/causal.py` | 20 | re-export shim，零引用 |
| `memory/forgetting.py` | 20 | re-export shim，零引用 |
| `memory/retrieval.py` | 21 | re-export shim，零引用 |
| `memory/capture.py` | 28 | DEPRECATED shim，零引用 |
| `domains/evolution.py` 6 方法 | ~60 | forget/cognify/causal×4 死镜像，facade 全走内联 |
| `storage/parametric/lora.py` | 361 | LoRA 参数化记忆，零生产调用 |
| `storage/activation/__init__.py` | 192 | Activation KV cache，零生产调用 |
| `storage/file_stores/markdown.py` | 234 | Markdown 文件存储，零生产调用 |
| `sync/drift.py` + `sync/synchronizer.py` | 253 | 同步器，零生产调用 |

## Background Evolution（后台演化）

- `evolution/scheduler.py` `EvolutionScheduler`：`threading.Timer` daemon 链，4 任务（forgetting 1h / conflict 5min / distillation 30min / graph 10min）。3 个任务委托 `Memory.evolution.X()`
- `SEPTMUSE_EVOLUTION_ENABLED=false` 禁用；`SEPTMUSE_EVOLUTION_INTERVAL`（默认 300s）控制轮询间隔
- 入口装配：`setup_mcp_server()` startup event 启动调度器；`run_stdio()` 在 try/finally 中启动/停止；`create_app()` lifespan 启动/停止
- `evolution/catchup.py` `LLMCatchUpQueue`：LLM 降级补跑队列（SQLite `septmuse_catchup_queue` 表）。`remember()` 在 LLM 不可用时入队 `fact_extraction`，调度器恢复后消费
- `ORMMemoryStore.get_user_ids()`：走 ABC（`state='active'` 过滤）
- 补跑策略：`fact_extraction` 走 `FactExtractor.extract_and_store()`；其他 task_type 规则版已处理，`mark_completed` 跳过

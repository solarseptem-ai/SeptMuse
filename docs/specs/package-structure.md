# SeptMuse 包结构与目录设计

> 配套文档：`docs/specs/agent-memory-architecture.md`（三维正交架构）
> 实现语言：Python（对齐 solarseptem-ai 生态 FastAPI + SQLModel）
> 调研参考：本地 `opensource/` 下 mem0 / letta / ReMe + context7 实查 MemOS / LangMem

---

## 1. 包名决策

**包名：`septmuse`**

| 候选 | 评估 |
|------|------|
| **`septmuse`** ✅ | 与项目名/生态一致（SeptKit/SeptLex/SeptMuse/SeptOrbit/SolAgent），lowercase 规范，导入 `from septmuse import ...` |
| `sept_muse` | Python 风格但与生态驼峰命名脱节 |
| `muse_memory` | 描述性但脱离生态 |

---

## 2. layout 决策

采用 **src/ layout**（借鉴 MemOS `src/memos/`），避免导入冲突，是现代 Python 包规范。
mem0/letta/ReMe 用扁平包名在根，但它们是成熟项目历史选择；新项目选 src/ layout 更稳。

---

## 3. 完整目录树

```
septmuse/
├── pyproject.toml
├── README.md
├── AGENTS.md
├── docs/
│   └── specs/
│       ├── agent-memory-architecture.md   # 已有: 三维正交架构
│       └── package-structure.md          # 本文档
├── opensource/                            # 已有: 参考库 mem0/letta/ReMe
├── alembic/                               # DB 迁移 (借鉴 letta/alembic)
│   └── versions/
├── scripts/
├── examples/
│   └── mvp_demo.py                        # 阶段1 最小闭环 demo
├── src/
│   └── septmuse/
│       ├── __init__.py
│       ├── py.typed
│       │
│       ├── api/                           # FastAPI + MCP (借鉴 mem0/server + openmemory/api/app/mcp_server.py)
│       │   ├── __init__.py
│       │   ├── rest.py                    # REST endpoints (§11.2 草案)
│       │   ├── deps.py                   # 依赖注入
│       │   └── mcp/                      # MCP server 子包 (源码参考 mem0 mcp_server.py)
│       │       ├── __init__.py
│       │       ├── server.py            # FastMCP 实例 + setup_mcp_server(app) 挂载
│       │       ├── tools.py             # @mcp.tool 工具集 (add/search/list/delete/remember/...)
│       │       ├── context.py           # contextvars 传 user_id/client_name (源码参考 mem0)
│       │       └── transports.py        # stdio + sse + streamable_http 三 transport
│       │
│       ├── configs/                      # 配置 (借鉴 mem0/configs, MemOS configs/)
│       │   ├── __init__.py
│       │   ├── settings.py              # 全局配置 (环境变量读取)
│       │   ├── defaults.py             # 【零配置】默认后端/LLM/embedder 选择
│       │   │                            #   默认: SQLite组合后端 + 本地嵌入 + 环境变量LLM
│       │   └── memory.py               # 记忆层配置 (各层参数)
│       │
│       ├── schemas/                       # 数据模型 (借鉴 letta/schemas/, ReMe/schema/)
│       │   ├── __init__.py
│       │   ├── block.py                  # 工作记忆 Block (借鉴 letta/schemas/block.py)
│       │   ├── episodic.py               # 情节: 时序事件+推理经验+raw log
│       │   ├── semantic.py               # 语义: 三元组事实+身份子类
│       │   ├── procedural.py             # 程序: helpful/harmful 规则 (借鉴 Cass Playbook)
│       │   ├── causal.py                 # 【自研】因果边
│       │   ├── strength.py                # 【自研】遗忘曲线强度
│       │   └── meta.py                   # 元认知状态 (打 meta 标签的语义记忆)
│       │
│       ├── content_types/                # 平面A: 内容类型 (核心, 借鉴 MemOS memories/)
│       │   ├── __init__.py
│       │   ├── working/                  # 工作记忆
│       │   │   ├── __init__.py
│       │   │   ├── block.py              # Block 自编辑 + XML 编译 (借鉴 letta core_memory_*)
│       │   │   └── eviction.py           # 超限驱逐到长时 (借鉴 Hermes char_limit)
│       │   ├── episodic/                 # 情节记忆
│       │   │   ├── __init__.py
│       │   │   ├── temporal.py           # 时序事件 (借鉴 Zep Episode+reference_time)
│       │   │   ├── reasoning.py          # 推理经验 obs/act/result (借鉴 LangMem Episode)
│       │   │   └── raw_log.py           # raw session log (借鉴 Cass Episodic)
│       │   ├── semantic/                 # 语义记忆
│       │   │   ├── __init__.py
│       │   │   ├── fact.py               # 三元组 CRUD
│       │   │   ├── identity.py           # 身份子类
│       │   │   └── extract.py            # cognify 抽取流水线 (借鉴 Cognee)
│       │   └── procedural/               # 程序记忆
│       │       ├── __init__.py
│       │       ├── playbook.py           # 规则退化 (借鉴 Cass helpful/harmful)
│       │       └── reflect.py           # reflect+curate 升华 (借鉴 Cass)
│       │
│       ├── storage/                      # 平面B: 存储形态后端 (借鉴 mem0 后端分目录)
│       │   ├── __init__.py
│       │   ├── base.py                   # 形态后端抽象基类
│       │   ├── block_store.py            # block 持久化
│       │   ├── sqlite/                  # 【零配置默认后端】单文件 SQLite 组合后端
│       │   │   ├── __init__.py
│       │   │   ├── store.py             # SQLiteStore: 实现 vector+graph+metadata 三接口
│       │   │   ├── vec.py              # sqlite-vec 向量检索 (零外部服务)
│       │   │   ├── graph.py            # 三元组表 (简化图, 无需 AGE/Neo4j)
│       │   │   └── metadata.py         # 元数据/历史表 (借鉴 mem0 SQLite history)
│       │   ├── vector/                   # 向量后端 (生产)
│       │   │   ├── __init__.py
│       │   │   ├── pgvector.py          # 生产主选 (复用 Postgres)
│       │   │   └── base.py
│       │   ├── graph/                    # 图后端
│       │   │   ├── __init__.py
│       │   │   ├── age.py               # Apache AGE 主选 (减少依赖)
│       │   │   ├── neo4j.py             # 重场景备用
│       │   │   └── base.py
│       │   ├── file/                    # 文件记忆 (借鉴 Basic Memory)
│       │   │   ├── __init__.py
│       │   │   ├── markdown.py          # Markdown+frontmatter+wikilinks
│       │   │   ├── sqlite_index.py      # SQLite 索引
│       │   │   └── sync.py              # 双向同步
│       │   ├── activation.py             # KV 张量 (借鉴 MemOS KVCacheMemory, 仅自托管)
│       │   └── parametric/              # 参数化 (借鉴 MemOS, 可选)
│       │       └── lora.py
│       │
│       ├── concerns/                     # 平面C: 横切关注点
│       │   ├── __init__.py
│       │   ├── capture/                  # 捕获方式
│       │   │   ├── __init__.py
│       │   │   ├── passive.py           # 被动 add
│       │   │   ├── autonomous.py        # agent 自治工具 (借鉴 letta core_memory_*)
│       │   │   ├── hooks.py             # PostToolUse hook (借鉴 Agent Memory)
│       │   │   └── pipeline.py          # SHA256去重→脱敏→压缩→双索引 (借鉴 Agent Memory)
│       │   ├── retrieval/               # 检索策略
│       │   │   ├── __init__.py
│       │   │   ├── progressive.py       # 渐进三层 (借鉴 ReMe meta→向量→历史)
│       │   │   ├── hybrid.py            # BM25+向量+图 (借鉴 Agent Memory)
│       │   │   ├── causal.py            # 【自研】反事实因果查询
│       │   │   └── forgetting.py        # 【自研】强度加权排序
│       │   ├── governance/              # 治理
│       │   │   ├── __init__.py
│       │   │   ├── token_budget.py     # token 预算 (借鉴 Hermes/Agent Memory)
│       │   │   ├── approval.py         # 写审批 (借鉴 Hermes write_approval)
│       │   │   ├── privacy.py          # 隐私脱敏 (借鉴 Agent Memory)
│       │   │   └── degradation.py      # 规则退化 (借鉴 Cass)
│       │   ├── evolution/              # 演化
│       │   │   ├── __init__.py
│       │   │   ├── zettel.py           # 链接生长 (借鉴 A-MEM Zettelkasten)
│       │   │   ├── reflect.py          # 反思升华 (借鉴 Cass)
│       │   │   └── dream.py            # Dream 整合 (借鉴 ReMe)
│       │   ├── sharing/                # 跨 agent 共享
│       │   │   ├── __init__.py
│       │   │   ├── user_id.py          # user_id 共享 (借鉴 Agno)
│       │   │   ├── unified_episodic.py # 统一 episodic 池 (借鉴 Cass)
│       │   │   └── rbac.py             # 【自研】权限治理
│       │   └── metacognition/          # 元认知 (横切, 状态持久化为语义)
│       │       ├── __init__.py
│       │       ├── router.py           # L0 路由 (借鉴 ReMe meta Layer0)
│       │       ├── coverage.py         # 【自研】L1 覆盖自描述
│       │       └── strategy.py         # 【自研】L2 策略自调
│       │
│       ├── sync/                        # 【自研】源同步器 (多形态一致性)
│       │   ├── __init__.py
│       │   ├── synchronizer.py          # 并行写+补偿+以图为权威源
│       │   └── drift.py                 # 漂移检测
│       │
│       ├── orchestration/               # 统一编排 API (借鉴 MemOS mem_os/mem_cube + mem0 Memory facade)
│       │   ├── __init__.py
│       │   ├── memory.py               # 【零配置 facade】Memory() 一行 API (借鉴 mem0 Memory)
│       │   ├── mem_os.py               # MOS 编排入口 (高级: 多类型编排)
│       │   ├── mem_cube.py            # MemCube 统一容器 (借鉴 MemOS)
│       │   └── registry.py            # 类型×形态 注册表
│       │
│       ├── providers/                   # 外部 provider 连接器 (借鉴 mem0/llms, letta/local_llm)
│       │   ├── __init__.py
│       │   ├── llms/
│       │   │   ├── openai.py
│       │   │   ├── anthropic.py
│       │   │   └── huggingface.py     # KVCache 必需
│       │   ├── embedders/
│       │   │   └── openai.py
│       │   └── rerankers/
│       │
│       ├── prompts/                     # 提示模板 (借鉴 letta/prompts/, MemOS templates/)
│       │   ├── __init__.py
│       │   ├── extract.py             # 记忆抽取提示
│       │   ├── causal_extract.py      # 【自研】因果边抽取
│       │   └── coverage_report.py     # 【自研】元认知覆盖报告生成
│       │
│       ├── utils/
│       │   ├── __init__.py
│       │   ├── hashing.py             # SHA-256 去重 (借鉴 Agent Memory)
│       │   ├── forgetting.py          # 【自研】Ebbinghaus 衰减计算
│       │   └── xml_compiler.py        # Block XML 编译 (借鉴 letta)
│       │
│       ├── observability/              # 可观测性 (借鉴 letta/monitoring + otel)
│       │   ├── __init__.py            # re-export: configure, get_logger, shutdown
│       │   ├── logging_utils.py       # 结构化日志 (structlog+滚动文件+彩色)
│       │   │                          #   ※ 暂内建, 待平台 solarseptem_core 建立后迁移
│       │   │                          #   ※ 不用 logging.py 名, 避免与标准库同名混淆
│       │   └── metrics.py             # 指标埋点 (未来: 检索命中率/衰减统计)
│       │
│       ├── cli/                        # 命令行入口 (借鉴 letta/cli)
│       │   ├── __init__.py
│       │   └── main.py               # `septmuse` 命令: init/search/add/dump
│       │
│       ├── exceptions/                 # 异常 (借鉴 letta/exceptions)
│       │   └── __init__.py
│       │
│       └── types/                     # 类型定义 (借鉴 MemOS types/, letta/types/)
│           └── __init__.py
│
└── tests/
    ├── __init__.py
    ├── unit/
    │   ├── test_block.py
    │   ├── test_episodic.py
    │   ├── test_causal.py             # 【自研】因果链测试集
    │   ├── test_forgetting.py         # 【自研】遗忘曲线测试
    │   └── test_metacognition.py      # 【自研】元认知测试
    ├── integration/
    │   ├── test_mvp_loop.py           # 阶段1 最小闭环
    │   ├── test_source_sync.py        # 多形态一致性
    │   └── test_cross_agent.py        # 跨 agent 共享
    └── e2e/
        └── test_full_memory_loop.py
```

---

## 4. 三维架构 → 目录映射

| 架构平面 | 目录 | 说明 |
|---------|------|------|
| 平面 A 内容类型 | `src/septmuse/content_types/` | 每类一子包，内含该类型特有逻辑 |
| 平面 B 存储形态 | `src/septmuse/storage/` | 按形态分后端，与类型解耦 |
| 平面 C 横切关注点 | `src/septmuse/concerns/` | 6 个关注点各一子包 |
| 数据模型 | `src/septmuse/schemas/` | SQLModel 定义，集中管理 |
| 编排入口 | `src/septmuse/orchestration/` | 统一 API + MemCube 容器 |
| 自研创新 | 散布（标注【自研】） | causal/strength/meta/sync/forgetting |

**解耦原则**：`content_types/` 只管"业务逻辑"（抽取、自编辑、时序、退化）；`storage/` 只管"形态后端"（读写具体载体）；交叉点（如 Block 既属工作记忆又属 block 形态）放在 `content_types/working/`，因其带强工作记忆特征。

---

## 5. 借鉴源映射表（每目录对应开源库）

| 目录 | 借鉴源 | 复用方式 |
|------|--------|---------|
| `schemas/block.py` | letta `schemas/block.py` `schemas/memory.py` | 移植 Block schema + core_memory_* |
| `content_types/working/block.py` | letta `functions/` core_memory_append/replace | 移植自编辑工具 |
| `content_types/working/eviction.py` | Hermes `memory_char_limit` | 采用硬上限+驱逐 |
| `content_types/episodic/temporal.py` | Zep/Graphiti Episode | 采用 reference_time 语义 |
| `content_types/episodic/reasoning.py` | LangMem `Episode(obs/act/result)` | 直接采用 schema |
| `content_types/episodic/raw_log.py` | Cass Episodic | 采用统一 raw log 理念 |
| `content_types/semantic/extract.py` | Cognee `cognify` 流水线 | 移植 classify→chunk→extract_graph |
| `content_types/procedural/playbook.py` | Cass Playbook helpful/harmful | 采用计数+deprecation |
| `storage/vector/pgvector.py` | mem0 `vector_stores/` | 采用后端抽象 |
| `storage/graph/age.py` | Graphiti ontology + fact | 采用三元组+边类型 |
| `storage/file/markdown.py` `sync.py` | Basic Memory | 采用 Markdown+SQLite+双向同步 |
| `storage/activation.py` | MemOS `KVCacheMemory` | 移植 extract/add/get_cache |
| `concerns/capture/hooks.py` `pipeline.py` | Agent Memory PostToolUse | 移植 hook 流水线 |
| `concerns/retrieval/progressive.py` | ReMe 三层 | 采用 meta→向量→历史 |
| `concerns/retrieval/hybrid.py` | Agent Memory BM25+向量+图 | 采用三路融合 |
| `concerns/governance/*` | Hermes + Agent Memory + Cass | 各取一治理点 |
| `concerns/evolution/zettel.py` | A-MEM Zettelkasten | 采用 add 时自动找关系 |
| `concerns/evolution/reflect.py` `dream.py` | Cass reflect + ReMe Dream | 采用反思+整合流程 |
| `concerns/sharing/user_id.py` | Agno user_id 共享 db | 采用同 db+user_id 模式 |
| `concerns/metacognition/router.py` | ReMe meta Layer0 ReadMetaMemoryOp | 采用命名空间索引 |
| `orchestration/mem_os.py` `mem_cube.py` | MemOS `mem_os/` `mem_cube/` | 采用编排+容器模式 |
| `providers/llms/*` | mem0 `llms/` letta `local_llm/` | 采用 provider 抽象 |
| `prompts/` | letta `prompts/` MemOS `templates/` | 采用模板组织 |
| `utils/hashing.py` | Agent Memory SHA-256 去重 | 直接采用 |
| `alembic/` | letta `alembic/` | 采用 DB 迁移 |

---

## 6. 自研模块落位（创新空白 + 编排）

| 自研模块 | 落位 | 对应架构章节 |
|---------|------|-------------|
| 因果链记忆 | `schemas/causal.py` + `concerns/retrieval/causal.py` + `prompts/causal_extract.py` | §6.1 |
| 遗忘曲线 | `schemas/strength.py` + `concerns/retrieval/forgetting.py` + `utils/forgetting.py` | §6.2 |
| 元认知 L1/L2 | `concerns/metacognition/coverage.py` `strategy.py` + `prompts/coverage_report.py` | §6.3 |
| 源同步器 | `sync/synchronizer.py` `drift.py` | §4.1 |
| 统一编排 API | `orchestration/mem_os.py` `mem_cube.py` `registry.py` | §11.2 |
| 跨 agent 权限 | `concerns/sharing/rbac.py` | §7.2 |
| 置信度+溯源 | `schemas/semantic.py` confidence+provenance 字段 | §3.2.2 |

---

## 7. 与 solarseptem-ai 平台集成

| 平台子系统 | 集成点 | 目录 |
|-----------|--------|------|
| SolAgent | 通过 `api/rest.py` 读写记忆 | `api/` |
| model_gateway | `providers/llms/` 调用 LLM | `providers/` |
| mcp_market | `api/mcp.py` 暴露记忆工具为 MCP | `api/mcp.py` |
| agent_runner | `concerns/capture/hooks.py` 接 PostToolUse | `concerns/capture/` |

---

## 8. 依赖清单（pyproject.toml 核心）

| 依赖 | 用途 | 必需性 |
|------|------|--------|
| fastapi | API 框架 | 必需 |
| sqlmodel | ORM | 必需 |
| psycopg2 | Postgres 驱动 | 必需 |
| pgvector | 向量扩展 | 必需 |
| alembic | 迁移 | 必需 |
| openai / anthropic | LLM provider | 必需 |
| numpy | KV 张量 | 仅激活记忆 |
| transformers | 自托管模型 (KVCache/LoRA) | 可选 |
| mcp | MCP server | 必需 |
| structlog | 结构化日志 | 必需 (observability/) |
| pydantic | 已随 sqlmodel | — |
| pytest | 测试 | dev |

> Neo4j 驱动仅当用 neo4j 后端时；默认 Apache AGE 走 Postgres，零额外依赖。

---

## 9. 落地建议

1. **先建骨架**：创建 `src/septmuse/` 空包 + `pyproject.toml` + `__init__.py`，确保 `python -c "import septmuse"` 通过。
2. **阶段1 MVP** 只填：`schemas/block.py` + `content_types/working/block.py` + `content_types/semantic/` + `storage/vector/pgvector.py` + `storage/graph/age.py` + `orchestration/mem_os.py` + `api/rest.py`。
3. 其余目录留 `__init__.py` 占位，按 §9 演进路线逐阶段填充。
4. 每填一个格子写对应单元测试（`tests/unit/`）。

---

## 10. 日志模块归属与命名

### 10.1 代码评估

用户提供的日志模块（`solarseptem_core.logging`，structlog + 滚动文件 + 彩色 + atexit 清理 + 线程锁）**可用，质量良好**，作为 SeptMuse 默认日志实现。

**3 个待优化点**：

| # | 点 | 处理 |
|---|----|------|
| 1 | 子模块名 `logging` 与标准库 `logging` 同名 | 落 SeptMuse 时改名 `logging_utils.py`，放 `observability/` 子包内（绝对导入不冲突，但避免 IDE/mypy 提示混淆） |
| 2 | 模块底部"导入即配置"副作用 | 保留开箱即用语义，但把 `_log_configured` 与 `_configured_once` 合并为单一 `_configured` 标志，并在 `configure()` 入口加幂等守卫（已配置则跳过默认副作用） |
| 3 | 两标志位冗余 | 同 #2，合并 |

### 10.2 归属决策

| 项 | 决策 | 理由 |
|----|------|------|
| 当前归属 | 暂内建于 `septmuse/observability/logging_utils.py` | 平台 `solarseptem_core` 尚不存在，SeptMuse 先自持可运行 |
| 命名 | `logging_utils` 而非 `logging` | 避免与标准库同名；`observability/` 子包隔离 |
| 迁移路径 | 平台 `solarseptem_core` 建立后，`logging_utils.py` 上移为 `solarseptem_core.logging_utils`，SeptMuse 改为 `from solarseptem_core.logging_utils import get_logger` | 日志是平台共享能力，不应各子系统各自维护 |
| 未来扩展 | `observability/metrics.py` 放检索命中率、衰减统计、复述触发数等指标 | 对齐 letta `monitoring/` + `otel/` |

### 10.3 使用约定（SeptMuse 内部）

```python
# 所有 septmuse 模块统一这样获取 logger
from septmuse.observability import get_logger

logger = get_logger(__name__)
logger.info("memory_added", memory_id=mid, type="semantic")
```

- 应用启动入口（`cli/main.py` / `api/rest.py`）调用一次 `configure()` 覆盖默认
- 业务模块只调 `get_logger()`，不重复 `configure()`
- structlog 事件用 kwarg 传结构化字段（如 `memory_id=` / `type=`），便于 `metrics.py` 聚合

---

## 11. 目录结构完整性自检

### 11.1 与三维架构的覆盖核对

| 架构要素 | 目录 | 状态 |
|---------|------|:----:|
| 平面A 工作记忆 | `content_types/working/` | ✅ |
| 平面A 情节 | `content_types/episodic/` | ✅ |
| 平面A 语义 | `content_types/semantic/` | ✅ |
| 平面A 程序 | `content_types/procedural/` | ✅ |
| 平面B block | `storage/block_store.py` | ✅ |
| 平面B 向量 | `storage/vector/` | ✅ |
| 平面B 图 | `storage/graph/` | ✅ |
| 平面B 文件 | `storage/file/` | ✅ |
| 平面B 激活 | `storage/activation.py` | ✅ |
| 平面B 参数化 | `storage/parametric/` | ✅ |
| 平面C 捕获 | `concerns/capture/` | ✅ |
| 平面C 检索 | `concerns/retrieval/` | ✅ |
| 平面C 治理 | `concerns/governance/` | ✅ |
| 平面C 演化 | `concerns/evolution/` | ✅ |
| 平面C 共享 | `concerns/sharing/` | ✅ |
| 平面C 元认知 | `concerns/metacognition/` | ✅ |
| 自研 源同步 | `sync/` | ✅ |
| 统一编排 | `orchestration/` | ✅ |
| 数据模型 | `schemas/` | ✅ |
| API | `api/` | ✅ |
| Providers | `providers/` | ✅ |
| Prompts | `prompts/` | ✅ |
| Utils | `utils/` | ✅ |
| **可观测性** | `observability/` | ✅ 新增 |
| **CLI** | `cli/` | ✅ 新增 |
| 异常 | `exceptions/` | ✅ |
| 类型 | `types/` | ✅ |
| DB 迁移 | `alembic/` | ✅ |
| 测试 | `tests/{unit,integration,e2e}/` | ✅ |

### 11.2 对照开源库补齐的目录

| 目录 | 参考开源库 | 为何需要 |
|------|-----------|---------|
| `observability/` | letta `monitoring/` + `otel/` | 日志/指标，原结构遗漏 |
| `cli/` | letta `cli/`、mem0 `cli/` | 命令行入口（init/search/dump），原结构遗漏 |
| `alembic/` | letta `alembic/` | SQLModel 迁移，原结构已列 |
| `prompts/` | letta `prompts/`、MemOS `templates/` | LLM 提示模板，原结构已列 |

### 11.3 结论

目录结构现已**完整覆盖三维架构 + 平台集成 + 工程基础设施**，无遗漏。新增 `observability/`（含日志）和 `cli/` 两块后，与 letta/mem0/MemOS 的工程目录完备度对齐。

**下一步建议**：
1. 按此结构生成脚手架（空 `__init__.py` + `pyproject.toml`），确保 `python -c "import septmuse"` 与 `python -c "from septmuse.observability import get_logger"` 通过
2. 把用户提供的日志代码（改名 `logging_utils.py` + 合并冗余标志位）落入 `observability/`
3. 进入阶段1 MVP 实现

---

## 12. 零配置开箱即用设计（pip install 即可用）

### 12.1 设计目标

`pip install septmuse` 后，用户**无需启动任何外部服务**（不要 Postgres / Neo4j / Qdrant / Redis），即可运行：

```python
from septmuse import Memory

m = Memory()                          # 零配置: 默认 SQLite 组合后端
m.add("我喜欢 Python 和异步编程", user_id="alice")
results = m.search("alice 喜欢什么", user_id="alice")
```

### 12.2 与 mem0 零配置对比

| 组件 | mem0 默认 | SeptMuse 默认 | 优势 |
|------|----------|--------------|------|
| 向量库 | 本地 Qdrant (/tmp/qdrant) | **SQLite + sqlite-vec** | 完全零外部进程，单文件 |
| 图库 | 无（仅向量） | **SQLite 三元组表** | 开箱即有关系推理 |
| 历史 | SQLite | SQLite | 对齐 |
| 嵌入 | OpenAI text-embedding-3-small（需 key） | **sentence-transformers 本地模型**（零 key） | 无 API 成本即可跑 |
| LLM | OpenAI gpt-5-mini（需 key） | 环境变量优先级解析 | 仍需 key，但可选 ollama 本地 |
| 文件记忆 | 无 | 默认关闭（可选开启） | 不污染用户磁盘 |
| 激活/参数化 | 无 | 默认关闭（需自托管模型） | — |

**差异化**：SeptMuse 默认**完全零外部服务**（连 Qdrant 都不要，单 `.septmuse.db` 文件），嵌入用本地模型零 API key。LLM key 仍需用户提供（无法避免，除非接 ollama）。

### 12.3 默认后端选择逻辑（`configs/defaults.py`）

```python
# 伪代码: 默认后端解析优先级
def resolve_defaults():
    return Defaults(
        # 1. 后端: 优先 SQLite 组合 (零外部服务)
        metadata_backend = "sqlite",          # ~/.septmuse/septmuse.db
        vector_backend   = env_or("sqlite_vec", "pgvector"),  # 默认 sqlite-vec
        graph_backend    = "sqlite_graph",    # SQLite 三元组表
        file_memory      = False,             # 默认关闭

        # 2. 嵌入: 优先本地 sentence-transformers (零 key)
        embedder = env_or("sentence-transformers/all-MiniLM-L6-v2",
                          "openai:text-embedding-3-small"),

        # 3. LLM: 从环境变量解析 (无法零 key, 但支持 ollama 本地)
        llm = resolve_llm_from_env(),  # OPENAI_API_KEY / ANTHROPIC_API_KEY / OLLAMA_BASE_URL

        # 4. 激活/参数化: 默认关闭
        activation = False,
        parametric  = False,
    )
```

### 12.4 facade API（`orchestration/memory.py`）

借鉴 mem0 `Memory` 类，提供一行 API：

```python
class Memory:
    """SeptMuse 记忆系统零配置入口。

    pip install septmuse 后直接:
        from septmuse import Memory
        m = Memory()
        m.add(messages, user_id="alice")
        m.search("query", user_id="alice")
    """

    def __init__(self, config: MemoryConfig | None = None) -> None:
        # config=None → 用 configs.defaults 的零配置默认
        # 内部组装 MemCube + MemOS + 默认 SQLite 后端
        ...

    def add(self, messages, *, user_id, metadata=None) -> dict:
        """写入: 触发抽取流水线 → 语义事实 + 情节事件 → 多形态存储"""
        ...

    def search(self, query: str, *, user_id, top_k=5) -> list:
        """检索: 元认知路由 → 混合检索 → token 预算注入"""
        ...

    def get_all(self, *, user_id) -> dict:
        """读取该用户全部记忆"""
        ...

    def remember(self, observation, thoughts, action, result, *, user_id) -> str:
        """写入情节推理经验 (借鉴 LangMem Episode)"""
        ...

    def forget(self, memory_id: str) -> None:
        """软删除 (标记 deprecated, 不物理删)"""
        ...
```

### 12.5 升级到生产（显式配置）

```python
from septmuse import Memory, MemoryConfig

m = Memory(config=MemoryConfig(
    metadata_backend="postgres",       # 升级到 Postgres
    vector_backend="pgvector",
    graph_backend="age",              # Apache AGE
    embedder="openai:text-embedding-3-large",
    llm="openai/gpt-4o",
    file_memory=True,                 # 开启 Markdown 人可读
    file_dir="./memories",
))
```

### 12.6 依赖分层（`pyproject.toml` extras）

```toml
[project]
name = "septmuse"
dependencies = [
    "fastapi", "sqlmodel", "structlog", "mcp", "pydantic",
    "sqlite-vec",              # 默认向量后端 (零外部服务)
    "sentence-transformers",  # 默认嵌入 (零 API key)
]

[project.optional-dependencies]
openai = ["openai>=1.0"]
anthropic = ["anthropic"]
postgres = ["psycopg2", "pgvector"]
graph = ["neo4j"]              # 仅 neo4j 后端
activation = ["transformers", "torch"]   # KV-cache/LoRA, 自托管
dev = ["pytest", "pytest-asyncio", "ruff", "mypy"]
all = ["septmuse[openai,anthropic,postgres,graph,activation,dev]"]
```

**默认安装**（`pip install septmuse`）即可跑：SQLite + sqlite-vec + sentence-transformers，零外部服务、零 API key（嵌入），仅 LLM key 需用户提供。

### 12.7 包顶层导出（`src/septmuse/__init__.py`）

```python
from septmuse.orchestration.memory import Memory
from septmuse.configs.defaults import MemoryConfig

__all__ = ["Memory", "MemoryConfig"]
__version__ = "0.1.0"
```

用户只需 `from septmuse import Memory`。

### 12.8 零配置可行性核对

| 要求 | 实现 | 风险 |
|------|------|------|
| 无外部服务 | SQLite 单文件 + sqlite-vec + 本地嵌入 | 低, 全部纯 Python 可装 |
| 无 API key（嵌入） | sentence-transformers 本地模型 | 低, 模型首次下载 ~80MB |
| LLM key 仍需 | 无法避免（除非 ollama） | 中, 提供清晰报错指引 |
| 一行 API | `Memory()` facade | 低, 借鉴 mem0 模式 |
| 升级路径 | `Memory(config=...)` 切后端 | 低, 抽象基类已设计 |

> sqlite-vec 是 SQLite 扩展，提供向量检索；若环境不支持加载扩展，回退 numpy 纯 Python 余弦相似（更慢但零依赖）。

---

## 13. MCP 服务设计（基于 mem0 源码实证）

### 13.1 结论：支持，且是核心交付

SeptMuse **必须**作为 MCP server 暴露记忆能力，让 Claude Code / Cursor / Codex / SolAgent 等任何 MCP client 直接调用。这是零配置理念的关键延伸：`pip install septmuse` 后无需起服务进程即可在编辑器里用记忆。

### 13.2 源码参考（实证，非自行发挥）

直接参考本地 `opensource/mem0/openmemory/api/app/mcp_server.py`（574 行，已通读）。mem0 的实测模式：

| 要素 | mem0 实测实现 | SeptMuse 对齐 |
|------|-------------|--------------|
| SDK | `from mcp.server.fastmcp import FastMCP` | 同（FastMCP 高层 API） |
| 实例 | `mcp = FastMCP("mem0-mcp-server")` | `mcp = FastMCP("septmuse-mcp-server")` |
| 工具 | `@mcp.tool(description=...)` 装饰器 | 同模式 |
| 用户上下文 | `contextvars.ContextVar("user_id")` + URL 路径参数 set | 同（contextvars） |
| SSE transport | `SseServerTransport("/mcp/messages/")` + `@router.get("/{client}/sse/{user}")` | 同 |
| Streamable HTTP | `StreamableHTTPServerTransport(mcp_session_id=None, is_json_response_enabled=True)` stateless | 同（新规范，替代 SSE） |
| 挂载 FastAPI | `setup_mcp_server(app): app.include_router(mcp_router)` | 同 |
| lazy client | `get_memory_client_safe()` 失败不崩 | 同（对接 §12 零配置 Memory） |
| 权限 + 访问日志 | `check_memory_access_permissions` + `MemoryAccessLog` | 同（对接 §concerns/sharing/rbac） |

### 13.3 SeptMuse 相对 mem0 的增量

| 增量 | mem0 无 | SeptMuse 有 |
|------|:------:|:----------:|
| **stdio transport**（本地编辑器零服务） | ✗（openmemory 需 docker 起 http） | ✅ `mcp.run(transport="stdio")` |
| 工具扩展（记忆类型全覆盖） | 仅 5 工具 | 9+ 工具（见 §13.4） |
| 因果/复述/覆盖工具 | ✗ | ✅ 三创新空白暴露为 MCP tool |

**stdio 是关键差异化**：mem0 的 MCP 走 http（需 docker compose 起 openmemory），SeptMuse 通过 `septmuse mcp` 或 `python -m septmuse.api.mcp` 直接 stdio 启动，Claude Code 配 `.mcp.json` 即用，无需任何服务进程。

### 13.4 MCP 工具集（`api/mcp/tools.py`）

对齐 mem0 5 工具 + SeptMuse 特有 4 工具：

```python
# === 基础工具 (对齐 mem0) ===
@mcp.tool(description="添加记忆。用户告知任何偏好/事实时调用。infer=False 存原文不抽取")
async def add_memories(text: str, infer: bool = True) -> str: ...

@mcp.tool(description="搜索记忆。每次用户提问时调用")
async def search_memory(query: str) -> str: ...

@mcp.tool(description="列出用户全部记忆")
async def list_memories() -> str: ...

@mcp.tool(description="按 ID 删除指定记忆")
async def delete_memories(memory_ids: list[str]) -> str: ...

@mcp.tool(description="删除用户全部记忆")
async def delete_all_memories() -> str: ...

# === SeptMuse 扩展工具 (创新空白暴露为 MCP tool) ===
@mcp.tool(description="记录成功交互的推理经验 (借鉴 LangMem Episode)")
async def remember_episode(observation: str, thoughts: str, action: str, result: str) -> str: ...

@mcp.tool(description="反事实因果查询: 若某事件未发生,结果是否仍成立")
async def causal_query(cause_event_id: str, hypothesized_effect: str) -> str: ...

@mcp.tool(description="触发主动复述强化低强度高价值记忆")
async def rehearse() -> str: ...

@mcp.tool(description="生成元认知覆盖报告: agent 记住了什么/记不住什么")
async def coverage_report() -> str: ...
```

### 13.5 三种 transport（`api/mcp/transports.py`）

| transport | 用途 | 启动 | 配置 |
|-----------|------|------|------|
| **stdio** | 本地编辑器（Claude Code/Cursor）零服务 | `septmuse mcp` 或 `python -m septmuse.api.mcp` | `.mcp.json: {"command":"septmuse","args":["mcp"]}` |
| **streamable_http** | 远程服务（新规范，替代 SSE） | `septmuse serve --transport http` | `.mcp.json: {"type":"http","url":"..."}` |
| **sse** | 兼容旧 client | `septmuse serve --transport sse` | `.mcp.json: {"type":"sse","url":"..."}` |

### 13.6 stdio 启动入口（`api/mcp/server.py`）

```python
# 源码模式参考 mem0, 加 stdio 入口
from mcp.server.fastmcp import FastMCP
from septmuse.orchestration.memory import Memory

mcp = FastMCP("septmuse-mcp-server")

def _get_memory() -> Memory:
    """lazy + 零配置, 失败不崩 (源码参考 mem0 get_memory_client_safe)"""
    try:
        return Memory()  # §12 零配置默认
    except Exception as e:
        return None

def setup_mcp_server(app: FastAPI) -> None:
    """挂载到 FastAPI (http/sse transport), 源码参考 mem0 setup_mcp_server"""
    from septmuse.api.mcp.transports import mount_http, mount_sse
    mount_http(app, mcp)
    mount_sse(app, mcp)

def run_stdio() -> None:
    """stdio 模式入口, Claude Code 等本地编辑器用"""
    mcp.run()  # FastMCP 默认 stdio

if __name__ == "__main__":
    run_stdio()
```

### 13.7 Claude Code 配置示例（用户侧）

`~/.claude/mcp.json` 或项目 `.mcp.json`：

```json
{
  "mcpServers": {
    "septmuse": {
      "command": "septmuse",
      "args": ["mcp"]
    }
  }
}
```

`pip install septmuse` + 上述配置 → Claude Code 直接获得 9 个记忆工具，零服务进程。

### 13.8 依赖补充

`pyproject.toml` §12.6 依赖表需补：

| 依赖 | 用途 | 必需性 |
|------|------|--------|
| `mcp` | MCP Python SDK (FastMCP/transport) | 必需 |
| `anyio` | streamable_http task group | 必需（mem0 同款） |

### 13.9 与 mcp_market 子系统集成

SeptMuse 的 MCP server 注册到平台 `mcp_market` 子系统，供 SolAgent 等发现并调用。注册元数据：工具清单、transport 端点、所需权限。

### 13.10 验证计划

- 单元测试：每个 `@mcp.tool` 单测（mock Memory）
- 集成测试：stdio 模式启动 + client 调工具往返
- 端到端：Claude Code 配 `.mcp.json` 实测 9 工具可用
- 源码对齐核对：与 `opensource/mem0/openmemory/api/app/mcp_server.py` 的 transport/context/lazy 模式逐项核对，禁止偏离源码模式自行发挥



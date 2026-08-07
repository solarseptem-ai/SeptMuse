# SeptMuse 优化计划 V2：从基础设施到顶层

> 日期：2026-08-05
> 前置文档：`docs/specs/opensource-gap-analysis.md`（差距分析，2026-07-23）
> 前置完成：`docs/plans/development-roadmap.md` P0-P4 全部完成
> 目标：在 P0-P4 功能补齐后，从基础设施层到生态层系统性优化

---

## 1. 差距重新评估（P0-P4 完成后）

### 1.1 能力雷达图对比

| 维度 | mem0 | letta | cognee | graphiti | MemOS | ReMe | SeptMuse(旧) | SeptMuse(现在) |
|------|------|-------|--------|----------|-------|------|-------------|---------------|
| 数据模型 | 6 | 8 | 9 | 9 | 10 | 7 | 7 | **8** |
| 检索策略 | 9 | 6 | 8 | 10 | 8 | 7 | 7 | **9** |
| 记忆治理 | 3 | 5 | 8 | 4 | 7 | 1 | 9 | **9** |
| 记忆演化 | 5 | 9 | 9 | 9 | 9 | 8 | 7 | **8** |
| 实体抽取/KG | 7 | 1 | 9 | 10 | 7 | 2 | 2 | **8** |
| 时态维度 | 2 | 5 | 7 | 10 | 7 | 4 | 5 | **8** |
| LLM 集成深度 | 8 | 9 | 9 | 8 | 8 | 8 | 3 | **6** |
| 多租户 | 7 | 9 | 9 | 5 | 8 | 1 | 5 | **5** |
| API 层 | 8 | 9 | 9 | 6 | 7 | 7 | 7 | **8** |
| 零配置 | 1 | 1 | 1 | 1 | 1 | 8 | 10 | **10** |
| **基础设施** | 7 | 7 | 8 | 7 | 8 | 6 | 5 | **5** |

### 1.2 已补齐的差距（14 项）

| # | 差距 | 补齐 Phase | 借鉴来源 |
|---|------|-----------|----------|
| 1 | 实体抽取 | P0 | mem0 spaCy + regex |
| 2 | 三元组 LLM 联合抽取 | P0 | graphiti |
| 3 | cognify 知识图谱构建 | P0 | cognee + graphiti |
| 4 | 实体向量库 + entity boost | P0+P1 | mem0 V3 去图化 |
| 5 | Reranker 框架（4 种） | P1 | graphiti + mem0 |
| 6 | BFS 图遍历检索 | P1 | graphiti |
| 7 | 预置检索 Recipes（7 种） | P1 | graphiti |
| 8 | 双时态建模 | P2 | graphiti EntityEdge |
| 9 | 时态区间查询 + NL 时间抽取 | P2 | cognee TemporalRetriever |
| 10 | 消息压缩 Summarizer | P2 | letta |
| 11 | LLM 事实抽取 ADDITIVE prompt | P3 | mem0 V3 |
| 12 | 冲突解决 + 实体去重 | P3 | graphiti 三段式 |
| 13 | Session 蒸馏两阶段 | P3 | cognee distill |
| 14 | V2 记忆导向 API | V2 架构 | cognee remember/recall/improve/forget |

### 1.3 仍存在的差距（按层分类）

**基础设施层（最大瓶颈）**：
- 向量检索全扫描（无 HNSW 近似索引），10K+ 记忆线性退化
- 检索三路串行（向量→BM25→entity），无并发
- BM25 无中文分词（纯空格分词，中文检索质量差）
- async 覆盖不全（V2Memory remember/recall/improve/forget 全同步）
- 无连接池管理（SQLite 每次创建 Session，无 WAL mode）
- 无查询结果缓存（相同 query 重复检索）

**LLM 集成层**：
- 无 LLM 自编辑记忆（letta apply_patch + rethink）
- 无后台异步处理（letta SleeptimeAgent / ReMe cron auto_dream）
- 无结构化输出 JSON schema 约束
- LLM 默认关闭（其他系统默认开）

**记忆演化层**：
- Dream 太简单（仅批量建链接，无 extract→integrate→topics→proactive）
- 无社区检测 + CommunityNode（graphiti label_propagation）
- 无 Saga 增量摘要（graphiti SagaNode）
- 无三权重演化（cognee feedback + importance + frequency）
- 无边冲突解决（graphiti 双时态失效）

**治理 + 多租户层**：
- 无 RBAC 角色系统（cognee/MemOS/letta 有）
- 无 Provenance 溯源删除（cognee source_ref）
- 无 Dataset/Cube 隔离

**生态层**：
- 无 TS/Rust SDK
- 无 Web UI
- 无 OpenAI/Anthropic 兼容端点
- 无多模态/参数化记忆（MemOS act_mem/para_mem）

---

## 2. 优化计划总览

| Phase | 名称 | 优先级 | 核心目标 | 预估工作量 |
|-------|------|--------|----------|-----------|
| **P0** | 基础设施加固 | 最高 | 检索性能 10x + async 全链路 + 中文分词 | 大 |
| **P1** | LLM 集成深化 | 高 | 自编辑 + 结构化输出 + 后台异步 | 中 |
| **P2** | 记忆演化深化 | 中 | Dream 升级 + 社区检测 + 三权重 | 中 |
| **P3** | 治理 + 多租户 | 中低 | RBAC + Provenance + Dataset 隔离 | 中 |
| **P4** | 生态扩展 | 低 | TS SDK + Web UI + 兼容端点 | 大 |

---

## Phase 0：基础设施加固

> **优先级：最高**
> **目标**：从"功能可用"升级到"生产级性能"，检索延迟降低 5-10x，支持 10K+ 记忆库

### P0-Task 1：HNSW 近似最近邻索引

**问题**：当前 `SQLAlchemyVectorStore` 用 numpy 余弦全扫描（`np.dot(query, vec) / norm`），10K 记忆 ~200ms，100K 记忆 ~2s。

**方案**：
- `SQLAlchemyVectorStore` 增加 HNSW 索引层（`hnswlib` 库，纯 Python + C++，无外部服务）
- 写入时同步更新 HNSW 索引（`index.add_items(vec, id)`）
- 检索时优先走 HNSW（`index.knn_query(vec, k=top_k)`），降级全扫描
- 索引持久化到 `~/.septmuse/hnsw.index`，启动时加载
- Chroma 后端已有 HNSW（`metadata={"hnsw:space": "cosine"}`），无需改

**验收标准**：
- 10K 记忆检索 < 20ms（vs 当前 ~200ms）
- 100K 记忆检索 < 50ms
- 索引重建 < 5s（10K 记忆）
- 无索引文件时降级全扫描不崩溃

### P0-Task 2：BM25 中文分词

**问题**：当前 `SQLiteBM25Index` 用空格分词，中文查询 "Alice的工作经历" 被当作一个 token，BM25 完全失效。

**方案**：
- `SQLiteBM25Index` 增加 jieba 分词后端（`jieba.lcut(text)`）
- `SEPTMUSE_TOKENIZER=space/jieba` 环境变量切换（默认 jieba，降级 space）
- BM25 索引重建支持增量更新（避免全量重建）
- IDF 全局统计持久化（`septmuse_bm25_stats` 表）

**验收标准**：
- "Alice的工作经历" 正确分词为 ["Alice", "的", "工作", "经历"]
- 中文 BM25 召回率提升 ≥ 50%（基准测试）
- 无 jieba 时降级空格分词不崩溃

### P0-Task 3：检索三路并发

**问题**：当前 `HybridRetriever.search` 串行执行 向量→BM25→entity boost，三路串行延迟叠加。

**方案**：
- 用 `concurrent.futures.ThreadPoolExecutor` 并发执行三路检索
- 向量检索 + BM25 检索 + entity boost 同时启动
- `asyncio.gather` 支持 async 路径
- 超时控制（每路 max 100ms，超时降级空结果）

**验收标准**：
- 三路并发延迟 = max(v, b, e) 而非 v + b + e
- 单路超时不阻塞其他路
- 基准测试延迟降低 ≥ 30%

### P0-Task 4：Async 全链路覆盖

**问题**：V2Memory 的 `remember`/`recall`/`improve`/`forget` 全同步，REST API 用 `async def` 但内部调 sync 方法。

**方案**：
- 新建 `memory/async_memory_v2.py` `AsyncV2Memory`
- 4 编排方法全部 async（`async def remember`/`recall`/`improve`/`forget`）
- 子组件 async 化：`AsyncCapturePipeline` / `AsyncHybridRetriever` / `AsyncEvolutionEngine`
- REST API 直接调 async 路径（去掉 `app.state.memory` sync 回退）
- MCP 工具用 async 路径

**验收标准**：
- REST API 全链路 async（无 sync 回退）
- 并发 100 请求延迟 < 1s（vs 当前串行 ~5s）
- `AsyncV2Memory` 通过全部 V2 测试

### P0-Task 5：连接池 + WAL mode

**问题**：SQLite 每次创建 Session，无 WAL mode，写操作阻塞读。

**方案**：
- `RelationalStoreFactory` 创建 engine 时设 `connect_args={"check_same_thread": False}`
- SQLite 启用 WAL mode（`PRAGMA journal_mode=WAL`）+ busy_timeout
- 连接池大小配置（`pool_size=10`，SQLite 用 `StaticPool` 单连接共享）
- 事务隔离级别（SQLite 默认 SERIALIZABLE → READ COMMITTED for PG/MySQL）

**验收标准**：
- 并发读写不阻塞（WAL mode）
- 连接池复用（无重复创建 Session）
- 写入 + 同时检索延迟 < 50ms

### P0-Task 6：查询结果缓存

**问题**：相同 query 重复检索，无缓存。

**方案**：
- `HybridRetriever` 增加 LRU 查询缓存（`functools.lru_cache` 或 `cachetools.TTLCache`）
- 缓存 key = `(query_hash, user_id, top_k, filters_hash)`
- TTL 5 分钟（记忆更新后缓存自动失效）
- `Memory.add`/`update`/`delete` 时清除用户级缓存

**验收标准**：
- 相同 query 二次检索 < 1ms
- 记忆变更后缓存失效
- 缓存命中率 > 60%（典型对话场景）

---

## Phase 1：LLM 集成深化

> **优先级：高**
> **目标**：LLM 从"抽取事实"深化到"自编辑 + 后台异步 + 结构化输出"

### P1-Task 1：LLM 自编辑记忆

**借鉴**：letta `memory_apply_patch`（统一 diff）+ `memory_rethink`（主动重写）

**方案**：
- `Memory.apply_patch(memory_id, diff)` — LLM 生成统一 diff，多块编辑
- `Memory.rethink(memory_id, instruction)` — LLM 主动重写记忆内容
- REST `PATCH /memories/{id}` 接受 diff 格式
- MCP `edit_memory` 工具
- 版本历史保留（history 表记录 diff）

### P1-Task 2：结构化输出 JSON schema

**借鉴**：mem0 `response_format={"type": "json_object"}` + graphiti JSON schema 约束

**方案**：
- LLM ABC 增加 `complete_structured(system, user, schema) -> dict`
- OpenAI provider 用 `response_format={"type": "json_schema", "json_schema": {...}}`
- Ollama provider 用 `format=schema`
- FactExtractor / TripletExtractor / ConflictResolver 全部用结构化输出
- 消除正则解析 LLM 输出的脆弱性

### P1-Task 3：后台异步处理

**借鉴**：letta `SleeptimeAgent` + ReMe `cron auto_dream`

**方案**：
- 新建 `concerns/background/scheduler.py` `BackgroundScheduler`
- 支持 cron 触发（`schedule.every().day.at("02:00").do(improve)`）
- 后台任务：Dream 链接生长、reflect 蒸馏、冲突解决、遗忘曲线衰减
- `Memory.start_background()` / `Memory.stop_background()`
- REST `POST /maintenance/run` 手动触发

### P1-Task 4：LLM 默认开启（可选）

**问题**：当前 `infer=False` 默认 verbatim 模式，其他系统默认开。

**方案**：
- `SEPTMUSE_INFER=true` 环境变量（已有）
- 文档引导用户配置 LLM provider
- 无 LLM 时降级 verbatim 模式（已有）
- 不改默认值（零配置原则），但改善 onboarding 文档

---

## Phase 2：记忆演化深化

> **优先级：中**
> **目标**：从"简单链接"升级到"社区检测 + Saga 摘要 + 三权重演化"

### P2-Task 1：Dream 升级

**借鉴**：ReMe `extract→integrate→topics→proactive` + MemOS `MotiveType` 驱动

**方案**：
- `DreamIntegrator` 从"仅批量建链接"升级为四阶段：
  1. `extract`：跨记忆抽取 memory units（LLM）
  2. `integrate`：CREATE/CORROBORATE/REFINE/CORRECT 四态写入
  3. `topics`：兴趣主题选择（LLM 聚类）
  4. `proactive`：主动暴露主题（interests.yaml 或 metadata）
- MotiveType 驱动：newness/frequency/conflict/feedback/fragmentation

### P2-Task 2：社区检测 + CommunityNode

**借鉴**：graphiti `label_propagation` + `CommunityNode.summary`

**方案**：
- 新建 `concerns/evolution/community.py` `CommunityDetector`
- label_propagation 算法（networkx）对 GraphStore 做社区聚类
- `CommunityNode` 存社区摘要（LLM 生成）
- 检索时社区级粒度（先社区匹配 → 再节点匹配）

### P2-Task 3：Saga 增量摘要

**借鉴**：graphiti `SagaNode` + `HasEpisodeEdge` + `NextEpisodeEdge`

**方案**：
- 新建 `concerns/evolution/saga.py` `SagaBuilder`
- 会话级增量摘要（每 N 条消息生成一个 Saga 节点）
- Saga 链表（NextEpisodeEdge 串联）
- 双 watermark（wall-clock + episode-time）

### P2-Task 4：三权重演化

**借鉴**：cognee `feedback_weight + importance_weight + frequency_weight`

**方案**：
- `SemanticFact` 增加 `feedback_weight`/`importance_weight`/`frequency_weight` 三字段
- 检索时加权：`final_score = relevance × (feedback + importance + frequency)`
- feedback_weight：用户反馈（thumb up/down）
- importance_weight：LLM 判定重要性
- frequency_weight：访问频次

---

## Phase 3：治理 + 多租户

> **优先级：中低**
> **目标**：从"API key 二态"升级到"RBAC + ACL + Dataset 隔离"

### P3-Task 1：RBAC 角色系统

**借鉴**：cognee `Principal + Role + ACL` + MemOS `4 级角色`

**方案**：
- 4 级角色：ROOT / ADMIN / USER / GUEST
- `septmuse_users` + `septmuse_roles` + `septmuse_user_roles` 表
- API key 绑定用户 + 角色
- 权限矩阵：read / write / delete / share / admin
- REST middleware 校验角色权限

### P3-Task 2：Provenance 溯源删除

**借鉴**：cognee `source_ref` + 3 种回滚维度

**方案**：
- `memories` 表增加 `source_ref` 字段（pipeline_run_id / source_id / dataset_id）
- `Memory.delete_by_source(source_ref)` 批量删除
- `Memory.rollback_by_pipeline(run_id)` 回滚整批
- `Memory.delete_by_dataset(dataset_id)` 数据集级删除

### P3-Task 3：Dataset 隔离

**借鉴**：cognee `Dataset` + MemOS `Cube`

**方案**：
- `septmuse_datasets` 表
- `memories.dataset_id` 字段
- 每个 dataset 独立向量索引
- 跨 dataset 共享需显式授权

---

## Phase 4：生态扩展

> **优先级：低**
> **目标**：从"Python 包"升级到"全栈生态"

### P4-Task 1：TypeScript SDK

**借鉴**：mem0 `mem0-ts` + letta `@letta-ai/letta-client`

**方案**：
- `sdk/typescript/` 目录
- 自动生成 REST client（从 OpenAPI schema）
- 支持 Node.js + 浏览器
- npm 发布 `@septmuse/sdk`

### P4-Task 2：Web UI

**借鉴**：mem0 `OpenMemory` (Next.js) + cognee `cognee-frontend`

**方案**：
- `web/` 目录
- Next.js + Tailwind
- 记忆浏览 / 搜索 / 编辑 / 删除
- 覆盖报告可视化
- 知识图谱可视化（d3.js）

### P4-Task 3：OpenAI/Anthropic 兼容端点

**借鉴**：letta `/openai/chat/completions`

**方案**：
- REST `POST /v1/chat/completions` 兼容 OpenAI 格式
- 自动注入记忆到 system prompt
- `POST /v1/messages` 兼容 Anthropic 格式
- drop-in 替换 OpenAI/Anthropic client

### P4-Task 4：多模态/参数化记忆（探索性）

**借鉴**：MemOS `act_mem` (KV cache) + `para_mem` (LoRA)

**方案**：
- `ActivationMemory` 已有 stub，补全 KV cache 注入
- `LoRAMemory` 已有 stub，补全 PEFT 微调
- 记忆不止存文本，还存进模型权重

---

## 3. 依赖关系

```
P0-Task 1 (HNSW) ──→ P0-Task 3 (并发检索) ──→ P0-Task 6 (缓存)
P0-Task 2 (中文分词) ──→ P0-Task 3 (并发检索)
P0-Task 4 (async) ──→ P1-Task 3 (后台异步)
P0-Task 5 (连接池) ──→ P0-Task 4 (async)

P1-Task 2 (结构化输出) ──→ P2-Task 1 (Dream 升级)
P1-Task 3 (后台异步) ──→ P2-Task 1 (Dream 升级)

P3-Task 1 (RBAC) ──→ P3-Task 3 (Dataset 隔离)
P3-Task 2 (Provenance) ──→ P3-Task 3 (Dataset 隔离)

P4 独立，无前置依赖
```

## 4. 推荐执行顺序

1. **P0-Task 1+2**（HNSW + 中文分词）— 检索性能 + 质量基础
2. **P0-Task 5**（连接池 + WAL）— 数据库基础
3. **P0-Task 3**（并发检索）— 性能叠加
4. **P0-Task 4**（async 全链路）— 并发基础
5. **P0-Task 6**（查询缓存）— 性能锦上添花
6. **P1-Task 2**（结构化输出）— LLM 基础
7. **P1-Task 1+3**（自编辑 + 后台异步）— LLM 深化
8. **P2**（演化深化）— 依赖 P1
9. **P3**（治理）— 独立
10. **P4**（生态）— 独立

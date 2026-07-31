# SeptMuse 整体开发计划：补齐差距 + 深化独创

> 日期：2026-07-23
> 前置文档：`docs/specs/opensource-gap-analysis.md`（差距分析报告）
> 目标：按优先级补齐与 6 个开源记忆系统的差距，同时深化 SeptMuse 独有优势
> 原则：每个 Phase 借鉴来源标注到具体文件/类，验收标准可量化

---

## 目录

1. [路线图总览](#1-路线图总览)
2. [Phase 0：实体抽取 + 知识图谱构建](#phase-0实体抽取--知识图谱构建)
3. [Phase 1：检索质量提升](#phase-1检索质量提升)
4. [Phase 2：时态建模 + 消息压缩](#phase-2时态建模--消息压缩)
5. [Phase 3：LLM 深度集成](#phase-3llm-深度集成)
6. [Phase 4：记忆演化深化](#phase-4记忆演化深化)
7. [Phase 5：运维治理 + 异步后台](#phase-5运维治理--异步后台)
8. [Phase 6：生态扩展](#phase-6生态扩展)
9. [依赖关系图](#依赖关系图)
10. [里程碑时间线](#里程碑时间线)

---

## 1. 路线图总览

| Phase | 名称 | 优先级 | 借鉴来源 | 核心目标 |
|-------|------|--------|----------|----------|
| P0 | 实体抽取 + 知识图谱构建 | 最高 | mem0 + cognee + graphiti | 补齐最大空白，从"链接"升级到"知识图谱" |
| P1 | 检索质量提升 | 高 | graphiti + mem0 | Reranker + Entity boost + BFS 图遍历 |
| P2 | 时态建模 + 消息压缩 | 高 | graphiti + letta | 双时态 KG + Summarizer |
| P3 | LLM 深度集成 | 中高 | mem0 + cognee + letta + ReMe | LLM 贯穿记忆全生命周期 |
| P4 | 记忆演化深化 | 中 | ReMe + MemOS + graphiti | Dream 升级 + 实体去重 + 冲突解决 |
| P5 | 运维治理 + 异步后台 | 中低 | cognee + letta + letta | RBAC + SleeptimeAgent + provenance |
| P6 | 生态扩展 | 低 | cognee + mem0 + letta | V2 API + Web UI + TS SDK |

---

## Phase 0：实体抽取 + 知识图谱构建

> **优先级：最高**
> **借鉴来源**：mem0 `entity_extraction.py` / cognee `cognify` 流水线 / graphiti 联合抽取
> **目标**：SeptMuse 从"Zettel 链接"升级到"知识图谱构建"，补齐最大空白

### P0-Task 1：实体抽取模块

**借鉴**：mem0 `opensource/mem0/mem0/utils/entity_extraction.py`

**实现**：
- 新建 `src/septmuse/concerns/extraction/entity.py`
- 用 spaCy NLP 抽取 4 类实体：PROPER（专有名词/NER）、QUOTED（引号文本）、TOPIC（名词短语）、IDENTIFIER（技术标识符）
- 实现 ~200 个泛化词黑名单（`_GENERIC_HEADS`/`_NON_SPECIFIC_ADJ`/`_GENERIC_CAPS`），过滤无意义实体
- 实现 span 去重冲突解决（`_resolve_candidates`）
- spaCy 不可用时返回空列表（优雅降级）
- 新增依赖：`spacy` + `en-core-web-sm` 模型（可选 extra `[ner]`）

**验收标准**：
- `extract_entities("Alice works at Google in London")` 返回 `[("PROPER", "Alice"), ("PROPER", "Google"), ("PROPER", "London")]`
- 泛化词（"the", "this", "person"）被过滤
- 无 spaCy 时不崩溃，返回空列表
- ≥10 个单元测试

### P0-Task 2：三元组 LLM 联合抽取

**借鉴**：graphiti `opensource/graphiti/graphiti_core/utils/maintenance/combined_extraction.py` `extract_nodes_and_edges`

**实现**：
- 新建 `src/septmuse/concerns/extraction/triplet.py`
- 单次 LLM 调用同时抽实体+边（三元组），优于分离抽取
- 孤儿节点丢弃（每实体至少有一条连接边，否则丢弃）
- 结构化输出：JSON schema 约束 LLM 输出 `{"entities": [...], "edges": [...]}`
- 复用已有 `FactExtractor` 的 LLM 抽象层
- 无 LLM 时 fallback 到 spaCy NER（P0-Task 1）

**验收标准**：
- `extract_triplets("Alice works at Google")` 返回 `[("Alice", "works_at", "Google")]`
- 孤儿实体被丢弃
- 无 LLM 时 fallback 到 spaCy NER
- ≥8 个单元测试

### P0-Task 3：cognify 知识图谱构建流水线

**借鉴**：cognee `opensource/cognee/api/v1/cognify/cognify.py` + `opensource/cognee/tasks/graph/extract_graph_from_data_v2.py`

**实现**：
- 新建 `src/septmuse/concerns/extraction/cognify.py`
- Pipeline DAG 流水线：`classify → chunk → extract_entities → extract_triplets → build_graph → summarize`
- 复用已有 `GraphStore`（SQLite/AGE/Neo4j）存储实体节点和关系边
- 复用已有 `ZettelLinker` 做增量链接
- 新建 `EntityNode` 和 `RelationEdge` schema（TypedMemoryStore 扩展）
- 支持 `run_in_background` 异步执行

**验收标准**：
- `m.cognify("Alice works at Google. Bob works at Google too.", user_id="u1")` 构建知识图谱：2 实体节点 + 2 关系边
- `m.search_entities("Google", user_id="u1")` 返回 Google 实体 + 关联实体
- 图遍历 `m.get_entity_neighbors("Google", user_id="u1")` 返回 Alice + Bob
- ≥15 个单元测试

### P0-Task 4：实体向量库（替代图数据库的可选方案）

**借鉴**：mem0 V3 "去图化" — 实体存独立 collection，用 `linked_memory_ids` 关联记忆

**实现**：
- `EntityStore`（独立 SQLite 表或独立 collection）
- `_upsert_entity`：精确文本匹配 → 语义匹配（score≥0.95）→ 新建
- `_remove_memory_from_entity_store`：删除记忆时清理实体引用
- 检索时 entity boost：从 query 抽取实体 → 实体库搜索 → 对匹配实体的 `linked_memory_ids` 加权

**验收标准**：
- 同一实体多次出现只存一次，`linked_memory_ids` 累积
- 删除记忆后实体引用清理
- ≥10 个单元测试

---

## Phase 1：检索质量提升

> **优先级：高**
> **借鉴来源**：graphiti 5 种重排器 / mem0 三信号融合 + reranker
> **目标**：检索质量从"双信号 RRF"升级到"三信号 + 多重排器"

### P1-Task 1：Reranker 框架

**借鉴**：graphiti `opensource/graphiti/graphiti_core/search/reranker/` + mem0 `opensource/mem0/mem0/reranker/`

**实现**：
- 新建 `src/septmuse/concerns/retrieval/reranker.py`
- 抽象基类 `Reranker`（ABC）
- 实现 4 种 reranker：
  1. `CrossEncoderReranker`：用 ONNX cross-encoder 模型重排（`modelscope` 下载）
  2. `MMRReranker`：最大边际相关性，去冗余（参数 `lambda_param`）
  3. `LLMReranker`：LLM 打分重排（复用已有 LLM 抽象）
  4. `NoopReranker`：透传（默认）
- `Memory.search(reranker="cross_encoder")` 可选启用
- `MemoryConfig` 新增 `reranker_backend` 字段

**验收标准**：
- `m.search("Python", user_id="u1", reranker="cross_encoder")` 返回重排结果
- MMR 去冗余：相似度 >0.9 的结果只保留一个
- ≥12 个单元测试

### P1-Task 2：Entity Boost 三信号融合

**借鉴**：mem0 `opensource/mem0/mem0/memory/main.py:1584-1769` `_search_vector_store` + `opensource/mem0/mem0/utils/scoring.py`

**实现**：
- 修改 `src/septmuse/storage/base.py` `hybrid_search` 方法
- 从三信号（semantic + BM25 + entity_boost）融合改为可选三信号
- 实体提升权重：`similarity × 0.5 × 1/(1+0.001×(n-1)²)`
- 自适应除数：`max_possible` 随激活信号数变化（1.0/2.0/2.5/1.5）
- threshold 门控语义分（低于 threshold 的候选即使 BM25/entity 高也排除）
- `explain=True` 返回 `score_details` 明细

**验收标准**：
- 三信号融合比双信号 RRF 召回率提升 ≥10%（基准测试）
- `explain=True` 返回 `{"semantic": 0.8, "bm25": 0.6, "entity_boost": 0.3, "combined": 0.57}`
- ≥10 个单元测试

### P1-Task 3：BFS 图遍历检索

**借鉴**：graphiti `opensource/graphiti/graphiti_core/search/search_utils/search_methods/bfs_search.py`

**实现**：
- 新建 `src/septmuse/concerns/retrieval/graph_search.py`
- `GraphSearcher` 类：从种子节点出发 BFS 遍历 `GraphStore`
- 参数：`max_depth`（默认 2）、`edge_filter`（关系类型过滤）
- 与向量/BM25 检索结果融合（RRF）
- 复用已有 `ZettelLinker.get_related_memories` 逻辑

**验收标准**：
- `m.search_graph(seed_memory_id="xxx", max_depth=2)` 返回 2 跳内邻居
- BFS 结果与向量结果 RRF 融合
- ≥8 个单元测试

### P1-Task 4：预置检索 Recipes

**借鉴**：graphiti `opensource/graphiti/graphiti_core/search/search_config_recipes.py`

**实现**：
- 新建 `src/septmuse/concerns/retrieval/recipes.py`
- 预置配置：
  - `HYBRID_RRF`：向量+BM25 RRF（当前默认）
  - `HYBRID_RRF_ENTITY`：三信号融合 + entity boost
  - `HYBRID_RRF_CROSS_ENCODER`：RRF + cross-encoder 重排
  - `HYBRID_RRF_MMR`：RRF + MMR 去冗余
  - `GRAPH_BFS`：纯图遍历
  - `PROGRESSIVE`：渐进三层
  - `FORGETTING`：遗忘曲线加权
- `m.search(query, recipe="HYBRID_RRF_CROSS_ENCODER")` 一键切换

**验收标准**：
- 7 种 recipe 均可正确执行
- recipe 参数可覆盖（`recipe="HYBRID_RRF", top_k=20`）
- ≥7 个单元测试

---

## Phase 2：时态建模 + 消息压缩

> **优先级：高**
> **借鉴来源**：graphiti 双时态 KG / letta Summarizer
> **目标**：补齐时态能力 + 长对话消息压缩

### P2-Task 1：双时态建模

**借鉴**：graphiti `opensource/graphiti/graphiti_core/edges.py` EntityEdge + `opensource/graphiti/graphiti_core/utils/maintenance/edge_operations.py` `resolve_edge_contradictions`

**实现**：
- 修改 `src/septmuse/schemas/` 添加双时态字段：
  - `valid_at`：事实开始为真的时间
  - `invalid_at`：停止为真的时间（被新事实失效时设置）
  - `expired_at`：系统标记失效的 wall-clock
  - `reference_time`：已有（episode 参考时间）
  - `created_at`：已有（写入时间）
- `SQLiteCompositeStore` ALTER TABLE 迁移添加 `valid_at`/`invalid_at`/`expired_at` 列
- 自动事实失效：新事实与旧事实冲突时，旧事实 `invalid_at = 新事实.valid_at`，`expired_at = utc_now()`
- 保留完整历史（失效不删除）

**验收标准**：
- `m.add("Alice works at Google", valid_at="2024-01-01")` 设置 valid_at
- `m.add("Alice works at Apple", valid_at="2025-01-01")` 自动失效旧记忆
- `m.search_at("2024-06-01", query="Alice works", user_id="u1")` 返回 Google（当时为真）
- `m.search_at("2025-06-01", query="Alice works", user_id="u1")` 返回 Apple
- ≥15 个单元测试

### P2-Task 2：时态区间查询

**借鉴**：graphiti `opensource/graphiti/graphiti_core/search/search_filters.py` + cognee `opensource/cognee/modules/retrieval/temporal_retriever.py`

**实现**：
- 新建 `src/septmuse/concerns/retrieval/temporal.py` `TemporalRetriever`
- `search_at(reference_time, query, user_id)`：查询某时刻为真的事实
- `search_interval(start, end, query, user_id)`：查询时间区间内的事实
- LLM 从自然语言查询抽取时间区间（"上周"、"2024年6月"→绝对日期）
- 时间过滤器：`valid_at <= reference_time AND (invalid_at IS NULL OR invalid_at > reference_time)`

**验收标准**：
- `m.search_at("2024-06-01", query="Alice", user_id="u1")` 正确过滤
- LLM 从"上周Alice在做什么"抽取时间区间
- 无时间信息时回退到普通检索
- ≥12 个单元测试

### P2-Task 3：消息压缩 Summarizer

**借鉴**：letta `opensource/letta/letta/services/summarizer/summarizer.py`

**实现**：
- 新建 `src/septmuse/concerns/compression/summarizer.py`
- 两种模式：
  1. `STATIC_BUFFER`：固定缓冲区，超限驱逐旧消息 + LLM 递归摘要
  2. `PARTIAL_EVICT`：驱逐 30% 旧消息 + LLM 生成摘要插入
- `SummaryMessage` schema：摘要文本 + 原始消息 ID 列表
- `Memory.compress(agent_id, mode="static", buffer_size=20)` 触发压缩
- 压缩后的摘要存入 `TypedMemoryStore`（EpisodicEvent, event_type="summary"）

**验收标准**：
- 50 条消息压缩到 20 条 + 1 条摘要
- 摘要保留关键信息（测试用 LLM mock 验证）
- ≥10 个单元测试

---

## Phase 3：LLM 深度集成

> **优先级：中高**
> **借鉴来源**：mem0 V3 单次抽取 / cognee session_distillation / letta self-editing / ReMe Dream LLM 决策
> **目标**：LLM 从"可选 FactExtractor"升级为"贯穿记忆全生命周期"

### P3-Task 1：LLM Provider 实现

**借鉴**：mem0 `opensource/mem0/mem0/llms/` 24 种 provider

**实现**：
- 实现 `src/septmuse/providers/llms/` 下具体 provider：
  - `openai.py`：OpenAI GPT（`openai` 库）
  - `anthropic.py`：Anthropic Claude（`anthropic` 库）
  - `ollama.py`：Ollama 本地（`ollama` 库，零配置）
  - `dashscope.py`：DashScope Qwen（`dashscope` 库）
- 每个 provider 实现 `LLM` ABC 的 `chat`/`complete` 方法
- 支持 `structured_output`（JSON schema 约束输出）
- 延迟 import（仅使用时加载）

**验收标准**：
- 4 个 provider 各 ≥5 个测试（mock 验证）
- `SEPTMUSE_LLM=openai` + `OPENAI_API_KEY` 环境变量配置
- `SEPTMUSE_LLM=ollama` 本地零配置可用
- ≥20 个单元测试

### P3-Task 2：单次 LLM 事实抽取（V3 模式）

**借鉴**：mem0 `opensource/mem0/mem0/configs/prompts.py` `ADDITIVE_EXTRACTION_PROMPT`

**实现**：
- 重构 `src/septmuse/content_types/semantic/extract.py` FactExtractor
- 单次 LLM 调用抽取事实（告别多轮 agentic loop）
- 实现 `ADDITIVE_EXTRACTION_PROMPT`（含 9 个 few-shot 示例）
- 输出 `linked_memory_ids` 跨记忆链接
- `infer=True` 默认开启（对齐 mem0/cognee/graphiti/MemOS/ReMe）
- verbatim 模式保留为 `infer=False` opt-out

**验收标准**：
- `m.add("I love Python and work at Google", user_id="u1", infer=True)` 抽取 2 条事实
- 抽取的事实质量 ≥ 直接存原文的检索效果
- ≥12 个单元测试

### P3-Task 3：冲突解决 + 实体去重

**借鉴**：graphiti `opensource/graphiti/graphiti_core/utils/maintenance/edge_operations.py` `resolve_extracted_edge` + `opensource/graphiti/graphiti_core/utils/maintenance/node_operations.py` 三段式去重

**实现**：
- 新建 `src/septmuse/concerns/evolution/conflict.py`
- 实体去重三段式：
  1. 精确归一化名匹配（恒跑）
  2. 模糊 MinHash/LSH（Jaccard ≥0.9，熵阈值门控）
  3. LLM 兜底（未解析项送 LLM 选 `duplicate_candidate_id`）
- 边冲突解决：LLM 判定 `duplicate_facts` + `contradicted_facts`
- 双时态失效：旧边 `invalid_at = 新边.valid_at`（复用 P2-Task 1）
- `_promote_resolved_node` type 升级（重复节点带更具体 type 时升级 canonical 节点）

**验收标准**：
- 同一实体不同写法（"Google"/"google"/"Google Inc"）自动合并
- 矛盾事实自动失效旧事实
- ≥15 个单元测试

### P3-Task 4：Session 蒸馏两阶段 LLM

**借鉴**：cognee `opensource/cognee/modules/session_distillation/distill.py`（curator → writer/rejecter）

**实现**：
- 重构 `src/septmuse/concerns/evolution/reflect.py` SessionReflector
- 两阶段：
  1. **curator**：LLM 批次提取课程（lessons）from 历史记忆
  2. **writer/rejecter**：LLM 决策每条课程写入/拒绝 + 实体锚定 + 新颖性搜索
- 每条课程独立文档存入 `TypedMemoryStore`（ProceduralRule）
- `m.reflect(user_id="u1", limit=50)` 触发蒸馏

**验收标准**：
- 从 50 条记忆蒸馏出 ≥3 条可执行规则
- 拒绝率合理（≥20% 的候选被拒）
- ≥10 个单元测试

### P3-Task 5：LLM 自编辑记忆

**借鉴**：letta `opensource/letta/letta/services/tool_executor/core_tool_executor.py` `memory_apply_patch`

**实现**：
- 新建 `src/septmuse/concerns/evolution/self_edit.py`
- `memory_apply_patch(agent_id, diff)`：统一 diff 多块编辑记忆
  - `*** Add Block` / `*** Delete Block` / `*** Update Block` / `*** Move Block`
- `memory_rethink(memory_id, new_content)`：LLM 主动重写
- `core_memory_append`/`core_memory_replace` 已有（对齐 letta）
- MCP 工具暴露：LLM agent 可通过 MCP 自主编辑记忆

**验收标准**：
- `m.memory_apply_patch(agent_id="a1", diff="*** Update Block\nhuman\n- old\n+ new")` 正确更新
- MCP `memory_apply_patch` 工具可调用
- ≥10 个单元测试

---

## Phase 4：记忆演化深化

> **优先级：中**
> **借鉴来源**：ReMe auto_dream / MemOS Dream 插件 / graphiti 社区检测 + Saga
> **目标**：Dream 从"仅建链接"升级到"extract→integrate→topics→proactive"

### P4-Task 1：Dream 升级（四阶段）

**借鉴**：ReMe `opensource/ReMe/reme/steps/evolve/dream/`（extract → integrate → topics → proactive）+ MemOS `opensource/MemOS/memos/dream/`

**实现**：
- 重构 `src/septmuse/concerns/evolution/dream.py` DreamIntegrator
- 四阶段：
  1. **dream_extract**：扫描近期记忆变更，LLM 全局抽取跨记忆 memory units + topics
  2. **dream_integrate**：每 unit 调 LLM 决策 CREATE/CORROBORATE/REFINE/CORRECT 写入 digest
  3. **dream_topics**：选 top-N 兴趣主题，7 天去重，存 `interests` 表
  4. **dream_proactive**：读 interests，暴露给 host agent 主动提及
- MotiveType 驱动（newness/frequency/conflict/feedback/fragmentation）
- `DreamMemoryLifecycle`：last_hit_at/hit_count/usefulness_score 衰减归档

**验收标准**：
- Dream 后产生新的整合记忆 + 兴趣主题
- 4 阶段均有独立测试
- ≥15 个单元测试

### P4-Task 2：社区检测 + 区域摘要

**借鉴**：graphiti `opensource/graphiti/graphiti_core/utils/maintenance/community_operations.py`

**实现**：
- 新建 `src/septmuse/concerns/evolution/community.py`
- `label_propagation` 算法做社区聚类
- `CommunityNode` schema：社区摘要 + 成员列表
- `build_communities` / `update_community` / `remove_communities`
- 检索时可按社区粒度返回

**验收标准**：
- 50 条记忆聚类出 ≥3 个社区
- 每个社区有 LLM 摘要
- ≥8 个单元测试

### P4-Task 3：Saga 增量摘要

**借鉴**：graphiti `opensource/graphiti/graphiti_core/nodes.py` SagaNode + `HasEpisodeEdge` + `NextEpisodeEdge`

**实现**：
- 新建 `src/septmuse/concerns/evolution/saga.py`
- `SagaNode` schema：summary + first/last_episode_uuid + last_summarized_at + last_summarized_episode_valid_at
- 增量摘要：用 watermark 增量摘要会话 episode 链
- 双 watermark：wall-clock（last_summarized_at）vs episode-time（last_summarized_episode_valid_at）

**验收标准**：
- 10 个 episode 增量摘要为 1 个 Saga
- 新增 episode 只摘要增量部分
- ≥8 个单元测试

### P4-Task 4：三权重演化

**借鉴**：cognee `opensource/cognee/infrastructure/engine/models/DataPoint.py`（feedback_weight + importance_weight + frequency_weight）

**实现**：
- 修改 `TypedMemoryStore` schema 添加三权重字段
- `feedback_weight`：用户反馈调整
- `importance_weight`：LLM 判定重要度
- `frequency_weight`：访问频率统计
- 检索时三权重综合排序

**验收标准**：
- 三权重可独立调整
- 检索结果按综合权重排序
- ≥8 个单元测试

---

## Phase 5：运维治理 + 异步后台

> **优先级：中低**
> **借鉴来源**：cognee RBAC + provenance / letta SleeptimeAgent / ReMe cron
> **目标**：补齐企业级运维能力

### P5-Task 1：RBAC 角色系统

**借鉴**：cognee `opensource/cognee/modules/users/models/`（Principal + Role + ACL）+ MemOS `opensource/MemOS/memos/mem_user/user_manager.py`（4 级角色）

**实现**：
- 新建 `src/septmuse/concerns/governance/rbac.py`
- `Principal` ABC + `User`/`Tenant` 子类
- `Role` + `UserRole` + `RoleDefaultPermissions`
- `ACL` 表：principal × permission × scope（read/write/delete/share）
- 4 级角色：ROOT/ADMIN/USER/GUEST
- REST API 端点：`POST /users`, `GET /users/{id}/roles`, `PUT /acl`

**验收标准**：
- GUEST 只读，USER 读写，ADMIN 管理用户
- ACL 表可细粒度授权
- ≥15 个单元测试

### P5-Task 2：Provenance 溯源删除

**借鉴**：cognee `opensource/cognee/infrastructure/databases/unified/provenance_delete_planner.py`

**实现**：
- `source_ref` 系统：source_ref_key/pipeline_run_id/dataset_id
- 三种回滚：`rollback_by_pipeline_run_id` / `delete_by_source_ref` / `delete_by_dataset_id`
- `provenance_delete_planner` 协调 graph+vector 清理

**验收标准**：
- 按 pipeline_run_id 回滚删除该次 cognify 产生的所有节点/边
- ≥8 个单元测试

### P5-Task 3：SleeptimeAgent 后台异步

**借鉴**：letta `opensource/letta/letta/services/` SleeptimeAgent

**实现**：
- 新建 `src/septmuse/concerns/async_/sleeptime.py`
- 主 agent 空闲时，后台 agent 异步处理记忆（抽取/整理/Dream/reflect）
- 用 `asyncio` 或 `concurrent.futures` 实现
- `Memory.enable_sleeptime(agent_id)` 启用

**验收标准**：
- 后台异步执行 cognify/dream/reflect 不阻塞主流程
- ≥8 个单元测试

### P5-Task 4：Cron 定时整合

**借鉴**：ReMe `opensource/ReMe/reme/config/default.yaml` auto_dream cron `0 23 * * *`

**实现**：
- 新建 `src/septmuse/concerns/async_/scheduler.py`
- 用 `apscheduler` 或 `croniter` 实现定时任务
- 默认配置：每天 23:00 执行 Dream 整合
- `Memory.enable_scheduler(cron="0 23 * * *", task="dream")` 配置

**验收标准**：
- cron 表达式正确解析
- 定时任务可启用/禁用
- ≥5 个单元测试

### P5-Task 5：过期机制

**借鉴**：mem0 `opensource/mem0/mem0/memory/main.py` `expiration_date` + graphiti `expired_at`

**实现**：
- `Memory.add(..., expiration_date="2025-12-31")` 设置过期
- `search`/`get_all` 默认过滤过期记忆
- `show_expired=True` 可查看过期记忆

**验收标准**：
- 过期记忆不在 search 结果中
- `show_expired=True` 可查看
- ≥5 个单元测试

---

## Phase 6：生态扩展

> **优先级：低**
> **借鉴来源**：cognee V2 API / mem0 OpenMemory / letta OpenAI 兼容
> **目标**：完善生态，提升开发者体验

### P6-Task 1：V2 记忆导向 API

**借鉴**：cognee `opensource/cognee/cognee/__init__.py` `remember`/`recall`/`improve`/`forget`

**实现**：
- 新建 `src/septmuse/v2/` 模块
- 4 大面向用户操作：
  - `remember(messages, user_id)` → 摄入 + cognify
  - `recall(query, user_id)` → 混合检索 + 重排
  - `improve(user_id)` → reflect + dream + distill
  - `forget(memory_id)` → 软删除 + 实体清理
- 同时保留 V1 原子 API（add/search/get/update/delete/...）

**验收标准**：
- V2 API 4 个操作均可正确执行
- V1 API 不受影响
- ≥12 个单元测试

### P6-Task 2：Web UI 仪表盘

**借鉴**：mem0 `opensource/mem0/openmemory/` OpenMemory Next.js + cognee `opensource/cognee/cognee-frontend/`

**实现**：
- 新建 `frontend/` 目录（Next.js + shadcn/ui）
- 页面：
  - 记忆列表/搜索
  - 知识图谱可视化（d3.js 或 react-flow）
  - 记忆演化时间线
  - 元认知覆盖报告
  - 用户/角色管理
- 连接 REST API

**验收标准**：
- 5 个页面均可正确渲染
- 知识图谱可视化可交互（缩放/拖拽）
- ≥5 个 e2e 测试

### P6-Task 3：TypeScript SDK

**借鉴**：mem0 `opensource/mem0/mem0-ts/` + letta `@letta-ai/letta-client`

**实现**：
- 新建 `ts-sdk/` 目录
- `@septmuse/sdk` npm 包
- 封装 REST API 调用
- 类型定义对齐 Python schema

**验收标准**：
- TypeScript SDK 可连接 REST API
- 类型安全
- ≥10 个测试

### P6-Task 4：OpenAI 兼容端点

**借鉴**：letta `opensource/letta/letta/server/rest_api/routers/openai/chat_completions/`

**实现**：
- REST API 新增 `/v1/chat/completions` 端点
- 自动注入记忆上下文到 system prompt
- 现有 OpenAI 客户端可直接 drop-in 替换

**验收标准**：
- OpenAI Python 库可直接连接
- 记忆自动注入
- ≥5 个 e2e 测试

---

## 依赖关系图

```
P0-Task 1 (实体抽取)
  ├── P0-Task 2 (三元组 LLM 抽取) ← 依赖 P0-Task 1
  ├── P0-Task 3 (cognify 流水线) ← 依赖 P0-Task 1+2
  └── P0-Task 4 (实体向量库) ← 依赖 P0-Task 1
        │
        ▼
P1-Task 2 (Entity boost) ← 依赖 P0-Task 4
P1-Task 1 (Reranker) ← 独立
P1-Task 3 (BFS 图遍历) ← 依赖 P0-Task 3 (知识图谱)
P1-Task 4 (检索 Recipes) ← 依赖 P1-Task 1+2+3
        │
        ▼
P2-Task 1 (双时态建模) ← 独立
P2-Task 2 (时态查询) ← 依赖 P2-Task 1
P2-Task 3 (消息压缩) ← 独立
        │
        ▼
P3-Task 1 (LLM Provider) ← 独立（但 P3-Task 2~5 都依赖它）
P3-Task 2 (单次抽取) ← 依赖 P3-Task 1
P3-Task 3 (冲突解决) ← 依赖 P0-Task 2 + P2-Task 1
P3-Task 4 (蒸馏课程) ← 依赖 P3-Task 1
P3-Task 5 (自编辑) ← 依赖 P3-Task 1
        │
        ▼
P4-Task 1 (Dream 升级) ← 依赖 P3-Task 1+2
P4-Task 2 (社区检测) ← 依赖 P0-Task 3
P4-Task 3 (Saga 摘要) ← 独立
P4-Task 4 (三权重) ← 独立
        │
        ▼
P5-Task 1 (RBAC) ← 独立
P5-Task 2 (Provenance) ← 依赖 P0-Task 3
P5-Task 3 (SleeptimeAgent) ← 依赖 P3/P4
P5-Task 4 (Cron) ← 依赖 P4-Task 1
P5-Task 5 (过期) ← 独立
        │
        ▼
P6-Task 1 (V2 API) ← 依赖 P0~P3
P6-Task 2 (Web UI) ← 依赖 P6-Task 1
P6-Task 3 (TS SDK) ← 独立
P6-Task 4 (OpenAI 兼容) ← 依赖 P6-Task 1
```

---

## 里程碑时间线

| 里程碑 | 完成 Phase | 交付物 | 测试基线 |
|--------|-----------|--------|----------|
| M0 | P0 | 实体抽取 + cognify 流水线 + 实体向量库 | 699 + ~43 = ~742 passed |
| M1 | P1 | Reranker + Entity boost + BFS + Recipes | ~742 + ~37 = ~779 passed |
| M2 | P2 | 双时态建模 + 时态查询 + 消息压缩 | ~779 + ~37 = ~816 passed |
| M3 | P3 | LLM Provider + 单次抽取 + 冲突解决 + 蒸馏 + 自编辑 | ~816 + ~57 = ~873 passed |
| M4 | P4 | Dream 升级 + 社区检测 + Saga + 三权重 | ~873 + ~39 = ~912 passed |
| M5 | P5 | RBAC + Provenance + SleeptimeAgent + Cron + 过期 | ~912 + ~41 = ~953 passed |
| M6 | P6 | V2 API + Web UI + TS SDK + OpenAI 兼容 | ~953 + ~32 = ~985 passed |

---

## 借鉴来源索引

| 借鉴来源 | 借鉴要点 | 用于 Phase |
|----------|----------|------------|
| **mem0** | spaCy NER + 工程化词表 + 实体向量库 + 三信号融合 + ADDITIVE_EXTRACTION_PROMPT + V3 ADD-only | P0, P1, P3 |
| **graphiti** | 双时态 KG + 联合抽取 + 三段式去重 + 边冲突解决 + 5 种 reranker + BFS 图遍历 + 社区检测 + Saga | P0, P1, P2, P3, P4 |
| **cognee** | cognify 流水线 + Pipeline DAG + session_distillation + truth_subspace + RBAC/ACL + provenance + V2 API | P0, P1, P3, P5, P6 |
| **letta** | Summarizer + SleeptimeAgent + memory_apply_patch + ContextWindowCalculator + OpenAI 兼容 | P2, P3, P5, P6 |
| **MemOS** | Dream 插件（MotiveType + DreamAction + Lifecycle）+ MemCube + 参数化记忆 + MemScheduler | P4, P5 |
| **ReMe** | auto_dream 四阶段（extract→integrate→topics→proactive）+ progressive retrieval + cron + 文件原生 | P3, P4, P5 |

---

## 验证策略

每个 Phase 完成后执行：

1. **lint**：`ruff check src/ tests/ examples/` + `ruff format --check src/ tests/ examples/`
2. **unit test**：`PYTHONPATH=src pytest tests/unit/ -q`
3. **e2e test**：`PYTHONPATH=src pytest tests/e2e/ -q`
4. **集成测试**（可选 extras）：`PYTHONPATH=src pytest tests/unit/ -q -m integration`
5. **CHANGELOG 更新**：记录 Added/Changed/Fixed
6. **AGENTS.md 更新**：更新环境变量表、测试怪癖、架构入口

**完成声明前必须有验证证据**（验证前置原则）。

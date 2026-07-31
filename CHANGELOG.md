# Changelog

All notable changes to SeptMuse will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Reranker 框架: NoopReranker/MMRReranker/CrossEncoderReranker/LLMReranker (原因: 补齐检索质量短板; 影响: 检索模块)
- Entity boost 三信号融合: 向量+BM25+entity boost (原因: 对齐 mem0 三信号; 影响: HybridRetriever)
- explain=True score_details: 返回 vector/bm25/entity_boost/combined 明细 (原因: 可观测性; 影响: HybridRetriever)
- SEPTMUSE_RERANKER 环境变量: noop/mmr/cross_encoder/llm (原因: 零配置; 影响: 全局配置)
- CLI --reranker / REST reranker / MCP reranker 参数 (原因: API 一致性; 影响: CLI/REST/MCP)
- pip install septmuse[reranker] extra: onnxruntime>=1.16 (原因: CrossEncoder 可选; 影响: pyproject.toml)
- 双时态建模: valid_at/invalid_at/expired_at 三列 + 手动失效 (原因: 补齐时态能力; 影响: 存储层)
- Memory.search_at(reference_time, query, user_id): 时态查询 (原因: 查询某时刻为真的事实; 影响: Memory facade)
- Memory.invalidate(memory_id): 手动标记事实不再为真 (原因: 矛盾检测降级为手动; 影响: Memory facade)
- Memory.add(valid_at=): 写入时设置事实有效期 (原因: 支持时态建模; 影响: Memory facade)
- CLI add --valid-at / invalidate 命令 (原因: API 一致性; 影响: CLI)
- REST POST /memories/{id}/invalidate (原因: API 一致性; 影响: REST)
- MCP invalidate_memory 工具 (原因: API 一致性; 影响: MCP)
- LLM Provider 框架: OllamaLLM/AnthropicLLM/DashScopeLLM + _resolve_llm 工厂 (原因: 解锁 LLM infer 模式; 影响: providers/llms/)
- SEPTMUSE_LLM_MODEL 环境变量: 覆盖 provider 默认模型 (原因: 灵活配置; 影响: MemoryConfig)
- Memory.__init__ 自动创建 LLM: llm_provider 配置时零配置可用 (原因: 零配置; 影响: Memory facade)
- pip install septmuse[dashscope] extra: dashscope>=1.17 (原因: DashScope 可选; 影响: pyproject.toml)
- 三元组 LLM 联合抽取: TripletExtractor + extract_triplets (原因: 补齐知识图谱构建能力; 影响: concerns/extraction/)
- 孤儿实体丢弃: 没有边连接的实体被过滤 (原因: 对齐 graphiti; 影响: TripletExtractor)
- 无 LLM fallback: EntityExtractor + 相邻实体规则生成三元组 (原因: 零配置可用; 影响: TripletExtractor)
- cognify 知识图谱构建流水线: CognifyPipeline + entity_relations 表 (原因: 补齐知识图谱构建; 影响: concerns/extraction/)
- Memory.cognify(text, user_id): 一键构建知识图谱 (存记忆→抽三元组→存实体/关系→建链接) (原因: API 一致性; 影响: Memory facade)
- Memory.get_entity_relations(entity_name, user_id): 实体间关系遍历 (双向) (原因: 图遍历检索; 影响: Memory facade)
- ZettelLinker 集成: cognify 后自动建记忆间向量链接 (原因: 复用已有链接生长; 影响: CognifyPipeline)
- BFS 图遍历检索: GraphSearcher + rrf_fuse (原因: 补齐图遍历检索; 影响: concerns/retrieval/)
- Memory.search_graph(seed_memory_id, max_depth): BFS 图遍历检索 (原因: API 一致性; 影响: Memory facade)
- Memory.search_graph_fused(query, user_id, seed_memory_id): BFS + 向量 RRF 融合 (原因: 融合检索; 影响: Memory facade)
- 预置检索 Recipes: 7 种配置一键切换 (HYBRID_RRF/HYBRID_RRF_ENTITY/HYBRID_RRF_CROSS_ENCODER/HYBRID_RRF_MMR/GRAPH_BFS/PROGRESSIVE/FORGETTING) (原因: 降低检索配置复杂度; 影响: concerns/retrieval/)
- Memory.search(recipe=): recipe 参数覆盖 hybrid/reranker/explain (原因: API 一致性; 影响: Memory facade)
- 时态区间查询: TemporalRetriever + search_interval + search_natural (原因: 补齐时态检索能力; 影响: concerns/retrieval/)
- store.get_temporal_interval(start, end, user_id): 存储层区间过滤 (原因: 支持区间查询; 影响: storage/)
- Memory.search_interval(start, end, query): 时间区间检索 (原因: API 一致性; 影响: Memory facade)
- Memory.search_natural(query): LLM 自然语言时间抽取 + 时态查询 (原因: 降低时态查询使用门槛; 影响: Memory facade)
- 无时间信息回退普通检索 (原因: 零配置可用; 影响: TemporalRetriever)
- 消息压缩 Summarizer: Summarizer + STATIC_BUFFER/PARTIAL_EVICT 模式 (原因: 补齐长对话压缩能力; 影响: concerns/compression/)
- Memory.compress(user_id, mode, buffer_size): 一键压缩消息 (原因: API 一致性; 影响: Memory facade)
- LLM 递归摘要 + 无 LLM 拼接降级 (原因: 零配置可用; 影响: Summarizer)
- 摘要存入 TypedMemoryStore EpisodicEvent event_type=summary (原因: 复用已有 schema; 影响: typed_store)
- ADDITIVE_EXTRACTION_PROMPT: 含 9 个 few-shot 的 V3 事实抽取提示 (原因: 对齐 mem0 V3; 影响: prompts/extract.py)
- FactExtractor 默认用 ADDITIVE_EXTRACTION_PROMPT (原因: 提升抽取质量; 影响: content_types/semantic/extract.py)
- linked_memory_ids: extract_and_store 输出跨记忆链接 (原因: 对齐 mem0 V3; 影响: FactExtractor)
- use_additive_prompt 参数: 可降级到 FACT_EXTRACTION_PROMPT (原因: 向后兼容; 影响: FactExtractor)
- 冲突解决 + 实体去重: ConflictResolver (原因: 补齐矛盾检测能力; 影响: concerns/evolution/)
- 矛盾事实检测: 相同 (subject, predicate) 不同 object → 自动失效旧事实 (原因: 对齐 graphiti; 影响: conflict.py)
- 实体去重三段式: 精确归一化 + 模糊相似度 (difflib) + LLM 兜底 (原因: 对齐 graphiti 三段式; 影响: conflict.py)
- TypedMemoryStore.soft_delete_fact: 软删除 SemanticFact (原因: 冲突解决需要; 影响: typed_store.py)
- Memory.resolve_conflicts + Memory.deduplicate_entities (原因: API 一致性; 影响: Memory facade)
- Session 蒸馏两阶段 LLM: curator (批次提取课程) + writer/rejecter (新颖性搜索 + LLM 判定) (原因: 对齐 cognee distill; 影响: concerns/evolution/reflect.py)
- 新颖性搜索: 检查已有规则避免重复写入 (原因: 规则去重; 影响: SessionReflector._accept_lesson)
- LLM writer/rejecter: 默认接受, 明确 reject 才拒绝 (原因: 向后兼容; 影响: SessionReflector._llm_accept)

### Added — HybridRetriever Entity Boost 三信号融合 + explain

- **HybridResult.entity_boost 字段** (`concerns/retrieval/hybrid.py`)：新增 `entity_boost: float = 0.0` 字段（位于 bm25_score 之后），默认 0.0 保证向后兼容（无 entity_store 时退化为双信号）。
- **HybridRetriever 三信号融合**：`__init__` 新增 `entity_extractor` / `entity_store` 可选参数（TYPE_CHECKING 守卫避免循环 import）。两者任一缺失时降级为双信号并记录警告；两者齐全时启用第三信号：抽取 query 实体 → EntityStore.search → 按 `boost = 0.5 / (1 + 0.001*(n-1)^2)` 衰减加权（n 为 linked_memory_ids 数，借鉴 mem0 _search_vector_store scoring），加性融合进 RRF fused score。
- **explain 参数**：`search()` 新增 `explain: bool = False`。为 True 时在每条结果 metadata 注入 `score_details`（含 vector/bm25/entity_boost/combined 四项），便于可解释性调试；False 时不变更 metadata。
- **LLMReranker 字段保留** (`concerns/retrieval/reranker.py`)：构造 HybridResult 时补传 `entity_boost=r.entity_boost`（Task 4 遗留），保证重排后第三信号字段不丢失。
- 新增 `tests/unit/test_hybrid_entity_boost.py`（8 测试：向后兼容降级 ×2、entity boost 提升/衰减/空库 ×3、explain 详情/默认关闭/含 boost ×3），全量 790 passed / 36 skipped。

### Added — CrossEncoderReranker + LLMReranker (重排器扩展)

- **CrossEncoderReranker** (`concerns/retrieval/reranker.py`)：ONNX cross-encoder 重排器（借鉴 graphiti BGERerankerClient + mem0 TS CrossEncoderReranker）。延迟 import onnxruntime，不可用时降级为 Noop + 日志警告；目标模型 BAAI/bge-reranker-v2-m3 ONNX 量化版（ModelScope 下载，实际推理待 P3/P4 补）。
- **LLMReranker** (`concerns/retrieval/reranker.py`)：LLM 逐条打分重排器（借鉴 mem0 LLMReranker）。`LLM.complete()` 打分 0-1，`_extract_score` 正则提取数字并 clamp [0,1]，无数字返回 0.5，无 LLM 实例抛 ValueError；构造新 HybridResult 保留原字段、更新 score 后按分数降序排序。
- **`_resolve_reranker` 工厂扩展**：新增 `"cross_encoder"`（传 `model_cache_dir`）与 `"llm"`（传 `llm`）两个 backend。
- **HybridResult 运行时 import**：`reranker.py` 将 `HybridResult` 从 `TYPE_CHECKING` 块移至运行时 import（LLMReranker 需运行时构造 HybridResult；`hybrid.py` 不依赖 `reranker`，无循环依赖）。
- 新增 `tests/unit/test_reranker.py::TestCrossEncoderReranker`（3 测试：onnxruntime 缺失降级、空输入、工厂解析）+ `TestLLMReranker`（8 测试：LLM 打分、无 LLM 抛错、分数 clamp 高/低、无数字默认 0.5、空输入、字段保留、工厂解析），全量 782 passed / 36 skipped。

### Added — MMRReranker (最大边际相关性 reranker)

- **MMRReranker** (`concerns/retrieval/reranker.py:71`)：贪心迭代 MMR 选择，`mmr = lambda * sim(query, doc) - (1-lambda) * max(sim(doc, selected))`，相似度 >0.9 的结果去冗余（借鉴 graphiti maximal_marginal_relevance）。保留原始 HybridResult 字段，更新 score 为 query-doc 余弦相似度。
- **`_resolve_reranker` 工厂扩展**：新增 `"mmr"` backend，需传 embedder 参数，默认 lambda=0.7（`reranker.py:153`）。
- 新增 `tests/unit/test_reranker.py::TestMMRReranker`（5 测试：去重、lambda 相关性、空输入、top_k 截断、字段保留），全量 771 passed / 36 skipped。

### Added — 实体抽取 + 实体向量库 (P0)

- **EntityExtractor** (`concerns/extraction/entity.py`)：纯 Python regex + 词表后端（默认，零配置），4 类实体（PROPER/QUOTED/TOPIC/IDENTIFIER）+ ~120 泛化词黑名单 + span 去重冲突解决。可选 spaCy 后端（`pip install septmuse[ner]`）。
- **EntityStore** (`storage/entity_store.py`)：独立 SQLite 表 `septmuse_entities`，upsert（精确匹配→语义匹配→新建）+ search + list + get_linked_memories + remove_memory_from_entities。借鉴 mem0 V3 去图化设计。
- **Memory facade 集成**：`add(auto_extract_entities=True)` 自动抽取实体，`delete()` 清理实体引用，新增 5 个方法（extract_entities/add_entity/search_entities/get_entity_neighbors/list_entities）。
- **MemoryConfig 新字段**：`entity_extractor_backend`（regex/spacy/none）。
- **环境变量**：`SEPTMUSE_ENTITY_EXTRACTOR`（regex/spacy/none）。
- **CLI 2 命令**：`septmuse entities <query>` / `septmuse entity-list`。
- **REST 2 端点**：`GET /entities` / `GET /entities/list`。
- **MCP 2 工具**：`search_entities` / `list_entities`。
- 新增 `tests/unit/test_entity_extractor.py`（19 测试 + 2 skip）、`test_entity_store.py`（16 测试）、`test_memory.py` 扩展（10 测试）、`tests/e2e/test_entity_e2e.py`（3 测试）。

### Added — 开源差距分析 + 整体开发计划

- **差距分析报告** (`docs/specs/opensource-gap-analysis.md`)：SeptMuse vs 6 个开源记忆系统（mem0/letta/cognee/graphiti/MemOS/ReMe）按 12 维度逐项对比，识别 8 个独有优势和 17 个短板，每个短板标注借鉴来源和借鉴要点。
- **整体开发计划** (`docs/plans/development-roadmap.md`)：7 个 Phase（P0-P6）共 28 个 Task 的详细实施计划，每个 Task 标注借鉴来源（文件/类级）、验收标准、依赖关系，含里程碑时间线和借鉴来源索引。

### Added — 嵌入升级 (P0-P2)

- **Memory.search 默认走 hybrid**：BM25 + 向量 RRF 融合检索成为默认路径，`hybrid=False` 可切回纯向量。HashEmbedder 质量差时 BM25 兜底关键词匹配（`orchestration/memory.py:184`）。
- **OnnxEmbedder** (`providers/embedders/onnx.py`)：onnxruntime + tokenizers，无 torch 依赖，模型从 ModelScope 下载量化 ONNX 缓存到 `~/.septmuse/models/`。`SEPTMUSE_EMBEDDER=onnx`（英文 all-MiniLM-L6-v2，384 dim，~23MB）/ `onnx-zh`（多语言 paraphrase-multilingual-MiniLM-L12-v2，384 dim，~50MB）。
- **AutoOnnxEmbedder** (`providers/embedders/auto.py`)：`SEPTMUSE_EMBEDDER=auto` — init 时语言检测（CJK 比例 > 30% → zh），选一个模型用于整个 session。默认 zh（中文优先项目）。`SEPTMUSE_LANG=zh/en` 可覆盖。
- **语言检测工具** (`providers/embedders/langdetect.py`)：`detect_language(text) -> "zh"|"en"`，纯 Python CJK 字符比例启发式，<1ms，无外部依赖。
- **`onnx` extra**：`pip install septmuse[onnx]` 安装 onnxruntime + tokenizers + modelscope。
- **MemoryConfig 新字段**：`embedder_backend` + `model_cache_dir`。
- 新增 `tests/unit/test_langdetect.py`（9 测试）、`test_onnx_embedder.py`（5 测试，skip 无 deps）、`test_auto_embedder.py`（4 测试，skip 无 deps）、`test_memory.py::TestSearchHybridDefault`（3 测试）、`TestResolveEmbedder`（4 测试）。

### Added — 生产就绪

- **API key 认证中间件** (`src/septmuse/api/auth.py`)：`SEPTMUSE_API_KEY` 环境变量控制开发/生产模式，支持 `Authorization: Bearer` + `X-API-Key` 双 header，Swagger UI + `/health` 豁免。REST + MCP server 均挂载。
- **Docker 部署方案**：`Dockerfile`（python:3.11-slim 单阶段，HashEmbedder 默认零模型下载）+ `docker/docker-compose.yml`（SQLite 默认 + `--profile prod` 启用 PostgreSQL/pgvector，build context 指向项目根）+ `.dockerignore`。
- **CI workflow** (`.github/workflows/ci.yml`)：ruff lint + pytest 矩阵（Python 3.10/3.11/3.12）+ build 验证（wheel LICENSE/METADATA 校验）。

### Fixed — MCP server 真实可用

- **MCP stdio tools/list 返回空**：根因 `python -m` 导致 server.py 双重执行，mcp 实例分裂。修复 `__main__` 块用 `sys.modules.setdefault` 注册当前模块避免二次 import（`server.py:103`）。
- **MCP SSE/HTTP tools/list 返回空**：根因 `setup_mcp_server` 漏 import tools 模块，`@mcp.tool` 装饰器未执行。修复加 `from septmuse.api.mcp import tools`（`server.py:69`）。
- **MCP stdio call_tool 卡死**：根因 `get_memory_safe` 用 `Memory()` 默认 sentence-transformers，stdio 子进程加载模型超时。修复默认 HashEmbedder + `SEPTMUSE_EMBEDDER` 环境变量可切换（`server.py:59`）。

### Changed

- **Memory facade 默认 HashEmbedder**：`_resolve_embedder` 默认返回 HashEmbedder（离线零模型加载，0.5s 初始化），sentence-transformers 作为 `SEPTMUSE_EMBEDDER=st` 显式升级。延迟 import sentence-transformers 避免启动时加载 transformers/torch。
- **`add_memories` 默认 `infer=False`**：对齐 mem0 阶段1 verbatim 模式，需要 LLM 抽取时显式传 `infer=True`。
- **MCP 工具参数命名统一**：`add_memories(text)` → `add_memories(content)`，`update_memory(text)` → `update_memory(content)`，与 `core_memory_append(content)` 一致，减少 LLM 调用混淆。

### Tests

- 新增 `tests/unit/test_auth.py`（9 测试）：开发模式不拦截、生产模式 401/200、Bearer/X-API-Key 双 header、环境变量、豁免路径、middleware 实例属性。
- 全量测试从 576 → **614 passed**（+9 auth + MCP 修复后历史失败转绿）。

### Added — P1 存储抽象层

- VectorStoreBase ABC (5 方法, 借鉴 mem0 精简)
- SQLiteVectorStore (默认零配置, numpy 余弦)
- ChromaVectorStore + QdrantVectorStore (extras=[chroma]/[qdrant])
- KeywordIndexBase ABC (4 方法, 借鉴 ReMe 改同步)
- SQLiteBM25Index (默认零配置, 纯 Python BM25)
- RankBM25Index (extras=[bm25])
- GraphStore.delete_edge + Neo4jGraphStore (extras=[neo4j])
- MemoryStore.keyword_search + hybrid_search (RRF 融合)
- SQLiteMemoryStore/PGVectorStore 重构为组合器模式
- MemoryConfig +3 backend 字段 (vector_backend/keyword_backend/graph_backend)
- pyproject.toml +4 extras (chroma/qdrant/neo4j/bm25) + all-backends

### Changed

- SQLiteMemoryStore 内部重构为组合器 (VectorStore + KeywordIndex), 旧签名不变
- memories 表保留 embedding 列 (双写迁移, 向后兼容)
- 全量测试 614 → 655 passed (+41 新测试), 22 skipped (integration)

### Added — P2 权限层

- MemoryState enum (active/paused/archived/deleted, 借鉴 mem0)
- memories 表 +state/app_id/archived_at/deleted_at 列 (ALTER TABLE 迁移, 向后兼容)
- memory_access_logs 表 + _record_access_log + get_access_logs
- check_memory_access_permissions 4 层权限检查 (借鉴 mem0 permissions.py)
- record_access 异步日志记录 (吞错, 不阻塞业务)
- REST API 权限检查 (403 授权) + 访问日志 + GET /memories/{id}/access-logs 端点
- MCP search_memory 访问日志记录
- 401/403 语义区分 (认证 vs 授权)

### Changed — P2

- delete() 同时设 is_deleted=1 + state='deleted' (双写兼容)
- search/get_all 过滤 state != 'active' 的记忆
- MemoryStore ABC +get_access_logs 默认实现 (返回空)
- 全量测试 655 → 686 passed (+31 新测试), 22 skipped

## [0.1.0] — 2026-07-20

### Added — 架构与脚手架

- 三维正交架构：内容类型（工作/情节/语义/程序/身份）× 存储形态（block/向量/图/文件/激活/参数化）× 横切关注点（捕获/检索/治理/演化/共享/元认知）。
- 零配置默认：SQLite 组合后端（`~/.septmuse/septmuse.db`）+ HashEmbedder（CLI/测试零模型加载）+ numpy 余弦检索回退。
- Memory facade：`add / search / get_all / get / update / get_history / delete` + 类型化记忆（`add_fact / update_fact / add_episode / update_episode / add_rule / update_rule`）+ 工作记忆 Block（`get_working_memory / get_blocks / update_block / core_memory_append / core_memory_replace`）。
- 长时记忆 update + History API：`MemoryStore.update / get_history` + REST `PUT /memories/{id}` + `GET /memories/{id}/history` + CLI `septmuse update / history`。

### Added — 三入口

- **CLI 10 命令**：`septmuse init / add / search / dump / update / history / block / serve / mcp / version`，argparse dispatch，HashEmbedder 默认零依赖。
- **REST API**：FastAPI 13 端点（`POST /memories`、`GET /memories`、`GET /memories/{id}`、`PUT /memories/{id}`、`DELETE /memories/{id}`、`GET /memories/{id}/history`、`POST /search`、Block 4 端点）。
- **MCP server 15 工具**：基础 5（add/search/list/delete/delete_all）+ 扩展 4（remember_episode/causal_query/rehearse/coverage_report）+ 新增 6（update_memory/update_block/core_memory_append/core_memory_replace/get_blocks/get_memory_history），支持 stdio / SSE / Streamable HTTP 三种 transport。

### Added — §6 三创新空白

- **因果链**：`add_causal_edge / find_causes / find_effects / counterfactual`，基于 GraphStore 的有向边 + 反事实查询。
- **遗忘曲线**：`rehearse / find_rehearse_candidates`，Ebbinghaus 间隔重复，记忆强度 decay + 最近复述时间加权。
- **元认知 L1+L2**：`coverage_report / adapt_strategy / meta_route`，L1 监控检索质量、L2 调整检索策略。

### Added — 横切关注点

- **捕获**：`capture` 零侵入注入对话消息 + `apply_token_budget` 滑动窗口 + `redact` 隐私脱敏（email/phone/id_card/credit_card 正则）。
- **检索**：`search_hybrid`（向量 + 关键词）+ `search_progressive`（渐进扩大 top_k）。
- **演化**：`reflect / dream`，反思生成新情节 + 做梦巩固记忆。
- **共享**：`list_agents / is_cross_agent / get_shared_memories`，跨 agent 共享 + 用户隔离。
- **治理**：`link_on_add / get_related`，自动实体链接 + 关联检索。

### Added — 存储后端

- SQLiteMemoryStore（默认，零配置）+ PGVectorStore（可选，需 Postgres + pgvector）。
- GraphStore ABC + SQLite（AGE）+ AGE driver（psycopg2 回退路径）。
- ActivationMemory（KV cache，依赖反转 `cache_builder` 回调）+ LoRAMemory（HuggingFace PEFT 集成）。
- TypedMemoryStore：Block CRUD（5 方法）+ SemanticFact / EpisodicEvent / ProceduralRule CRUD + `update_fact / update_episode / update_rule`。

### Added — Provider

- HashEmbedder（默认，零模型加载，中文按字切分）+ OpenAIEmbedder（可选，`text-embedding-3-small`）。
- sentence-transformers（可选 extra，零 API key）。

### Added — 可观测性

- structlog 结构化日志，输出到 stderr（MCP stdio 协议安全 + Unix 惯例）。
- `configure / get_logger / shutdown` 三函数 API。

### Added — 测试

- **unit 553 测试**：store / facade / REST / CLI / MCP / typed / history / capture / forgetting / causal / metacognition / evolution / source_sync / sharing_meta / governance / retrieval / activation / parametric / file_memory / graph_store / openai_embedder / pgvector / block / extract / mem_os / facade_integration。
- **e2e 23 测试**：跨会话偏好召回率 ≥ 80% + 用户隔离 + 认知分层（情节时序 + 程序退化 + Block XML）+ 跨 agent 共享 + 零侵入捕获 + 隐私脱敏 + token 预算 + 因果链 + 遗忘曲线 + 元认知 + 源同步。
- **9 skipped**：PGVector / AGE 集成（需 `SEPTMUSE_TEST_PG_DSN` 环境变量 + 真实 Postgres 实例）。

### Known Issues

- 13 个 FastAPI/Starlette 版本错误（`on_startup` 参数被移除）—— 非阻塞，修复需 pin Starlette 版本或迁移 FastAPI lifespan API。
- pgvector + AGE 集成测试需真实 Postgres 实例。
- `test_evolution.py` 的 LSP 类型错误（graph_store: `GraphStore | None` → `GraphStore`），pytest 通过但 LSP 不通过。

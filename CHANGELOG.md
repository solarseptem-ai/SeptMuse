# Changelog

All notable changes to SeptMuse will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added — Reranker 优化 (对齐 mem0)

- **LLMReranker + BatchLLMReranker prompt 升级** (`rerankers/llm.py` + `rerankers/batch_llm.py`)：prompt 加详细 0.0-1.0 评分标准（1.0=完美 / 0.8-0.9=高度相关 / 0.6-0.7=中等 / 0.4-0.5=轻微 / 0.0-0.3=不相关），对齐 mem0 LLMReranker。加"Do not include any explanation"指令。(原因: LLM 打分质量; 影响: rerankers/llm.py + batch_llm.py)。
- **CrossEncoderReranker normalize 配置** (`rerankers/cross_encoder.py`)：加 `normalize=True/False` 参数，默认 sigmoid 归一化到 [0,1]，可选原始 logit。对齐 mem0 HuggingFaceReranker。(原因: 灵活性; 影响: rerankers/cross_encoder.py)。
- **CrossEncoderReranker device 配置** (`rerankers/cross_encoder.py`)：加 `device=None/"cuda"/"cpu"` 参数，None 时自动检测 CUDA，cuda 用 `["CUDAExecutionProvider", "CPUExecutionProvider"]` fallback。对齐 mem0 HuggingFaceReranker device 自动检测。(原因: GPU 性能; 影响: rerankers/cross_encoder.py)。
- **SentenceTransformerReranker 新增** (`rerankers/sentence_transformer.py`，新建)：sentence-transformers CrossEncoder（`cross-encoder/ms-marco-MiniLM-L-6-v2` 默认），延迟 import，不可用降级 noop + warning。支持 `normalize` + `device` + `batch_size` + `show_progress_bar` 配置。与 `cross_encoder`（ONNX 轻量版）功能重叠但模型选择更丰富。注册到 `services/registry.py` + `rerankers/__init__.py` + `retrieval/reranker.py` 兼容层。(原因: 功能覆盖对齐 mem0; 影响: rerankers/sentence_transformer.py + configs/rerankers/sentence_transformer.py + services/registry.py)。
- 新增 `tests/unit/test_reranker.py` (+12: LLM prompt 评分标准 + CrossEncoder normalize/device + SentenceTransformer 降级/config/resolve)。全量 reranker 测试 **37 passed / 0 退化**。

### Added — 优化计划 V2: Phase 0 基础设施 (P0-Task 5)

- **SQLite WAL mode + busy_timeout** (`configs/database.py`)：`sqlite_pragmas` 默认加 `busy_timeout: 5000`（5s 写锁等待）。WAL mode 让并发读写不阻塞，`busy_timeout` 避免锁争用立即抛 `database is locked` 错误。(原因: 并发读写性能; 影响: configs/database.py)。
- **PG/MySQL 连接池配置** (`services/database/service.py`)：PG/MySQL engine 创建时用 `pool_size` + `max_overflow` + `pool_timeout`（从 config 读取，默认 pool_size=5, max_overflow=10, timeout=30s）。`:memory:` SQLite 用 `StaticPool`（已有），文件 SQLite 用默认 `QueuePool`。(原因: 生产级连接池; 影响: services/database/service.py)。
- 新增 `tests/unit/test_database_service.py` (+5: WAL PRAGMA + busy_timeout + synchronous + StaticPool + 自定义 PRAGMA 覆盖)。全量 **1319 passed / 16 failed (pre-existing) / 23 skipped / 0 退化**。

### Added — 优化计划 V2: Phase 0 基础设施 (P0-Task 3)

- **检索三路并发** (`retrieval/hybrid.py`)：`HybridRetriever.search` 用 `ThreadPoolExecutor(max_workers=3)` 并发执行向量检索 + BM25 关键词检索 + entity boost。延迟从 `v + b + e` 降为 `max(v, b, e)`。每路超时 5s 降级空结果 (`SEARCH_TIMEOUT` + `_await_future` helper)。BM25 降级路径 (keyword_search 抛异常时) 依赖候选集，在向量路径完成后串行执行。(原因: 检索延迟优化; 影响: retrieval/hybrid.py)。
- **`:memory:` SQLite StaticPool** (`services/database/service.py`)：`:memory:` SQLite 用 `poolclass=StaticPool` 共享单连接，跨线程并发检索看到同一内存库。修复 ThreadPoolExecutor 中 `:memory:` 每线程独立空库问题。(原因: 并发检索前提; 影响: services/database/service.py)。
- 新增 `tests/unit/test_hybrid_concurrent.py` (+4: 向量异常降级/BM25异常降级/向量超时降级/三路成功)。全量 **1314 passed / 16 failed (pre-existing) / 23 skipped / 0 退化**。

### Added — 优化计划 V2: Phase 0 基础设施 (P0-Task 1 + P0-Task 2)

- **chromadb 降级** (`storage/relational_stores/factory.py`)：`_resolve_vector_store()` 方法 — chromadb ImportError 时降级到 SQLAlchemyVectorStore + 日志警告。生产环境 chroma 不可用时不崩溃, 降级到纯 SQL 向量检索 (全扫描, 小数据集可用)。(原因: 零配置可用性; 影响: storage/relational_stores/factory.py)。
- **pgvector HNSW 索引** (`storage/vector_stores/pgvector_store.py`)：`_init_pgvector()` 加 `CREATE INDEX IF NOT EXISTS ... USING hnsw (vector vector_cosine_ops) WITH (m=16, ef_construction=64)`。ANN 检索从全扫描 O(n) 加速到 O(log n)。对齐 pgvector 官方推荐参数。(原因: 大数据集 ANN 性能; 影响: storage/vector_stores/pgvector_store.py)。
- **统一中文分词模块** (`core/tokenizer.py`, 新建)：`tokenize(text)` 函数 — jieba 可用时按词切分 ("我喜欢编程" → ["我", "喜欢", "编程"]), 不可用时降级正则按字 ("我喜欢编程" → ["我", "喜", "欢", "编", "程"])。`SEPTMUSE_TOKENIZER` env var 控制 (`jieba`/`space`/`auto`, 默认 `auto`)。消除三处重复 `_tokenize` 函数 (sqlite_bm25 / rank_bm25 / hybrid)。(原因: 中文检索质量 — BM25 分词质量决定召回; 影响: core/tokenizer.py + storage/keyword_stores/ + retrieval/)。
- **jieba 默认依赖** (`pyproject.toml`)：加 `jieba>=0.42` 到 `[project] dependencies`。(原因: 中文分词零配置可用; 影响: pyproject.toml)。
- 新增 `tests/unit/test_relational_store_factory.py` (+2: chromadb 降级 + chroma 可用) + `tests/unit/test_pgvector_vector_store.py` (+1: HNSW 索引 SQL) + `tests/unit/test_tokenizer.py` (+10: space/jieba/auto/降级)。全量 **1310 passed / 16 failed (pre-existing LLM) / 23 skipped / 0 退化**。

### Added — 向量存储层重构 + Embedding/Reranker 优化 + bge-zh 默认模型

- **向量存储层重构** (`storage/vector_stores/`)：统一 `VectorStoreBase` 抽象（5 方法：add/query/get/delete/ensure_dim），实现三后端 — `ChromaVectorStore`（默认配置，cosine，metadata None 过滤，`$and` 多键 where，`upsert()` 替代 `add()`）、`SQLAlchemyVectorStore`（SQLite/MySQL，`json_extract` WHERE payload 过滤）、`PgvectorVectorStore`（Postgres，`payload @> ::jsonb` WHERE 过滤）。`SQLiteCompositeStore` 重构为组合器（委托 vector_store + keyword_store + graph_store）。（原因: 生产级向量后端 + 对齐 mem0 抽象; 影响: storage/vector_stores/）。
- **Embedder ABC `embed_batch` 非抽象默认实现** (`embedders/base.py`)：`embed_batch` 改为基类默认实现（循环 `embed()`），子类可 override 做真批量。对齐 mem0 `EmbedderInterface.embed_many`。（原因: 基类不该强制子类实现批量; 影响: embedders/）。
- **OnnxEmbedder 真批量推理** (`embedders/onnx.py`)：`embed_batch` 改为 batch encode → 单次 `session.run` → 向量化 mean pool + L2，batch_size=32 分块。新增 `max_length` 参数（BGE 512，MiniLM 256）。`BGE_ZH_MODEL` 常量 + `_MODEL_FILE_OVERRIDES` 支持非 Xenova 文件结构（Maiteka `model_qint8.onnx` 在根目录）。`_ensure_model_files` 返回 `(onnx_path, tokenizer_path)` tuple。（原因: 批量推理 10x 加速 + bge-zh 模型适配; 影响: embedders/onnx.py）。
- **HashEmbedder dim 对齐** (`embedders/hash.py`)：dim 384 → 128 对齐 config。（原因: 配置一致性; 影响: embedders/hash.py）。
- **MMRReranker numpy 向量化** (`rerankers/mmr.py`)：sim_matrix 预计算 + `np.argmax` 内层循环，用 `embed_batch` 替代逐个 `embed`。（原因: 大候选池 reranker 性能; 影响: rerankers/mmr.py）。
- **CrossEncoderReranker 批量推理** (`rerankers/cross_encoder.py`)：batch encode pairs → 单次 `session.run` → `np.sigmoid`，batch_size=32 分块。（原因: 批量推理加速; 影响: rerankers/cross_encoder.py）。
- **CachedEmbedder** (`embedders/cached.py`，新建)：LRU cache 透明包装 `embed` + `embed_batch`（maxsize=256），`threading.Lock` 保证 async/sync 并发安全，返回 `list(vec)` 浅拷贝防缓存污染。（原因: 避免重复嵌入开销; 影响: embedders/cached.py）。
- **集中 `resolve_embedder`** (`embedders/resolver.py`，新建)：消除 `memory/main.py` / `memory/async_main.py` / `services/embedder/service.py` 三处重复 embedder 解析逻辑。新增 `bge-zh` 后端，onnxruntime 不可用时降级到 HashEmbedder。（原因: DRY + bge-zh 后端注册; 影响: embedders/resolver.py）。
- **Reranker 实例缓存** (`memory/main.py`)：`_get_reranker` dict 缓存，对齐 mem0 init-once 模式，避免每次 search 重新实例化。（原因: 性能; 影响: memory/main.py）。
- **默认 Embedding 切换到 bge-zh** (`configs/embeddings/base.py` + `configs/vector_stores/base.py` + `services/registry.py`)：`backend` 默认 `"hash"` → `"bge-zh"`，`embedding_model_dims` 默认 128 → 512，`_DEFAULTS["embedder"]` → `"bge-zh"`，新增 `bge-zh` BackendEntry。模型 `Maiteka/bge-small-zh-v1.5-onnx`（512 dim，ModelScope 下载）。onnxruntime 不可用时降级 HashEmbedder。测试 conftest 强制 `hash` + `dim=128` 避免模型下载。（原因: 中文语义嵌入质量; 影响: configs/ + services/ + tests/conftest.py）。
- **`SEPTMUSE_EMBEDDING_DIMS` 环境变量** (`configs/base.py`)：新增 alias 覆盖向量维度。（原因: 灵活配置; 影响: configs/base.py）。
- **AutoOnnxEmbedder 显式 max_length** (`embedders/auto.py`)：传 `max_length=256` 给 OnnxEmbedder。（原因: 防止默认值不匹配; 影响: embedders/auto.py）。

### Fixed — 细微缺陷修复 (12 项)

- **AI 幻觉修正 — CachedEmbedder 缓存污染** (`embedders/cached.py`)：返回可变 list 引用导致调用方修改污染 LRU 缓存。修复：返回 `list(vec)` 浅拷贝。（原因: 缓存完整性; 影响: embedders/cached.py）。
- **CachedEmbedder 线程不安全** (`embedders/cached.py`)：async/sync 并发访问 LRU cache 导致 crash。修复：加 `threading.Lock`，读/写分离加锁。（原因: 并发安全; 影响: embedders/cached.py）。
- **Entity boost 量级劫持排序** (`retrieval/hybrid.py`)：entity boost `0.5` 比 RRF score 大 ~43 倍，排序被 boost 主导。修复：`0.5` → `0.5/(RRF_K+1)` ≈ 0.008，与 RRF 可比。（原因: 排序公平性; 影响: retrieval/hybrid.py）。
- **SQLAlchemyVectorStore 全量加载** (`storage/vector_stores/sqlalchemy_vec.py`)：`_fetch_rows` 全量加载后 Python 侧 payload 过滤，大数据集 OOM。修复：payload 过滤推到 SQL `json_extract` WHERE（SQLite/MySQL），只加载匹配行。（原因: 性能; 影响: storage/vector_stores/sqlalchemy_vec.py）。
- **PgvectorStore Python 侧 payload 过滤** (`storage/vector_stores/pgvector_store.py`)：同上，payload 过滤在 Python 侧导致 top_k 不足。修复：`payload @> '{"key":value}'::jsonb` SQL WHERE。（原因: 性能; 影响: storage/vector_stores/pgvector_store.py）。
- **BM25 IDF 小样本不可靠** (`retrieval/hybrid.py`)：BM25 IDF 在候选池小样本上计算不可靠。修复：优先用 `store.keyword_search`（全局 IDF），降级用候选池 BM25。（原因: 检索质量; 影响: retrieval/hybrid.py）。
- **HybridRetriever 每次重建** (`memory/main.py`)：每次 search 重新实例化 HybridRetriever。修复：`self._retriever` 延迟缓存。（原因: 性能; 影响: memory/main.py）。
- **ORMMemoryStore over-fetch 不足** (`storage/relational_stores/orm_store.py`)：over-fetch 3x 过滤后 top_k 不足。修复：3x → 5x。（原因: 检索完整性; 影响: storage/relational_stores/orm_store.py）。
- **MemoryConfig docstring 过时** (`configs/base.py`)：docstring 写 "HashEmbedder" 但默认已改 bge-zh。修复：更新为 "bge-zh"。（原因: 文档准确性; 影响: configs/base.py）。
- **CachedEmbedder.embed_batch 返回行误导** (`embedders/cached.py`)：`[r for r in results if r is not None]` 静默丢弃 None。修复：`assert all(r is not None)` + 直接返回。（原因: 契约清晰; 影响: embedders/cached.py）。
- 测试更新：`test_hybrid_entity_boost.py` 断言 `combined >= entity_boost`（RRF 归一化后不同 scale），`test_reranker.py` mock 加 `embed_batch`，`test_config_merge.py` / `test_service_provider.py` / `test_memory.py` 适配 bge-zh 默认。
- 全量测试 **1297 passed / 16 failed（全为预先存在的 LLM/OpenAI API key 问题）/ 23 skipped / 0 新增失败**，e2e 32 passed。

### Added — V2 记忆架构 (ABC 分层 + V2Memory 编排入口 + 10 子组件)

- **记忆 ABC 分层** (`memory/base.py`)：MemoryABC (根抽象, 类型标记) + ShortTermMemory (compile_to_prompt/get_limit/evict_overflow) + LongTermMemory (invalidate/get_history/get_all)。ABC 只做类型标记 + 各层特有方法, 不强制统一 add/search (原因: 参数签名不同; 影响: memory/)。
- **V2Memory 编排入口** (`memory/memory_v2.py`)：remember/recall/improve/forget 4 编排方法。组合 Memory 实例 (不继承), 持有 10 子组件。零 LLM 降级: 无 SEPTMUSE_LLM 时 remember 只存 raw_log, improve 跳过 reflect (原因: 零配置可用; 影响: memory/)。
- **平面 A 4 子组件**：memory/working_memory.py (WorkingMemory 继承 ShortTermMemory, 委托 WorkingMemoryStore) + memory/semantic.py (SemanticMemory 继承 LongTermMemory) + memory/episodic.py (EpisodicMemory 继承 LongTermMemory) + memory/procedural.py (ProceduralMemory 继承 LongTermMemory)。数据模型共享 models/, 操作类全新定义 (原因: ABC 注册 + 不依赖 models/ 操作类; 影响: memory/)。
- **平面 C 6 子组件**：memory/capture.py (CapturePipeline 薄包装) + memory/retrieval.py (HybridRetriever + TokenBudget 薄包装) + memory/meta.py (MetacognitionLayer 聚合 L0+L1+L2) + memory/evolution.py (EvolutionEngine 聚合 Dream+reflect+冲突) + memory/causal.py (CausalGraph 薄包装 CausalRetriever) + memory/forgetting.py (ForgettingManager 薄包装 ForgettingRetriever) (原因: V2Memory 从 memory/ import, 不直接 import 各功能目录; 影响: memory/)。
- **工作记忆独立后端** (`storage/working_memory_stores/`)：WorkingMemoryStore ABC + SQLiteWorkingMemoryStore (共享 engine, 独立实现) + factory。Block CRUD 独立于 typed_store (原因: 决策 3 彻底分库; 影响: storage/)。
- **Redis 工作记忆后端** (`storage/working_memory_stores/redis_store.py`)：RedisWorkingMemoryStore 实现 WorkingMemoryStore ABC, Redis hash 存储 (key=`septmuse:wm:{agent_id}`, field=label, value=JSON Block)。延迟 import redis, factory 未装/未配时自动 fallback 到 SQLite (原因: 可选生产后端; 影响: storage/working_memory_stores/)。
- **remember 编排**：捕获去重+脱敏 → 情节 raw_log (恒做) → 语义事实 (仅 LLM 时) → 工作 block (可选)。不直接产程序规则 (原因: 程序规则留给 improve 蒸馏; 影响: V2Memory)。
- **recall 编排**：L0 路由 → 三信号检索 (over-fetch) → 遗忘曲线加权 → token 预算裁剪 → L2 策略自调 (仅 L1 报告存在时, 决策 6) → block+规则注入 (原因: 元认知驱动检索; 影响: V2Memory)。
- **improve 编排**：Dream 链接生长 → reflect 蒸馏 (仅 LLM 时) → 冲突解决 → L1 报告持久化为 SemanticFact(tags=["meta","coverage"]) (原因: 离线元认知; 影响: V2Memory)。
- **forget 编排**：先 invalidate (标记不再为真, 保留双时态历史) 再 delete (软删除) + 实体清理 + 图边清理 (决策 5, 原因: 彻底遗忘但保留历史轨迹; 影响: V2Memory)。
- **L1 报告 fallback** (决策 6)：首次使用时 improve 还没跑过, recall 跳过 L2 策略自调, 正常检索; improve 跑过后 L1 报告存在, recall 才用 L2 策略 (原因: L1 是离线生成, recall 不被阻塞; 影响: V2Memory)。
- 新增 `tests/unit/test_v2_memory.py` (18 用例: 创建/remember/recall/improve/forget/零LLM降级) + `tests/unit/test_memory_abc.py` (11 用例: ABC 契约 + 子组件注册 + isinstance), 全量 1259 passed / 37 skipped / 13 failed (API key)。

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

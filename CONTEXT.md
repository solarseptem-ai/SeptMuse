# CONTEXT.md — SeptMuse 术语表

> 纯术语表 + 领域关系。实现细节见代码，决策见 [docs/adr/](docs/adr/)，架构见 [docs/architecture.md](docs/architecture.md)。

## 记忆架构

- **Memory** — 零配置 [facade](#facade) 入口。持有所有组件（[store](#typedmemorystore)/[embedder](#embedder)/llm/[forgetting](#forgetting) 等），对外暴露 [remember](#remember)/[recall](#recall)/[search](#searchpipeline)/[improve](#improve)/[forget](#forget)。`src/septmuse/memory/main.py`。<a id="memory"></a>
- **facade** — Memory 的角色：统一入口，委托 [管线层](#pipeline) 执行，不自实现编排。<a id="facade"></a>
- **pipeline** — 统一管线编排层。`src/septmuse/pipelines/`。[PipelineRegistry](#componentregistry) 集中构造+缓存。<a id="pipeline"></a>
- **TypedMemoryStore** — 类型化记忆存储（[SemanticFact](#semanticfact) / [EpisodicEvent](#episodicevent) / ProceduralRule），共享 engine + facts vector_store。
- **SemanticFact** — 语义事实类型。`Memory.add(infer=True)` 走 [FactExtractor](#factextractor) 抽取。<a id="semanticfact"></a>
- **EpisodicEvent** — 情节事件类型。[Summarizer](#summarizer) 的摘要存为 `event_type="summary"`。<a id="episodicevent"></a>
- **HybridRetriever** — 混合检索器（向量 + BM25 [RRF](#rrf) 融合 + [entity boost](#entity-boost)）。[Memory](#memory) 拥有，懒构造。<a id="hybridretriever"></a>
- **RRF** — Reciprocal Rank Fusion。`k=60`，[score](#score) 统一为相似度 [0,1]，公式 `1/(k+rank+1)`。
- **entity boost** — [HybridRetriever](#hybridretriever) 的第三信号。量级 `0.5/(RRF_K+1)` ≈ 0.008，不劫持排序。<a id="entity-boost"></a>
- **score** — 统一为相似度 [0,1]，越高越相似。向量 cosine、BM25 归一化、[RRF](#rrf) 融合都遵守。<a id="score"></a>
- **SearchPipeline** — 检索管线（深化模块）。拥有 [search](#search)() 的 6 步链路：recipe 解析 → 检索 → [reranker](#reranker) → 过期过滤 → 后处理。[Memory](#memory).search() 委托给它。<a id="searchpipeline"></a>
- **search** — 检索操作。委托 [SearchPipeline](#searchpipeline)。<a id="search"></a>
- **Retriever** — 检索器 Protocol（一个 `search()` 方法，返回 `list[HybridResult]`）。[HybridRetriever](#hybridretriever) 和 [VectorRetrieverAdapter](#vectorretrieveradapter) 两个实现证明 seam 为真。<a id="retriever"></a>
- **VectorRetrieverAdapter** — 纯向量检索适配器。包装 `store.search()`（返回 `list[dict]`），适配为 `list[HybridResult]`。<a id="vectorretrieveradapter"></a>
- **RecallPipeline** — 召回管线（深化模块）。拥有 [recall](#recall)() 的 7 步链路：[compress](#compress) → [search](#search) → [token_budget](#token_budget) → [inject_prompt](#inject_prompt) → route → strategy。[Memory](#memory).recall() 委托给它。持有 [SearchPipeline](#searchpipeline) 实例作为依赖。<a id="recallpipeline"></a>
- **RememberPipeline** — 写入管线（深化模块）。拥有 [remember](#remember)() 的 10 步链路：价值评估 → 类型分类 → 语义去重 → 捕获预处理 → raw_log → verbatim 存储 → 实体抽取 → 事实抽取 → [冲突检测](#conflict-detection) → 工作记忆更新。[Memory](#memory).remember() 委托给它。<a id="rememberpipeline"></a>
- **ComponentRegistry** — 组件注册表（未来深化候选）。统一 `_init_v2_components()` 的 7× try/import/except 模式。<a id="componentregistry"></a>

## 模块化架构（目标，2026-08-20 spec）

> 详见 [docs/specs/2026-08-20-module-protocol-architecture.md](docs/specs/2026-08-20-module-protocol-architecture.md)。当前 `Memory` 正拆分为 5 个职责单一的 Protocol 模块。

- **MemorySession** — 薄 facade，组合 5 个模块，暴露 4 个核心操作（[remember](#remember) / [recall](#recall) / [forget](#forget) / [improve](#improve)）。只做依赖注入 + 路由。`src/septmuse/session.py`。<a id="memorysession"></a>
- **MemoryStore** — 持久化层 Protocol。统一管理 verbatim 记忆、typed 记忆（[SemanticFact](#semanticfact) / [EpisodicEvent](#episodicevent) / ProceduralRule）、实体、图谱、工作记忆 Block。合并 [ORMMemoryStore](#ormmemorystore) + [TypedMemoryStore](#typedmemorystore) + [EntityStore](#entitystore) + [GraphStore](#graphstore)。<a id="memorystore-protocol"></a>
- **MemoryIndex** — 检索层 Protocol。统一检索 + 后处理（[reranker](#reranker) + [forgetting](#forgetting) + [token_budget](#token_budget)）。合并 [SearchPipeline](#searchpipeline) + [HybridRetriever](#hybridretriever) + [GraphSearcher](#graphsearcher) + [ForgettingRetriever](#forgettingretriever) + [Reranker](#reranker)。<a id="memoryindex-protocol"></a>
- **MemoryIngest** — 摄入层 Protocol。捕获 + 抽取 + 实体链接的完整流水线。合并 [CapturePipeline](#capturepipeline) + [AddPipeline](#addpipeline) + [RememberPipeline](#rememberpipeline) + [FactExtractor](#factextractor) + [EntityExtractor](#entityextractor) + [CognifyPipeline](#cognifypipeline)。<a id="memoryingest-protocol"></a>
- **MemoryEvolve** — 演化层 Protocol。后台演化任务（遗忘 / 冲突解决 / 反思蒸馏 / 实体去重 / 消息压缩），调度器内置。合并 [SelfEvolutionLoop](#selfevolutionloop) + [EvolutionScheduler](#evolutionscheduler) + [ConflictResolver](#conflictresolver) + [SessionReflector](#sessionreflector) + [Summarizer](#summarizer)。<a id="memoryevolve-protocol"></a>
- **MemoryGovern** — 治理层 Protocol。访问控制 + 审计 + 限流 + API key 验证。合并 [governance/access](#governance-access) + [governance/audit](#governance-audit) + [api/auth](#api-auth) + [api/rate_limit](#api-rate-limit)。<a id="memorygovern-protocol"></a>

## 检索后处理

- **forgetting** — 遗忘曲线加权。`ForgettingRetriever.apply_strength()` 对检索结果按 `final_score = 0.7*relevance + 0.3*strength` 加权，测试效应自动刷新 `last_accessed`。<a id="forgetting"></a>
- **token_budget** — token 预算裁剪。`TokenBudget.fit()` 按加权后 [score](#score) 贪心选择，总 token 不超预算。用于 [RecallPipeline](#recallpipeline)。<a id="token_budget"></a>
- **inject_prompt** — prompt 注入。将检索结果格式化为 `<memory>` 块注入 prompt。用于 [RecallPipeline](#recallpipeline)。<a id="inject_prompt"></a>
- **compress** — 消息压缩。[Summarizer](#summarizer) 的 `static`/`partial` 两种模式。用于 [RecallPipeline](#recallpipeline)。<a id="compress"></a>
- **hyde** — Hypothetical Document Embeddings（no-op，[Recipe 断链状态](#recipe-disconnection)，未来实现）。<a id="hyde"></a>
- **query_rewrite** — 查询重写（no-op，[Recipe 断链状态](#recipe-disconnection)，未来实现）。<a id="query_rewrite"></a>
- **reranker** — 重排序器。`noop`/`mmr`/`cross_encoder`/`llm`/`sentence_transformer`。集成在 [SearchPipeline](#searchpipeline)。<a id="reranker"></a>
- **Recipe disconnection** — [SearchPipeline](#searchpipeline) 只读 `hybrid`/`reranker`/`explain`/`forgetting` 4 字段。`graph_bfs`/`progressive`/[hyde](#hyde)/[query_rewrite](#query_rewrite) 未路由到对应检索器（log 提示 future）。<a id="recipe-disconnection"></a>

## 编排 API

- **remember** — 捕获文本 → 存 raw_log → 抽事实 → 更新工作记忆。委托 [RememberPipeline](#rememberpipeline)。<a id="remember"></a>
- **recall** — [检索](#search) → [token 裁剪](#token_budget) → [注入 prompt](#inject_prompt) → L0 路由 + L2 策略。委托 [RecallPipeline](#recallpipeline)。<a id="recall"></a>
- **improve** — Dream 链接生长 + [reflect](#reflect) 蒸馏 + 覆盖报告 + [冲突解决](#conflict-resolution)。<a id="improve"></a>
- **forget** — invalidate → delete → 图清理。<a id="forget"></a>

## 演化

- **FactExtractor** — LLM 事实抽取器。`extract_and_store()` 输出 `linked_memory_ids`。[Memory](#memory).add(infer=True) 走它，infer=False verbatim 存原文。<a id="factextractor"></a>
- **Summarizer** — 消息压缩器。`compress(user_id, mode, buffer_size)`：`static`/`partial`。LLM 递归摘要；无 LLM 用拼接降级。摘要存入 [TypedMemoryStore](#typedmemorystore)。<a id="summarizer"></a>
- **reflect** — 两阶段 LLM 蒸馏：curator 批次提取 lessons → writer/rejecter 新颖性搜索 + LLM 判定（默认接受，明确 `reject` 才拒绝）。规则存入 [TypedMemoryStore](#typedmemorystore)（ProceduralRule）。<a id="reflect"></a>
- **conflict detection** — 相同 (subject, predicate) 不同 object → 冲突。<a id="conflict-detection"></a>
- **conflict resolution** — [conflict detection](#conflict-detection) + 软删除旧 fact + invalidate verbatim memory + [实体去重](#entity-dedup)。<a id="conflict-resolution"></a>
- **entity dedup** — 三段式：精确归一化 + 模糊（difflib ≥0.75）+ LLM 兜底。<a id="entity-dedup"></a>

## Embedder

- **Embedder** — 嵌入器。`SEPTMUSE_EMBEDDER` 环境变量选。bge-zh 默认，onnxruntime 不可用降级 HashEmbedder。[CachedEmbedder](#cachedembedder) 透明包装。<a id="embedder"></a>
- **CachedEmbedder** — LRU 256 + `threading.Lock`，返回浅拷贝。[Memory](#memory) facade 默认包装。<a id="cachedembedder"></a>

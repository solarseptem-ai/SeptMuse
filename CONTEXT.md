# CONTEXT.md — SeptMuse 术语表

> 纯术语表 + 领域关系。实现细节见代码，决策见 docs/adr/。

## 记忆架构

- **Memory** — 零配置 facade 入口。持有所有组件（store/embedder/llm/forgetting 等），对外暴露 add/search/recall/remember/improve/forget/compress。
- **TypedMemoryStore** — 类型化记忆存储（SemanticFact / EpisodicEvent / ProceduralRule），共享 engine + facts vector_store。
- **HybridRetriever** — 混合检索器（向量 + BM25 RRF 融合 + entity boost）。Memory 拥有，懒构造。
- **SearchPipeline** — 检索管线（深化模块）。拥有 search() 的 6 步链路：recipe 解析 → 检索 → reranker → 过期过滤 → 后处理。Memory.search() 委托给它。
- **Retriever** — 检索器 Protocol（一个 `search()` 方法，返回 `list[HybridResult]`）。HybridRetriever 和 VectorRetrieverAdapter 两个实现证明 seam 为真。
- **VectorRetrieverAdapter** — 纯向量检索适配器。包装 `store.search()`（返回 `list[dict]`），适配为 `list[HybridResult]`（vector_score=score, bm25_score=0, entity_boost=0）。
- **RecallPipeline** — 召回管线（深化模块）。拥有 recall() 的 7 步链路：compress → search → budget → prompt → route → strategy。Memory.recall() 委托给它。持有 SearchPipeline 实例作为依赖。
- **ComponentRegistry** — 组件注册表（未来深化候选）。统一 _init_v2_components() 的 7× try/import/except 模式。

## 检索后处理

- **forgetting** — 遗忘曲线加权。ForgettingRetriever.apply_strength() 对检索结果按 `final_score = 0.7*relevance + 0.3*strength` 加权，测试效应自动刷新 last_accessed。
- **token_budget** — token 预算裁剪。TokenBudget.fit() 按加权后 score 贪心选择，总 token 不超预算。
- **inject_prompt** — prompt 注入。将检索结果格式化为 `<memory>` 块注入 prompt。
- **hyde** — Hypothetical Document Embeddings（no-op，未来实现）。
- **query_rewrite** — 查询重写（no-op，未来实现）。

## 编排 API

- **remember** — 捕获文本 → 存 raw_log → 抽事实 → 更新工作记忆。
- **recall** — 检索 → token 裁剪 → 注入 prompt → L0 路由 + L2 策略。
- **improve** — Dream 链接生长 + reflect 蒸馏 + 覆盖报告 + 冲突解决。
- **forget** — invalidate → delete → 图清理。

# AGENTS.md — SeptMuse

SeptMuse 是 Python agent 记忆系统（包名 `septmuse`，src/ layout）。三维正交架构：内容类型（工作/情节/语义/程序）× 存储形态（block/向量/图/文件/激活/参数化）× 横切关注点（捕获/检索/治理/演化/共享/元认知）。零配置 `pip install septmuse` 即可用，默认 SQLite ORM + Qdrant 本地嵌入（bge-zh embedder，onnxruntime 不可用时降级 HashEmbedder），无 API key、无外部服务。

架构详情见 [docs/architecture.md](docs/architecture.md)，测试策略见 [docs/testing.md](docs/testing.md)，术语见 [CONTEXT.md](CONTEXT.md)，决策记录见 [docs/adr/](docs/adr/)。

## 开发命令

```bash
# 安装开发依赖
pip install -e ".[dev,server]"

# 测试（详见 docs/testing.md）
$env:PYTHONPATH = "src"; $env:SEPTMUSE_EMBEDDER = "hash"; $env:SEPTMUSE_TOKENIZER = "space"
$env:SEPTMUSE_RATE_LIMIT = "0"; $env:SEPTMUSE_VECTOR_BACKEND = "sqlite"

PYTHONPATH=src pytest tests/unit/ -q              # 单元 + 集成
PYTHONPATH=src pytest tests/e2e/ -q               # 端到端

# Lint + format（line-length 120）
ruff check src/ tests/ examples/
ruff format --check src/ tests/ examples/

# 运行入口
PYTHONPATH=src python -m septmuse.cli.main init              # CLI
PYTHONPATH=src python -m septmuse.cli.main serve              # HTTP API（:8000，MCP + REST）
PYTHONPATH=src python -m septmuse.api.mcp.server               # MCP server（stdio）
```

**命令顺序**：lint → test → 声明完成。CI 跑 `ruff check` + `ruff format --check` + `pytest` + `python -m build`。

## 仓库边界

- **`opensource/`** 是只读参考库（mem0/letta/ReMe/cognee/graphiti/MemOS 源码）— 禁止修改、禁止 import。所有"借鉴 X"在 src/ 内重写。
- **迁移机制**：`storage/migrations/runner.py` `MigrationRunner`，m001-m006 + `schema_version` 表追踪。无 alembic 目录。
- **`docs/specs/`** 设计规格、`docs/plans/` 实施计划、`.sdd/` subagent-driven ledger — 改动前先读对应 spec。
- **`.codegraph/`** 已索引 — 优先用 `codegraph_explore` 查代码，再考虑 grep/Read。

## 框架与工具链怪癖

### Ruff（line-length 120）

- `select = ["E","F","I","W","UP","B","SIM","RUF"]`，`ignore = ["E501","RUF001","RUF002","RUF003"]`（后三者：中文全角标点）。`known-first-party = ["septmuse"]`。
- **Windows 致命 bug**：`ruff format <file>` 在 Windows 上曾 2 次清空文件。安全做法：`ruff format --stdin-filename <path>` 经 Python subprocess，或格式化后检查文件大小。CI 跑 `ruff format --check`（只读）不触发。

### MCP tools.py

- **禁止** `from __future__ import annotations`（FastMCP `func_metadata` 把返回注解当字符串解析会炸）。工具签名用具体类型（`list[str]` 而非 `list`），`user_id` 默认从 `contextvars` 读，错误时返回 `"Error: ..."` 非 异常。

### SQLite

- 默认 `~/.septmuse/septmuse.db`；`":memory:"` 跨线程陷阱见 [docs/testing.md](docs/testing.md)。`record_access` 吞错（`hasattr` 向后兼容旧 store）。

### Embedder（`embedders/resolver.py`）

- `SEPTMUSE_EMBEDDER`：`bge-zh`（默认，OnnxEmbedder 512 dim，onnxruntime 不可用降级 HashEmbedder）/ `hash`（测试）/ `onnx`（384 dim）/ `onnx-zh`（多语言）/ `auto`（语言检测）/ `st`（sentence-transformers）。
- 语言检测 init 时一次，不 per-query 切换。不要在生产路径强制加载 sentence-transformers。
- `CachedEmbedder`（`embedders/cached.py`）：LRU 256 + `threading.Lock`，返回浅拷贝。Memory facade 默认包装。
- `embed_batch`：基类循环 `embed()`，OnnxEmbedder override 真批量推理（batch_size=32）。

### Tokenizer（`core/tokenizer.py`）

- `SEPTMUSE_TOKENIZER`：`auto`（jieba 默认）/ `jieba`（按词切分）/ `space`（按字切分，测试默认）。
- `tokenize(text)` 四处共用（sqlite_bm25/rank_bm25/hybrid/sqlalchemy_keyword）。小写化 + 去标点 + 去停用词。

### Entity Extractor（`extraction/entity.py`）

- `SEPTMUSE_ENTITY_EXTRACTOR`：`regex`（默认，4 类实体 + ~120 黑名单）/ `spacy`（`[ner]` extra）/ `none`。spaCy 不可用降级 regex。
- 实体存独立表 `septmuse_entities`，`linked_memory_ids` 关联。`Memory.add(auto_extract_entities=True)` 默认自动抽取，`Memory.delete()` 自动清理。

### Triplet Extractor（`extraction/triplet.py`）

- `TripletExtractor(llm=, entity_extractor=)`：有 LLM 走单次联合抽取（`{"entities":[...], "edges":[...]}`），无 LLM fallback EntityExtractor + 相邻规则。孤儿实体丢弃。

### Cognify Pipeline（`extraction/cognify.py`）

- `cognify(text, user_id)`：存记忆→抽三元组→upsert 实体→存 entity_relations→ZettelLinker 建链接。`entity_relations` 表 UNIQUE 约束幂等。
- `Memory.cognify()` / `Memory.get_entity_relations()` 在 facade 暴露。

### BFS Graph Search（`retrieval/graph_search.py`）

- `GraphSearcher(graph_store, store)`：BFS 遍历，去重防环。`search_graph` 返回 depth + graph_score（`1/2^depth`）。
- `rrf_fuse`：k=60，score 统一相似度 [0,1]。`Memory.search_graph()` / `Memory.search_graph_fused()` 在 facade 暴露。

### Search Recipes（`retrieval/recipes.py`）

- 8 种预置配置：`HYBRID_RRF`（默认）/`HYBRID_RRF_ENTITY`/`HYBRID_RRF_CROSS_ENCODER`/`HYBRID_RRF_MMR`/`GRAPH_BFS`/`PROGRESSIVE`/`FORGETTING`/`OPTIMAL`。
- **Recipe 断链状态**：SearchPipeline 只读 `hybrid`/`reranker`/`explain`/`forgetting` 4 字段。`graph_bfs`/`progressive`/`hyde`/`query_rewrite` 未路由（log 提示 future）。

### Temporal（`retrieval/temporal.py`）

- `search_interval(start, end, query, user_id)`：`valid_at <= end AND (invalid_at IS NULL OR invalid_at > start)`。`extract_time_range(query)` LLM 抽时间区间，无 LLM 返回 None。

### Summarizer（`evolution/summarizer.py`）

- `compress(user_id, mode, buffer_size)`：`static`（固定缓冲区）/ `partial`（驱逐 30%）。LLM 递归摘要驱逐消息；无 LLM 用拼接降级。摘要存入 `TypedMemoryStore`（`EpisodicEvent`, `event_type="summary"`）。

### LLM Fact Extraction（`prompts/extract.py`）

- `FactExtractor(llm, embedder, typed_store, verbatim_store, use_additive_prompt=True)`：默认 `ADDITIVE_EXTRACTION_PROMPT`（9 few-shot），`use_additive_prompt=False` 降级 `FACT_EXTRACTION_PROMPT`。
- `extract_and_store()` 输出 `linked_memory_ids`。`Memory.add(infer=True)` 走 FactExtractor，`infer=False` verbatim 存原文。

### Conflict Resolution（`evolution/conflict.py`）

- `detect_conflicts(user_id)`：相同 (subject, predicate) 不同 object → 冲突。`resolve_conflicts(user_id)`：软删除旧 fact + invalidate verbatim memory。
- `deduplicate_entities(user_id)`：三段式——精确归一化 + 模糊（difflib ≥0.75）+ LLM 兜底。

### Session Distillation（`evolution/reflect.py`）

- 两阶段 LLM 蒸馏：curator 批次提取 lessons → writer/rejecter 新颖性搜索 + LLM 判定（默认接受，明确 `reject` 才拒绝）。

### Reranker（`rerankers/`）

- `SEPTMUSE_RERANKER`：`noop`（默认）/ `mmr`（相似度 >0.9 只留一个）/ `cross_encoder`（ONNX `BAAI/bge-reranker-v2-m3`，`[reranker]` extra）/ `llm`（逐条打分 0-1）/ `sentence_transformer`（需 torch）。不可用降级 noop。
- Entity boost 集成在 `HybridRetriever`，量级 `0.5/(RRF_K+1)` ≈ 0.008（不劫持排序）。`Memory.search(reranker="mmr")` 可覆盖。

### Vector Store（`storage/vector_stores/`）

- `SEPTMUSE_VECTOR_BACKEND`：`qdrant`（默认，HNSW ANN）/ `sqlite`（JSON + numpy 暴力）/ `chroma`（`[chroma]` extra）/ `pgvector`。`qdrant-client` 不可用降级 SQLAlchemyVectorStore。
- Qdrant 本地嵌入：`SEPTMUSE_QDRANT_PATH` 默认 `~/.septmuse/qdrant`，设 `SEPTMUSE_QDRANT_HOST` 走远程。可选 BM25 稀疏向量（`enable_bm25=True`，需 `fastembed`）。
- Filter 10 操作符（eq/ne/gt/gte/lt/lte/in/nin/contains/icontains）+ AND/OR/NOT。简单 `{"k": "v"}` 向后兼容。ABC 12 方法。

### Bitemporal（双时态）

- `memories` 表 `valid_at`/`invalid_at`/`expired_at` 三列。`Memory.add(valid_at=)` 设置事实开始为真的时间，`Memory.invalidate()` 标记不再为真。
- `search_at(reference_time, query, user_id)`：`valid_at <= time AND (invalid_at IS NULL OR invalid_at > time)`。`valid_at=None` 视为无时间约束，始终返回。

### LLM Provider（`providers/llms/`）

- `SEPTMUSE_LLM`：`openai`（`gpt-4o-mini`，`[openai]` extra）/ `ollama`（`qwen2.5:7b`，零配置）/ `anthropic`（`claude-3-5-haiku`，`[anthropic]` extra）/ `dashscope`（`qwen-plus`，`[dashscope]` extra）。`SEPTMUSE_LLM_MODEL` 覆盖默认模型。
- LLM ABC：`complete(system_prompt, user_prompt) -> str`，JSON 输出靠 prompt 工程。`Memory(config)` 当 `llm` 未注入但 `llm_provider` 已设时自动用 `_resolve_llm` 创建。

### Observability（`observability/`）

- `SEPTMUSE_METRICS=true` 启用 Prometheus + `/metrics` 端点。未启用时所有埋点 no-op。
- 三层埋点：REST 中间件（RED）+ facade `@track_operation` + ABC 模板方法（`Embedder._embed()` / `LLM._complete()` 子类实现，`embed()`/`complete()` 基类自动计时）。
- `BusinessMetricsCollector` pull-on-scrape，每次 scrape 查 DB 算值。`/metrics` 免 API key。

## 环境变量

| 变量 | 默认 | 作用 |
|------|------|------|
| `SEPTMUSE_DB_PATH` | `~/.septmuse/septmuse.db` | SQLite 路径；`:memory:` 内存库 |
| `SEPTMUSE_EMBEDDER` | `bge-zh` | `bge-zh`/`hash`/`onnx`/`onnx-zh`/`auto`/`st` |
| `SEPTMUSE_EMBEDDING_DIMS` | 未设 | 覆盖向量维度 |
| `SEPTMUSE_API_KEY` | 未设 | 未设=开发模式；已设=生产模式（401 未认证） |
| `SEPTMUSE_USER_ID` | `default` | CLI/MCP 默认 user_id |
| `SEPTMUSE_VECTOR_BACKEND` | `qdrant` | `qdrant`/`sqlite`/`chroma`/`pgvector` |
| `SEPTMUSE_QDRANT_PATH` | `~/.septmuse/qdrant` | Qdrant 本地嵌入路径 |
| `SEPTMUSE_QDRANT_HOST` | 未设 | Qdrant 远程 host |
| `SEPTMUSE_QDRANT_PORT` | `6333` | Qdrant 远程端口 |
| `SEPTMUSE_QDRANT_API_KEY` | 未设 | Qdrant Cloud API key |
| `SEPTMUSE_QDRANT_URL` | 未设 | Qdrant 完整 URL |
| `SEPTMUSE_KEYWORD_BACKEND` | `sqlite_bm25` | `sqlite_bm25`/`rank_bm25`/`none` |
| `SEPTMUSE_GRAPH_BACKEND` | `sqlite` | `sqlite`/`age`/`neo4j` |
| `SEPTMUSE_LLM` | 未设 | LLM provider |
| `SEPTMUSE_LLM_MODEL` | 未设 | 覆盖 provider 默认模型 |
| `SEPTMUSE_INFER` | `false` | `true` 启用 LLM 抽取事实 |
| `SEPTMUSE_LANG` | 未设 | `zh`/`en`（仅 `auto` 模式生效） |
| `SEPTMUSE_MODEL_CACHE` | `~/.septmuse/models/` | ONNX 模型缓存目录 |
| `SEPTMUSE_ENTITY_EXTRACTOR` | `regex` | `regex`/`spacy`/`none` |
| `SEPTMUSE_RERANKER` | `noop` | `noop`/`mmr`/`cross_encoder`/`llm`/`sentence_transformer` |
| `SEPTMUSE_TOKENIZER` | `auto` | `auto`/`jieba`/`space` |
| `SEPTMUSE_TEST_PG_DSN` | 未设 | 测试用 Postgres DSN |
| `SEPTMUSE_TEST_NEO4J_URI` | 未设 | 测试用 Neo4j URI |
| `SEPTMUSE_METRICS` | 未设 | `true` 启用 /metrics |
| `SEPTMUSE_METRICS_PATH` | `/metrics` | 端点路径 |
| `SEPTMUSE_EVOLUTION_ENABLED` | `true` | `false` 禁用后台演化 |
| `SEPTMUSE_EVOLUTION_INTERVAL` | `300` | 演化调度器轮询间隔（秒） |

## 关键约定

- **401 vs 403**：401=认证（`api/auth.py` ApiKeyMiddleware）；403=授权（`governance/access.py`，记忆 state 不允许访问）。REST get/delete/list 都做权限检查。
- **state 状态机**：`memories.state`（active/paused/archived/deleted，默认 active）。`delete()` 双写 `is_deleted=1` + `state='deleted'`。`search`/`get_all` 过滤 `state != 'active'`。
- **access-logs**：`memory_access_logs` 表 + `get_access_logs`；REST `GET /memories/{id}/access-logs`。
- **score 统一为相似度 [0,1]**：越高越相似。向量 cosine、BM25 归一化、RRF 融合（k=60）都遵守。
- **Entity boost 量级**：`0.5/(RRF_K+1)` ≈ 0.008，不劫持排序。
- **MCP 工具 `search_memory` 带 `app_id`**：多租户权限隔离 + `record_access` 审计。
- **Docker**：`docker/docker-compose.yml`（默认 SQLite+HashEmbedder，`--profile prod` 加 Postgres/pgvector）；`Dockerfile` 在根目录。

## 架构审查执行协议

> 执行 `/improve-codebase-architecture` 时的硬约束。两轮审查中均出现过偏离，固化在此防复发。

- **报告后停**：HTML 报告产出后只问"挑哪个探索"——不分组、不排执行序、不替用户选、不提方案。
- **grilling 不可跳过**：用户选定候选后必须跑 `/grilling` 走决策树——不自主评估替代方案后推荐。grilling 的价值是把隐性约束逼成显性决策。
- **ADR + CONTEXT.md 是副产物不是可选**：难逆转 + 缺上下文会惊讶 + 真实权衡（三条全中）的决策必须出 `docs/adr/implemented/architecture/` Agent Note；新命名 / 术语锐化必须当场写回 `CONTEXT.md`。
- **词汇纪律**：用 `/codebase-design` 词汇（module / interface / depth / seam / adapter / leverage / locality），不漂移到 component / service / boundary。
- **探索按摩擦驱动不按目录分区**：sub-agent 的任务是"找按摩擦点"（浅模块、locality 缺失、seam 泄漏、难测试），不是"扫描某目录下所有文件"。

## 仓库状态

- **是 git 仓库**（远程 `git@github.com:solarseptem-ai/SeptMuse.git`）— 可以跑 git commit/push。
- **README 测试数已过时**（写的是 576 unit，实际 1833 collected）— 以 `pytest --co -q` 实际收集数为准。
- **`run_opencode.bat`** 是会话恢复快捷方式，非构建脚本。
- **`scripts/`** 目录不存在。
- **`dist/`** 是 `python -m build` 产物，勿手动改。

## 输出语言

与用户交互、任务汇报、报错、结果说明强制使用简体中文（工具返回的英文日志/字段可保留原文，解读转中文）。代码内部注释可用英文。

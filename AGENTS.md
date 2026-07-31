# AGENTS.md — SeptMuse

SeptMuse 是 Python agent 记忆系统（包名 `septmuse`，src/ layout）。三维正交架构：内容类型（工作/情节/语义/程序）× 存储形态（block/向量/图/文件/激活/参数化）× 横切关注点（捕获/检索/治理/演化/共享/元认知）。零配置 `pip install septmuse` 即可用，默认 SQLite + HashEmbedder，无 API key、无外部服务。

## 开发命令

```bash
# 安装开发依赖（不 pip install -e .，用 PYTHONPATH=src 跑测试）
pip install -e ".[dev,server]"

# 测试（必须设 PYTHONPATH=src，否则 import septmuse 失败）
$env:PYTHONPATH = "src"        # PowerShell
PYTHONPATH=src pytest tests/unit/ -q          # 单元 + 集成（686 passed, 22 skipped 基线）
PYTHONPATH=src pytest tests/e2e/ -q           # 端到端（23 passed）
PYTHONPATH=src pytest tests/unit/test_memory.py::test_add_returns_id -q   # 单测试

# Lint + format（line-length 120）
ruff check src/ tests/ examples/
ruff format --check src/ tests/ examples/

# 运行入口
PYTHONPATH=src python -m septmuse.cli.main init            # CLI（argparse，10 命令）
PYTHONPATH=src python -m septmuse.cli.main serve            # REST API（:8000/docs）
PYTHONPATH=src python -m septmuse.api.mcp.server            # MCP server（stdio）
```

**命令顺序**：lint → test → 声明完成。CI（`.github/workflows/ci.yml`）跑 `ruff check` + `ruff format --check` + `pytest tests/unit/ tests/e2e/`（Python 3.10/3.11/3.12 矩阵）+ `python -m build`。

## 架构入口

- ** facade**：`src/septmuse/orchestration/memory.py` `Memory` 类（零配置入口，借鉴 mem0）— `_resolve_embedder` 读 `SEPTMUSE_EMBEDDER` 环境变量
- **三 API 入口**：
  - CLI：`src/septmuse/cli/main.py`（argparse，非 Typer；10 命令：init/add/search/dump/update/history/block/serve/mcp/version）
  - REST：`src/septmuse/api/rest/__init__.py` `create_app()` → FastAPI（~17 端点：/memories, /memories/{id}, /memories/{id}/access-logs, /memories/search, /agents/{user_id}/memories, /health 等）
  - MCP：`src/septmuse/api/mcp/tools.py`（15 工具，FastMCP `@mcp.tool`）+ `transports.py`（stdio/SSE/Streamable HTTP 三 transport，挂载到 FastAPI）
- **存储抽象**：`src/septmuse/storage/base.py` `MemoryStore` ABC + `storage/vector/base.py` `VectorStoreBase` + `storage/keyword/base.py` `KeywordIndexBase` + `storage/graph/base.py` `GraphStore` ABC
- **默认后端**：`storage/sqlite/store.py` `SQLiteCompositeStore`（组合 vector + keyword + graph，双写迁移，`ALTER TABLE` 在代码内非 alembic）
- **配置**：`src/septmuse/configs/defaults.py` `MemoryConfig`（pydantic）+ `default_config()` 读环境变量
- **治理**：`src/septmuse/concerns/governance/permissions.py` `MemoryState` enum + `check_memory_access_permissions`（4 层）；`access_log.py` `record_access`（吞错，`hasattr` 向后兼容）

## 仓库边界

- **`opensource/`** 是只读参考库（mem0/letta/ReMe/cognee/graphiti/MemOS 源码）— **禁止修改、禁止 import**。所有"借鉴 X"的实现在 src/ 内重写，不直接调 opensource。
- **`alembic/`** 是空壳（`versions/` 空，无 `alembic.ini`）— 迁移靠 `SQLiteCompositeStore._migrate_add_state_columns` 等运行时 `ALTER TABLE`，不要走 alembic。
- **`docs/specs/`** 设计规格、`docs/plans/` 实施计划、`.sdd/` subagent-driven ledger — 改动前先读对应 spec。
- **`.codegraph/`** 已索引 — 优先用 `codegraph_explore` 查代码，再考虑 grep/Read。

## 框架与工具链怪癖

### Ruff（line-length 120）

- `select = ["E","F","I","W","UP","B","SIM","RUF"]`，`ignore = ["E501","RUF001","RUF002","RUF003"]`（后三者：中文全角标点）
- `known-first-party = ["septmuse"]`（isort）
- **Windows 致命 bug**：`ruff format <file>` 在 Windows 上曾 2 次清空文件（已实证）。安全做法：`ruff format --stdin-filename <path>` 经 Python subprocess 调用，或格式化后立即检查文件大小，空了就重写。CI 跑 `ruff format --check`（只读），不会触发此 bug。

### MCP tools.py

- **禁止** `from __future__ import annotations`（FastMCP `func_metadata` 把返回注解当字符串解析会炸）— 文件顶部有显式注释。
- 工具签名必须用具体类型（`list[str]` 而非 `list`），`user_id` 默认从 `contextvars` 读，错误时返回字符串 `"Error: ..."`（非异常）。

### SQLite

- 默认 `~/.septmuse/septmuse.db`；`db_path=None` 触发此默认；`":memory:"` 可用但 **FastAPI TestClient 跨线程会连到新空库** — REST/e2e 测试一律用 `tmp_path / "test.db"` 文件路径（见 `tests/e2e/*.py` 的 `db = str(tmp_path / "e2e.db")` 模式）。
- `record_access` 吞错（日志失败不阻塞业务），用 `hasattr` 检查 store 是否有 `get_access_logs` 方法（向后兼容旧 store）。

### Embedder

- `SEPTMUSE_EMBEDDER=hash`（默认，HashEmbedder，离线零模型加载，0.5s 初始化）— CLI/MCP server/测试默认。
- `SEPTMUSE_EMBEDDER=onnx` — Xenova/all-MiniLM-L6-v2 ONNX 量化版（384 dim，~23MB，无 torch，CPU <50ms，ModelScope 下载）。
- `SEPTMUSE_EMBEDDER=onnx-zh` — Xenova/paraphrase-multilingual-MiniLM-L12-v2（384 dim，多语言，中英文均支持，ModelScope 下载）。
- `SEPTMUSE_EMBEDDER=auto` — init 时语言检测自动选 onnx-zh（默认）或 onnx。`SEPTMUSE_LANG=zh/en` 可覆盖。
- `SEPTMUSE_EMBEDDER=st` — sentence-transformers（延迟 import，启动慢 ~30s，需模型缓存）。
- **语言检测策略**：init 时一次，不 per-query 切换（不同模型投影到不同语义空间）。
- 不要在生产路径强制加载 sentence-transformers。

### Entity Extractor

- `SEPTMUSE_ENTITY_EXTRACTOR=regex`（默认，纯 Python regex + 词表，零配置）— 4 类实体（PROPER/QUOTED/TOPIC/IDENTIFIER）+ ~120 泛化词黑名单 + span 去重。
- `SEPTMUSE_ENTITY_EXTRACTOR=spacy` — spaCy NER + noun_chunks（`pip install septmuse[ner]`，模型首次使用时自动下载）。
- `SEPTMUSE_ENTITY_EXTRACTOR=none` — 禁用实体抽取。
- spaCy 不可用时自动降级到 regex + 日志警告。
- 实体存独立 SQLite 表 `septmuse_entities`，用 `linked_memory_ids` 关联记忆（借鉴 mem0 V3 去图化）。
- `Memory.add(auto_extract_entities=True)` 默认自动抽取，`Memory.delete()` 自动清理引用。

### Triplet Extractor (P0-Task 2)

- `concerns/extraction/triplet.py`：三元组 LLM 联合抽取（借鉴 graphiti `extract_nodes_and_edges`）。
- `TripletExtractor(llm=, entity_extractor=)`：有 LLM 走单次联合抽取（输出 `{"entities":[...], "edges":[...]}`），无 LLM fallback 到 EntityExtractor + 相邻实体规则。
- `extract_triplets(text, llm=, entity_extractor=)`：便捷函数。
- 孤儿实体丢弃（没有边连接的实体被过滤，对齐 graphiti）。
- `Triplet` dataclass：`subject`/`predicate`/`object` + `as_tuple()`。

### Cognify Pipeline (P0-Task 3)

- `concerns/extraction/cognify.py`：知识图谱构建流水线（借鉴 cognee cognify + graphiti）。
- `CognifyPipeline(store, graph_store, embedder, entity_store, llm, entity_extractor)`：Pipeline DAG。
- `cognify(text, user_id)`：存记忆→抽三元组→upsert 实体→存 entity_relations→ZettelLinker 建链接。
- `entity_relations` 表：实体间关系边（source_entity, relation, target_entity, user_id, UNIQUE 约束幂等）。
- `search_entities` 复用已有 `EntityStore.search`。
- `get_entity_relations(entity_name, user_id)`：实体间关系遍历（双向，区别于 `get_entity_neighbors(entity_id)` 返回 linked_memory_ids）。
- `Memory.cognify()` / `Memory.get_entity_relations()` 在 facade 暴露。

### BFS Graph Search (P1-Task 3)

- `concerns/retrieval/graph_search.py`：BFS 图遍历检索（借鉴 graphiti bfs_search）。
- `GraphSearcher(graph_store, store)`：从种子节点 BFS 遍历 GraphStore。
- `bfs(seed_memory_id, max_depth, relation)`：返回 `[{"id", "depth"}]`，去重防环。
- `search_graph(seed_memory_id, max_depth)`：返回记忆内容 + depth + graph_score（`1/2^depth` 衰减）。
- `rrf_fuse(vector_results, graph_results)`：RRF 融合（`k=60`，score 统一为相似度 [0,1]）。
- `Memory.search_graph(seed_memory_id, max_depth)` / `Memory.search_graph_fused(query, ...)` 在 facade 暴露。

### Search Recipes (P1-Task 4)

- `concerns/retrieval/recipes.py`：7 种预置检索配置（借鉴 graphiti search_config_recipes）。
- `SearchRecipe` dataclass：`name`/`hybrid`/`reranker`/`explain`/`graph_bfs`/`forgetting`/`progressive`。
- `get_recipe(name)` / `list_recipes()`：获取和列出 recipe。
- 7 种：`HYBRID_RRF`（默认）、`HYBRID_RRF_ENTITY`（+explain）、`HYBRID_RRF_CROSS_ENCODER`（+cross-encoder）、`HYBRID_RRF_MMR`（+MMR）、`GRAPH_BFS`（纯图遍历）、`PROGRESSIVE`（渐进三层）、`FORGETTING`（遗忘曲线）。
- `Memory.search(recipe="HYBRID_RRF_MMR")` 一键切换，覆盖 hybrid/reranker/explain。

### Temporal Interval Query (P2-Task 2)

- `concerns/retrieval/temporal.py`：时态区间查询 + LLM 自然语言时间抽取（借鉴 cognee temporal_retriever）。
- `TemporalRetriever(store, embedder, llm)`：时态检索器。
- `search_interval(start, end, query, user_id)`：查询 [start, end) 内为真的记忆（`valid_at <= end AND (invalid_at IS NULL OR invalid_at > start)`）。
- `extract_time_range(query)`：LLM 从自然语言抽取时间区间（`"上周" → {"start","end"}`），无 LLM 返回 None。
- `search_natural(query, user_id)`：自然语言时态查询（先抽时间→有时态过滤→无回退普通检索）。
- `store.get_temporal_interval(start, end, user_id)`：存储层区间过滤。
- `Memory.search_interval()` / `Memory.search_natural()` 在 facade 暴露。

### Summarizer (P2-Task 3)

- `concerns/compression/summarizer.py`：消息压缩 Summarizer（借鉴 letta Summarizer）。
- `Summarizer(store, typed_store, llm)`：两种压缩模式。
- `compress(user_id, mode, buffer_size)`：`static`（固定缓冲区，超限驱逐旧消息）/ `partial`（驱逐 30%）。
- LLM 递归摘要驱逐消息；无 LLM 用拼接降级。
- 摘要存入 `TypedMemoryStore`（`EpisodicEvent`, `event_type="summary"`）。
- `Memory.compress(user_id, mode="static", buffer_size=20)` 在 facade 暴露。

### LLM Fact Extraction (P3-Task 2)

- `prompts/extract.py`：`ADDITIVE_EXTRACTION_PROMPT`（含 9 个 few-shot，对齐 mem0 V3）+ `FACT_EXTRACTION_PROMPT`（精简版）。
- `FactExtractor(llm, embedder, typed_store, verbatim_store, use_additive_prompt=True)`：默认用 ADDITIVE prompt。
- `extract_and_store()` 输出 `linked_memory_ids`（跨记忆链接，对齐 mem0 V3）。
- `use_additive_prompt=False` 降级到 FACT_EXTRACTION_PROMPT（向后兼容）。
- `Memory.add(infer=True)` 走 FactExtractor 抽取事实（`infer=False` verbatim 存原文）。

### Conflict Resolution (P3-Task 3)

- `concerns/evolution/conflict.py`：冲突解决 + 实体去重（借鉴 graphiti edge_operations + node_operations）。
- `ConflictResolver(typed_store, store, llm)`：矛盾检测 + 实体去重。
- `detect_conflicts(user_id)`：相同 (subject, predicate) 不同 object → 冲突。
- `resolve_conflicts(user_id)`：软删除旧 fact + invalidate verbatim memory（复用 P2-Task 1）。
- `deduplicate_entities(user_id)`：三段式去重——精确归一化 + 模糊相似度（difflib, ≥0.75）+ LLM 兜底（可选）。
- `TypedMemoryStore.soft_delete_fact(fact_id)`：软删除 SemanticFact。
- `Memory.resolve_conflicts()` / `Memory.deduplicate_entities()` 在 facade 暴露。

### Session Distillation (P3-Task 4)

- `concerns/evolution/reflect.py`：重构为两阶段 LLM 蒸馏（借鉴 cognee `distill.py` curator → writer/rejecter）。
- **curator**：LLM 批次提取课程（lessons）from 历史记忆。
- **writer/rejecter**：新颖性搜索（检查已有规则避免重复）+ LLM 判定（默认接受，明确 `reject` 才拒绝）。
- `_is_similar(a, b)`：简单子串匹配做新颖性搜索。
- `_llm_accept(statement)`：LLM 判定 lesson 是否值得写入。
- `Memory.reflect(user_id, limit=50)` 触发蒸馏，规则存入 `TypedMemoryStore`（`ProceduralRule`）。

### Reranker

- `SEPTMUSE_RERANKER=noop`（默认，透传）— 不改变顺序，零开销。
- `SEPTMUSE_RERANKER=mmr` — 最大边际相关性，去冗余（相似度 >0.9 只留一个），纯数学无依赖。
- `SEPTMUSE_RERANKER=cross_encoder` — ONNX cross-encoder（`BAAI/bge-reranker-v2-m3`），`pip install septmuse[reranker]`，不可用时降级为 noop。
- `SEPTMUSE_RERANKER=llm` — LLM 逐条打分 0-1，需 `SEPTMUSE_LLM` 配置 LLM provider。
- `Memory.search(reranker="mmr")` 可覆盖配置。
- Entity boost 集成在 `HybridRetriever`（第三信号），`Memory.search(explain=True)` 返回 `score_details`。

### Bitemporal (双时态)

- `memories` 表有 `valid_at`/`invalid_at`/`expired_at` 三列（P2-Task 1 迁移）。
- `Memory.add(valid_at="2024-01-01")`：设置事实开始为真的时间。
- `Memory.invalidate(memory_id)`：手动标记事实不再为真（设置 invalid_at + expired_at，不删除）。
- `Memory.search_at(reference_time, query, user_id)`：时态查询，过滤 `valid_at <= time AND (invalid_at IS NULL OR invalid_at > time)`。
- `valid_at=None` 的记忆视为"无时间约束"，search_at 始终返回（向后兼容）。
- LLM 自动矛盾检测留给 P3-Task 3（在 add() 中插入矛盾检测步骤，不改存储层）。

### LLM Provider

- `SEPTMUSE_LLM=openai` — OpenAI GPT（`gpt-4o-mini` 默认），`OPENAI_API_KEY` 必填，`pip install septmuse[openai]`。
- `SEPTMUSE_LLM=ollama` — Ollama 本地（`qwen2.5:7b` 默认），零配置 `localhost:11434`，`pip install septmuse[ollama]`。
- `SEPTMUSE_LLM=anthropic` — Anthropic Claude（`claude-3-5-haiku-latest` 默认），`ANTHROPIC_API_KEY` 必填，`pip install septmuse[anthropic]`。
- `SEPTMUSE_LLM=dashscope` — DashScope Qwen（`qwen-plus` 默认），`DASHSCOPE_API_KEY` 必填，`pip install septmuse[dashscope]`。
- `SEPTMUSE_LLM_MODEL` 覆盖 provider 默认模型。
- `Memory(config)` 当 `llm` 未注入但 `llm_provider` 已设时，自动用 `_resolve_llm` 创建。
- `OpenAILLM` 已在 `providers/llms/openai.py` 实现（支持 `OPENAI_BASE_URL` 兼容端点）。
- LLM ABC：`complete(system_prompt, user_prompt) -> str`，JSON 输出靠 prompt 工程。

## 测试怪癖

- **36 skipped 是正常的**：`pytest.mark.integration` 标记的测试（chroma/qdrant/neo4j/rank-bm25）需装对应 extras；pgvector + AGE 测试需 `SEPTMUSE_TEST_PG_DSN`；Neo4j 需 `SEPTMUSE_TEST_NEO4J_URI`；parametric 需 torch+peft；activation 需 torch；onnx/auto embedder 需 `pip install septmuse[onnx]`。
- **e2e 测试**：`tests/e2e/` 4 文件 23 测试，全部用 `tmp_path` 文件 DB，测跨会话持久化。`pytest tests/e2e/` 必须通过。
- **`pytest_asyncio_mode = "auto"`**：async 测试无需 `@pytest.mark.asyncio`。
- **`--strict-markers --strict-config`**：新增 marker 必须在 `pyproject.toml [tool.pytest.ini_options] markers` 注册。
- **测试保护规则**：现有单元/接口测试案例固定不动，禁止改测试代码绕过业务缺陷；仅可新增测试覆盖新功能。

## 环境变量

| 变量 | 默认 | 作用 |
|------|------|------|
| `SEPTMUSE_DB_PATH` | `~/.septmuse/septmuse.db` | SQLite 路径；`:memory:` 内存库 |
| `SEPTMUSE_EMBEDDER` | `hash` | `hash`/`onnx`/`onnx-zh`/`auto`/`st` |
| `SEPTMUSE_API_KEY` | 未设 | 未设=开发模式（无认证，警告一次）；已设=生产模式（401 未认证） |
| `SEPTMUSE_USER_ID` | `default`（CLI） | CLI/MCP 默认 user_id |
| `SEPTMUSE_VECTOR_BACKEND` | `sqlite` | `sqlite`/`pgvector`/`chroma`/`qdrant` |
| `SEPTMUSE_KEYWORD_BACKEND` | `sqlite_bm25` | `sqlite_bm25`/`rank_bm25`/`none` |
| `SEPTMUSE_GRAPH_BACKEND` | `sqlite` | `sqlite`/`age`/`neo4j` |
| `SEPTMUSE_LLM` | 未设 | LLM provider（verbatim 模式不需要） |
| `SEPTMUSE_LLM_MODEL` | 未设 | 覆盖 provider 默认模型 |
| `SEPTMUSE_INFER` | `false` | `true` 启用 LLM 抽取事实 |
| `SEPTMUSE_LANG` | 未设 | `zh`/`en`（仅 `auto` 模式生效，未设时默认 `zh`） |
| `SEPTMUSE_MODEL_CACHE` | `~/.septmuse/models/` | ONNX 模型缓存目录 |
| `SEPTMUSE_ENTITY_EXTRACTOR` | `regex` | `regex`/`spacy`/`none` |
| `SEPTMUSE_RERANKER` | `noop` | `noop`/`mmr`/`cross_encoder`/`llm` |
| `SEPTMUSE_TEST_PG_DSN` | 未设 | 测试用 Postgres DSN |
| `SEPTMUSE_TEST_NEO4J_URI` | 未设 | 测试用 Neo4j URI |

## 关键约定

- **401 vs 403**：401=认证（`api/auth.py` ApiKeyMiddleware，API key 缺失/错误）；403=授权（`concerns/governance/permissions.py`，记忆 state 不允许访问）。REST get/delete/list 都做权限检查。
- **state 状态机**：`memories.state` 列（active/paused/archived/deleted，默认 active，`ALTER TABLE` 迁移自动兼容旧数据）。`delete()` 双写 `is_deleted=1` + `state='deleted'`。`search`/`get_all` 过滤 `state != 'active'`。
- **access-logs**：`memory_access_logs` 表 + `_record_access_log` + `get_access_logs`；REST `GET /memories/{id}/access-logs` 端点查询审计日志。
- **score 统一为相似度 [0,1]**：越高越相似。向量 cosine、BM25 归一化、RRF 融合（k=60，`alpha` 是向量权重 pure-mode）都遵守此约定。不要引入"距离越小越相似"的歧义。
- **MCP 工具 `search_memory` 带 `app_id` 参数**：用于多租户权限隔离 + `record_access` 审计。
- **Docker**：`docker/docker-compose.yml`（默认 SQLite+HashEmbedder，`--profile prod` 加 Postgres/pgvector）；`Dockerfile` 在根目录，镜像零配置可用。

## 仓库状态

- **不是 git 仓库**（文件快照模式）— 不要跑 git commit/push，改动靠文件对比。
- **README 测试数已过时**（写的是 576 unit，实际 708 collected / 686 passed + 22 skipped）— 以 `pytest --co -q` 实际收集数为准。
- **`run_opencode.bat`** 是会话恢复快捷方式（`opencode --session <id>`），非构建脚本。
- **`scripts/`** 当前为空。
- **`dist/`** 是 `python -m build` 产物，勿手动改。

## 输出语言

与用户交互、任务汇报、报错、结果说明强制使用简体中文（工具返回的英文日志/字段可保留原文，解读转中文）。代码内部注释可用英文。

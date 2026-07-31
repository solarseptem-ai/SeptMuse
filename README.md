<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/septmuse-banner.svg">
  <source media="(prefers-color-scheme: light)" srcset="assets/septmuse-banner-light.svg">
  <img alt="SeptMuse — Agent 记忆系统" src="assets/septmuse-banner.svg">
</picture>

---

> Agent 记忆系统 — 零配置开箱即用，生产级架构。

## 为什么用 SeptMuse

AI Agent 的核心瓶颈不是推理能力，而是**记忆**。LLM 上下文窗口有限，对话一结束记忆就消失。Agent 无法跨会话记住用户偏好、无法从历史错误中学习、无法追踪事实随时间的演变。

SeptMuse 解决这个问题：

- **零配置启动**：`pip install septmuse` → SQLite + 离线嵌入，无 API key、无外部服务、无 Docker。30 秒内给 Agent 加上持久记忆。
- **生产级架构**：三维正交设计（内容类型 × 存储形态 × 横切关注点），每个能力独立演进。sync + async 双版本 API，REST/MCP/CLI 三入口，可插拔后端（SQLite/PG/Chroma/Qdrant/Neo4j）。
- **不只是存储**：实体抽取 → 知识图谱 → 冲突检测 → 遗忘曲线 → 会话蒸馏 → 元认知覆盖报告。记忆会自我治理、自我演化。

### 与其他记忆库的区别

| 能力 | SeptMuse | 传统向量库 |
|------|----------|-----------|
| 零配置 | SQLite + 离线嵌入，离线可用 | 需 API key 或外部服务 |
| 混合检索 | 向量 + BM25 RRF 融合 + 实体 boost | 仅向量 |
| 知识图谱 | 三元组抽取 + BFS 图遍历 + 实体关系 | 无 |
| 双时态建模 | valid_at / invalid_at / expired_at | 无 |
| 冲突检测 | 相同 (subject, predicate) 不同 object → 软删除旧 fact | 无 |
| 遗忘曲线 | Ebbinghaus 间隔重复 + 主动复述 | 无 |
| 会话蒸馏 | LLM 批量提取 lessons → 新颖性搜索 → 规则沉淀 | 无 |
| 工作记忆 | Block XML 编译 + 自动持久化 + 自编辑 | 无 |
| 权限审计 | 4 层权限 + 访问日志 + state 状态机 | 无 |
| 数据迁移 | 版本追踪 + 有序迁移 + CLI 手动触发 | 无 |
| sync + async | AsyncMemory 9 方法 + async 权限/日志 | 仅 sync |
| 三入口 | REST (21 端点) + MCP (15 工具) + CLI (12 命令) | 通常 1 个 |

## Quickstart

### 安装

```bash
pip install septmuse                # 核心：SQLite + 离线嵌入，零依赖
pip install "septmuse[onnx]"       # ONNX 嵌入（推荐生产，CPU <50ms，无 torch）
pip install "septmuse[server]"     # uvicorn（REST serve 需要）
pip install "septmuse[openai]"     # OpenAI LLM + Embedder
pip install "septmuse[litellm]"    # litellm 统一代理（100+ provider）
pip install "septmuse[postgres]"   # PGVector 生产后端
```

### Python API（3 行起步）

```python
from septmuse import Memory

memory = Memory()  # 零配置：SQLite ~/.septmuse/septmuse.db + 离线嵌入
memory.add("我喜欢 Python 和 vim 键位", user_id="alice")
results = memory.search("alice 喜欢什么编辑器", user_id="alice")
# [{"id": "mem-...", "memory": "我喜欢 Python 和 vim 键位", "score": 0.85, ...}]
```

### Async API（高并发场景）

```python
import asyncio
from septmuse.memory.async_main import AsyncMemory
from septmuse.embedders.hash import HashEmbedder
from septmuse.storage.async_sqlite.store import AsyncSQLiteMemoryStore

store = AsyncSQLiteMemoryStore(db_path="mem.db")
memory = AsyncMemory(embedder=HashEmbedder(), store=store)

async def main():
    await memory.add("hello world", user_id="alice")
    results = await memory.search("hello", user_id="alice")
    await memory.close()

asyncio.run(main())
```

### CLI

```bash
septmuse init                          # 初始化 ~/.septmuse/
septmuse add "我喜欢 Python" --user alice
septmuse search "alice 喜欢什么" --user alice
septmuse update <id> "新内容" --user alice
septmuse history <id>                  # 变更历史
septmuse block alice-assistant         # 工作记忆 Block
septmuse backends                      # 查看可用后端
septmuse config show                   # 查看当前配置
septmuse migrate                       # 运行数据库迁移
septmuse serve                         # REST API (:8000/docs)
septmuse mcp                           # MCP server (stdio)
septmuse version
```

### REST API

```bash
septmuse serve  # FastAPI :8000

# 添加
curl -X POST http://localhost:8000/memories \
  -H "Content-Type: application/json" \
  -d '{"content": "我喜欢 Python", "user_id": "alice"}'

# 混合检索（向量 + BM25）
curl -X POST http://localhost:8000/memories/search \
  -H "Content-Type: application/json" \
  -d '{"query": "编程偏好", "user_id": "alice", "top_k": 5}'

# 访问审计日志
curl http://localhost:8000/memories/{id}/access-logs
```

### MCP Server（Claude Code / Cursor 接入）

```bash
septmuse mcp   # stdio transport，15 工具
# 或 septmuse serve → MCP 自动挂载到 /mcp/sse 和 /mcp/http
```

## 核心竞争力

### 1. 零配置离线可用

默认 SQLite + HashEmbedder（离线哈希嵌入，零模型加载，0.5s 初始化）。无 API key、无 Docker、无 Neo4j、无 Qdrant。`pip install` 后直接 `Memory()` 即可读写记忆。

生产切 ONNX 嵌入：`SEPTMUSE_EMBEDDER=onnx`（384 维，~23MB，CPU <50ms，ModelScope 下载，无 torch）。

### 2. 三维正交架构

```
内容类型          存储形态          横切关注点
─────────        ─────────        ─────────
工作记忆          block             捕获
情节记忆          向量              检索
语义记忆          图                治理
程序记忆          文件              演化
                  激活              共享
                  参数化            元认知
```

每个记忆能力是三个平面某一格的组合。新增能力只需在对应格子填实现，不破坏其他平面。

### 3. ServiceProvider 能力切换

8 个能力 × 多后端，声明式注册表 + 环境变量切换：

```bash
SEPTMUSE_EMBEDDER=onnx           # hash / onnx / onnx-zh / auto / st
SEPTMUSE_LLM=litellm             # openai / ollama / anthropic / dashscope / litellm / groq / gemini / deepseek
SEPTMUSE_RERANKER=mmr            # noop / mmr / cross_encoder / llm
SEPTMUSE_VECTOR_BACKEND=pgvector  # sqlite / pgvector / chroma / qdrant
SEPTMUSE_GRAPH_BACKEND=neo4j     # sqlite / age / neo4j
```

```python
from septmuse.services.providers import llm_provider
print(llm_provider.list_backends())
# ['openai', 'ollama', 'anthropic', 'dashscope', 'litellm', 'groq', 'gemini', 'deepseek']
```

### 4. 混合检索 + 知识图谱

- **向量检索**：numpy 余弦相似度（默认）或 pgvector
- **关键词检索**：纯 Python BM25（默认）或 rank-bm25
- **RRF 融合**：`alpha` 参数控制向量/关键词权重
- **实体 boost**：第三信号，记忆命中的实体数提升检索分数
- **BFS 图遍历**：从种子记忆出发，按关系边遍历，`1/2^depth` 衰减打分
- **7 种检索配方**：`HYBRID_RRF` / `HYBRID_RRF_ENTITY` / `HYBRID_RRF_CROSS_ENCODER` / `HYBRID_RRF_MMR` / `GRAPH_BFS` / `PROGRESSIVE` / `FORGETTING`

```python
# 一键切换检索配方
results = memory.search("query", user_id="alice", recipe="HYBRID_RRF_MMR")
```

### 5. 双时态建模 + 冲突检测

```python
# 记录事实开始为真的时间
memory.add("Alice 住在北京", user_id="alice", valid_at="2020-01-01")

# 标记事实不再为真（搬家了）
memory.invalidate(memory_id, invalid_at="2024-06-01")

# 时态查询：某时刻为真的记忆
memory.search_at("2023-01-01", "Alice 住哪", user_id="alice")

# 自然语言时态查询
memory.search_natural("上周 Alice 住哪", user_id="alice")

# 冲突检测：相同 (subject, predicate) 不同 object
memory.resolve_conflicts(user_id="alice")  # 软删除旧 fact

# 实体去重
memory.deduplicate_entities(user_id="alice")  # 精确 + 模糊 + LLM 三段式
```

### 6. 记忆自演化

```python
# 会话蒸馏：LLM 从历史记忆提取 lessons → 新颖性搜索 → 规则沉淀
memory.reflect(user_id="alice", limit=50)

# 消息压缩：超限驱逐旧消息 → LLM 递归摘要
memory.compress(user_id="alice", mode="static", buffer_size=20)

# 知识图谱构建：存记忆 → 抽三元组 → upsert 实体 → 建关系边
memory.cognify("Alice 是后端工程师，擅长 Python", user_id="alice")
memory.get_entity_relations("Alice", user_id="alice")
```

### 7. 权限 + 审计 + 迁移

- **4 层权限**：记忆存在 → state=active → app_id 校验 → 白名单
- **访问日志**：每次 search/get/delete 记录 MemoryAccessLog
- **state 状态机**：active / paused / archived / deleted
- **数据迁移**：`schema_version` 表 + 5 个有序迁移模块 + `septmuse migrate` CLI

```bash
# 手动迁移
septmuse migrate --db-path ~/.septmuse/septmuse.db
# 已应用 5 个迁移:
#   001 - initial schema (memories + history)
#   002 - add state/deleted_at/app_id columns
#   003 - add session_id column
#   004 - add temporal columns
#   005 - create memory_access_logs table
```

### 8. sync + async 双版本

| 路径 | sync | async |
|------|------|-------|
| Store | SQLiteMemoryStore (sqlite3) | AsyncSQLiteMemoryStore (aiosqlite) |
| Facade | Memory (9 方法) | AsyncMemory (9 方法) |
| 权限 | check_memory_access_permissions | async_check_memory_access_permissions |
| 日志 | record_access | async_record_access |
| REST API | — | 9 核心端点用 AsyncMemory + 12 实验端点保持 sync |

embedder/LLM 等 sync 组件用 `asyncio.to_thread` 包装，store 层真 async。

## 配置

### 环境变量

| 变量 | 默认 | 说明 |
|------|------|------|
| `SEPTMUSE_DB_PATH` | `~/.septmuse/septmuse.db` | SQLite 路径 |
| `SEPTMUSE_EMBEDDER` | `hash` | hash / onnx / onnx-zh / auto / st |
| `SEPTMUSE_LLM` | 未设 | openai / ollama / anthropic / dashscope / litellm / groq / gemini / deepseek |
| `SEPTMUSE_LLM_MODEL` | 未设 | 覆盖 provider 默认模型 |
| `SEPTMUSE_RERANKER` | `noop` | noop / mmr / cross_encoder / llm |
| `SEPTMUSE_VECTOR_BACKEND` | `sqlite` | sqlite / pgvector / chroma / qdrant |
| `SEPTMUSE_KEYWORD_BACKEND` | `sqlite_bm25` | sqlite_bm25 / rank_bm25 / none |
| `SEPTMUSE_GRAPH_BACKEND` | `sqlite` | sqlite / age / neo4j |
| `SEPTMUSE_ENTITY_EXTRACTOR` | `regex` | regex / spacy / none |
| `SEPTMUSE_INFER` | `false` | true 启用 LLM 抽取事实 |
| `SEPTMUSE_API_KEY` | 未设 | 未设=开发模式；已设=生产模式（401 认证） |
| `SEPTMUSE_MODEL_CACHE` | `~/.septmuse/models/` | ONNX 模型缓存目录 |

### YAML 配置

```yaml
# ~/.septmuse/config.yaml
database:
  db_path: ~/.septmuse/septmuse.db
embedder:
  backend: onnx
llm:
  backend: litellm
  model: groq/llama-3.1-70b-versatile
reranker:
  backend: mmr
```

```python
from septmuse import Memory, MemoryConfig

config = MemoryConfig(_yaml_file="config.yaml")
memory = Memory(config=config)
```

## 测试

```bash
# PowerShell
$env:PYTHONPATH = "src"
python -m pytest tests/unit/ tests/e2e/ -q

# 全量：1076 passed + 36 skipped + 23 failed（预存在）
```

| 层级 | 测试数 | 覆盖 |
|------|--------|------|
| unit | 1058+ | store / facade / REST / CLI / MCP / 权限 / 迁移 / async / LLM provider / 混合检索 |
| e2e | 32 | 跨会话偏好召回 + 用户隔离 + 认知分层 + 跨 agent 共享 |
| skipped | 36 | PGVector / AGE / Neo4j / ONNX / spaCy / torch（需 extras） |

## License

[Apache License 2.0](./LICENSE) — Copyright 2026 The sonhhxg0529 Authors. All Rights Reserved.

# OpenAI Embedder + PGVector Store 设计规格

- **日期**: 2026-07-20
- **状态**: 已批准（用户确认）
- **范围**: 阶段6 可选生产后端补完 —— OpenAI 嵌入 provider + Postgres/pgvector 向量后端
- **关联文档**: `docs/specs/agent-memory-architecture.md` §3 §6 §12, `docs/specs/package-structure.md` §12.2 §12.6

---

## 1. 背景

SeptMuse 阶段1-5 已完成 387 测试用例全部通过，核心功能（三维正交架构 + 3 创新空白 + 横切关注点 + MemOS 编排 + REST API）100% 实现并验证。零配置默认后端（SQLite + sentence-transformers + HashEmbedder 回退）满足离线/测试场景。

生产场景需替换为：
- **OpenAI 嵌入**（text-embedding-3-small）—— 高质量语义检索，零本地模型加载
- **pgvector 向量后端**（Postgres 扩展）—— 复用 solarseptem 平台主库 Postgres，减少依赖

架构文档 §12.2、§12.6 已预留位置；`pyproject.toml` 已声明 `openai = ["openai>=1.30"]` 与 `postgres = ["psycopg2-binary>=2.9", "pgvector>=0.3"]` 可选依赖。本次补完这两个 provider 的实现代码。

## 2. 目标

1. **新增** `providers/embedders/openai.py` —— 实现 `Embedder` ABC，调用 OpenAI Embeddings API
2. **新增** `storage/base.py` —— `MemoryStore` 抽象基类，定义存储后端契约
3. **新增** `storage/vector/pgvector.py` —— 实现 `MemoryStore` ABC，Postgres + pgvector 向量后端
4. **重构** 6 个模块的 `SQLiteMemoryStore` 类型注解为 `MemoryStore`（行为零变更；zettel/dream/user_id 保持 SQLiteMemoryStore，因直接访问 SQLite 内部属性）
5. **测试** —— 新增 OpenAI embedder 单元测试（mock client）+ pgvector store 单元测试（psycopg 不可用时 skipif）
6. **回归** —— 现有 387 测试全绿，ruff check 0 错误

## 3. 非目标

- 不实现 Azure OpenAI / Vertex AI / Gemini 等其他嵌入 provider
- 不实现 Qdrant / Milvus / Pinecone 等其他向量后端
- 不修改 `MemoryConfig` 或 `configs/defaults.py` 的默认后端选择逻辑（保持 SQLite 默认）
- 不修改 REST API 端点签名
- 不触及激活/参数化记忆（KVCache/LoRA）

## 4. 设计

### 4.1 OpenAI Embedder

**文件**: `src/septmuse/providers/embedders/openai.py`

**继承**: `septmuse.providers.embedders.base.Embedder`（已存在，定义 `dimension` + `embed` + `embed_batch`）

**借鉴源**: `opensource/mem0/mem0/embeddings/openai.py` 的 `OpenAIEmbedding` 类

**类签名**:

```python
class OpenAIEmbedder(Embedder):
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "text-embedding-3-small",
        base_url: str | None = None,
        embedding_dims: int | None = None,
        **kwargs: Any,
    ) -> None: ...
```

**行为细节**:

| 项 | 设计 | 借鉴 mem0 |
|---|---|---|
| 默认模型 | `text-embedding-3-small` | ✅ 同 |
| 默认维度 | 1536（text-embedding-3-small 原生维度） | ✅ 同 |
| 零配置 key | `api_key or os.getenv("OPENAI_API_KEY")`，缺失抛 `ValueError` | ✅ 同 |
| 零配置 base_url | `base_url or os.getenv("OPENAI_BASE_URL")`，未设则用 OpenAI 默认 | ✅ 同 |
| matryoshka 检测 | 仅当 `embedding_dims is not None` 时向 API 传 `dimensions` 参数（兼容 vLLM/Voyage 等非 matryoshka 后端） | ✅ 同 |
| `embed(text)` | `text.replace("\n", " ")` → `client.embeddings.create(input=[text], model=..., encoding_format="float")` → `.data[0].embedding` | ✅ 同 |
| `embed_batch(texts)` | 100 一批分块 → 批量 API → 按 `.index` 排序 → 数量校验 | ✅ 同 |
| `dimension` 属性 | `self._dim`（构造时记录 `embedding_dims or 1536`） | ✅ 同 |
| 依赖缺失 | `ImportError` 提示 `pip install septmuse[openai]` | ✅ 同（对齐 `providers/llms/openai.py` 既有模式） |
| 日志 | `structlog` `embedder_loading` / `embedder_ready`（对齐 `sentence_transformers.py`） | — |

**关键差异（与 mem0）**:

- SeptMuse 的 `Embedder` ABC 用 `list[float]` 返回（非 numpy），不传 `memory_action` 参数（SeptMuse 不区分 add/search 嵌入）
- SeptMuse 用 `structlog`（非 stdlib `logging`）
- SeptMuse 维度属性是 `@property`（mem0 是 `self.config.embedding_dims`）

### 4.2 MemoryStore ABC

**文件**: `src/septmuse/storage/base.py`（新建）

**目的**: 定义存储后端契约，让 SQLiteMemoryStore 和 PGVectorStore 都实现同一接口，实现可插拔。

**借鉴源**: mem0 `vector_stores/base.py` 的 `VectorStoreBase`（抽象方法集合），但 SeptMuse 的接口契约是 `SQLiteMemoryStore` 已有的方法签名（不照搬 mem0 的 `create_col`/`insert`/`list_cols` 等集合语义——SeptMuse 是单表语义）。

**ABC 定义**:

```python
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any


class MemoryStore(ABC):
    """记忆存储后端抽象基类。

    所有存储后端（SQLiteMemoryStore / PGVectorStore / 未来 Qdrant 等）
    实现此接口，保证 capture/retrieval/evolution 等横切关注点可插拔。

    方法签名严格对齐 SQLiteMemoryStore 既有实现，不破坏现有行为。
    """

    @abstractmethod
    def add(
        self,
        content: str,
        embedding: list[float],
        *,
        user_id: str,
        agent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """添加记忆，返回 memory_id。"""
        ...

    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        *,
        user_id: str,
        top_k: int = 5,
        threshold: float = 0.1,
    ) -> list[dict[str, Any]]:
        """向量检索，返回 [{"id", "memory", "score", "metadata", "created_at"}]。

        score 含义：相似度（越高越相似，范围 [0, 1]）。
        - SQLiteMemoryStore: numpy 余弦点积（embedder 已归一化）
        - PGVectorStore: max(0, 1 - cosine_distance)
        """
        ...

    @abstractmethod
    def get_all(self, *, user_id: str) -> list[dict[str, Any]]:
        """列出该用户全部未删除记忆。"""
        ...

    @abstractmethod
    def get(self, memory_id: str) -> dict[str, Any] | None:
        """取单条，不存在返回 None。"""
        ...

    @abstractmethod
    def delete(self, memory_id: str) -> None:
        """软删除（标记 is_deleted + history 记录）。"""
        ...

    @abstractmethod
    def close(self) -> None:
        """释放连接资源。"""
        ...
```

**SQLiteMemoryStore 改动**:

`storage/sqlite/store.py` 第 53 行：
```python
# 前
class SQLiteMemoryStore:
# 后
class SQLiteMemoryStore(MemoryStore):
```

加 `from septmuse.storage.base import MemoryStore` 导入。**行为零变更**（已满足契约）。

### 4.3 PGVectorStore

**文件**: `src/septmuse/storage/vector/pgvector.py`（新建）

**继承**: `septmuse.storage.base.MemoryStore`

**借鉴源**: `opensource/mem0/mem0/vector_stores/pgvector.py` 的 `PGVector` 类

**类签名**:

```python
class PGVectorStore(MemoryStore):
    def __init__(
        self,
        *,
        collection_name: str = "memories",
        embedding_model_dims: int = 1536,
        dbname: str | None = None,
        user: str | None = None,
        password: str | None = None,
        host: str = "localhost",
        port: int = 5432,
        connection_string: str | None = None,
        minconn: int = 1,
        maxconn: int = 5,
        sslmode: str | None = None,
    ) -> None: ...
```

**行为细节**:

| 项 | 设计 | 借鉴 mem0 |
|---|---|---|
| 驱动优先级 | psycopg3 优先，psycopg2 回退，都不可用抛 `ImportError` 提示 `pip install septmuse[postgres]` | ✅ 同（但 mem0 抛 psycopg2/psycopg 都缺） |
| 连接池 | `ConnectionPool`（psycopg3 `psycopg_pool.ConnectionPool` / psycopg2 `ThreadedConnectionPool`） | ✅ 同 |
| 连接字符串 | `connection_string` 优先；否则拼 `postgresql://{user}:{password}@{host}:{port}/{dbname}`；`sslmode` 注入 | ✅ 同（`_with_sslmode`） |
| 建表 | `CREATE EXTENSION IF NOT EXISTS vector` + `memories` 表（id TEXT PK, user_id, content, embedding `vector(N)`, metadata JSONB, created_at, updated_at, is_deleted）+ `history` 表（对齐 SQLiteMemoryStore 字段）+ `idx_memories_user` 索引 | 🔄 改造：mem0 用 UUID PK + 单表，SeptMuse 用 TEXT PK + 双表（对齐 SQLiteMemoryStore） |
| `_ensure_collection` | 首次操作时建表（lazy），`_collection_ensured` 标志位避免重复 | ✅ 同 |
| `_get_cursor` | contextmanager，psycopg3 自动 commit/rollback，psycopg2 手动 getconn/putconn | ✅ 同 |
| `add` | INSERT memories + INSERT history（event="ADD"），返回 `mem-{uuid4()}` | 🔄 改造：对齐 SQLiteMemoryStore 的 `mem-` 前缀 + history 表结构 |
| `search` | `SELECT id, content, embedding, metadata, created_at FROM memories WHERE user_id=? AND is_deleted=0` → `vector <=> %s::vector AS distance` → `score = max(0, 1 - distance)` → 按 score 降序 top_k + threshold 过滤 | 🔄 改造：mem0 用 `ORDER BY distance LIMIT top_k`，SeptMuse 加 threshold 过滤（对齐 SQLiteMemoryStore） |
| `get_all` | `SELECT id, content, metadata, created_at, updated_at FROM memories WHERE user_id=? AND is_deleted=0` | ✅ 同（字段对齐 SQLiteMemoryStore） |
| `get` | 单条查询，不存在返回 None | ✅ 同 |
| `delete` | 软删除：`UPDATE memories SET is_deleted=1, updated_at=?` + `INSERT history (event="DELETE")` | 🔄 改造：mem0 物理删除，SeptMuse 软删除（对齐 SQLiteMemoryStore） |
| `close` | 关闭连接池（psycopg3 `pool.close()` / psycopg2 `pool.closeall()`） | ✅ 同 |
| `__del__` | `contextlib.suppress(Exception)` 包裹 close | ✅ 同 |
| 日志 | `structlog` `pgvector_store_ready` / `memory_added` / `memory_deleted`（对齐 SQLiteMemoryStore） | — |

**关键差异（与 mem0）**:

- SeptMuse 不用 `sql.Identifier` 拼表名（collection_name 固定为 "memories"，简化 SQL）
- SeptMuse 用 `mem-{uuid4()}` ID 前缀（对齐 SQLiteMemoryStore，mem0 用纯 UUID）
- SeptMuse `history` 表字段对齐 SQLiteMemoryStore（`id/memory_id/old_memory/new_memory/event/created_at/is_deleted`）
- SeptMuse `search` 加 `threshold` 过滤（mem0 只 top_k）
- SeptMuse `delete` 软删除（mem0 物理删除）
- SeptMuse 不实现 `keyword_search` / `update` / `list_cols` / `delete_col` / `col_info` / `list` / `reset`（不在 MemoryStore ABC 中，YAGNI）

### 4.4 重构 6 个模块 type hints

**目的**: 把 `SQLiteMemoryStore` 类型注解改为 `MemoryStore`（ABC），让横切关注点接受任何 `MemoryStore` 实现。

**原则**: 只改 type hints，**行为零变更**。SQLiteMemoryStore 仍是默认实现。

**涉及文件**（6 个，不直接访问 SQLite 内部 `_lock`/`conn` 的模块）:

| 文件 | 改动 |
|---|---|
| `concerns/capture/pipeline.py` | `store: SQLiteMemoryStore` → `store: MemoryStore` |
| `concerns/retrieval/hybrid.py` | 同上 |
| `concerns/retrieval/progressive.py` | 同上 |
| `orchestration/memory.py` | `store: SQLiteMemoryStore \| None` → `store: MemoryStore \| None`；`self.store` 类型注解 |
| `content_types/semantic/extract.py` | `verbatim_store: SQLiteMemoryStore \| None` → `verbatim_store: MemoryStore \| None` |
| `concerns/metacognition/coverage.py` | `store: SQLiteMemoryStore` → `store: MemoryStore` |

**保持 `SQLiteMemoryStore` 的模块**（3 个，直接访问 SQLite 内部 `_lock`/`conn` 管理 `memory_links` 等独立表）:

| 文件 | 原因 |
|---|---|
| `concerns/evolution/zettel.py` | 直接访问 `store._lock` 和 `store.conn` 管理 `memory_links` 表（SQLite 专有） |
| `concerns/evolution/dream.py` | 同上（Dream 整合批量建链接） |
| `concerns/sharing/user_id.py` | 同上（跨 agent 共享查询） |

这 3 个模块是 SQLite 专有功能，PGVectorStore 不支持（需用 Postgres 原生图查询实现，未来方向）。

**不涉及**（用 TypedMemoryStore 的，不在此重构范围）:
- `storage/typed_store.py`、`content_types/semantic/fact.py`、`content_types/episodic/`、`content_types/procedural/`、`concerns/retrieval/causal.py`、`concerns/retrieval/forgetting.py`、`concerns/evolution/reflect.py`、`api/rest/__init__.py`（通过 `Memory` facade 间接使用，不直接 import store）

## 5. 数据流

### 5.1 OpenAI 嵌入数据流

```
用户调用 Memory(embedder=OpenAIEmbedder())
  → capture_pipeline.capture(content)
    → OpenAIEmbedder.embed(content)
      → openai.OpenAI.embeddings.create(model, input=[text])
      → 返回 list[float]（1536 维）
    → store.add(content, embedding, user_id=...)
```

### 5.2 PGVector 检索数据流

```
用户调用 Memory(store=PGVectorStore(connection_string="..."))
  → memory.search("query", user_id="alice")
    → OpenAIEmbedder.embed("query")  # 或任意 Embedder
    → PGVectorStore.search(query_embedding, user_id="alice", top_k=5)
      → SQL: SELECT ... FROM memories WHERE user_id='alice' AND is_deleted=0
      → ORDER BY (vector <=> query::vector) ASC
      → score = max(0, 1 - distance)
      → threshold 过滤 + top_k
    → 返回 [{"id", "memory", "score", "metadata", "created_at"}]
```

## 6. 错误处理

| 场景 | 处理 |
|---|---|
| `openai` 包未安装 | `ImportError("openai package required: pip install septmuse[openai]")`（对齐 `providers/llms/openai.py`） |
| `OPENAI_API_KEY` 未设 | `ValueError("OPENAI_API_KEY not set: pass api_key or set env var")`（对齐 `providers/llms/openai.py`） |
| OpenAI API 调用失败 | 捕获 `Exception`，`logger.error` 记录后 re-raise（对齐 `providers/llms/openai.py`） |
| `embed_batch` 数量不一致 | `ValueError`（对齐 mem0） |
| `psycopg` / `psycopg2` 都未安装 | `ImportError("psycopg/psycopg2 required: pip install septmuse[postgres]")` |
| Postgres 连接失败 | psycopg3 `ConnectionPool.open(wait=False)` 不阻塞；首次操作时 `_get_cursor` 抛异常，由调用方处理 |
| pgvector 扩展未安装 | `CREATE EXTENSION vector` 失败，异常向上传播（需 DBA 预装） |

## 7. 测试策略

### 7.1 OpenAI Embedder 测试

**文件**: `tests/unit/test_openai_embedder.py`

| 测试 | 方法 |
|---|---|
| `test_embed_returns_normalized_vector` | Mock `openai.OpenAI`，验证 `embed("text")` 返回 list[float]，长度 = dimension |
| `test_embed_replaces_newlines` | Mock client，验证传入 `input` 的文本 `\n` 被替换为空格 |
| `test_embed_batch_chunks_100` | 150 条文本 → 验证 2 次 API 调用（100 + 50） |
| `test_embed_batch_sorts_by_index` | Mock 返回乱序 `.data`，验证按 index 排序 |
| `test_embed_batch_count_mismatch` | Mock 返回数量 ≠ 输入 → `ValueError` |
| `test_dimension_property` | 验证 `dimension` 返回构造时设置的值 |
| `test_matryoshka_passes_dimensions` | `embedding_dims=256` → 验证 API 收到 `dimensions=256` |
| `test_non_matryoshka_omits_dimensions` | `embedding_dims=None` → 验证 API 未收到 `dimensions` |
| `test_zero_config_reads_env_key` | `monkeypatch.setenv("OPENAI_API_KEY", "sk-test")` → 构造成功 |
| `test_missing_key_raises` | `monkeypatch.delenv("OPENAI_API_KEY")` + `api_key=None` → `ValueError` |
| `test_import_error_without_openai` | Mock `import openai` 失败 → `ImportError` 提示 `pip install septmuse[openai]` |
| `test_base_url_env` | `monkeypatch.setenv("OPENAI_BASE_URL", "https://custom")` → 验证传给 client |

### 7.2 PGVector Store 测试

**文件**: `tests/unit/test_pgvector_store.py`

| 测试 | 方法 |
|---|---|
| `test_import_error_without_psycopg` | Mock psycopg/psycopg2 都不可用 → `ImportError` 提示 `pip install septmuse[postgres]` |
| `test_connection_string_priority` | `connection_string=` 优先于 user/password/host 拼接 |
| `test_sslmode_injection` | `sslmode="require"` → 验证连接字符串含 `sslmode=require` |

**集成测试（`@pytest.mark.skipif(not HAS_POSTGRES)`）**:

| 测试 | 方法 |
|---|---|
| `test_add_and_search` | add 3 条 → search → 验证 top_k 顺序 + score |
| `test_search_threshold` | 低相似度被 threshold 过滤 |
| `test_get_all_by_user` | user_id 隔离 |
| `test_get_returns_none_for_missing` | 不存在的 id → None |
| `test_delete_is_soft` | delete 后 get_all 不返回，但 get 仍可查（软删除） |
| `test_history_recorded` | add/delete 后 history 表有记录 |
| `test_close_releases_pool` | close 后再操作抛异常 |

`HAS_POSTGRES` 检测：`pytest.importorskip("psycopg")` 或环境变量 `SEPTMUSE_TEST_PG_DSN`。

### 7.3 回归测试

- 现有 387 测试全绿（SQLiteMemoryStore 加 `(MemoryStore)` 继承不破坏行为）
- 6 个模块 type hints 改 `MemoryStore` 后，SQLiteMemoryStore 仍是运行时实例，行为不变
- ruff check 0 错误

## 8. 依赖变更

**无新增依赖**。`pyproject.toml` 已有：

```toml
[project.optional-dependencies]
openai = ["openai>=1.30"]
postgres = ["psycopg2-binary>=2.9", "pgvector>=0.3"]
```

**注意**: mem0 用 psycopg3 优先 + psycopg2 回退。SeptMuse 的 `postgres` extra 当前只声明 `psycopg2-binary`。为实现 psycopg3 优先逻辑，需在 `postgres` extra 加 `psycopg[binary,pool]`。

**变更**:

```toml
# 前
postgres = ["psycopg2-binary>=2.9", "pgvector>=0.3"]

# 后
postgres = ["psycopg[binary,pool]>=3.1", "psycopg2-binary>=2.9", "pgvector>=0.3"]
```

psycopg3 是现代驱动（mem0 优先用），psycopg2-binary 作为回退兼容旧环境。

## 9. 验收标准

- [ ] `src/septmuse/providers/embedders/openai.py` 实现 `Embedder` ABC
- [ ] `src/septmuse/storage/base.py` 定义 `MemoryStore` ABC
- [ ] `src/septmuse/storage/vector/pgvector.py` 实现 `MemoryStore` ABC
- [ ] `src/septmuse/storage/sqlite/store.py` 的 `SQLiteMemoryStore` 继承 `MemoryStore`
- [ ] 6 个模块 type hints 改为 `MemoryStore`（zettel/dream/user_id 保持 SQLiteMemoryStore）
- [ ] `tests/unit/test_openai_embedder.py` 12 测试用例
- [ ] `tests/unit/test_pgvector_store.py` 3 单元 + 7 集成测试（集成测试 skipif）
- [ ] `pyproject.toml` `postgres` extra 加 `psycopg[binary,pool]`
- [ ] `ruff check` 0 错误
- [ ] `pytest` 现有 387 + 新增 ~15 测试全绿
- [ ] 所有新文件含 Apache 2.0 license header

## 10. 文件清单

**新建**:
- `src/septmuse/providers/embedders/openai.py`
- `src/septmuse/storage/base.py`
- `src/septmuse/storage/vector/pgvector.py`
- `tests/unit/test_openai_embedder.py`
- `tests/unit/test_pgvector_store.py`

**修改**:
- `src/septmuse/storage/sqlite/store.py`（加继承）
- `src/septmuse/concerns/capture/pipeline.py`（type hint）
- `src/septmuse/concerns/retrieval/hybrid.py`（type hint）
- `src/septmuse/concerns/retrieval/progressive.py`（type hint）
- `src/septmuse/concerns/evolution/zettel.py`（type hint）
- `src/septmuse/concerns/evolution/dream.py`（type hint）
- `src/septmuse/concerns/sharing/user_id.py`（type hint）
- `src/septmuse/orchestration/memory.py`（type hint）
- `src/septmuse/content_types/semantic/extract.py`（type hint)
- `src/septmuse/concerns/metacognition/coverage.py`（type hint）
- `pyproject.toml`（postgres extra 加 psycopg3）

## 11. 风险与缓解

| 风险 | 缓解 |
|---|---|
| psycopg3 在 Windows 安装问题 | `psycopg[binary]` 用预编译 wheel；psycopg2-binary 回退 |
| pgvector 扩展未预装 | 文档明确要求 DBA 预装；`CREATE EXTENSION` 失败时异常清晰 |
| 6 模块重构破坏行为 | 只改 type hints，SQLiteMemoryStore 仍是运行时实例；387 测试回归保护 |
| OpenAI API 速率限制 | `embed_batch` 100 一批（对齐 mem0，OpenAI 单次上限） |
| matryoshka 维度参数误传 | 检测 `embedding_dims is not None` 才传，兼容非 matryoshka 后端 |

## 12. 后续可选方向

本次完成后，剩余可选 provider（架构文档 §9 标注"可选"）：
- `storage/graph/age.py`（Apache AGE 图后端）
- `storage/activation.py`（KVCache 激活记忆）
- `storage/parametric/lora.py`（LoRA 参数化记忆）
- `providers/embedders/azure_openai.py` / `vertexai.py` / `gemini.py`（其他嵌入 provider）
- `providers/llms/anthropic.py` / `ollama.py`（其他 LLM provider）

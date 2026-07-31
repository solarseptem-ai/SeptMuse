# 设计规格：AsyncMemory + litellm 云 Provider

> 日期：2026-07-31
> 状态：待实施
> 范围：新建 AsyncMemory 异步记忆类（9 方法）+ AsyncMemoryStore 异步存储层 + litellm/groq/gemini/deepseek 4 个 LLM 后端

---

## 1. 背景与动机

SeptMuse 功能缺口分析发现两个关键短板：

1. **async/sync 双版本**：行业主流记忆系统提供 async/sync 双版本 API（add/delete/get_all/search/update 等），SeptMuse 的 Memory facade 全 sync（0 个 async 方法）。FastAPI 的 21 个 async 端点直接调 sync Memory，阻塞事件循环。
2. **LLM provider 覆盖**：行业主流记忆系统支持 20+ LLM provider，SeptMuse 只有 4 个（openai/ollama/anthropic/dashscope），缺 groq/gemini/deepseek 等热门 provider。

### 设计目标

- **AsyncMemory 新类**：不修改现有 Memory，新建独立异步 facade，9 个核心 async 方法，提供 async/sync 双版本 API。
- **AsyncMemoryStore**：新建异步存储 ABC + aiosqlite 实现，真正异步的存储层。
- **混合异步策略**：store 层真 async（aiosqlite），embedder/LLM 层用 `asyncio.to_thread` 包装（I/O 等待释放 GIL）。
- **litellm 统一入口**：一个依赖覆盖 100+ provider。
- **逐个云 provider**：groq/gemini/deepseek 各自独立类，用户可选 litellm 或直连。

---

## 2. 设计决策

| 决策点 | 选择 | 理由 |
|--------|------|------|
| async 实现 | 新建 AsyncMemory 类 | 不污染 Memory，职责分离，对齐已有继承模式 |
| 存储层 async | 新建 AsyncMemoryStore ABC + aiosqlite | 真正异步的 SQLite，不阻塞事件循环 |
| embedder/LLM async | asyncio.to_thread 包装 | I/O 等待释放 GIL，不改 Embedder/LLM ABC |
| async 方法范围 | 核心 9 个 | add/search/update/delete/delete_all/get/get_all/get_history/close |
| litellm | 加 LitellmLLM + manifest 条目 | 一个依赖覆盖 100+ provider |
| 云 provider | groq/gemini/deepseek 独立类 | 用户可选 litellm 或直连 |

---

## 3. 架构概览

```
┌─ sync 路径（不动）─────────────────────────────┐
│  MemoryStore (ABC, sync)                        │
│    └─ SQLiteMemoryStore (sqlite3)               │
│  Memory (sync facade)                           │
│    └─ ExperimentalMemory(Memory) (+49 方法)     │
└─────────────────────────────────────────────────┘

┌─ async 路径（新建）────────────────────────────┐
│  AsyncMemoryStore (ABC, async)                 │
│    └─ AsyncSQLiteMemoryStore (aiosqlite)        │
│  AsyncMemory (async facade, 9 方法)             │
└─────────────────────────────────────────────────┘
        │ 共享
        ▼
┌─ 共享组件（sync，async 内用 to_thread 包装）──┐
│  Embedder (hash/onnx/openai...)                │
│  LLM (openai/ollama/litellm/groq/gemini/...)   │
│  EntityExtractor / Reranker                    │
└─────────────────────────────────────────────────┘
```

### 数据流（async add 示例）

```
AsyncMemory.add(messages, user_id)
  → await asyncio.to_thread(embedder.embed_batch, texts)   # embedder sync，to_thread 包装
  → await self.store.add(text, emb, user_id)               # store 真 async（aiosqlite）
  → await asyncio.to_thread(entity_extractor.extract, text) # extractor sync，to_thread
  → 返回 {"results": [...], "relations": []}
```

---

## 4. AsyncMemoryStore ABC（`storage/async_base.py`）

```python
from abc import ABC, abstractmethod
from typing import Any


class AsyncMemoryStore(ABC):
    """异步记忆存储后端抽象。

    所有方法为 async def，使用 aiosqlite/asyncpg 等异步驱动。
    sync MemoryStore 的对偶，方法签名保持一致。
    """

    @abstractmethod
    async def add(
        self,
        content: str,
        embedding: list[float],
        *,
        user_id: str,
        agent_id: str | None = None,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        valid_at: str | None = None,
    ) -> str:
        """添加记忆，返回 memory_id。"""
        ...

    @abstractmethod
    async def search(
        self,
        query_embedding: list[float],
        *,
        user_id: str,
        session_id: str | None = None,
        top_k: int = 5,
        threshold: float = 0.1,
    ) -> list[dict[str, Any]]:
        """向量检索，返回 [{"id", "memory", "score", ...}]。"""
        ...

    @abstractmethod
    async def get_all(self, *, user_id: str, session_id: str | None = None) -> list[dict[str, Any]]:
        """列出该用户全部未删除记忆。"""
        ...

    @abstractmethod
    async def get(self, memory_id: str) -> dict[str, Any] | None:
        """取单条。"""
        ...

    @abstractmethod
    async def delete(self, memory_id: str) -> None:
        """软删除。"""
        ...

    @abstractmethod
    async def update(
        self,
        memory_id: str,
        content: str,
        embedding: list[float],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """更新记忆。"""
        ...

    @abstractmethod
    async def get_history(self, memory_id: str) -> list[dict[str, Any]]:
        """获取变更历史。"""
        ...

    @abstractmethod
    async def close(self) -> None:
        """释放连接资源。"""
        ...

    # ── 默认实现（子类可覆盖）──

    async def keyword_search(
        self, query: str, *, user_id: str, session_id: str | None = None, top_k: int = 5
    ) -> list[dict[str, Any]]:
        """关键词检索。默认返回空。"""
        return []

    async def hybrid_search(
        self,
        query: str,
        query_embedding: list[float],
        *,
        user_id: str,
        session_id: str | None = None,
        top_k: int = 5,
        alpha: float = 0.5,
    ) -> list[dict[str, Any]]:
        """混合检索（向量 + 关键词 RRF 融合）。"""
        vec = await self.search(query_embedding, user_id=user_id, session_id=session_id, top_k=top_k * 2)
        kw = await self.keyword_search(query, user_id=user_id, session_id=session_id, top_k=top_k * 2)
        return _rrf_fuse(vec, kw, alpha=alpha)[:top_k]

    async def get_access_logs(self, memory_id: str, limit: int = 100) -> list[dict[str, Any]]:
        """查询访问日志。默认返回空。"""
        return []

    async def get_temporal_valid(
        self, reference_time: str, *, user_id: str, session_id: str | None = None
    ) -> list[dict[str, Any]]:
        """查询某时刻为真的记忆。默认返回空。"""
        return []

    async def get_temporal_interval(
        self, start: str, end: str, *, user_id: str, session_id: str | None = None
    ) -> list[dict[str, Any]]:
        """查询时间区间内为真的记忆。默认返回空。"""
        return []

    async def invalidate(self, memory_id: str, *, invalid_at: str | None = None) -> dict[str, Any]:
        """标记事实不再为真。默认不支持。"""
        raise NotImplementedError(f"{type(self).__name__} 不支持 invalidate")
```

### 设计要点

- 所有方法 `async def`，签名与 sync `MemoryStore` 一致（便于对照）。
- 默认实现方法（keyword_search/hybrid_search/get_access_logs 等）也是 async，子类按需覆盖。
- `_rrf_fuse` 函数复用 sync 版（纯计算，无 I/O，不需 async）。

---

## 5. AsyncSQLiteMemoryStore（`storage/async_sqlite/store.py`）

用 `aiosqlite` 替代 `sqlite3`，所有方法真正异步。

### 表结构（与 sync SQLiteMemoryStore 一致）

复用 sync 版的 `_create_tables` 逻辑，但用 aiosqlite 执行 DDL。

### 核心方法

```python
import aiosqlite

class AsyncSQLiteMemoryStore(AsyncMemoryStore):
    """异步 SQLite 记忆存储（aiosqlite）。"""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._db_path = str(db_path) if db_path else str(Path.home() / ".septmuse" / "septmuse.db")
        self._conn: aiosqlite.Connection | None = None  # 延迟初始化

    async def _ensure_conn(self) -> aiosqlite.Connection:
        """延迟打开连接（首次操作时）。"""
        if self._conn is None:
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = aiosqlite.connect(self._db_path)
            await self._conn.__aenter__()
            await self._create_tables()
        return self._conn

    async def _create_tables(self) -> None:
        """建表（与 sync 版 DDL 一致）。"""
        assert self._conn is not None
        await self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS memories (...);
            CREATE TABLE IF NOT EXISTS history (...);
            ...
        """)
        await self._conn.commit()

    async def add(self, content, embedding, *, user_id, agent_id=None, session_id=None,
                  metadata=None, valid_at=None) -> str:
        conn = await self._ensure_conn()
        mid = f"mem-{uuid.uuid4()}"
        now = _utcnow_iso()
        await conn.execute(
            "INSERT INTO memories (id, user_id, ...) VALUES (?, ?, ...)",
            (mid, user_id, ...),
        )
        await conn.execute("INSERT INTO history (...) VALUES (...)", ...)
        await conn.commit()
        # 双写 vector_store + keyword_index（用 asyncio.to_thread 包装 sync 组件）
        await asyncio.to_thread(self._vector_store.insert_vectors, [embedding], [mid])
        await asyncio.to_thread(self._keyword_index.add_docs, {mid: content})
        return mid

    async def search(self, query_embedding, *, user_id, session_id=None, top_k=5, threshold=0.1):
        conn = await self._ensure_conn()
        # SQLite 查询（async）
        cursor = await conn.execute(
            "SELECT id, content, metadata, created_at, embedding FROM memories WHERE ...",
            (...,),
        )
        rows = await cursor.fetchall()
        # numpy 余弦计算（CPU 密集，用 to_thread）
        scored = await asyncio.to_thread(self._score_rows, query_embedding, rows)
        return [r for r in scored if r["score"] >= threshold][:top_k]

    # ... async_get_all / async_get / async_delete / async_update / async_get_history / async_close
```

### 设计要点

- **延迟初始化**：`_ensure_conn()` 在首次操作时打开 aiosqlite 连接。
- **双写组件**：vector_store（SQLiteVectorStore sync）和 keyword_index（SQLiteBM25Index sync）用 `asyncio.to_thread` 包装。
- **numpy 计算用 to_thread**：余弦相似度计算是 CPU 密集型，不阻塞事件循环。
- **表结构与 sync 版一致**：同一个 DB 文件，sync 和 async 可共享数据。

---

## 6. AsyncMemory facade（`memory/async_main.py`）

```python
import asyncio
from typing import Any

from septmuse.configs.base import MemoryConfig
from septmuse.configs.defaults import default_config
from septmuse.core.logging import get_logger
from septmuse.embedders.base import Embedder
from septmuse.llms.base import LLM
from septmuse.storage.async_base import AsyncMemoryStore
from septmuse.storage.async_sqlite.store import AsyncSQLiteMemoryStore
from septmuse.services.providers import embedder_provider, llm_provider

logger = get_logger(__name__)


class AsyncMemory:
    """异步记忆 facade — 9 个 async 方法，提供 async/sync 双版本 API。

    用法:
        mem = AsyncMemory()
        result = await mem.add("hello", user_id="alice")
        results = await mem.search("hello", user_id="alice")

    REST API 用 AsyncMemory，CLI/MCP 用 Memory（sync）。
    """

    def __init__(
        self,
        config: MemoryConfig | None = None,
        *,
        embedder: Embedder | None = None,
        store: AsyncMemoryStore | None = None,
        llm: LLM | None = None,
    ) -> None:
        self.config = config or default_config()
        self.embedder = embedder or embedder_provider.resolve(
            self.config.embedder.backend, config=self.config.embedder
        )
        self.store = store or AsyncSQLiteMemoryStore(db_path=self.config.db_path)
        self.llm = llm
        if self.llm is None and self.config.llm is not None:
            self.llm = llm_provider.resolve(self.config.llm.backend, config=self.config.llm)
        logger.info("async_memory_init", db_path=str(self.config.db_path))

    async def add(self, messages, *, user_id, agent_id=None, session_id=None,
                  metadata=None, infer=None, valid_at=None,
                  auto_extract_entities=True) -> dict[str, Any]:
        """异步添加记忆。"""
        texts = _normalize_messages(messages)
        if not texts:
            return {"results": [], "relations": []}

        # embedder sync，用 to_thread 包装
        embeddings = await asyncio.to_thread(self.embedder.embed_batch, texts)

        results = []
        for text, emb in zip(texts, embeddings, strict=True):
            # store 真 async
            mid = await self.store.add(
                text, emb, user_id=user_id, agent_id=agent_id,
                session_id=session_id, metadata=metadata, valid_at=valid_at,
            )
            results.append({"id": mid, "memory": text, "event": "ADD"})

        logger.info("async_memory_add", user_id=user_id, count=len(results))
        return {"results": results, "relations": []}

    async def search(self, query: str, *, user_id: str, top_k: int = 5,
                     threshold: float = 0.1) -> list[dict[str, Any]]:
        """异步检索记忆。"""
        emb = await asyncio.to_thread(self.embedder.embed, query)
        return await self.store.search(emb, user_id=user_id, top_k=top_k, threshold=threshold)

    async def update(self, memory_id: str, content: str, *,
                     metadata: dict[str, Any] | None = None) -> bool:
        """异步更新记忆。"""
        emb = await asyncio.to_thread(self.embedder.embed, content)
        return await self.store.update(memory_id, content, emb, metadata=metadata)

    async def delete(self, memory_id: str) -> None:
        """异步软删除。"""
        await self.store.delete(memory_id)

    async def delete_all(self, *, user_id: str) -> int:
        """异步批量删除该用户所有记忆。"""
        memories = await self.store.get_all(user_id=user_id)
        for m in memories:
            await self.store.delete(m["id"])
        return len(memories)

    async def get(self, memory_id: str) -> dict[str, Any] | None:
        """异步取单条。"""
        return await self.store.get(memory_id)

    async def get_all(self, *, user_id: str) -> list[dict[str, Any]]:
        """异步列出全部。"""
        return await self.store.get_all(user_id=user_id)

    async def get_history(self, memory_id: str) -> list[dict[str, Any]]:
        """异步获取变更历史。"""
        return await self.store.get_history(memory_id)

    async def close(self) -> None:
        """异步释放资源。"""
        await self.store.close()
```

### 设计要点

- **不继承 Memory**：独立类，不引入 sync 方法的污染。
- **embedder/LLM 用 to_thread**：这些 ABC 是 sync 的，to_thread 释放 GIL。
- **store 真 async**：await self.store.add/search/... 直接调用 aiosqlite。
- **方法签名简化**：不支持 typed memory（fact/episode/rule）和 infer 模式——这些留后续。核心是 verbatim 模式的 async 对偶。

---

## 7. REST API 切换

`create_app` 改为用 `AsyncMemory`：

```python
def create_app(memory: AsyncMemory | MemoryConfig | None = None) -> FastAPI:
    if memory is None:
        memory = AsyncMemory(config=MemoryConfig(db_path=":memory:"), embedder=HashEmbedder())
    elif isinstance(memory, MemoryConfig):
        memory = AsyncMemory(config=memory, embedder=HashEmbedder())

    app = FastAPI(title="SeptMuse Memory API", ...)
    register_routes(app, memory)
    return app
```

21 个端点改为 `await memory.add(...)` / `await memory.search(...)` 等。

### 向后兼容

- `create_app` 仍接受 `MemoryConfig`（自动创建 AsyncMemory）。
- 如果传入 `Memory`（sync）实例，包装为 `AsyncMemory`（to_thread 包装所有调用）——或报错提示用 AsyncMemory。

---

## 8. litellm LLM Provider

### `llms/litellm.py`

```python
from septmuse.llms.base import LLM

class LitellmLLM(LLM):
    """litellm 统一 LLM 代理 — 一个依赖覆盖 100+ provider。

    用法:
        llm = LitellmLLM(model="groq/llama-3.1-70b-versatile", api_key="...")
        result = llm.complete(system_prompt, user_prompt)
    """

    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None,
                 base_url: str | None = None, **kwargs) -> None:
        import litellm
        self._litellm = litellm
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self._extra_kwargs = kwargs

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """委托 litellm.completion。"""
        response = self._litellm.completion(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            api_key=self.api_key,
            api_base=self.base_url,
            **self._extra_kwargs,
        )
        return response.choices[0].message.content
```

### `configs/llms/litellm.py`

```python
from septmuse.configs.llms.base import BaseLLMConfig

class LitellmLLMConfig(BaseLLMConfig):
    """litellm LLM 配置。"""
    backend: str = "litellm"
    model: str = "gpt-4o-mini"
    api_key: str | None = None
    base_url: str | None = None
```

---

## 9. 云 Provider（groq / gemini / deepseek）

### groq

```python
# llms/groq.py
class GroqLLM(LLM):
    """Groq LLM — 超低延迟推理。"""
    def __init__(self, model="llama-3.1-70b-versatile", api_key=None, **kwargs):
        from groq import Groq
        self._client = Groq(api_key=api_key)
        self.model = model

    def complete(self, system_prompt, user_prompt):
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system_prompt},
                       {"role": "user", "content": user_prompt}],
        )
        return response.choices[0].message.content
```

### gemini

```python
# llms/gemini.py
class GeminiLLM(LLM):
    """Google Gemini LLM。"""
    def __init__(self, model="gemini-1.5-flash", api_key=None, **kwargs):
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(model)
        self.model = model

    def complete(self, system_prompt, user_prompt):
        response = self._model.generate_content(f"{system_prompt}\n\n{user_prompt}")
        return response.text
```

### deepseek

```python
# llms/deepseek.py
class DeepSeekLLM(LLM):
    """DeepSeek LLM — OpenAI 兼容 API。"""
    def __init__(self, model="deepseek-chat", api_key=None, base_url="https://api.deepseek.com", **kwargs):
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def complete(self, system_prompt, user_prompt):
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system_prompt},
                       {"role": "user", "content": user_prompt}],
        )
        return response.choices[0].message.content
```

---

## 10. manifest 新增条目

```python
# services/registry.py 的 "llm" 部分新增 4 条
"llm": {
    # 现有 4 个...
    "litellm":   BackendEntry("septmuse.llms.litellm",   "LitellmLLM",   LitellmLLMConfig,   ("litellm",)),
    "groq":      BackendEntry("septmuse.llms.groq",      "GroqLLM",      GroqLLMConfig,      ("groq",)),
    "gemini":    BackendEntry("septmuse.llms.gemini",    "GeminiLLM",    GeminiLLMConfig,    ("google-generativeai",)),
    "deepseek":  BackendEntry("septmuse.llms.deepseek",  "DeepSeekLLM",  DeepSeekLLMConfig,  ("openai",)),
}
```

### pyproject.toml 新增可选依赖

```toml
litellm = ["litellm>=1.40"]
groq = ["groq>=0.11"]
gemini = ["google-generativeai>=0.7"]
deepseek = ["openai>=1.30"]  # OpenAI 兼容，已有 openai extra
```

---

## 11. 测试策略

| 测试类型 | 覆盖点 | 文件 |
|---------|--------|------|
| AsyncMemoryStore ABC | async 方法签名 + 默认实现 | `tests/unit/test_async_store_base.py` |
| AsyncSQLiteMemoryStore | async add/search/get/delete/update/history | `tests/unit/test_async_sqlite_store.py` |
| AsyncMemory facade | 9 个 async 方法 + to_thread 包装 | `tests/unit/test_async_memory.py` |
| REST async 端点 | 21 个端点用 await | `tests/unit/test_async_rest.py` |
| litellm LLM | complete 委托 litellm | `tests/unit/test_litellm_llm.py` |
| groq/gemini/deepseek | 各自 complete | `tests/unit/test_cloud_llms.py` |
| manifest 完整性 | llm 的 8 个后端 | 扩展 `test_manifest.py` |
| 现有测试不破坏 | sync 路径不动 | 1028 passed 基线 |

### 测试原则

- async 测试用 `pytest_asyncio_mode = "auto"`（已在 pyproject.toml 配置）。
- LLM 测试 mock 外部 API（不真实调用）。
- AsyncSQLiteMemoryStore 测试用 `tmp_path` 文件 DB（不用 :memory:，aiosqlite 跨连接问题）。

---

## 12. 迁移批次（5 批）

| 批次 | 内容 | 验证 |
|------|------|------|
| 1 | litellm + groq + gemini + deepseek（4 LLM 后端）+ manifest + pyproject | LLM 单元测试 + manifest 完整性 |
| 2 | AsyncMemoryStore ABC（`storage/async_base.py`）+ 测试 | ABC 签名测试 |
| 3 | AsyncSQLiteMemoryStore（`storage/async_sqlite/store.py`）+ aiosqlite 依赖 + 测试 | 存储层 async 测试 |
| 4 | AsyncMemory facade（`memory/async_main.py`）+ 9 方法测试 | facade async 测试 |
| 5 | REST API 切换 AsyncMemory + 21 端点改 await + ruff + pytest 全量 | REST async 测试 + 基线不退化 |

### 批次依赖

```
批次 1（LLM 后端）           ← 独立
批次 2（AsyncMemoryStore ABC）← 独立
  └→ 批次 3（AsyncSQLiteMemoryStore）← 依赖 ABC
       └→ 批次 4（AsyncMemory facade）← 依赖 store
            └→ 批次 5（REST 切换）← 依赖 facade
```

批次 1 和 2 可并行。

---

## 13. 不做的事（YAGNI）

- **不改 Embedder/LLM ABC 加 async 方法**——用 to_thread 包装够了，避免连带改动。
- **AsyncMemory 不支持 typed memory（fact/episode/rule）**——核心是 verbatim 模式的 async 对偶，typed 留后续。
- **AsyncMemory 不支持 infer 模式**——FactExtractor 依赖 LLM sync 调用，async 化复杂，留后续。
- **PGVectorStore 不加 async 版本**——psycopg3 有 async 支持，但不在这批改，留后续。
- **不改 MCP server 为 async**——MCP 用 stdio/SSE，sync 够了。
- **不删 sync Memory**——两者并存，用户按需选。

---

## 14. 验收标准

- [ ] `SEPTMUSE_LLM=litellm SEPTMUSE_LLM_MODEL=groq/llama-3.1-70b-versatile` 可用
- [ ] `SEPTMUSE_LLM=groq` 可用
- [ ] `SEPTMUSE_LLM=gemini` 可用
- [ ] `SEPTMUSE_LLM=deepseek` 可用
- [ ] `septmuse backends` 输出 llm 有 8 个后端
- [ ] `AsyncMemory()` 可实例化，`await mem.add("hello", user_id="alice")` 返回结果
- [ ] `await mem.search("hello", user_id="alice")` 返回结果
- [ ] REST API `POST /memories` 用 `await memory.add(...)`
- [ ] 现有 sync 测试不退化（1028 passed 基线）
- [ ] async 测试全通过
- [ ] `ruff check src/ tests/` 全绿

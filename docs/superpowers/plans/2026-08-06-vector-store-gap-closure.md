# 向量数据库差距补齐 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 补齐 SeptMuse 向量数据库与 mem0 的差距——ABC 5→12 方法、Qdrant 本地嵌入默认化、丰富 Filter 操作符、BM25 稀疏向量、现有后端适配。

**Architecture:** `VectorStoreBase` ABC 扩展 7 方法（2 抽象 + 5 非抽象 fallback）。`QdrantVectorStore` 完全重写为本地嵌入模式 + BM25 + 丰富 Filter + batch search。Factory 默认 Qdrant，降级 SQLAlchemy。HybridRetriever 检测 VectorStore 内置 BM25。

**Tech Stack:** qdrant-client (核心依赖), fastembed (optional, BM25 编码), SQLAlchemy (fallback)

**Spec:** `docs/superpowers/specs/2026-08-06-vector-store-gap-closure-design.md`

## Global Constraints

- **包名** `septmuse`，src/ layout，`PYTHONPATH=src` 运行测试
- **ruff line-length 120**，`ruff check` 必须通过。**禁用 `ruff format`**（Windows bug）
- **代码注释中文**，不暴露开源库参考来源
- **测试**：开发完统一测试
- **conda 环境**：`SeptMuse`，Python 3.12
- **测试保护**：现有测试固定不动，禁止改测试绕过缺陷
- **类名**：`Embedder`（非 EmbedderBase），`LLM`（非 LLMBase），`VectorStoreBase`
- **score 统一**：相似度 [0,1]，越高越相似
- **qdrant-client**：核心依赖（`[project.dependencies]`）
- **fastembed**：optional extra（`[fastembed]`），未装时 `keyword_search()` 返回 None

---

## File Structure

| 文件 | 职责 |
|------|------|
| `src/septmuse/storage/vector_stores/base.py` | ABC 扩展：7 新方法 + Filter 文档 |
| `src/septmuse/storage/vector_stores/qdrant.py` | 完全重写：本地嵌入 + BM25 + 丰富 Filter + batch |
| `src/septmuse/storage/vector_stores/sqlalchemy_vec.py` | 新方法：update_vector + delete_collection + Filter 操作符 |
| `src/septmuse/storage/vector_stores/pgvector_store.py` | 新方法：update_vector + delete_collection |
| `src/septmuse/storage/vector_stores/chroma.py` | 新方法：update_vector + delete_collection |
| `src/septmuse/storage/relational_stores/factory.py` | `_resolve_vector_store` 默认 qdrant + 降级 |
| `src/septmuse/configs/vector_stores/base.py` | `backend` 默认 "chroma" → "qdrant" |
| `src/septmuse/retrieval/hybrid.py` | `_keyword_path` 检测 VectorStore 内置 BM25 |
| `pyproject.toml` | `qdrant-client` 核心依赖 + `fastembed` extra |

---

### Task 1: pyproject.toml 依赖 + VectorStoreBase ABC 扩展

**Files:**
- Modify: `pyproject.toml` (加 qdrant-client 核心依赖)
- Modify: `src/septmuse/storage/vector_stores/base.py` (ABC 扩展)

**Interfaces:**
- Produces: `VectorStoreBase.update_vector()`, `VectorStoreBase.search_batch()`, `VectorStoreBase.keyword_search()`, `VectorStoreBase.list_collections()`, `VectorStoreBase.delete_collection()`, `VectorStoreBase.get_collection_info()`, `VectorStoreBase.reset_collection()`

- [ ] **Step 1: 添加 qdrant-client 到 pyproject.toml**

在 `[project.dependencies]` 列表中添加 `"qdrant-client>=1.12.0"`。

在 `[project.optional-dependencies]` 中添加/更新：
```toml
"fastembed" = ["fastembed>=0.3.0"]
```

- [ ] **Step 2: 安装依赖**

Run: `pip install qdrant-client`

- [ ] **Step 3: 扩展 VectorStoreBase ABC**

在 `src/septmuse/storage/vector_stores/base.py` 中，在现有 5 方法之后添加：

```python
@abstractmethod
def update_vector(self, vector_id: str, vector: list[float], payload: dict[str, Any] | None = None) -> bool:
    """原地更新向量 + payload。True=更新成功，False=不存在。"""
    ...

@abstractmethod
def delete_collection(self) -> None:
    """删除整个 collection（所有向量 + payload）。"""
    ...

def search_batch(
    self,
    queries: list[str],
    vectors_list: list[list[float]],
    top_k: int = 5,
    filters: dict[str, Any] | None = None,
) -> list[list[VectorSearchResult]]:
    """批量搜索。默认循环 search_vectors，子类可 override 做原生批量。"""
    return [self.search_vectors(v, top_k, filters) for v in vectors_list]

def keyword_search(
    self, query: str, top_k: int = 5, filters: dict[str, Any] | None = None
) -> list[VectorSearchResult] | None:
    """BM25 关键词搜索。默认返回 None（不支持），Qdrant 后端 override。"""
    return None

def list_collections(self) -> list[str]:
    """列出所有 collection。默认返回当前 collection 名。"""
    return [getattr(self, "collection_name", "default")]

def get_collection_info(self) -> dict[str, Any]:
    """collection 元信息。默认返回 name + count。"""
    return {"name": getattr(self, "collection_name", "default"), "count": 0}

def reset_collection(self) -> None:
    """重置 collection（删除 + 重新创建）。"""
    self.delete_collection()
```

- [ ] **Step 4: 验证 import**

Run: `$env:PYTHONPATH = "src"; python -c "from septmuse.storage.vector_stores.base import VectorStoreBase; print('OK')"`
Expected: OK

Run: `ruff check src/septmuse/storage/vector_stores/base.py`
Expected: All checks passed

---

### Task 2: QdrantVectorStore 完全重写

**Files:**
- Modify: `src/septmuse/storage/vector_stores/qdrant.py` (完全重写)

**Interfaces:**
- Consumes: `VectorStoreBase` from Task 1
- Produces: `QdrantVectorStore(collection_name, embedding_model_dims, path, host, port, url, api_key, enable_bm25)`

- [ ] **Step 1: 重写 QdrantVectorStore**

完全重写 `src/septmuse/storage/vector_stores/qdrant.py`。关键结构：

**构造函数** — 三种连接模式（本地嵌入默认）：
```python
class QdrantVectorStore(VectorStoreBase):
    def __init__(
        self,
        collection_name: str = "septmuse",
        embedding_model_dims: int = 512,
        path: str | None = None,
        host: str | None = None,
        port: int | None = None,
        url: str | None = None,
        api_key: str | None = None,
        enable_bm25: bool = True,
    ):
        from qdrant_client import QdrantClient

        self.collection_name = collection_name
        self._dim = embedding_model_dims
        self._enable_bm25 = enable_bm25
        self._has_bm25_slot = False
        self._bm25_encoder = None  # 延迟加载
        self._collection_ensured = False

        # 连接模式：host/url/api_key 设了走远程，否则走本地 path
        if host or url:
            params = {}
            if host:
                params["host"] = host
            if port:
                params["port"] = port
            if url:
                params["url"] = url
            if api_key:
                params["api_key"] = api_key
            self.client = QdrantClient(**params)
            self.is_local = False
        else:
            # 默认本地嵌入
            from pathlib import Path
            qpath = path or str(Path.home() / ".septmuse" / "qdrant")
            self.client = QdrantClient(path=qpath)
            self.is_local = True
```

**_ensure_collection** — 创建 collection + BM25 slot + payload 索引：
```python
def _ensure_collection(self, dim: int):
    if self._collection_ensured:
        return
    from qdrant_client.models import Distance, SparseVectorParams, VectorParams
    from qdrant_client.models import Modifier

    collections = self.client.get_collections().collections
    exists = any(c.name == self.collection_name for c in collections)
    if not exists:
        sparse_config = {}
        if self._enable_bm25:
            sparse_config = {"bm25": SparseVectorParams(modifier=Modifier.IDF)}
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            sparse_vectors_config=sparse_config or None,
        )
        self._has_bm25_slot = self._enable_bm25
    else:
        info = self.client.get_collection(self.collection_name)
        sparse_cfg = info.config.params.sparse_vectors
        self._has_bm25_slot = bool(sparse_cfg and "bm25" in sparse_cfg)
    self._create_payload_indexes()
    self._collection_ensured = True
```

**_build_filter** — 10 操作符 + AND/OR/NOT（参考 mem0 Qdrant `_create_filter`，约 100 行）：

需要支持的操作符映射：
- 精确 `{"k": "v"}` → `MatchValue(value="v")`
- `{"k": {"eq": "v"}}` → `MatchValue`
- `{"k": {"ne": "v"}}` → `MatchExcept`
- `{"k": {"gt": N}}` → `Range(gt=N)`
- `{"k": {"gte": N}}` → `Range(gte=N)`
- `{"k": {"lt": N}}` → `Range(lt=N)`
- `{"k": {"lte": N}}` → `Range(lte=N)`
- `{"k": {"in": [...]}}` → `MatchAny`
- `{"k": {"nin": [...]}}` → `MatchExcept`
- `{"k": {"contains": "x"}}` → `MatchText`
- `{"k": {"icontains": "x"}}` → `MatchText`
- `{"AND": [...]}` → `Filter(must=[...])`
- `{"OR": [...]}` → `Filter(should=[...])`
- `{"NOT": [...]}` → `Filter(must_not=[...])`

递归处理 AND/OR/NOT 的嵌套 filter dict。

**insert_vectors** — 同时写 dense + BM25 sparse：
```python
def insert_vectors(self, vectors, ids, payloads=None):
    if not vectors:
        return
    self._ensure_collection(len(vectors[0]))
    from qdrant_client.models import PointStruct, SparseVector

    # 预计算 BM25
    bm25_list = [None] * len(vectors)
    if self._has_bm25_slot and payloads:
        bm25_list = self._batch_encode_bm25(payloads)

    points = []
    for vid, vec, payload, sparse in zip(ids, vectors, payloads or [{}]*len(ids), bm25_list, strict=True):
        named = {"": vec}
        if sparse is not None:
            named["bm25"] = sparse
        points.append(PointStruct(id=vid, vector=named, payload=payload))
    self.client.upsert(collection_name=self.collection_name, points=points)
```

**search_vectors** — 用 `query_points` + 丰富 Filter（现有逻辑基本不变，`_build_filter` 升级）

**search_batch** — Qdrant 原生 `query_batch_points`：
```python
def search_batch(self, queries, vectors_list, top_k=5, filters=None):
    from qdrant_client.models import QueryRequest
    query_filter = self._build_filter(filters) if filters else None
    requests = [QueryRequest(query=vec, filter=query_filter, limit=top_k, with_payload=True) for vec in vectors_list]
    try:
        results = self.client.query_batch_points(collection_name=self.collection_name, requests=requests)
        return [[VectorSearchResult(id=str(p.id), score=float(p.score), payload=p.payload) for p in r.points] for r in results]
    except Exception:
        return [self.search_vectors(v, top_k, filters) for v in vectors_list]
```

**keyword_search** — BM25 稀疏向量搜索（fastembed 可选）：
```python
def keyword_search(self, query, top_k=5, filters=None):
    if not self._has_bm25_slot:
        return None
    sparse = self._encode_bm25(query)
    if sparse is None:
        return None
    hits = self.client.query_points(
        collection_name=self.collection_name, query=sparse, using="bm25",
        query_filter=self._build_filter(filters) if filters else None, limit=top_k,
    )
    return [VectorSearchResult(id=str(p.id), score=float(p.score), payload=p.payload) for p in hits.points]
```

**BM25 编码器**：
```python
def _get_bm25_encoder(self):
    if self._bm25_encoder is None:
        try:
            from fastembed import SparseTextEmbedding
            self._bm25_encoder = SparseTextEmbedding(model_name="Qdrant/bm25")
        except ImportError:
            self._bm25_encoder = False
    return self._bm25_encoder if self._bm25_encoder is not False else None

def _encode_bm25(self, text):
    encoder = self._get_bm25_encoder()
    if encoder is None:
        return None
    from qdrant_client.models import SparseVector
    results = list(encoder.embed([text]))
    if results:
        s = results[0]
        return SparseVector(indices=s.indices.tolist(), values=s.values.tolist())
    return None

def _batch_encode_bm25(self, payloads):
    encoder = self._get_bm25_encoder()
    if encoder is None:
        return [None] * len(payloads)
    from qdrant_client.models import SparseVector
    texts = [p.get("data", "") or p.get("text", "") or str(p) for p in payloads]
    results = list(encoder.embed(texts))
    return [SparseVector(indices=r.indices.tolist(), values=r.values.tolist()) if r else None for r in results]
```

**update_vector** — 原地更新 dense + BM25 + payload：
```python
def update_vector(self, vector_id, vector, payload=None):
    from qdrant_client.models import PointStruct
    named = {"": vector}
    if self._has_bm25_slot and payload:
        text = payload.get("data", "") or payload.get("text", "")
        if text:
            sparse = self._encode_bm25(text)
            if sparse is not None:
                named["bm25"] = sparse
    self.client.upsert(collection_name=self.collection_name, points=[PointStruct(id=vector_id, vector=named, payload=payload or {})])
    return True
```

**Collection 管理方法**：
```python
def list_collections(self):
    return [c.name for c in self.client.get_collections().collections]

def delete_collection(self):
    self.client.delete_collection(collection_name=self.collection_name)
    self._collection_ensured = False

def get_collection_info(self):
    info = self.client.get_collection(collection_name=self.collection_name)
    return {"name": self.collection_name, "count": info.points_count or 0,
            "dim": self._dim, "distance": "COSINE"}

def reset_collection(self):
    self.delete_collection()
    self._collection_ensured = False
```

**_create_payload_indexes** — 远程模式为 user_id/agent_id/session_id 创建索引：
```python
def _create_payload_indexes(self):
    if self.is_local:
        return
    for field in ["user_id", "agent_id", "session_id"]:
        try:
            self.client.create_payload_index(
                collection_name=self.collection_name, field_name=field, field_schema="keyword")
        except Exception:
            pass
```

保留现有的 `_extract_vector` 静态方法、`delete_vector`、`get_vector`、`list_vectors`、`close` 方法（逻辑不变，`_build_filter` 升级）。

- [ ] **Step 2: 验证本地嵌入**

Run: `$env:PYTHONPATH = "src"; python -c "from septmuse.storage.vector_stores.qdrant import QdrantVectorStore; s = QdrantVectorStore(path='C:/Users/Sikh0/AppData/Local/temp/opencode/test_qdrant'); s.insert_vectors([[0.1]*512], ['m1'], [{'user_id':'alice'}]); r = s.search_vectors([0.1]*512, top_k=1, filters={'user_id':'alice'}); print(r[0].id, r[0].score); s.delete_collection(); s.close(); print('OK')"`
Expected: m1 <score> OK

Run: `ruff check src/septmuse/storage/vector_stores/qdrant.py`
Expected: All checks passed

---

### Task 3: SQLAlchemyVectorStore + PgvectorVectorStore 适配

**Files:**
- Modify: `src/septmuse/storage/vector_stores/sqlalchemy_vec.py`
- Modify: `src/septmuse/storage/vector_stores/pgvector_store.py`

- [ ] **Step 1: SQLAlchemyVectorStore 新方法**

在 `sqlalchemy_vec.py` 中添加：

```python
def update_vector(self, vector_id, vector, payload=None):
    """原地更新 — DELETE + INSERT 两步（跨方言 upsert）。"""
    with Session(self._engine) as session:
        session.execute(text("DELETE FROM vector_entries WHERE id = :id").bindparams(id=vector_id))
        session.execute(text("INSERT INTO vector_entries (id, vector, payload) VALUES (:id, :vec, :payload)").bindparams(
            id=vector_id, vec=json.dumps(vector), payload=json.dumps(payload or {})))
        session.commit()
    return True

def delete_collection(self):
    """DROP TABLE。"""
    with self._engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS vector_entries"))
        conn.commit()

def get_collection_info(self):
    """SELECT COUNT(*)。"""
    with self._engine.connect() as conn:
        result = conn.execute(text("SELECT COUNT(*) FROM vector_entries"))
        count = result.scalar() or 0
    return {"name": "vector_entries", "count": count}
```

同时扩展 `_fetch_rows` 的 filter 解析，支持 `{"k": {"gte": N}}` 等操作符语法（递归构建 SQL WHERE）。

- [ ] **Step 2: PgvectorVectorStore 新方法**

在 `pgvector_store.py` 中添加：

```python
def update_vector(self, vector_id, vector, payload=None):
    """ON CONFLICT upsert。"""
    if not self._pgvector_available:
        return self._fallback.update_vector(vector_id, vector, payload)
    with self._engine.connect() as conn:
        conn.execute(text(
            "INSERT INTO vector_entries (id, vector, payload) VALUES (:id, :vec::vector, :payload::jsonb) "
            "ON CONFLICT (id) DO UPDATE SET vector = :vec::vector, payload = :payload::jsonb"
        ).bindparams(id=vector_id, vec=json.dumps(vector), payload=json.dumps(payload or {})))
        conn.commit()
    return True

def delete_collection(self):
    if not self._pgvector_available:
        return self._fallback.delete_collection()
    with self._engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS vector_entries"))
        conn.commit()

def get_collection_info(self):
    if not self._pgvector_available:
        return self._fallback.get_collection_info()
    with self._engine.connect() as conn:
        result = conn.execute(text("SELECT COUNT(*) FROM vector_entries"))
        count = result.scalar() or 0
    return {"name": "vector_entries", "count": count, "dim": self._dim, "distance": "COSINE"}
```

- [ ] **Step 3: 验证**

Run: `$env:PYTHONPATH = "src"; python -c "from septmuse.storage.vector_stores.sqlalchemy_vec import SQLAlchemyVectorStore; from sqlalchemy import create_engine; e = create_engine('sqlite:///:memory:'); s = SQLAlchemyVectorStore(e); s.insert_vectors([[0.1]*4], ['m1'], [{'user_id':'a'}]); print(s.update_vector('m1', [0.2]*4, {'user_id':'b'})); print(s.get_collection_info()); s.delete_collection(); print('OK')"`
Expected: True {'name':'vector_entries','count':1} OK

Run: `ruff check src/septmuse/storage/vector_stores/sqlalchemy_vec.py src/septmuse/storage/vector_stores/pgvector_store.py`

---

### Task 4: ChromaVectorStore 适配

**Files:**
- Modify: `src/septmuse/storage/vector_stores/chroma.py`

- [ ] **Step 1: 添加新方法**

在 `chroma.py` 中添加：

```python
def update_vector(self, vector_id, vector, payload=None):
    """Chroma collection.upsert 原地更新。"""
    import json
    chroma_meta = {k: v for k, v in (payload or {}).items() if v is not None} or {"_id": vector_id}
    doc = json.dumps(payload or {}, ensure_ascii=False)
    self.collection.upsert(
        ids=[vector_id], embeddings=[vector], metadatas=[chroma_meta], documents=[doc])
    return True

def delete_collection(self):
    """删除整个 collection。"""
    self.client.delete_collection(name=self.collection_name)

def get_collection_info(self):
    """collection 元信息。"""
    return {"name": self.collection_name, "count": self.collection.count()}
```

- [ ] **Step 2: 验证 import**

Run: `ruff check src/septmuse/storage/vector_stores/chroma.py`

---

### Task 5: Factory + Config 改造

**Files:**
- Modify: `src/septmuse/storage/relational_stores/factory.py` (默认 qdrant + 降级)
- Modify: `src/septmuse/storage/vector_stores/factory.py` (支持 qdrant)
- Modify: `src/septmuse/configs/vector_stores/base.py` (默认 "chroma" → "qdrant")

- [ ] **Step 1: Config 默认值变更**

在 `src/septmuse/configs/vector_stores/base.py` 中，将 `backend` 默认值从 `"chroma"` 改为 `"qdrant"`：

```python
class BaseVectorStoreConfig(BaseModel):
    backend: str = Field(default="qdrant", description="向量后端")
    embedding_model_dims: int = Field(default=512, description="嵌入维度")
```

- [ ] **Step 2: RelationalStoreFactory 改造**

在 `src/septmuse/storage/relational_stores/factory.py` 的 `_resolve_vector_store` 中，在现有 chroma 分支之前添加 qdrant 分支：

```python
@staticmethod
def _resolve_vector_store(config, engine, dialect) -> VectorStoreBase:
    backend = config.vector_store.backend

    if backend == "qdrant":
        try:
            from septmuse.storage.vector_stores.qdrant import QdrantVectorStore
            return QdrantVectorStore(
                collection_name="septmuse",
                embedding_model_dims=config.vector_store.embedding_model_dims or 512,
                path=os.getenv("SEPTMUSE_QDRANT_PATH"),
                host=os.getenv("SEPTMUSE_QDRANT_HOST"),
                port=int(os.getenv("SEPTMUSE_QDRANT_PORT", "6333")) if os.getenv("SEPTMUSE_QDRANT_HOST") else None,
                url=os.getenv("SEPTMUSE_QDRANT_URL"),
                api_key=os.getenv("SEPTMUSE_QDRANT_API_KEY"),
                enable_bm25=True,
            )
        except ImportError:
            logger.warning("qdrant_not_available_fallback_sqlite")

    if backend == "chroma":
        # 现有逻辑不变
        ...

    # fallback: dialect-based
    from septmuse.storage.vector_stores.factory import create_vector_store
    return create_vector_store(engine, dialect)
```

- [ ] **Step 3: 验证默认后端**

Run: `$env:PYTHONPATH = "src"; python -c "from septmuse.configs import default_config; c = default_config(); print(c.vector_store.backend)"`
Expected: qdrant

---

### Task 6: HybridRetriever 改造

**Files:**
- Modify: `src/septmuse/retrieval/hybrid.py`

- [ ] **Step 1: 修改 _keyword_path 检测 VectorStore 内置 BM25**

在 `hybrid.py` 的 `_keyword_path` 内部函数中，优先检查 VectorStore 是否有内置 `keyword_search`：

```python
def _keyword_path() -> list[dict[str, Any]] | None:
    with time_block("hybrid_search_components_seconds", {"component": "keyword"}):
        # 优先用 VectorStore 内置 keyword_search
        vs = getattr(self.store, '_vector_store', None)
        if vs is not None and hasattr(vs, 'keyword_search'):
            results = vs.keyword_search(query, top_k=internal_limit, filters=vs_filters)
            if results is not None:
                return [{"id": r.id, "score": r.score, "memory": r.payload.get("data", "")} for r in results]
        # 回退到外部 KeywordIndexBase
        try:
            return self.store.keyword_search(query, user_id=user_id, session_id=session_id, top_k=internal_limit)
        except Exception:
            return None
```

- [ ] **Step 2: 验证 import**

Run: `$env:PYTHONPATH = "src"; python -c "from septmuse.retrieval.hybrid import HybridRetriever; print('OK')"`

Run: `ruff check src/septmuse/retrieval/hybrid.py`

---

### Task 7: 测试

**Files:**
- Create: `tests/unit/test_vector_stores/conftest.py`
- Create: `tests/unit/test_vector_stores/test_qdrant_local.py`
- Create: `tests/unit/test_vector_stores/test_qdrant_filters.py`
- Create: `tests/unit/test_vector_stores/test_qdrant_bm25.py`
- Create: `tests/unit/test_vector_stores/test_qdrant_batch.py`
- Create: `tests/unit/test_vector_stores/test_abc_new_methods.py`
- Create: `tests/unit/test_vector_stores/test_factory_qdrant_default.py`

- [ ] **Step 1: 创建 conftest.py**

```python
"""向量存储测试 conftest。"""
import pytest
from pathlib import Path


@pytest.fixture
def qdrant_path(tmp_path):
    """临时 Qdrant 本地嵌入路径。"""
    return str(tmp_path / "qdrant_test")


@pytest.fixture
def qdrant_store(qdrant_path):
    """临时 QdrantVectorStore（512 dim，本地嵌入）。"""
    from septmuse.storage.vector_stores.qdrant import QdrantVectorStore
    store = QdrantVectorStore(
        collection_name="test",
        embedding_model_dims=512,
        path=qdrant_path,
        enable_bm25=False,  # 测试默认关 BM25（fastembed 可选）
    )
    yield store
    try:
        store.delete_collection()
    except Exception:
        pass
    store.close()
```

- [ ] **Step 2: 创建 test_qdrant_local.py — CRUD + 本地路径**

```python
"""Qdrant 本地嵌入全量测试。"""
import pytest
from septmuse.storage.vector_stores.qdrant import QdrantVectorStore


def test_insert_and_search(qdrant_store):
    """插入 + 搜索基本流程。"""
    qdrant_store.insert_vectors([[0.1] * 512], ["m1"], [{"user_id": "alice"}])
    results = qdrant_store.search_vectors([0.1] * 512, top_k=1, filters={"user_id": "alice"})
    assert len(results) == 1
    assert results[0].id == "m1"
    assert 0.0 <= results[0].score <= 1.0


def test_delete(qdrant_store):
    qdrant_store.insert_vectors([[0.1] * 512], ["m1"], [{"user_id": "alice"}])
    assert qdrant_store.delete_vector("m1") is True
    assert qdrant_store.delete_vector("m1") is False


def test_get_vector(qdrant_store):
    qdrant_store.insert_vectors([[0.1] * 512], ["m1"], [{"user_id": "alice"}])
    entry = qdrant_store.get_vector("m1")
    assert entry is not None
    assert entry.id == "m1"
    assert len(entry.vector) == 512


def test_update_vector(qdrant_store):
    qdrant_store.insert_vectors([[0.1] * 512], ["m1"], [{"user_id": "alice"}])
    assert qdrant_store.update_vector("m1", [0.2] * 512, {"user_id": "bob"}) is True


def test_list_collections(qdrant_store):
    cols = qdrant_store.list_collections()
    assert "test" in cols


def test_get_collection_info(qdrant_store):
    qdrant_store.insert_vectors([[0.1] * 512], ["m1"], [{"user_id": "alice"}])
    info = qdrant_store.get_collection_info()
    assert info["name"] == "test"
    assert info["count"] >= 1


def test_reset_collection(qdrant_store):
    qdrant_store.insert_vectors([[0.1] * 512], ["m1"], [{"user_id": "alice"}])
    qdrant_store.reset_collection()
    info = qdrant_store.get_collection_info()
    assert info["count"] == 0
```

- [ ] **Step 3: 创建 test_qdrant_filters.py — 10 操作符 + 逻辑组合**

```python
"""Qdrant Filter 操作符测试。"""
import pytest


def _setup_data(store):
    """插入测试数据。"""
    store.insert_vectors(
        [[0.1] * 512, [0.2] * 512, [0.3] * 512],
        ["m1", "m2", "m3"],
        [{"user_id": "alice", "score": 5, "tags": "work"},
         {"user_id": "alice", "score": 10, "tags": "personal"},
         {"user_id": "bob", "score": 15, "tags": "work"}],
    )


def test_filter_exact(qdrant_store):
    _setup_data(qdrant_store)
    results = qdrant_store.search_vectors([0.1] * 512, top_k=5, filters={"user_id": "alice"})
    ids = [r.id for r in results]
    assert "m1" in ids
    assert "m3" not in ids


def test_filter_eq(qdrant_store):
    _setup_data(qdrant_store)
    results = qdrant_store.search_vectors([0.1] * 512, top_k=5, filters={"user_id": {"eq": "bob"}})
    assert all(r.id == "m3" for r in results)


def test_filter_ne(qdrant_store):
    _setup_data(qdrant_store)
    results = qdrant_store.search_vectors([0.1] * 512, top_k=5, filters={"user_id": {"ne": "bob"}})
    ids = {r.id for r in results}
    assert "m3" not in ids


def test_filter_gte(qdrant_store):
    _setup_data(qdrant_store)
    results = qdrant_store.search_vectors([0.1] * 512, top_k=5, filters={"score": {"gte": 10}})
    ids = {r.id for r in results}
    assert "m2" in ids
    assert "m3" in ids
    assert "m1" not in ids


def test_filter_in(qdrant_store):
    _setup_data(qdrant_store)
    results = qdrant_store.search_vectors([0.1] * 512, top_k=5, filters={"user_id": {"in": ["alice", "bob"]}})
    assert len(results) == 3


def test_filter_and(qdrant_store):
    _setup_data(qdrant_store)
    results = qdrant_store.search_vectors([0.1] * 512, top_k=5,
        filters={"AND": [{"user_id": "alice"}, {"score": {"gte": 10}}]})
    ids = {r.id for r in results}
    assert "m2" in ids
    assert "m1" not in ids
```

- [ ] **Step 4: 创建 test_qdrant_bm25.py — BM25 稀疏向量**

```python
"""Qdrant BM25 稀疏向量测试。fastembed 未装时 skip。"""
import pytest

try:
    import fastembed  # noqa: F401
    FASTEMBED_AVAILABLE = True
except ImportError:
    FASTEMBED_AVAILABLE = False


@pytest.fixture
def bm25_store(tmp_path):
    """带 BM25 的 Qdrant store。"""
    from septmuse.storage.vector_stores.qdrant import QdrantVectorStore
    store = QdrantVectorStore(
        collection_name="test_bm25",
        embedding_model_dims=512,
        path=str(tmp_path / "qdrant_bm25"),
        enable_bm25=True,
    )
    yield store
    try:
        store.delete_collection()
    except Exception:
        pass
    store.close()


@pytest.mark.skipif(not FASTEMBED_AVAILABLE, reason="fastembed not installed")
def test_bm25_keyword_search(bm25_store):
    """BM25 关键词搜索。"""
    bm25_store.insert_vectors(
        [[0.1] * 512, [0.2] * 512],
        ["m1", "m2"],
        [{"user_id": "alice", "data": "我喜欢编程"},
         {"user_id": "alice", "data": "天气很好今天"}],
    )
    results = bm25_store.keyword_search("编程", top_k=2)
    assert results is not None
    assert len(results) > 0
    assert results[0].id == "m1"


def test_bm25_returns_none_when_disabled(qdrant_store):
    """enable_bm25=False 时 keyword_search 返回 None。"""
    assert qdrant_store.keyword_search("test") is None
```

- [ ] **Step 5: 创建 test_qdrant_batch.py**

```python
"""Qdrant search_batch 测试。"""


def test_search_batch(qdrant_store):
    qdrant_store.insert_vectors(
        [[0.1] * 512, [0.2] * 512],
        ["m1", "m2"],
        [{"user_id": "alice"}, {"user_id": "bob"}],
    )
    results = qdrant_store.search_batch(
        ["q1", "q2"],
        [[0.1] * 512, [0.2] * 512],
        top_k=2,
    )
    assert len(results) == 2
    assert len(results[0]) > 0
    assert len(results[1]) > 0
```

- [ ] **Step 6: 创建 test_abc_new_methods.py — SQLAlchemy 新方法**

```python
"""VectorStoreBase 新方法测试（用 SQLAlchemyVectorStore 验证 fallback）。"""
from sqlalchemy import create_engine
from septmuse.storage.vector_stores.sqlalchemy_vec import SQLAlchemyVectorStore


def test_update_vector():
    engine = create_engine("sqlite:///:memory:")
    store = SQLAlchemyVectorStore(engine)
    store.insert_vectors([[0.1] * 4], ["m1"], [{"user_id": "a"}])
    assert store.update_vector("m1", [0.2] * 4, {"user_id": "b"}) is True


def test_delete_collection():
    engine = create_engine("sqlite:///:memory:")
    store = SQLAlchemyVectorStore(engine)
    store.insert_vectors([[0.1] * 4], ["m1"], [{"user_id": "a"}])
    store.delete_collection()
    info = store.get_collection_info()
    assert info["count"] == 0


def test_keyword_search_returns_none():
    """SQLAlchemy 后端 keyword_search 默认返回 None。"""
    engine = create_engine("sqlite:///:memory:")
    store = SQLAlchemyVectorStore(engine)
    assert store.keyword_search("test") is None


def test_search_batch_default():
    """search_batch 默认循环 search_vectors。"""
    engine = create_engine("sqlite:///:memory:")
    store = SQLAlchemyVectorStore(engine)
    store.insert_vectors([[0.1] * 4, [0.2] * 4], ["m1", "m2"], [{"user_id": "a"}, {"user_id": "a"}])
    results = store.search_batch(["q1", "q2"], [[0.1] * 4, [0.2] * 4], top_k=2)
    assert len(results) == 2
```

- [ ] **Step 7: 创建 test_factory_qdrant_default.py**

```python
"""Factory 默认 Qdrant 测试。"""
import os
from septmuse.configs import default_config


def test_default_backend_is_qdrant():
    config = default_config()
    assert config.vector_store.backend == "qdrant"


def test_factory_creates_qdrant(tmp_path):
    """Factory 默认创建 QdrantVectorStore。"""
    os.environ["SEPTMUSE_QDRANT_PATH"] = str(tmp_path / "qdrant")
    try:
        from septmuse.storage.relational_stores.factory import RelationalStoreFactory
        config = default_config()
        config.database.db_url = f"sqlite:///{tmp_path / 'test.db'}"
        store = RelationalStoreFactory.create(config)
        from septmuse.storage.vector_stores.qdrant import QdrantVectorStore
        assert isinstance(store._vector_store, QdrantVectorStore)
    finally:
        os.environ.pop("SEPTMUSE_QDRANT_PATH", None)
```

- [ ] **Step 8: 跑测试**

Run: `$env:PYTHONPATH = "src"; python -m pytest tests/unit/test_vector_stores/ -v --tb=short`
Expected: ALL PASSED (BM25 tests skipped if fastembed not installed)

Run: `ruff check tests/unit/test_vector_stores/`

---

### Task 8: 全量回归 + AGENTS.md

**Files:**
- Modify: `AGENTS.md`

- [ ] **Step 1: ruff check 全量**

Run: `ruff check src/ tests/`
Expected: All checks passed

- [ ] **Step 2: pytest 全量**

Run: `$env:PYTHONPATH = "src"; python -m pytest tests/unit/ tests/e2e/ -q --tb=no --timeout=300`
Expected: 与基线一致（1457+ passed / 16 failed pre-existing / 23+ skipped）+ 新增 vector store 测试全绿

- [ ] **Step 3: 更新 AGENTS.md**

- `SEPTMUSE_VECTOR_BACKEND` 默认值改为 `qdrant`
- 新增 Qdrant 本地嵌入段落
- 新增 `SEPTMUSE_QDRANT_*` 环境变量表
- ABC 方法列表更新（12 方法）
- 新增 Filter 操作符语法说明

- [ ] **Step 4: 最终验证**

Run: `$env:PYTHONPATH = "src"; $env:SEPTMUSE_QDRANT_PATH = "C:/Users/Sikh0/AppData/Local/temp/opencode/test_final"; python -c "from septmuse.configs import default_config; c = default_config(); print(c.vector_store.backend)"`

Expected: qdrant

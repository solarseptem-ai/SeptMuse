# 向量数据库差距补齐设计

> 日期：2026-08-06
> 前置：`docs/superpowers/specs/2026-08-06-observability-metrics-design.md`（已完成）
> 对标：`opensource/mem0/mem0/vector_stores/`（25+ 后端）
> 范围：ABC 扩展（5→12 方法）+ Qdrant 默认化（本地嵌入）+ 丰富 Filter + BM25 稀疏向量 + 现有后端适配

---

## 1. 目标

补齐 SeptMuse 向量数据库与 mem0 的差距，分四个子项目：

| 子项目 | 范围 | 本 spec |
|--------|------|:---:|
| **1. ABC 扩展 + Qdrant 默认化** | ABC 12 方法 + Qdrant 增强 + 现有后端适配 + Factory 改造 + HybridRetriever 改造 | ✓ |
| 2. FAISS + numpy 矩阵优化 | FAISS 后端 + SQLAlchemy 矩阵化 + distance strategy | 后续 |
| 3. 主流托管后端 | Pinecone / Milvus / Weaviate / MongoDB / Redis / Elasticsearch | 后续 |
| 4. 长尾后端 | Supabase / Azure AI / Baidu / Cassandra 等 13 个 | 后续 |

**非目标**：不做 FAISS 后端（子项目 2）、不做托管后端（子项目 3/4）、不改 SQLite 的 ANN 能力（子项目 2 的 numpy 矩阵优化）。

## 2. 设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 默认向量后端 | Qdrant 本地嵌入（`QdrantClient(path=...)`） | 零配置保持，ANN HNSW 索引，对齐 mem0 |
| `qdrant-client` 依赖位置 | 核心依赖（`[project.dependencies]`） | 默认后端必须可用，~5MB |
| `fastembed` 依赖位置 | optional extra（`[fastembed]`） | BM25 编码器 ~10MB，未装时降级 |
| BM25 路径 | VectorStore 内置 `keyword_search()` + HybridRetriever 检测 | 对齐 mem0，Qdrant 原生 BM25 质量优于 SQLite BM25 |
| Filter 操作符 | 10 种操作符 + AND/OR/NOT 逻辑组合 | 对齐 mem0，消除 MemoryStore 层 workaround |
| 现有后端兼容 | 新方法全部有默认 fallback（非抽象） | 现有后端零改动即可工作 |

## 3. VectorStoreBase ABC 扩展

`src/septmuse/storage/vector_stores/base.py` — 从 5 方法扩展到 12 方法。

### 3.1 完整方法清单

| 方法 | 类型 | 说明 |
|------|------|------|
| `insert_vectors(vectors, ids, payloads)` | 已有抽象 | 不变 |
| `search_vectors(query_vector, top_k, filters)` | 已有抽象 | filters 支持丰富操作符语法 |
| `delete_vector(vector_id)` | 已有抽象 | 不变 |
| `get_vector(vector_id)` | 已有抽象 | 不变 |
| `list_vectors(filters, limit)` | 已有抽象 | 不变 |
| **`update_vector(vector_id, vector, payload)`** | 新增抽象 | 原地更新，替代 delete+insert |
| **`search_batch(queries, vectors_list, top_k, filters)`** | 新增非抽象 | 默认循环 `search_vectors`，Qdrant 原生 override |
| **`keyword_search(query, top_k, filters)`** | 新增非抽象 | 默认返回 None，Qdrant BM25 override |
| **`list_collections()`** | 新增非抽象 | 默认返回 `[self.collection_name]` |
| **`delete_collection()`** | 新增抽象 | 删除整个 collection |
| **`get_collection_info()`** | 新增非抽象 | 默认返回 `{"name":..., "count":...}` |
| **`reset_collection()`** | 新增非抽象 | 默认 delete_collection + 重新创建 |

### 3.2 新增抽象方法签名

```python
@abstractmethod
def update_vector(self, vector_id: str, vector: list[float], payload: dict[str, Any] | None = None) -> bool:
    """原地更新向量 + payload。True=更新成功，False=不存在。"""
    ...

@abstractmethod
def delete_collection(self) -> None:
    """删除整个 collection（所有向量 + payload）。"""
    ...
```

### 3.3 新增非抽象方法签名

```python
def search_batch(
    self, queries: list[str], vectors_list: list[list[float]], top_k: int = 5, filters: dict[str, Any] | None = None
) -> list[list[VectorSearchResult]]:
    """批量搜索。默认循环 search_vectors，子类可 override 做原生批量。"""
    return [self.search_vectors(v, top_k, filters) for v in vectors_list]

def keyword_search(self, query: str, top_k: int = 5, filters: dict[str, Any] | None = None) -> list[VectorSearchResult] | None:
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

### 3.4 Filter 操作符语法

`search_vectors` 的 `filters` 参数扩展，支持丰富操作符（与 mem0 对齐）：

```python
# 简单精确匹配（向后兼容）
filters = {"user_id": "alice"}

# 丰富操作符
filters = {
    "user_id": "alice",                     # 精确匹配
    "score": {"gte": 0.5},                  # 范围
    "tags": {"in": ["work", "personal"]},   # 包含
    "content": {"contains": "hello"},       # 子串
    "AND": [{"x": 1}, {"y": 2}],            # 逻辑组合
}
```

**操作符清单**：

| 操作符 | 语法 | 说明 |
|--------|------|------|
| 精确匹配 | `{"k": "v"}` | 等价 `{"k": {"eq": "v"}}` |
| `eq` | `{"k": {"eq": "v"}}` | 等于 |
| `ne` | `{"k": {"ne": "v"}}` | 不等于 |
| `gt` | `{"k": {"gt": 5}}` | 大于 |
| `gte` | `{"k": {"gte": 5}}` | 大于等于 |
| `lt` | `{"k": {"lt": 10}}` | 小于 |
| `lte` | `{"k": {"lte": 10}}` | 小于等于 |
| `in` | `{"k": {"in": ["a","b"]}}` | 包含在列表中 |
| `nin` | `{"k": {"nin": ["a"]}}` | 不包含在列表中 |
| `contains` | `{"k": {"contains": "x"}}` | 子串包含 |
| `icontains` | `{"k": {"icontains": "x"}}` | 子串包含（大小写不敏感） |
| `AND` | `{"AND": [{...}, {...}]}` | 逻辑与 |
| `OR` | `{"OR": [{...}, {...}]}` | 逻辑或 |
| `NOT` | `{"NOT": [{...}]}` | 逻辑非 |

每个后端自己解析——Qdrant 用 `Filter/FieldCondition`，SQLAlchemy 用 SQL WHERE，不支持的 fallback Python 侧过滤。简单 `{"k": "v"}` 向后兼容。

## 4. QdrantVectorStore 增强

`src/septmuse/storage/vector_stores/qdrant.py` — 完全重写，对齐 mem0 Qdrant 后端。

### 4.1 构造函数

```python
class QdrantVectorStore(VectorStoreBase):
    def __init__(
        self,
        collection_name: str = "septmuse",
        embedding_model_dims: int = 512,
        # 本地嵌入（默认）
        path: str | None = None,       # None → "~/.septmuse/qdrant"
        # 远程服务
        host: str | None = None,
        port: int | None = None,
        url: str | None = None,
        api_key: str | None = None,
        # BM25
        enable_bm25: bool = True,
    ):
        ...
```

三种连接模式：
- **本地嵌入**（默认）：`QdrantClient(path="~/.septmuse/qdrant")`，零配置，持久化到本地目录
- **远程服务**：`QdrantClient(host=..., port=..., url=..., api_key=...)`
- **环境变量驱动**：`SEPTMUSE_QDRANT_HOST` 设了走远程，否则走本地 path

### 4.2 Collection 创建

```python
def _create_collection(self, dim: int):
    """创建 collection — dense 向量 + 可选 BM25 sparse 向量 slot。"""
    self.client.create_collection(
        collection_name=self.collection_name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        sparse_vectors_config={"bm25": SparseVectorParams(modifier=Modifier.IDF)} if self._enable_bm25 else None,
    )
    self._has_bm25_slot = self._enable_bm25
    self._create_payload_indexes()
```

### 4.3 丰富 Filter — `_build_filter()`

`_build_filter(filters)` 解析操作符语法，构建 Qdrant `Filter` 对象：

| 操作符 | Qdrant 映射 |
|--------|------------|
| 精确 `{"k": "v"}` | `MatchValue(value="v")` |
| `eq` | `MatchValue(value=...)` |
| `ne` | `MatchExcept(except=[...])` |
| `gt/gte/lt/lte` | `Range(gt=.../gte=.../lt=.../lte=...)` |
| `in` | `MatchAny(any=[...])` |
| `nin` | `MatchExcept(except=[...])` |
| `contains/icontains` | `MatchText(text=...)` |
| `AND` | `Filter(must=[...])` |
| `OR` | `Filter(should=[...])` |
| `NOT` | `Filter(must_not=[...])` |

### 4.4 insert_vectors 增强

同时写 dense + BM25 sparse 向量：

```python
def insert_vectors(self, vectors, ids, payloads):
    # 预计算 BM25 sparse 向量（批量 fastembed）
    bm25_vectors = self._batch_encode_bm25(payloads) if self._has_bm25_slot else [None] * len(vectors)
    points = []
    for vid, vec, payload, sparse in zip(ids, vectors, payloads, bm25_vectors, strict=True):
        named = {"": vec}
        if sparse is not None:
            named["bm25"] = sparse
        points.append(PointStruct(id=vid, vector=named, payload=payload))
    self.client.upsert(collection_name=self.collection_name, points=points)
```

### 4.5 search_vectors 增强

使用 `query_points` + 丰富 Filter：

```python
def search_vectors(self, query_vector, top_k=5, filters=None):
    self._ensure_collection(len(query_vector))
    query_filter = self._build_filter(filters)
    hits = self.client.query_points(
        collection_name=self.collection_name,
        query=query_vector,
        query_filter=query_filter,
        limit=top_k,
        with_payload=True,
    )
    return [VectorSearchResult(id=str(p.id), score=float(p.score), payload=p.payload) for p in hits.points]
```

### 4.6 search_batch — Qdrant 原生批量

```python
def search_batch(self, queries, vectors_list, top_k=5, filters=None):
    query_filter = self._build_filter(filters) if filters else None
    requests = [QueryRequest(query=vec, filter=query_filter, limit=top_k) for vec in vectors_list]
    try:
        results = self.client.query_batch_points(collection_name=self.collection_name, requests=requests)
        return [[VectorSearchResult(...) for p in r.points] for r in results]
    except Exception:
        # 降级到逐个搜索
        return [self.search_vectors(v, top_k, filters) for v in vectors_list]
```

### 4.7 keyword_search — BM25 稀疏向量

```python
def keyword_search(self, query, top_k=5, filters=None):
    """BM25 稀疏向量搜索。fastembed 未装或 collection 无 bm25 slot 时返回 None。"""
    if not self._has_bm25_slot:
        return None
    sparse = self._encode_bm25(query)
    if sparse is None:
        return None
    hits = self.client.query_points(
        collection_name=self.collection_name,
        query=sparse,
        using="bm25",
        query_filter=self._build_filter(filters),
        limit=top_k,
    )
    return [VectorSearchResult(id=str(p.id), score=float(p.score), payload=p.payload) for p in hits.points]
```

### 4.8 BM25 编码器

```python
def _get_bm25_encoder(self):
    """延迟加载 fastembed BM25 编码器。未装返回 None。"""
    if self._bm25_encoder is None:
        try:
            from fastembed import SparseTextEmbedding
            self._bm25_encoder = SparseTextEmbedding(model_name="Qdrant/bm25")
        except ImportError:
            self._bm25_encoder = False  # sentinel: tried and failed
    return self._bm25_encoder if self._bm25_encoder is not False else None

def _encode_bm25(self, text: str) -> SparseVector | None:
    """单条文本 → BM25 稀疏向量。"""
    encoder = self._get_bm25_encoder()
    if encoder is None:
        return None
    results = list(encoder.embed([text]))
    if results:
        sparse = results[0]
        return SparseVector(indices=sparse.indices.tolist(), values=sparse.values.tolist())
    return None

def _batch_encode_bm25(self, payloads: list[dict]) -> list[SparseVector | None]:
    """批量 BM25 编码。"""
    encoder = self._get_bm25_encoder()
    if encoder is None:
        return [None] * len(payloads)
    texts = [p.get("data", "") or p.get("text", "") for p in payloads]
    results = list(encoder.embed(texts))
    return [SparseVector(indices=r.indices.tolist(), values=r.values.tolist()) if r else None for r in results]
```

### 4.9 update_vector

```python
def update_vector(self, vector_id, vector, payload=None):
    """原地更新 dense 向量 + payload + BM25 sparse。"""
    named = {"": vector}
    if self._has_bm25_slot and payload:
        text = payload.get("data", "") or payload.get("text", "")
        if text:
            sparse = self._encode_bm25(text)
            if sparse is not None:
                named["bm25"] = sparse
    point = PointStruct(id=vector_id, vector=named, payload=payload or {})
    self.client.upsert(collection_name=self.collection_name, points=[point])
    return True
```

### 4.10 Collection 管理

```python
def list_collections(self):
    return [c.name for c in self.client.get_collections().collections]

def delete_collection(self):
    self.client.delete_collection(collection_name=self.collection_name)

def get_collection_info(self):
    info = self.client.get_collection(collection_name=self.collection_name)
    return {
        "name": self.collection_name,
        "count": info.points_count or 0,
        "dim": info.config.params.vectors.size if info.config.params.vectors else None,
        "distance": str(info.config.params.vectors.distance) if info.config.params.vectors else None,
    }

def reset_collection(self):
    self.delete_collection()
    self._collection_ensured = False
    if self._dim is not None:
        self._ensure_collection(self._dim)
```

### 4.11 Payload 索引

远程模式自动为 `user_id`/`agent_id`/`session_id` 创建 payload 索引（加速 filter）：

```python
def _create_payload_indexes(self):
    if self.is_local:
        return  # 本地 Qdrant 不支持 payload index
    for field in ["user_id", "agent_id", "session_id"]:
        try:
            self.client.create_payload_index(
                collection_name=self.collection_name, field_name=field, field_schema="keyword"
            )
        except Exception:
            pass  # 可能已存在
```

## 5. 现有后端适配

### 5.1 SQLAlchemyVectorStore

| 新方法 | 实现 |
|--------|------|
| `update_vector` | DELETE + INSERT 两步（现有 insert_vectors 的 upsert 模式） |
| `delete_collection` | `DROP TABLE IF EXISTS vector_entries` |
| `list_collections` | 返回 `["vector_entries"]` |
| `get_collection_info` | `SELECT COUNT(*) FROM vector_entries` |
| `reset_collection` | DROP + `_create_table()` |
| `search_batch` | 默认循环（非抽象 fallback） |
| `keyword_search` | 返回 None（非抽象 fallback） |

### 5.2 PgvectorVectorStore

| 新方法 | 实现 |
|--------|------|
| `update_vector` | ON CONFLICT upsert（现有 insert_vectors 的 PG 分支） |
| `delete_collection` | `DROP TABLE IF EXISTS vector_entries` |
| `list_collections` | 返回 `["vector_entries"]` |
| `get_collection_info` | `SELECT COUNT(*) FROM vector_entries` |
| `reset_collection` | DROP + `_init_pgvector()` |
| `search_batch` | 默认循环 |
| `keyword_search` | 返回 None |

### 5.3 ChromaVectorStore

| 新方法 | 实现 |
|--------|------|
| `update_vector` | chroma collection `update(ids=..., embeddings=..., metadatas=...)` |
| `delete_collection` | chroma `delete_collection()` |
| `list_collections` | chroma `list_collections()` |
| `get_collection_info` | chroma `count()` |
| `reset_collection` | delete + recreate |

### 5.4 Filter 操作符适配

**SQLAlchemyVectorStore** — `_fetch_rows()` 扩展：
- 精确匹配：现有 `json_extract WHERE` 不变
- 丰富操作符：解析 dict 值，生成对应 SQL（`>=`/`<=`/`IN`/`LIKE` 等）
- AND/OR/NOT：递归构建 SQL WHERE + `AND`/`OR`/`NOT`

**PgvectorVectorStore** — `search_vectors()` 扩展：
- 精确匹配：现有 `payload @> '{"k":"v"}'::jsonb` 不变
- 丰富操作符：PG JSONB 操作符（`->>`/`@>`/`?` 等）

**ChromaVectorStore** — `search_vectors()` 扩展：
- chroma `where` 参数支持 `$eq`/`$ne`/`$gt`/`$gte`/`$lt`/`$lte`/`$in`/`$nin`

**QdrantVectorStore** — `_build_filter()` 全操作符支持（§4.3）。

## 6. Factory 改造

`src/septmuse/storage/relational_stores/factory.py` — `RelationalStoreFactory._resolve_vector_store()` 改造：

```python
@staticmethod
def _resolve_vector_store(config, engine, dialect) -> VectorStoreBase:
    backend = config.vector_store.backend  # "qdrant"(默认) / "sqlite" / "chroma" / "pgvector"

    if backend == "qdrant":
        try:
            from septmuse.storage.vector_stores.qdrant import QdrantVectorStore
            return QdrantVectorStore(
                collection_name="septmuse",
                embedding_model_dims=config.embedder.embedding_dims or 512,
                path=os.getenv("SEPTMUSE_QDRANT_PATH"),
                host=os.getenv("SEPTMUSE_QDRANT_HOST"),
                port=int(os.getenv("SEPTMUSE_QDRANT_PORT", "6333")),
                url=os.getenv("SEPTMUSE_QDRANT_URL"),
                api_key=os.getenv("SEPTMUSE_QDRANT_API_KEY"),
                enable_bm25=True,
            )
        except ImportError:
            logger.warning("qdrant_not_available_fallback_sqlite")
            from septmuse.storage.vector_stores.sqlalchemy_vec import SQLAlchemyVectorStore
            return SQLAlchemyVectorStore(engine)

    if backend == "chroma":
        # 现有逻辑不变
        ...

    # sqlite / pgvector / mysql — dialect-based
    return create_vector_store(engine, dialect)
```

默认值变更：`config.vector_store.backend` 默认从 `"chroma"` → `"qdrant"`。

## 7. HybridRetriever 改造

`src/septmuse/retrieval/hybrid.py` — `_keyword_path()` 检测 VectorStore 内置 BM25：

```python
def _keyword_path(self) -> list[dict[str, Any]] | None:
    with time_block("hybrid_search_components_seconds", {"component": "keyword"}):
        # 优先用 VectorStore 内置 keyword_search
        if hasattr(self.store._vector_store, 'keyword_search'):
            results = self.store._vector_store.keyword_search(query, top_k=internal_limit, filters=vs_filters)
            if results is not None:
                return [{"id": r.id, "score": r.score, "memory": r.payload.get("data", "")} for r in results]
        # 回退到外部 KeywordIndexBase
        try:
            return self.store.keyword_search(query, user_id=user_id, session_id=session_id, top_k=internal_limit)
        except Exception:
            return None
```

检测逻辑：`keyword_search()` 返回非 None → 用 VectorStore 内置 BM25；返回 None → 回退外部 `KeywordIndexBase`。

## 8. 依赖变更

### 8.1 核心依赖

`[project.dependencies]` 新增：
```toml
"qdrant-client>=1.12.0",
```

### 8.2 Optional 依赖

```toml
[project.optional-dependencies]
"fastembed" = ["fastembed>=0.3.0"]  # BM25 稀疏向量编码
"vector-stores" = ["qdrant-client>=1.12.0", "chromadb>=0.5.0", "pgvector>=0.3.0", "fastembed>=0.3.0"]
```

## 9. 环境变量

| 变量 | 默认 | 作用 |
|------|------|------|
| `SEPTMUSE_VECTOR_BACKEND` | `qdrant` | `qdrant`/`sqlite`/`chroma`/`pgvector`（原默认 `chroma` → 改 `qdrant`） |
| `SEPTMUSE_QDRANT_PATH` | `~/.septmuse/qdrant` | Qdrant 本地嵌入路径 |
| `SEPTMUSE_QDRANT_HOST` | 未设 | Qdrant 远程 host（设了走远程，覆盖 path） |
| `SEPTMUSE_QDRANT_PORT` | `6333` | Qdrant 远程端口 |
| `SEPTMUSE_QDRANT_API_KEY` | 未设 | Qdrant Cloud API key |
| `SEPTMUSE_QDRANT_URL` | 未设 | Qdrant 完整 URL（Cloud/自定义端点） |

## 10. 测试策略

```
tests/unit/test_vector_stores/
├── conftest.py                     # 通用 fixture（临时 Qdrant path、Chroma path）
├── test_qdrant_store.py            # Qdrant 本地嵌入全量测试（CRUD + 本地路径）
├── test_qdrant_filters.py          # 10 种操作符 + AND/OR/NOT 逻辑组合
├── test_qdrant_bm25.py             # BM25 稀疏向量 search（fastembed 可选，skip if 未装）
├── test_qdrant_batch.py            # search_batch 原生批量搜索
├── test_abc_new_methods.py         # update_vector / list_collections / col_info / reset
├── test_sqlalchemy_new_methods.py  # SQLAlchemyVectorStore 新方法
├── test_chroma_new_methods.py      # ChromaVectorStore 新方法
└── test_factory_qdrant_default.py  # Factory 默认 Qdrant + 降级路径
```

### 10.1 测试要点

- Qdrant 本地嵌入用 `tmp_path` 目录（不污染 `~/.septmuse/qdrant`），测试完自动清理
- BM25 测试用 `@pytest.mark.skipif(not fastembed_available)` 标记
- Filter 操作符测试覆盖所有 10 种 + 3 种逻辑组合
- Factory 测试：默认 backend=qdrant，`qdrant-client` 未装时降级 SQLAlchemy
- HybridRetriever 测试：Qdrant 后端走内置 keyword_search，SQLAlchemy 后端走外部 KeywordIndex

### 10.2 测试保护

- 现有 `test_qdrant_vector_store.py` 测试需适配新构造函数（`path=` 参数）
- 现有 `test_sqlalchemy_vector_store.py` 测试零改动（新方法有默认 fallback）
- 现有 `test_vector_factory.py` 测试需适配 Qdrant 默认

## 11. 修改范围

| 文件 | 改动 |
|------|------|
| `src/septmuse/storage/vector_stores/base.py` | ABC 扩展：7 新方法 + Filter 操作符文档 |
| `src/septmuse/storage/vector_stores/qdrant.py` | 完全重写：本地嵌入 + 丰富 Filter + BM25 + batch + collection 管理 |
| `src/septmuse/storage/vector_stores/sqlalchemy_vec.py` | 新方法：update_vector + delete_collection + col_info + reset + Filter 操作符 |
| `src/septmuse/storage/vector_stores/pgvector_store.py` | 新方法：update_vector + delete_collection + col_info + reset |
| `src/septmuse/storage/vector_stores/chroma.py` | 新方法：update_vector + delete_collection + col_info + reset |
| `src/septmuse/storage/vector_stores/factory.py` | `create_vector_store` 支持 qdrant backend |
| `src/septmuse/storage/relational_stores/factory.py` | `_resolve_vector_store` 默认 qdrant + 降级 |
| `src/septmuse/retrieval/hybrid.py` | `_keyword_path` 检测 VectorStore 内置 BM25 |
| `src/septmuse/configs/base.py` | `vector_store.backend` 默认改 `qdrant` |
| `pyproject.toml` | `qdrant-client` 核心依赖 + `fastembed` extra |
| `AGENTS.md` | 环境变量 + ABC 方法 + Qdrant 段落 |
| `tests/unit/test_vector_stores/` | 9 测试文件 |
| **总计** | ~18 文件 |

## 12. 未来扩展（子项目 2-4）

- **子项目 2**：FAISS 后端 + SQLAlchemyVectorStore numpy 矩阵优化 + distance strategy 可配置
- **子项目 3**：Pinecone / Milvus / Weaviate / MongoDB / Redis / Elasticsearch（6 个托管后端）
- **子项目 4**：Supabase / Azure AI / Baidu / Cassandra / Databricks / LangChain / Neptune / OpenSearch / S3 / Turbopuffer / Upstash / Valkey / Vertex AI（13 个长尾后端）

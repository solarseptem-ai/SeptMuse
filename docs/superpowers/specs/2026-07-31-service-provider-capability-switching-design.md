# 设计规格：ServiceProvider 能力后端切换机制

> 日期：2026-07-31
> 状态：待实施
> 范围：8 个能力（vector_store / embedder / llm / reranker / entity_extractor / keyword_index / graph_store / search_recipe）的后端注册、解析、配置合并、自省

---

## 1. 背景与动机

SeptMuse 的核心卖点是"零配置"——不设任何环境变量也能跑（默认 SQLite + HashEmbedder）。但能力后端（向量库 / 向量模型 / LLM / reranker 等）的切换机制存在三个问题：

1. **工厂逻辑分散重复**：7 个 `_resolve_*` 函数散落在 4 处（`configs/defaults.py` / `memory/main.py` / `llms/__init__.py` / `retrieval/reranker.py` / `extraction/entity.py`），同一套 match case 逻辑重复 2-3 遍。加新后端要改 4-5 处。
2. **硬编码 match case，无注册表**：加后端要改工厂函数的 match 分支 + 加 config 类 + 加 env 解析，没有统一的注册入口。
3. **配置来源单一**：只能通过环境变量在 init 时切换，无 YAML 配置文件，多环境/profile 不友好。

### 设计目标

- **方便**：加新后端改 1-2 处（新建文件 + manifest 声明一行），不碰工厂 match case。
- **实用**：不过度工程，manifest 是纯数据声明，集中可见。
- **高效**：启动零开销（manifest 是数据不 import 后端模块），resolve 时按需 import。
- **零配置兼容**：无 YAML 无 env 时用代码默认（hash/sqlite），现有扁平环境变量继续工作。

---

## 2. 设计决策

| 决策点 | 选择 | 理由 |
|--------|------|------|
| 工厂模式 | ServiceProvider 容器 | 统一 register/resolve 入口，替代散落 _resolve_* |
| 切换时机 | init 时选（不运行时动态切换） | 简单，避免连接迁移/数据兼容复杂度 |
| 配置来源 | YAML 配置文件 | 多环境/profile 友好 |
| env vs YAML 优先级 | 环境变量覆盖 YAML 覆盖代码默认 | 三层优先级，保留零配置 + 向后兼容 |
| 注册机制 | manifest 声明式注册表 | 零启动开销 + 集中可见 + 测试隔离好 |
| 配置合并 | pydantic-settings BaseSettings | 零代码合并逻辑，原生支持 env > yaml > default |
| 自省 | list_backends + is_available + CLI | 切换前预判可用性 |

### 能力范围（8 个）

| 能力 | manifest key | 代码默认后端 | 可选后端 |
|------|-------------|-------------|---------|
| 向量数据库 | `vector_store` | `sqlite` | sqlite / qdrant / chroma / pgvector |
| 向量模型 | `embedder` | `hash` | hash / onnx / onnx-zh / auto / openai / st |
| LLM | `llm` | `null`（不使用） | openai / ollama / anthropic / dashscope |
| Reranker | `reranker` | `noop` | noop / mmr / cross_encoder / llm |
| 实体抽取器 | `entity_extractor` | `regex` | regex / spacy / none |
| 关键词索引 | `keyword_index` | `sqlite_bm25` | sqlite_bm25 / rank_bm25 / none |
| 图存储 | `graph_store` | `sqlite` | sqlite / age / neo4j |
| 检索配方 | `search_recipe` | `HYBRID_RRF` | HYBRID_RRF / HYBRID_RRF_ENTITY / HYBRID_RRF_CROSS_ENCODER / HYBRID_RRF_MMR / GRAPH_BFS / PROGRESSIVE / FORGETTING |

---

## 3. 架构概览

三层结构，在现有 services/ 层下加 ServiceProvider 层：

```
┌─ services/ 层（保持不变）──────────────────────────────────┐
│  Service ABC + ServiceFactory + ServiceManager + deps.py  │  ← 粗粒度服务级
└───────────────────────────────────────────────────────────┘
        │ ServiceFactory.create(config) 内部委托
        ▼
┌─ services/providers.py + services/registry.py（新建）──────┐
│  ServiceProvider[T] + manifest 声明 + 8 个全局实例        │  ← 细粒度后端级
│  resolve(backend, config=) → 按需 import + 实例化          │
│  list_backends() / is_available() / available_backends()  │
└───────────────────────────────────────────────────────────┘
        ▲ manifest 声明后端 → module_path + class_name
        │
┌─ 各后端模块（embedders/ storage/ retrieval/ ...）─────────┐
│  sqlite_vec.py / qdrant.py / chroma.py / onnx.py / ...    │  ← 后端实现
└───────────────────────────────────────────────────────────┘
```

### 数据流

```
YAML/env → MemoryConfig (BaseSettings 自动合并)
  → ServiceFactory.create(config)
    → ServiceProvider.resolve(backend, config=)   ← 不再 match case
      → manifest 查 entry → importlib.import_module → getattr(class)
        → 实例化（lazy import 后端库）
          → 实例缓存到 ServiceManager
```

### 关键变化

`EmbedderService._resolve_embedder()` 从 40 行 match case → 3 行委托：

```python
# 之前：40 行 match case 硬编码
# 之后：
def _resolve_embedder(self) -> Embedder:
    return embedder_provider.resolve(
        self._config.embedder.backend,
        config=self._config.embedder,
    )
```

---

## 4. manifest 声明式注册表（`services/registry.py`）

纯数据声明，每条 4-tuple `(module_path, class_name, config_cls, optional_deps)`：

```python
from dataclasses import dataclass
from septmuse.configs.vector_stores.sqlite import SQLiteVectorConfig
from septmuse.configs.vector_stores.qdrant import QdrantVectorConfig
# ... 其他 config 类 import

@dataclass(frozen=True)
class BackendEntry:
    module: str           # "septmuse.storage.vector.sqlite_vec"
    cls: str              # "SQLiteVectorStore"
    config_cls: type      # SQLiteVectorConfig
    deps: tuple[str, ...] # ("qdrant_client",) 外部依赖，空 tuple = 零依赖

BACKEND_MANIFEST: dict[str, dict[str, BackendEntry]] = {
    "vector_store": {
        "sqlite":   BackendEntry("septmuse.storage.vector.sqlite_vec", "SQLiteVectorStore",   SQLiteVectorConfig,   ()),
        "qdrant":   BackendEntry("septmuse.storage.vector.qdrant",     "QdrantVectorStore",   QdrantVectorConfig,   ("qdrant_client",)),
        "chroma":   BackendEntry("septmuse.storage.vector.chroma",      "ChromaVectorStore",   ChromaVectorConfig,   ("chromadb",)),
        "pgvector": BackendEntry("septmuse.storage.vector.pgvector",     "PGVectorStore",       PgVectorConfig,        ("psycopg",)),
    },
    "embedder": {
        "hash":    BackendEntry("septmuse.embedders.hash",                "HashEmbedder",              HashEmbedderConfig,   ()),
        "onnx":    BackendEntry("septmuse.embedders.onnx",                "OnnxEmbedder",              OnnxEmbedderConfig,   ("onnxruntime",)),
        "onnx-zh": BackendEntry("septmuse.embedders.onnx",                "OnnxEmbedder",              OnnxEmbedderConfig,   ("onnxruntime",)),
        "auto":    BackendEntry("septmuse.embedders.auto",                "AutoOnnxEmbedder",          OnnxEmbedderConfig,   ("onnxruntime",)),
        "openai":  BackendEntry("septmuse.embedders.openai",             "OpenAIEmbedder",            OpenAIEmbedderConfig, ("openai",)),
        "st":      BackendEntry("septmuse.embedders.sentence_transformers","SentenceTransformerEmbedder", STEConfig,        ("sentence_transformers",)),
    },
    "llm": {
        "openai":    BackendEntry("septmuse.llms.openai",     "OpenAILLM",     OpenAILLMConfig,     ("openai",)),
        "ollama":    BackendEntry("septmuse.llms.ollama",     "OllamaLLM",     OllamaLLMConfig,     ("ollama",)),
        "anthropic": BackendEntry("septmuse.llms.anthropic",  "AnthropicLLM",  AnthropicLLMConfig,  ("anthropic",)),
        "dashscope": BackendEntry("septmuse.llms.dashscope",  "DashScopeLLM",  DashScopeLLMConfig,  ("dashscope",)),
    },
    "reranker": {
        "noop":          BackendEntry("septmuse.retrieval.reranker",       "NoopReranker",          NoopRerankerConfig,        ()),
        "mmr":           BackendEntry("septmuse.retrieval.reranker",       "MMRReranker",            MMRRerankerConfig,          ()),
        "cross_encoder": BackendEntry("septmuse.retrieval.cross_encoder",  "CrossEncoderReranker",  CrossEncoderConfig,        ("onnxruntime",)),
        "llm":           BackendEntry("septmuse.retrieval.reranker",       "LLMReranker",            LLMRerankerConfig,          ()),
    },
    "entity_extractor": {
        "regex": BackendEntry("septmuse.extraction.entity", "RegexEntityExtractor", RegexExtractorConfig,  ()),
        "spacy": BackendEntry("septmuse.extraction.entity", "SpacyEntityExtractor", SpacyExtractorConfig,  ("spacy",)),
        "none":  BackendEntry("septmuse.extraction.entity", "NullEntityExtractor",  BaseEntityExtractorConfig, ()),
    },
    "keyword_index": {
        "sqlite_bm25": BackendEntry("septmuse.storage.keyword.sqlite_bm25", "SQLiteBM25Index", SQLiteBM25Config,  ()),
        "rank_bm25":   BackendEntry("septmuse.storage.keyword.rank_bm25",   "RankBM25Index",   RankBM25Config,    ("rank_bm25",)),
        "none":        BackendEntry("septmuse.storage.keyword.base",        "NullKeywordIndex",BaseKeywordIndexConfig, ()),
    },
    "graph_store": {
        "sqlite": BackendEntry("septmuse.storage.graph.sqlite", "SQLiteGraphStore", SQLiteGraphConfig, ()),
        "age":    BackendEntry("septmuse.storage.graph.age",     "AGEGraphStore",    AgeGraphConfig,    ("psycopg",)),
        "neo4j":  BackendEntry("septmuse.storage.graph.neo4j",   "Neo4jGraphStore",  Neo4jGraphConfig,  ("neo4j",)),
    },
    "search_recipe": {
        "HYBRID_RRF":               BackendEntry("septmuse.retrieval.recipes", "get_recipe", None, ()),
        "HYBRID_RRF_ENTITY":        BackendEntry("septmuse.retrieval.recipes", "get_recipe", None, ()),
        "HYBRID_RRF_CROSS_ENCODER":BackendEntry("septmuse.retrieval.recipes", "get_recipe", None, ()),
        "HYBRID_RRF_MMR":           BackendEntry("septmuse.retrieval.recipes", "get_recipe", None, ()),
        "GRAPH_BFS":                BackendEntry("septmuse.retrieval.recipes", "get_recipe", None, ()),
        "PROGRESSIVE":              BackendEntry("septmuse.retrieval.recipes", "get_recipe", None, ()),
        "FORGETTING":               BackendEntry("septmuse.retrieval.recipes", "get_recipe", None, ()),
    },
}

# 代码默认后端（零配置 fallback）
_DEFAULTS: dict[str, str] = {
    "vector_store": "sqlite",
    "embedder": "hash",
    "llm": "",              # 空字符串 = 不创建 LLM（verbatim 模式）
    "reranker": "noop",
    "entity_extractor": "regex",
    "keyword_index": "sqlite_bm25",
    "graph_store": "sqlite",
    "search_recipe": "HYBRID_RRF",
}
```

### manifest 设计要点

- `module` 是完整 Python 模块路径（`septmuse.storage.vector.sqlite_vec`），resolve 时 `importlib.import_module` 按需加载。
- `cls` 是模块内的类名或函数名（search_recipe 用 `get_recipe` 函数而非类）。
- `config_cls` 是对应 config 类，用于从 YAML 字段实例化 config 对象。`None` 表示该后端不需要 config（如 search_recipe）。
- `deps` 是外部依赖库名列表，`is_available()` 用 `importlib.util.find_spec` 检查，不 import 后端模块。空 tuple = 零依赖。

---

## 5. ServiceProvider API（`services/providers.py`）

```python
from __future__ import annotations

import importlib
import importlib.util
from typing import Generic, TypeVar

from septmuse.services.registry import BACKEND_MANIFEST, _DEFAULTS, BackendEntry

T = TypeVar("T")

class ServiceProvider(Generic[T]):
    """能力后端容器 — manifest 声明 + 按需 import + 健康检查。

    用法:
        embedder_provider.resolve("onnx", config=onnx_config) -> Embedder
        embedder_provider.list_backends() -> ["hash", "onnx", ...]
        embedder_provider.is_available("qdrant") -> False
        embedder_provider.available_backends() -> ["hash", "onnx", "onnx-zh", "auto"]
    """

    def __init__(self, capability: str) -> None:
        if capability not in BACKEND_MANIFEST:
            raise ValueError(f"Unknown capability: {capability}")
        self.capability = capability
        self._manifest: dict[str, BackendEntry] = BACKEND_MANIFEST[capability]
        self._class_cache: dict[str, type] = {}

    def resolve(self, backend: str | None = None, *, config=None, **kwargs) -> T:
        """解析后端 → 按需 import → 实例化。

        Args:
            backend: 后端名称，None 用代码默认（_DEFAULTS）。
            config: 后端专属 config 对象（如 OnnxEmbedderConfig），可选。
            **kwargs: 直接传给后端构造函数的参数。

        Returns:
            后端实例。

        Raises:
            ValueError: backend 不在 manifest 中。
            ImportError: 后端外部依赖未安装。
        """
        name = backend if backend is not None else _DEFAULTS.get(self.capability, "")
        if name == "":
            return None  # llm 默认空字符串 = 不创建
        entry = self._manifest.get(name)
        if entry is None:
            raise ValueError(
                f"Unknown {self.capability} backend: {name}. "
                f"Available: {self.list_backends()}"
            )
        cls = self._import_class(entry.module, entry.cls)
        if config is not None:
            return cls(**_config_to_kwargs(config, entry))
        if entry.config_cls is None and not kwargs:
            # 无 config 的后端（如 search_recipe），backend 名称作为参数
            kwargs = {"name": name}
        return cls(**kwargs)

    def list_backends(self) -> list[str]:
        """列出所有已注册后端（不论依赖是否安装）。"""
        return list(self._manifest.keys())

    def is_available(self, backend: str) -> bool:
        """检查后端外部依赖是否安装（find_spec，不 import 后端模块）。"""
        entry = self._manifest.get(backend)
        if entry is None:
            return False
        return all(importlib.util.find_spec(dep) is not None for dep in entry.deps)

    def available_backends(self) -> list[str]:
        """已装依赖的可用后端。"""
        return [b for b in self.list_backends() if self.is_available(b)]

    def default_backend(self) -> str:
        """代码默认后端（零配置 fallback）。"""
        return _DEFAULTS.get(self.capability, "")

    def _import_class(self, module_path: str, class_name: str) -> type:
        """按需 import 模块 + getattr 类，结果缓存。"""
        cache_key = f"{module_path}.{class_name}"
        if cache_key in self._class_cache:
            return self._class_cache[cache_key]
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        self._class_cache[cache_key] = cls
        return cls


def _config_to_kwargs(config, entry: BackendEntry) -> dict:
    """从 config 对象提取后端构造函数参数。

    config 是 pydantic model（如 OnnxEmbedderConfig），用 model_dump() 转 dict。
    search_recipe 的 entry.cls 是函数（get_recipe），特殊处理。
    """
    if entry.config_cls is None:
        # search_recipe 等无 config 的后端，config 本身就是参数
        if hasattr(config, "model_dump"):
            return config.model_dump()
        return {"name": config} if isinstance(config, str) else {}
    if hasattr(config, "model_dump"):
        return config.model_dump(exclude_none=True)
    return {}


# 8 个全局 Provider 实例
vector_store_provider        = ServiceProvider("vector_store")
embedder_provider            = ServiceProvider("embedder")
llm_provider                 = ServiceProvider("llm")
reranker_provider            = ServiceProvider("reranker")
entity_extractor_provider   = ServiceProvider("entity_extractor")
keyword_index_provider       = ServiceProvider("keyword_index")
graph_store_provider         = ServiceProvider("graph_store")
search_recipe_provider       = ServiceProvider("search_recipe")

# 便捷访问
ALL_PROVIDERS: dict[str, ServiceProvider] = {
    "vector_store":       vector_store_provider,
    "embedder":           embedder_provider,
    "llm":                llm_provider,
    "reranker":           reranker_provider,
    "entity_extractor":   entity_extractor_provider,
    "keyword_index":      keyword_index_provider,
    "graph_store":        graph_store_provider,
    "search_recipe":      search_recipe_provider,
}
```

### 核心特性

| 特性 | 实现 |
|------|------|
| 零启动开销 | manifest 是数据，不 import 任何后端模块 |
| 按需 import | `resolve()` 首次调用才 `importlib.import_module`，结果缓存到 `_class_cache` |
| 健康检查 | `is_available()` 用 `find_spec` 查外部依赖，不 import 后端模块 |
| 能力自省 | `list_backends()` + `available_backends()` 列出全部/可用 |
| 统一签名 | 8 个 Provider 同构，`resolve(backend, config=)` 一致 |

---

## 6. pydantic-settings 配置合并

### MemoryConfig 改继承 BaseSettings

```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class MemoryConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SEPTMUSE_",          # SEPTMUSE_INFER → infer 字段
        env_nested_delimiter="__",       # SEPTMUSE_EMBEDDER__MODEL → embedder.model
        yaml_file=["./septmuse.yaml", "~/.septmuse/config.yaml"],  # 多路径 fallback
        extra="ignore",
    )
    database: DatabaseConfig = DatabaseConfig()
    vector_store: VectorStoreConfig = SQLiteVectorConfig()
    embedder: EmbedderConfig = HashEmbedderConfig()
    llm: LLMConfig | None = None
    reranker: RerankerConfig = NoopRerankerConfig()
    entity_extractor: EntityExtractorConfig = RegexExtractorConfig()
    keyword_index: KeywordIndexConfig = SQLiteBM25Config()
    graph_store: GraphStoreConfig = SQLiteGraphConfig()
    search_recipe: str = "HYBRID_RRF"
    infer: bool = False
```

### 三层优先级

```
环境变量 (SEPTMUSE_*)     ← 最高，覆盖一切
    ▼ 无则读
YAML (septmuse.yaml)      ← 中间层
    ▼ 无则用
代码默认 (hash/sqlite/...)  ← 零配置 fallback
```

### 向后兼容：扁平环境变量别名

现有用户用 `SEPTMUSE_EMBEDDER=onnx`（扁平），不是 `SEPTMUSE_EMBEDDER__BACKEND=onnx`（嵌套）。用 `model_validator` 做别名映射：

```python
@model_validator(mode="before")
@classmethod
def _flat_env_aliases(cls, data: dict) -> dict:
    """旧版扁平环境变量 → 嵌套字段（向后兼容）。

    SEPTMUSE_EMBEDDER=onnx → embedder.backend="onnx"
    SEPTMUSE_VECTOR_BACKEND=qdrant → vector_store.backend="qdrant"
    ...
    """
    aliases = {
        "SEPTMUSE_EMBEDDER":          ("embedder", "backend"),
        "SEPTMUSE_VECTOR_BACKEND":    ("vector_store", "backend"),
        "SEPTMUSE_KEYWORD_BACKEND":   ("keyword_index", "backend"),
        "SEPTMUSE_GRAPH_BACKEND":     ("graph_store", "backend"),
        "SEPTMUSE_RERANKER":          ("reranker", "backend"),
        "SEPTMUSE_ENTITY_EXTRACTOR":  ("entity_extractor", "backend"),
        "SEPTMUSE_LLM":               ("llm", "backend"),
    }
    for env_name, (section, field) in aliases.items():
        val = os.getenv(env_name)
        if val and isinstance(data, dict) and section in data:
            data[section][field] = val
    return data
```

### YAML 配置文件格式

```yaml
# septmuse.yaml — 项目根目录 或 ~/.septmuse/config.yaml
database:
  db_path: ~/.septmuse/septmuse.db

vector_store:
  backend: sqlite           # sqlite/qdrant/chroma/pgvector
  qdrant:                   # 后端参数，仅当 backend=qdrant 生效
    host: localhost
    port: 6333

embedder:
  backend: hash             # hash/onnx/onnx-zh/auto/openai/st
  onnx:
    model: Xenova/all-MiniLM-L6-v2

llm:
  backend: null             # null/openai/ollama/anthropic/dashscope
  openai:
    model: gpt-4o-mini

reranker:
  backend: noop             # noop/mmr/cross_encoder/llm

entity_extractor:
  backend: regex            # regex/spacy/none

keyword_index:
  backend: sqlite_bm25      # sqlite_bm25/rank_bm25/none

graph_store:
  backend: sqlite           # sqlite/age/neo4j

search_recipe:
  name: HYBRID_RRF          # HYBRID_RRF/.../FORGETTING

infer: false
```

### 实现时需验证的 API

> 用 context7 确认以下 pydantic-settings API 真实存在（避免幻觉）：
> 1. `SettingsConfigDict(yaml_file=...)` 是否原生支持 YAML（可能需 `pydantic-settings[yaml]` extra）
> 2. `yaml_file` 是否支持多路径 fallback 列表
> 3. `model_validator(mode="before")` 在 BaseSettings 中的执行时机（env 解析之前还是之后）
> 4. pydantic-settings 版本要求（YAML 支持是 v2.x 特性）

---

## 7. ConfigService 增强

```python
class ConfigService(Service):
    """配置服务 — YAML 加载 + env 覆盖 + 自省。"""

    def __init__(self, config: MemoryConfig | None = None) -> None:
        self._config = config or MemoryConfig()  # BaseSettings 自动合并
        self._reload_count = 0
        self.set_ready()

    @property
    def config(self) -> MemoryConfig:
        return self._config

    def reload(self) -> MemoryConfig:
        """重新从 YAML + env 加载。"""
        self._config = MemoryConfig()
        self._reload_count += 1
        return self._config

    def describe(self) -> dict:
        """当前生效配置自省（哪些来源生效）。

        返回每个能力的 backend 值 + 来源（env/yaml/code_default）。

        示例返回:
            {
                "embedder": {"backend": "onnx", "source": "env"},
                "vector_store": {"backend": "sqlite", "source": "yaml"},
                "llm": {"backend": "null", "source": "code_default"},
                ...
            }
        """
        ...
```

---

## 8. 现有 `_resolve_*` 迁移路径

### 迁移命运表

| 现有函数 | 位置 | 命运 |
|---------|------|------|
| `_resolve_embedder()` | `configs/defaults.py` | **删除**，match case 搬到 manifest |
| `_resolve_llm()` | `configs/defaults.py` | **删除** |
| `_resolve_reranker()` | `configs/defaults.py` | **删除** |
| `_resolve_vector_store()` | `configs/defaults.py` | **删除** |
| `_resolve_keyword_index()` | `configs/defaults.py` | **删除** |
| `_resolve_graph_store()` | `configs/defaults.py` | **删除** |
| `_resolve_entity_extractor()` | `configs/defaults.py` | **删除** |
| `_resolve_embedder(config)` | `memory/main.py` | **删除**，改 `embedder_provider.resolve()` |
| `_resolve_llm(config)` | `llms/__init__.py` | **删除**，改 `llm_provider.resolve()` |
| `_resolve_reranker(backend, ...)` | `retrieval/reranker.py` | **删除**，改 `reranker_provider.resolve()` |
| `_resolve_entity_extractor(config)` | `extraction/entity.py` | **删除**，改 `entity_extractor_provider.resolve()` |

### default_config() 简化

```python
# 之前：default_config() 调 7 个 _resolve_* 组装 MemoryConfig
# 之后：
def default_config() -> MemoryConfig:
    """零配置默认 — BaseSettings 自动从 env + yaml + default 合并。"""
    return MemoryConfig()
```

### ServiceFactory.create() 变化

`EmbedderService._resolve_embedder()` 从 40 行 match case → 3 行委托：

```python
class EmbedderService(Service):
    def _resolve_embedder(self) -> Embedder:
        return embedder_provider.resolve(
            self._config.embedder.backend,
            config=self._config.embedder,
        )
```

同理 `LLMService` / `RetrievalService`（如实现）/ 各 Service 的 `_resolve_*` 均改为委托 provider。

---

## 9. CLI 自省命令

### `septmuse backends` — 列出所有能力 + 可用后端

```
$ septmuse backends
vector_store:     sqlite[✓] qdrant[✗] chroma[✗] pgvector[✗]
embedder:         hash[✓] onnx[✓] onnx-zh[✓] auto[✓] openai[✗] st[✗]
llm:              openai[✗] ollama[✗] anthropic[✗] dashscope[✗]
reranker:         noop[✓] mmr[✓] cross_encoder[✗] llm[✗]
entity_extractor: regex[✓] spacy[✗] none[✓]
keyword_index:    sqlite_bm25[✓] rank_bm25[✗] none[✓]
graph_store:      sqlite[✓] age[✗] neo4j[✗]
search_recipe:    HYBRID_RRF[✓] HYBRID_RRF_ENTITY[✓] HYBRID_RRF_CROSS_ENCODER[✓]
                  HYBRID_RRF_MMR[✓] GRAPH_BFS[✓] PROGRESSIVE[✓] FORGETTING[✓]
```

### `septmuse config show` — 当前生效配置 + 来源

```
$ septmuse config show
embedder:        backend=onnx     source=env       # SEPTMUSE_EMBEDDER 覆盖
vector_store:    backend=sqlite   source=yaml      # septmuse.yaml 声明
llm:             backend=null     source=code_default
reranker:        backend=noop     source=code_default
search_recipe:   name=HYBRID_RRF  source=code_default
infer:           false            source=code_default
```

### CLI 实现位置

在 `cli/main.py` 新增 `backends` 和 `config` 子命令（argparse），调用 `ALL_PROVIDERS` 遍历输出。

---

## 10. 测试策略

| 测试类型 | 覆盖点 | 文件 |
|---------|--------|------|
| manifest 完整性 | 8 个能力的 manifest 都有代码默认后端（零配置能跑） | `tests/unit/test_manifest.py` |
| ServiceProvider | resolve/list_backends/is_available/available_backends/default_backend 五方法 | `tests/unit/test_service_provider.py` |
| 配置三层合并 | env > yaml > default 覆盖优先级 | `tests/unit/test_config_merge.py` |
| 向后兼容 | 旧扁平 `SEPTMUSE_EMBEDDER=onnx` 继续工作 | `tests/unit/test_config_compat.py` |
| CLI 自省 | `septmuse backends` / `config show` 输出格式 | `tests/unit/test_cli_backends.py` |
| 现有测试不破坏 | `_resolve_*` 删除后，686 passed + 23 e2e 基线不退化 | 现有测试不动 |

### 测试原则

- 现有全部单元测试、接口测试案例**固定不动**，禁止改测试代码绕过缺陷。
- 仅新增测试覆盖新功能。
- `is_available` 测试不依赖真实安装（mock `find_spec`）。

---

## 11. 迁移批次（6 批，每批可独立验证）

| 批次 | 内容 | 验证 | 预计 |
|------|------|------|------|
| 1 | 新建 `services/registry.py` + `services/providers.py`（manifest + ServiceProvider） | 单元测试五方法 + manifest 完整性 | 中 |
| 2 | `MemoryConfig` 改 `BaseSettings` + YAML 加载 + `model_validator` 别名 | 配置合并测试 + 向后兼容测试 | 大 |
| 3 | `configs/defaults.py` 7 个 `_resolve_*` 删除，`default_config()` 简化 | import 测试 + ruff | 小 |
| 4 | `memory/main.py` + `llms/__init__.py` + `retrieval/reranker.py` + `extraction/entity.py` 的 `_resolve_*` 替换为 provider.resolve() | pytest 基线不退化（686 passed） | 中 |
| 5 | CLI 加 `backends` + `config show` 命令 | CLI 测试 | 小 |
| 6 | ruff + pytest 全量验证 | 全绿 | 小 |

### 批次依赖

```
批次 1（registry + providers）
  └→ 批次 2（MemoryConfig BaseSettings）← 依赖 manifest 的 config_cls
       └→ 批次 3（删 _resolve_* + default_config 简化）
            └→ 批次 4（调用方迁移 provider.resolve()）
                 ├→ 批次 5（CLI 自省）
                 └→ 批次 6（全量验证）
```

---

## 12. 风险与缓解

| 风险 | 缓解 |
|------|------|
| pydantic-settings YAML 支持可能需 extra 或版本限制 | 实现前用 context7 验证 API；如不支持 YAML，fallback 到手写 YAML 加载 + model_validator 合并 |
| 现有测试引用 `_resolve_*` 内部函数（如 `_resolve_embedder`） | grep 确认引用点，改用 provider.resolve() 或保留 thin wrapper 兼容 |
| manifest 的 config_cls 需 import 所有 config 类 → 启动开销 | config 类是 pydantic BaseModel，import 开销极小（无外部依赖）；与后端实现模块解耦 |
| search_recipe 的 entry.cls 是函数（`get_recipe`）而非类 | `_config_to_kwargs` 特殊处理：config 是字符串时传 `{"name": config}` |
| `llm` 默认空字符串 = 不创建 LLM | `resolve()` 在 `name == ""` 时返回 None，调用方需处理 None |

---

## 13. 不做的事（YAGNI）

- **运行时动态切换**：不实现 `memory.swap("vector_store", "qdrant")`。init 时选够了，运行时切换要处理连接迁移/数据兼容，复杂度不值得。
- **entry_points 插件发现**：SeptMuse 是单包项目，第三方插件发现过度工程。
- **配置热重载**：`ConfigService.reload()` 存在但只手动调用，不监听文件变化自动重载。
- **多 YAML profile**（dev.yaml/prod.yaml）：一个 YAML 够了，多环境用环境变量覆盖。
- **后端实例池/连接池**：ServiceManager 单例缓存够了，不引入连接池。

---

## 14. 验收标准

- [ ] `septmuse backends` 命令输出 8 个能力的可用后端列表
- [ ] `septmuse config show` 输出当前生效配置 + 来源
- [ ] 无 YAML 无 env 时，`Memory()` 零配置启动（SQLite + HashEmbedder）
- [ ] `SEPTMUSE_EMBEDDER=onnx` 旧环境变量继续工作
- [ ] `septmuse.yaml` 中 `embedder.backend: onnx` 生效
- [ ] 环境变量覆盖 YAML（`SEPTMUSE_EMBEDDER=st` + yaml `embedder.backend: onnx` → 用 st）
- [ ] 加新后端只需 2 处改动（新建后端文件 + manifest 加一行）
- [ ] `ruff check src/ tests/` 全绿
- [ ] `pytest tests/unit/ tests/e2e/` 基线不退化（686 passed + 22 skipped + 23 e2e）

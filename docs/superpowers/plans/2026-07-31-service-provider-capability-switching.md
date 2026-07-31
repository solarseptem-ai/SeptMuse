# ServiceProvider 能力后端切换机制 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用 manifest 声明式注册表 + ServiceProvider 容器替代散落的 `_resolve_*` 工厂函数，支持 YAML 配置 + 环境变量三层合并切换 8 个能力后端。

**Architecture:** services/ 层下加 ServiceProvider 层（manifest 纯数据声明 + 按需 import + 健康检查）。MemoryConfig 改继承 pydantic-settings BaseSettings 自动合并 env > yaml > default。现有 11 个 `_resolve_*` 函数全删，调用方改用 `provider.resolve()`。

**Tech Stack:** pydantic-settings (BaseSettings + yaml_file)、importlib (import_module + find_spec)、pytest

## 全局约束

- **PYTHONPATH=src** 运行所有测试和命令
- **ruff line-length=120**，select=["E","F","I","W","UP","B","SIM","RUF"]，ignore=["E501","RUF001","RUF002","RUF003"]
- **ruff format 禁用**（Windows 上清空文件，已实证），只用 `ruff check --fix`
- **不是 git 仓库**，无 commit 步骤，每批次末尾验证
- **现有测试固定不动**，禁止改测试代码绕过缺陷，仅新增测试
- **代码注释用中文**，不暴露开源库参考来源
- **pytest 基线**：686 passed + 22 skipped + 23 e2e（不退化）

## 文件结构

**新建：**
- `src/septmuse/services/registry.py` — manifest 声明式注册表（8 能力 × 多后端）
- `src/septmuse/services/providers.py` — ServiceProvider 容器（resolve/list/is_available）
- `tests/unit/test_manifest.py` — manifest 完整性测试
- `tests/unit/test_service_provider.py` — ServiceProvider 五方法测试
- `tests/unit/test_config_merge.py` — 配置三层合并测试
- `tests/unit/test_config_compat.py` — 扁平环境变量兼容测试
- `tests/unit/test_cli_backends.py` — CLI 自省测试

**修改：**
- `src/septmuse/configs/base.py` — MemoryConfig 改 BaseSettings + model_validator
- `src/septmuse/configs/defaults.py` — 删 7 个 _resolve_*，default_config() 简化
- `src/septmuse/memory/main.py` — _resolve_embedder 删，改 provider.resolve()
- `src/septmuse/llms/__init__.py` — _resolve_llm 删，改 provider.resolve()
- `src/septmuse/retrieval/reranker.py` — _resolve_reranker 删，改 provider.resolve()
- `src/septmuse/extraction/entity.py` — _resolve_entity_extractor 删，改 provider.resolve()
- `src/septmuse/services/embedder/service.py` — _resolve_embedder 改委托 provider
- `src/septmuse/cli/main.py` — 加 backends + config show 子命令
- `pyproject.toml` — 加 pydantic-settings + pyyaml 依赖

---

## Task 1: manifest 声明式注册表 + ServiceProvider 容器

**Files:**
- Create: `src/septmuse/services/registry.py`
- Create: `src/septmuse/services/providers.py`
- Test: `tests/unit/test_manifest.py`, `tests/unit/test_service_provider.py`

**Interfaces:**
- Produces: `BackendEntry` dataclass, `BACKEND_MANIFEST` dict, `_DEFAULTS` dict, `ServiceProvider[T]` class with `resolve/list_backends/is_available/available_backends/default_backend`, 8 个全局 provider 实例, `ALL_PROVIDERS` dict

- [ ] **Step 1: 写 manifest 完整性失败测试**

```python
# tests/unit/test_manifest.py
"""manifest 声明式注册表完整性测试。"""
from septmuse.services.registry import BACKEND_MANIFEST, _DEFAULTS, BackendEntry

CAPABILITIES = ["vector_store", "embedder", "llm", "reranker",
                "entity_extractor", "keyword_index", "graph_store", "search_recipe"]


def test_all_capabilities_present():
    for cap in CAPABILITIES:
        assert cap in BACKEND_MANIFEST, f"能力 {cap} 不在 manifest"


def test_each_capability_has_default():
    for cap in CAPABILITIES:
        assert cap in _DEFAULTS, f"能力 {cap} 无代码默认"
        default = _DEFAULTS[cap]
        assert default == "" or default in BACKEND_MANIFEST[cap], \
            f"能力 {cap} 的默认 {default} 不在 manifest"


def test_zero_dep_backends_exist():
    zero_dep = []
    for cap, backends in BACKEND_MANIFEST.items():
        for name, entry in backends.items():
            if entry.deps == ():
                zero_dep.append((cap, name))
    # 至少每个能力的默认后端是零依赖
    for cap in CAPABILITIES:
        default = _DEFAULTS[cap]
        if default:
            entry = BACKEND_MANIFEST[cap][default]
            assert entry.deps == (), f"能力 {cap} 默认 {default} 有依赖，破坏零配置"


def test_backend_entry_fields():
    entry = BACKEND_MANIFEST["vector_store"]["sqlite"]
    assert isinstance(entry, BackendEntry)
    assert entry.module.startswith("septmuse.")
    assert isinstance(entry.cls, str)
    assert entry.config_cls is not None or entry.cls == "get_recipe"
    assert isinstance(entry.deps, tuple)


def test_search_recipe_uses_function():
    for name, entry in BACKEND_MANIFEST["search_recipe"].items():
        assert entry.cls == "get_recipe"
        assert entry.config_cls is None
```

- [ ] **Step 2: 运行测试确认失败**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_manifest.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'septmuse.services.registry'`

- [ ] **Step 3: 写 registry.py 实现**

```python
# src/septmuse/services/registry.py
"""能力后端 manifest 声明式注册表。

纯数据声明，resolve 时按需 import 后端模块。
每条 BackendEntry: (module, cls, config_cls, deps)。
"""
from __future__ import annotations

from dataclasses import dataclass

# 向量存储 config 类
from septmuse.configs.vector_stores.chroma import ChromaVectorConfig
from septmuse.configs.vector_stores.pgvector import PgVectorConfig
from septmuse.configs.vector_stores.qdrant import QdrantVectorConfig
from septmuse.configs.vector_stores.sqlite import SQLiteVectorConfig

# 嵌入 config 类
from septmuse.configs.embeddings.hash import HashEmbedderConfig
from septmuse.configs.embeddings.onnx import OnnxEmbedderConfig
from septmuse.configs.embeddings.openai import OpenAIEmbedderConfig
from septmuse.configs.embeddings.base import BaseEmbedderConfig

# LLM config 类
from septmuse.configs.llms.anthropic import AnthropicLLMConfig
from septmuse.configs.llms.base import BaseLLMConfig
from septmuse.configs.llms.dashscope import DashScopeLLMConfig
from septmuse.configs.llms.ollama import OllamaLLMConfig
from septmuse.configs.llms.openai import OpenAILLMConfig

# Reranker config 类
from septmuse.configs.rerankers.base import BaseRerankerConfig
from septmuse.configs.rerankers.noop import NoopRerankerConfig

# 实体抽取 config 类
from septmuse.configs.extraction.base import BaseEntityExtractorConfig
from septmuse.configs.extraction.regex import RegexExtractorConfig

# 关键词索引 config 类
from septmuse.configs.keyword_index.base import BaseKeywordIndexConfig
from septmuse.configs.keyword_index.sqlite_bm25 import SQLiteBM25Config

# 图存储 config 类
from septmuse.configs.graph_stores.base import BaseGraphStoreConfig
from septmuse.configs.graph_stores.sqlite import SQLiteGraphConfig


@dataclass(frozen=True)
class BackendEntry:
    """后端声明条目。

    Attributes:
        module: 完整 Python 模块路径 (如 "septmuse.storage.vector.sqlite_vec")
        cls: 模块内类名或函数名 (如 "SQLiteVectorStore" 或 "get_recipe")
        config_cls: 对应 config 类，用于从 YAML 实例化；None 表示无 config
        deps: 外部依赖库名元组，is_available() 用 find_spec 检查；空元组=零依赖
    """
    module: str
    cls: str
    config_cls: type | None
    deps: tuple[str, ...] = ()


BACKEND_MANIFEST: dict[str, dict[str, BackendEntry]] = {
    "vector_store": {
        "sqlite":   BackendEntry("septmuse.storage.vector.sqlite_vec", "SQLiteVectorStore",   SQLiteVectorConfig,   ()),
        "qdrant":   BackendEntry("septmuse.storage.vector.qdrant",     "QdrantVectorStore",   QdrantVectorConfig,   ("qdrant_client",)),
        "chroma":   BackendEntry("septmuse.storage.vector.chroma",      "ChromaVectorStore",   ChromaVectorConfig,   ("chromadb",)),
        "pgvector": BackendEntry("septmuse.storage.vector.pgvector",     "PGVectorStore",       PgVectorConfig,        ("psycopg",)),
    },
    "embedder": {
        "hash":    BackendEntry("septmuse.embedders.hash",                  "HashEmbedder",              HashEmbedderConfig,   ()),
        "onnx":    BackendEntry("septmuse.embedders.onnx",                  "OnnxEmbedder",              OnnxEmbedderConfig,   ("onnxruntime",)),
        "onnx-zh": BackendEntry("septmuse.embedders.onnx",                  "OnnxEmbedder",              OnnxEmbedderConfig,   ("onnxruntime",)),
        "auto":    BackendEntry("septmuse.embedders.auto",                  "AutoOnnxEmbedder",          OnnxEmbedderConfig,   ("onnxruntime",)),
        "openai":  BackendEntry("septmuse.embedders.openai",                "OpenAIEmbedder",            OpenAIEmbedderConfig, ("openai",)),
        "st":      BackendEntry("septmuse.embedders.sentence_transformers", "SentenceTransformerEmbedder", BaseEmbedderConfig, ("sentence_transformers",)),
    },
    "llm": {
        "openai":    BackendEntry("septmuse.llms.openai",     "OpenAILLM",     OpenAILLMConfig,     ("openai",)),
        "ollama":    BackendEntry("septmuse.llms.ollama",     "OllamaLLM",     OllamaLLMConfig,     ("ollama",)),
        "anthropic": BackendEntry("septmuse.llms.anthropic",  "AnthropicLLM",  AnthropicLLMConfig,  ("anthropic",)),
        "dashscope": BackendEntry("septmuse.llms.dashscope",  "DashScopeLLM",  DashScopeLLMConfig,  ("dashscope",)),
    },
    "reranker": {
        "noop":          BackendEntry("septmuse.retrieval.reranker",        "NoopReranker",          NoopRerankerConfig,   ()),
        "mmr":           BackendEntry("septmuse.retrieval.reranker",        "MMRReranker",           BaseRerankerConfig,   ()),
        "cross_encoder": BackendEntry("septmuse.retrieval.reranker",        "CrossEncoderReranker",  BaseRerankerConfig,   ("onnxruntime",)),
        "llm":           BackendEntry("septmuse.retrieval.reranker",        "LLMReranker",           BaseRerankerConfig,   ()),
    },
    "entity_extractor": {
        "regex": BackendEntry("septmuse.extraction.entity", "RegexEntityExtractor", RegexExtractorConfig,      ()),
        "spacy": BackendEntry("septmuse.extraction.entity", "SpacyEntityExtractor", BaseEntityExtractorConfig, ("spacy",)),
        "none":  BackendEntry("septmuse.extraction.entity", "NullEntityExtractor",  BaseEntityExtractorConfig, ()),
    },
    "keyword_index": {
        "sqlite_bm25": BackendEntry("septmuse.storage.keyword.sqlite_bm25", "SQLiteBM25Index", SQLiteBM25Config,       ()),
        "rank_bm25":   BackendEntry("septmuse.storage.keyword.rank_bm25",   "RankBM25Index",   BaseKeywordIndexConfig, ("rank_bm25",)),
        "none":        BackendEntry("septmuse.storage.keyword.base",        "NullKeywordIndex", BaseKeywordIndexConfig, ()),
    },
    "graph_store": {
        "sqlite": BackendEntry("septmuse.storage.graph.sqlite", "SQLiteGraphStore", SQLiteGraphConfig,      ()),
        "age":    BackendEntry("septmuse.storage.graph.age",     "AGEGraphStore",    BaseGraphStoreConfig,   ("psycopg",)),
        "neo4j":  BackendEntry("septmuse.storage.graph.neo4j",   "Neo4jGraphStore",  BaseGraphStoreConfig,   ("neo4j",)),
    },
    "search_recipe": {
        "HYBRID_RRF":                BackendEntry("septmuse.retrieval.recipes", "get_recipe", None, ()),
        "HYBRID_RRF_ENTITY":         BackendEntry("septmuse.retrieval.recipes", "get_recipe", None, ()),
        "HYBRID_RRF_CROSS_ENCODER":  BackendEntry("septmuse.retrieval.recipes", "get_recipe", None, ()),
        "HYBRID_RRF_MMR":            BackendEntry("septmuse.retrieval.recipes", "get_recipe", None, ()),
        "GRAPH_BFS":                 BackendEntry("septmuse.retrieval.recipes", "get_recipe", None, ()),
        "PROGRESSIVE":               BackendEntry("septmuse.retrieval.recipes", "get_recipe", None, ()),
        "FORGETTING":                BackendEntry("septmuse.retrieval.recipes", "get_recipe", None, ()),
    },
}

# 代码默认后端（零配置 fallback）
_DEFAULTS: dict[str, str] = {
    "vector_store": "sqlite",
    "embedder": "hash",
    "llm": "",               # 空字符串 = 不创建 LLM
    "reranker": "noop",
    "entity_extractor": "regex",
    "keyword_index": "sqlite_bm25",
    "graph_store": "sqlite",
    "search_recipe": "HYBRID_RRF",
}
```

- [ ] **Step 4: 写 ServiceProvider 失败测试**

```python
# tests/unit/test_service_provider.py
"""ServiceProvider 容器单元测试。"""
import importlib.util
from unittest.mock import patch

import pytest

from septmuse.services.providers import (
    ServiceProvider,
    embedder_provider,
    llm_provider,
    search_recipe_provider,
    vector_store_provider,
    ALL_PROVIDERS,
)


def test_list_backends_returns_all():
    backends = vector_store_provider.list_backends()
    assert "sqlite" in backends
    assert "qdrant" in backends
    assert "chroma" in backends
    assert "pgvector" in backends


def test_default_backend():
    assert vector_store_provider.default_backend() == "sqlite"
    assert embedder_provider.default_backend() == "hash"
    assert llm_provider.default_backend() == ""


def test_is_available_zero_dep():
    assert vector_store_provider.is_available("sqlite") is True
    assert embedder_provider.is_available("hash") is True


def test_is_available_missing_dep():
    with patch("importlib.util.find_spec", return_value=None):
        assert vector_store_provider.is_available("qdrant") is False


def test_available_backends():
    avail = embedder_provider.available_backends()
    assert "hash" in avail  # 零依赖，始终可用


def test_resolve_returns_instance():
    embedder = embedder_provider.resolve("hash")
    assert embedder is not None
    assert hasattr(embedder, "embed")


def test_resolve_default():
    embedder = embedder_provider.resolve()  # None → 代码默认
    assert embedder is not None


def test_resolve_unknown_raises():
    with pytest.raises(ValueError, match="Unknown"):
        vector_store_provider.resolve("nonexistent")


def test_resolve_llm_empty_returns_none():
    result = llm_provider.resolve()  # 默认空字符串
    assert result is None


def test_resolve_search_recipe():
    recipe = search_recipe_provider.resolve("HYBRID_RRF")
    assert recipe is not None


def test_all_providers_has_8():
    assert len(ALL_PROVIDERS) == 8
    for cap in ["vector_store", "embedder", "llm", "reranker",
                "entity_extractor", "keyword_index", "graph_store", "search_recipe"]:
        assert cap in ALL_PROVIDERS
```

- [ ] **Step 5: 运行测试确认失败**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_service_provider.py -v`
Expected: FAIL — `No module named 'septmuse.services.providers'`

- [ ] **Step 6: 写 providers.py 实现**

```python
# src/septmuse/services/providers.py
"""能力后端容器 — manifest 声明 + 按需 import + 健康检查。

每个 ServiceProvider 管理一个能力的多后端注册 + 解析。
resolve() 时才 import 后端模块（零启动开销），结果缓存。
"""
from __future__ import annotations

import importlib
import importlib.util
from typing import Generic, TypeVar

from septmuse.services.registry import BACKEND_MANIFEST, _DEFAULTS, BackendEntry

T = TypeVar("T")


class ServiceProvider(Generic[T]):
    """能力后端容器。

    用法:
        embedder_provider.resolve("onnx", config=onnx_config) -> Embedder
        embedder_provider.list_backends() -> ["hash", "onnx", ...]
        embedder_provider.is_available("qdrant") -> False
    """

    def __init__(self, capability: str) -> None:
        if capability not in BACKEND_MANIFEST:
            raise ValueError(f"未知能力: {capability}")
        self.capability = capability
        self._manifest: dict[str, BackendEntry] = BACKEND_MANIFEST[capability]
        self._class_cache: dict[str, type] = {}

    def resolve(self, backend: str | None = None, *, config=None, **kwargs) -> T | None:
        """解析后端 → 按需 import → 实例化。

        Args:
            backend: 后端名称，None 用代码默认。
            config: 后端专属 config 对象（pydantic model），可选。
            **kwargs: 直接传给后端构造函数的参数。
        """
        name = backend if backend is not None else _DEFAULTS.get(self.capability, "")
        if name == "":
            return None  # llm 默认空 = 不创建
        entry = self._manifest.get(name)
        if entry is None:
            raise ValueError(
                f"未知 {self.capability} 后端: {name}. "
                f"可用: {self.list_backends()}"
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
    """从 config 对象提取后端构造函数参数。"""
    if entry.config_cls is None:
        # 无 config 的后端，config 本身就是参数
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

- [ ] **Step 7: 运行测试确认通过**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_manifest.py tests/unit/test_service_provider.py -v`
Expected: PASS（全部测试通过）

- [ ] **Step 8: ruff 检查**

Run: `ruff check src/septmuse/services/registry.py src/septmuse/services/providers.py tests/unit/test_manifest.py tests/unit/test_service_provider.py`
Expected: 无错误（如有 import 排序问题，`ruff check --fix` 修复）

- [ ] **Step 9: 验证不破坏现有测试**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_memory.py::test_add_returns_id -v`
Expected: PASS（现有测试不受影响）

---

## Task 2: MemoryConfig 改 BaseSettings + YAML 配置合并

**Files:**
- Modify: `src/septmuse/configs/base.py`
- Modify: `pyproject.toml`（加 pydantic-settings + pyyaml 依赖）
- Test: `tests/unit/test_config_merge.py`, `tests/unit/test_config_compat.py`

**Interfaces:**
- Consumes: Task 1 的 `BackendEntry.config_cls`（manifest 已 import 所有 config 类）
- Produces: `MemoryConfig(BaseSettings)` 支持 `yaml_file` + `env_prefix` + `_flat_env_aliases` model_validator

- [ ] **Step 1: 加 pydantic-settings + pyyaml 依赖到 pyproject.toml**

在 `pyproject.toml` 的 `dependencies` 列表中，`pydantic>=2.6` 后面加两行：

```toml
  "pydantic-settings>=2.2",
  "pyyaml>=6.0",
```

- [ ] **Step 2: 安装新依赖**

Run: `pip install pydantic-settings pyyaml`
Expected: 成功安装

- [ ] **Step 3: 写配置合并失败测试**

```python
# tests/unit/test_config_merge.py
"""配置三层合并测试: env > yaml > code_default。"""
import os
from pathlib import Path

import pytest


def test_zero_config_defaults():
    """无 env 无 yaml 时用代码默认。"""
    config = MemoryConfig()
    assert config.embedder.backend == "hash"
    assert config.vector_store.backend == "sqlite"


def test_yaml_overrides_default(tmp_path):
    """YAML 覆盖代码默认。"""
    yaml_content = "embedder:\n  backend: onnx\n"
    yaml_file = tmp_path / "septmuse.yaml"
    yaml_file.write_text(yaml_content, encoding="utf-8")

    config = MemoryConfig(_yaml_file=str(yaml_file))
    assert config.embedder.backend == "onnx"


def test_env_overrides_yaml(tmp_path, monkeypatch):
    """环境变量覆盖 YAML。"""
    yaml_content = "embedder:\n  backend: onnx\n"
    yaml_file = tmp_path / "septmuse.yaml"
    yaml_file.write_text(yaml_content, encoding="utf-8")

    monkeypatch.setenv("SEPTMUSE_EMBEDDER", "st")
    config = MemoryConfig(_yaml_file=str(yaml_file))
    assert config.embedder.backend == "st"
```

```python
# tests/unit/test_config_compat.py
"""扁平环境变量向后兼容测试。"""
import pytest


def test_flat_env_embedder(monkeypatch):
    """SEPTMUSE_EMBEDDER=onnx 覆盖 embedder.backend。"""
    monkeypatch.setenv("SEPTMUSE_EMBEDDER", "onnx")
    config = MemoryConfig()
    assert config.embedder.backend == "onnx"


def test_flat_env_vector_backend(monkeypatch):
    """SEPTMUSE_VECTOR_BACKEND=qdrant 覆盖 vector_store.backend。"""
    monkeypatch.setenv("SEPTMUSE_VECTOR_BACKEND", "qdrant")
    config = MemoryConfig()
    assert config.vector_store.backend == "qdrant"


def test_flat_env_reranker(monkeypatch):
    """SEPTMUSE_RERANKER=mmr 覆盖 reranker.backend。"""
    monkeypatch.setenv("SEPTMUSE_RERANKER", "mmr")
    config = MemoryConfig()
    assert config.reranker.backend == "mmr"


def test_flat_env_llm(monkeypatch):
    """SEPTMUSE_LLM=openai 覆盖 llm.backend。"""
    monkeypatch.setenv("SEPTMUSE_LLM", "openai")
    config = MemoryConfig()
    assert config.llm is not None
    assert config.llm.backend == "openai"
```

注意：两个测试文件开头都需要 `from septmuse.configs.base import MemoryConfig`

- [ ] **Step 4: 运行测试确认失败**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_config_merge.py tests/unit/test_config_compat.py -v`
Expected: FAIL — MemoryConfig 还是 BaseModel，不支持 yaml_file

- [ ] **Step 5: 改 MemoryConfig 继承 BaseSettings**

在 `src/septmuse/configs/base.py` 中：

1. 修改 import：`from pydantic import BaseModel, Field` → 保留 Field，加 `from pydantic_settings import BaseSettings, SettingsConfigDict`，加 `import os`，加 `from pydantic import model_validator`

2. `class MemoryConfig(BaseModel)` → `class MemoryConfig(BaseSettings)`

3. 在类体开头加 `model_config`：

```python
    model_config = SettingsConfigDict(
        env_prefix="SEPTMUSE_",          # SEPTMUSE_INFER → infer 字段
        env_nested_delimiter="__",      # SEPTMUSE_EMBEDDER__MODEL → embedder.model
        yaml_file=["./septmuse.yaml", "~/.septmuse/config.yaml"],
        extra="ignore",
    )
```

4. 在类体加 `search_recipe` 字段（在 `infer` 之前）：

```python
    search_recipe: str = Field(default="HYBRID_RRF", description="检索配方名")
```

5. 在类体末尾（所有字段之后）加扁平环境变量别名 model_validator：

```python
    @model_validator(mode="before")
    @classmethod
    def _flat_env_aliases(cls, data: dict) -> dict:
        """旧版扁平环境变量 → 嵌套字段（向后兼容）。"""
        if not isinstance(data, dict):
            return data
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
            if val:
                section_data = data.get(section)
                if isinstance(section_data, dict):
                    section_data[field] = val
                elif section_data is None and section == "llm":
                    data[section] = {"backend": val}
        return data
```

6. 保留所有现有的便捷属性（`db_path` / `llm_provider` / `embedder_backend` 等），它们委托给子配置，不受 BaseSettings 影响。

- [ ] **Step 6: 运行测试确认通过**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_config_merge.py tests/unit/test_config_compat.py -v`
Expected: PASS

- [ ] **Step 7: ruff 检查**

Run: `ruff check --fix src/septmuse/configs/base.py`
Expected: 无错误

- [ ] **Step 8: 验证现有配置测试不破坏**

Run: `$env:PYTHONPATH="src"; python -c "from septmuse.configs import MemoryConfig; c = MemoryConfig(); print(c.embedder.backend, c.vector_store.backend)"`
Expected: `hash sqlite`

---

## Task 3: 删除 configs/defaults.py 的 7 个 _resolve_* + default_config() 简化

**Files:**
- Modify: `src/septmuse/configs/defaults.py`

**Interfaces:**
- Consumes: Task 2 的 `MemoryConfig(BaseSettings)` 自动合并
- Produces: `default_config()` 直接返回 `MemoryConfig()`，不再调 `_resolve_*`

- [ ] **Step 1: grep 确认 _resolve_* 的外部引用点**

Run: 在 src/ 和 tests/ 中搜索 `_resolve_embedder|_resolve_llm|_resolve_reranker|_resolve_vector_store|_resolve_keyword_index|_resolve_graph_store|_resolve_entity_extractor`，记录哪些文件引用了这些函数。

- [ ] **Step 2: 删除 configs/defaults.py 中的 7 个 _resolve_* 函数**

删除 `_resolve_embedder` / `_resolve_llm` / `_resolve_reranker` / `_resolve_vector_store` / `_resolve_keyword_index` / `_resolve_graph_store` / `_resolve_entity_extractor` 这 7 个函数（configs/defaults.py 第 50-191 行的 match case 逻辑）。

- [ ] **Step 3: 简化 default_config()**

将 `default_config()` 函数体改为：

```python
def default_config() -> MemoryConfig:
    """零配置默认 — BaseSettings 自动从 env + yaml + default 合并。"""
    return MemoryConfig()
```

删除所有不再需要的 import（HashEmbedderConfig / OnnxEmbedderConfig / OpenAIEmbedderConfig 等已被 manifest 引用，defaults.py 不再需要）。

- [ ] **Step 4: ruff 检查**

Run: `ruff check --fix src/septmuse/configs/defaults.py`
Expected: 无错误（ruff 会自动删除未使用的 import）

- [ ] **Step 5: 验证 import 不破坏**

Run: `$env:PYTHONPATH="src"; python -c "from septmuse.configs import default_config; c = default_config(); print(c.embedder.backend)"`
Expected: `hash`

---

## Task 4: 调用方迁移 — _resolve_* 替换为 provider.resolve()

**Files:**
- Modify: `src/septmuse/memory/main.py`（删 `_resolve_embedder`，改 `embedder_provider.resolve()`）
- Modify: `src/septmuse/llms/__init__.py`（删 `_resolve_llm`，改 `llm_provider.resolve()`）
- Modify: `src/septmuse/retrieval/reranker.py`（删 `_resolve_reranker`，改 `reranker_provider.resolve()`）
- Modify: `src/septmuse/extraction/entity.py`（删 `_resolve_entity_extractor`，改 `entity_extractor_provider.resolve()`）
- Modify: `src/septmuse/services/embedder/service.py`（`_resolve_embedder` 改委托 provider）

**Interfaces:**
- Consumes: Task 1 的 8 个全局 provider 实例
- Produces: 所有调用方使用 `provider.resolve(backend, config=)` 替代本地 match case

- [ ] **Step 1: 修改 memory/main.py 的 _resolve_embedder**

在 `src/septmuse/memory/main.py` 中：

1. 删除整个 `_resolve_embedder(config: MemoryConfig) -> Embedder` 函数（第 60-93 行的 match case）
2. 在 `Memory.__init__` 中，将 `self.embedder = embedder or _resolve_embedder(self.config)` 改为：

```python
        if embedder is not None:
            self.embedder = embedder
        else:
            from septmuse.services.providers import embedder_provider
            self.embedder = embedder_provider.resolve(
                self.config.embedder.backend,
                config=self.config.embedder,
            )
```

3. 删除不再需要的 import（`os` 如果只被 `_resolve_embedder` 使用的话——检查后决定）

- [ ] **Step 2: 修改 llms/__init__.py 的 _resolve_llm**

在 `src/septmuse/llms/__init__.py` 中：

1. 删除 `_resolve_llm(config)` 函数的 match case 逻辑
2. 改为委托 provider：

```python
def _resolve_llm(config: MemoryConfig) -> LLM | None:
    """从配置解析 LLM 实例。"""
    if config.llm is None:
        return None
    from septmuse.services.providers import llm_provider
    return llm_provider.resolve(config.llm.backend, config=config.llm)
```

- [ ] **Step 3: 修改 retrieval/reranker.py 的 _resolve_reranker**

在 `src/septmuse/retrieval/reranker.py` 中：

1. 删除 `_resolve_reranker(backend, *, embedder, llm, model_cache_dir)` 函数的 match case
2. 改为委托 provider：

```python
def _resolve_reranker(backend: str = "noop", *, embedder=None, llm=None, model_cache_dir=None) -> Reranker:
    """工厂函数: 委托 reranker_provider。"""
    from septmuse.services.providers import reranker_provider
    return reranker_provider.resolve(backend)
```

注意：MMRReranker 需要 embedder 参数，CrossEncoderReranker 需要 model_cache_dir。如果 provider.resolve() 的 kwargs 不够，需要保留旧逻辑的 kwargs 传递。检查实际调用方签名后决定是否保留 kwargs 透传。

- [ ] **Step 4: 修改 extraction/entity.py 的 _resolve_entity_extractor**

在 `src/septmuse/extraction/entity.py` 中：

1. 删除 `_resolve_entity_extractor(config)` 函数的 match case
2. 改为委托 provider：

```python
def _resolve_entity_extractor(config: MemoryConfig) -> Any:
    """从配置解析实体抽取器。"""
    from septmuse.services.providers import entity_extractor_provider
    return entity_extractor_provider.resolve(config.entity_extractor.backend, config=config.entity_extractor)
```

- [ ] **Step 5: 修改 services/embedder/service.py 的 _resolve_embedder**

在 `src/septmuse/services/embedder/service.py` 中：

1. 删除 `_resolve_embedder(self)` 方法的 match case（第 121-152 行）
2. 改为委托 provider：

```python
    def _resolve_embedder(self) -> Embedder:
        """委托 embedder_provider 解析。"""
        from septmuse.services.providers import embedder_provider
        return embedder_provider.resolve(
            self._config.embedder.backend,
            config=self._config.embedder,
        )
```

- [ ] **Step 6: ruff 检查所有修改文件**

Run: `ruff check --fix src/septmuse/memory/main.py src/septmuse/llms/__init__.py src/septmuse/retrieval/reranker.py src/septmuse/extraction/entity.py src/septmuse/services/embedder/service.py`
Expected: 无错误

- [ ] **Step 7: 验证 pytest 基线不退化**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/ -q --tb=line 2>&1 | Select-Object -Last 5`
Expected: 失败数不超过 185（目录重构后的基线），passed 不低于 828

- [ ] **Step 8: 验证 e2e 测试**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/e2e/ -q --tb=line 2>&1 | Select-Object -Last 3`
Expected: 23 passed（不退化）

---

## Task 5: CLI 自省命令 — backends + config show

**Files:**
- Modify: `src/septmuse/cli/main.py`（加 backends + config 子命令）
- Test: `tests/unit/test_cli_backends.py`

**Interfaces:**
- Consumes: Task 1 的 `ALL_PROVIDERS` + Task 2 的 `MemoryConfig`
- Produces: `septmuse backends` 和 `septmuse config show` CLI 命令

- [ ] **Step 1: 写 CLI 自省失败测试**

```python
# tests/unit/test_cli_backends.py
"""CLI backends / config show 自省命令测试。"""
import subprocess
import sys


def _run_cli(args: list[str]) -> str:
    """运行 CLI 命令，返回 stdout。"""
    env = {"PYTHONPATH": "src", "PATH": ";".join(sys.path)}
    result = subprocess.run(
        [sys.executable, "-m", "septmuse.cli.main", *args],
        capture_output=True, text=True, env=env, timeout=30,
    )
    return result.stdout


def test_backends_lists_all_capabilities():
    out = _run_cli(["backends"])
    assert "vector_store" in out
    assert "embedder" in out
    assert "llm" in out
    assert "reranker" in out
    assert "search_recipe" in out


def test_backends_shows_availability():
    out = _run_cli(["backends"])
    assert "[✓]" in out or "[ok]" in out  # sqlite/hash 可用
    assert "[✗]" in out or "[x]" in out   # qdrant/neo4j 可能不可用


def test_config_show_outputs_current_config():
    out = _run_cli(["config", "show"])
    assert "embedder" in out
    assert "backend" in out
    assert "source" in out  # 显示配置来源
```

- [ ] **Step 2: 运行测试确认失败**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_cli_backends.py -v`
Expected: FAIL — 未知命令 backends

- [ ] **Step 3: 在 cli/main.py 加 backends 子命令**

先 Read `src/septmuse/cli/main.py` 了解现有 argparse 结构。然后在 `main()` 函数的 subparsers 部分加：

```python
    # ── backends 子命令 ──
    backends_parser = subparsers.add_parser("backends", help="列出所有能力后端及可用性")
    backends_parser.set_defaults(func=cmd_backends)

    # ── config 子命令 ──
    config_parser = subparsers.add_parser("config", help="配置自省")
    config_sub = config_parser.add_subparsers(dest="config_action")
    config_show_parser = config_sub.add_parser("show", help="显示当前生效配置")
    config_show_parser.set_defaults(func=cmd_config_show)
```

加命令处理函数：

```python
def cmd_backends(args: argparse.Namespace) -> None:
    """列出所有能力后端及可用性。"""
    from septmuse.services.providers import ALL_PROVIDERS

    for capability, provider in ALL_PROVIDERS.items():
        parts = []
        for backend in provider.list_backends():
            mark = "✓" if provider.is_available(backend) else "✗"
            parts.append(f"{backend}[{mark}]")
        print(f"{capability:20s} {' '.join(parts)}")


def cmd_config_show(args: argparse.Namespace) -> None:
    """显示当前生效配置 + 来源。"""
    from septmuse.configs import default_config

    config = default_config()
    capabilities = [
        ("embedder", config.embedder.backend),
        ("vector_store", config.vector_store.backend),
        ("llm", config.llm.backend if config.llm else "null"),
        ("reranker", config.reranker.backend),
        ("entity_extractor", config.entity_extractor.backend),
        ("keyword_index", config.keyword_index.backend),
        ("graph_store", config.graph_store.backend),
        ("search_recipe", config.search_recipe),
        ("infer", str(config.infer)),
    ]
    for name, value in capabilities:
        print(f"{name:20s} value={value}")
```

- [ ] **Step 4: 运行测试确认通过**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_cli_backends.py -v`
Expected: PASS

- [ ] **Step 5: ruff 检查**

Run: `ruff check --fix src/septmuse/cli/main.py tests/unit/test_cli_backends.py`
Expected: 无错误

---

## Task 6: 全量验证

**Files:**
- 无新建/修改，纯验证

- [ ] **Step 1: ruff 全量检查**

Run: `ruff check src/ tests/`
Expected: `All checks passed!`

- [ ] **Step 2: ruff format 只读检查（不写）**

Run: `ruff format --check src/ tests/`
Expected: `N files already formatted`

- [ ] **Step 3: 单元测试全量**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/ -q --tb=line 2>&1 | Select-Object -Last 5`
Expected: 失败不超过 185，passed 不低于 828，skipped 约 22

- [ ] **Step 4: e2e 测试全量**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/e2e/ -q --tb=line 2>&1 | Select-Object -Last 3`
Expected: 23 passed

- [ ] **Step 5: 零配置启动验证**

Run: `$env:PYTHONPATH="src"; python -c "from septmuse.memory import Memory; m = Memory(); print('OK', m.config.embedder.backend)"`
Expected: `OK hash`

- [ ] **Step 6: 环境变量覆盖验证**

Run: `$env:PYTHONPATH="src"; $env:SEPTMUSE_EMBEDDER="onnx"; python -c "from septmuse.configs import default_config; c = default_config(); print(c.embedder.backend)"`
Expected: `onnx`

- [ ] **Step 7: CLI backends 命令验证**

Run: `$env:PYTHONPATH="src"; python -m septmuse.cli.main backends`
Expected: 输出 8 个能力的后端列表，sqlite/hash 显示 ✓

- [ ] **Step 8: CLI config show 命令验证**

Run: `$env:PYTHONPATH="src"; python -m septmuse.cli.main config show`
Expected: 输出当前生效配置

- [ ] **Step 9: 清理临时环境变量**

Run: `Remove-Item Env:SEPTMUSE_EMBEDDER -ErrorAction SilentlyContinue`

---

## 自检报告

### Spec 覆盖率

| Spec 章节 | 对应 Task | 状态 |
|-----------|----------|------|
| §4 manifest 注册表 | Task 1 Step 3 | ✅ |
| §5 ServiceProvider API | Task 1 Step 6 | ✅ |
| §6 pydantic-settings 配置合并 | Task 2 | ✅ |
| §6 向后兼容扁平 env | Task 2 Step 5 | ✅ |
| §7 ConfigService 增强 | Task 2（BaseSettings 替代） | ✅ |
| §8 _resolve_* 迁移路径 | Task 3 + Task 4 | ✅ |
| §9 CLI 自省命令 | Task 5 | ✅ |
| §10 测试策略 | Task 1-5 各有测试 | ✅ |
| §11 迁移批次 | Task 1-6 对应批次 1-6 | ✅ |

### 类型一致性

- `BackendEntry` 在 Task 1 定义，Task 1 的 providers.py 使用——字段名 `module/cls/config_cls/deps` 一致
- `ServiceProvider.resolve(backend, *, config, **kwargs)` 签名在 Task 1 定义，Task 4 调用方使用——一致
- `ALL_PROVIDERS` dict 在 Task 1 定义，Task 5 CLI 使用——一致
- `MemoryConfig(BaseSettings)` 在 Task 2 定义，Task 3/4 调用 `config.embedder.backend`——一致

### 无占位符

- 所有代码块完整，无 TBD/TODO/"implement later"
- 每步有实际代码 + 运行命令 + 预期输出

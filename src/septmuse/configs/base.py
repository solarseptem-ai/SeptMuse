#  Copyright 2026 The sonhhxg0529 Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""SeptMuse 配置 — 子配置基类 + 顶层 MemoryConfig 组合。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict, YamlConfigSettingsSource


class DatabaseConfig(BaseModel):
    """数据库连接配置。"""

    db_url: str | None = Field(
        default=None,
        description="数据库连接 URL; None=用 db_path 回退 SQLite; mysql:// / postgresql:// 切换后端",
    )
    db_path: str | Path | None = Field(
        default=None,
        description="SQLite 路径; None → 默认 ~/.septmuse/septmuse.db; ':memory:' → 内存库",
    )
    connection_pool_size: int = Field(default=5, description="连接池大小 (PG 模式)")
    connection_max_overflow: int = Field(default=10, description="连接池溢出上限 (PG 模式)")
    connect_timeout: float = Field(default=30.0, description="连接超时秒数")
    sqlite_pragmas: dict[str, Any] = Field(
        default_factory=lambda: {"journal_mode": "WAL", "synchronous": "NORMAL", "busy_timeout": 5000},
        description="SQLite PRAGMA 设置 (WAL 模式提升并发读写, busy_timeout 毫秒级等待)",
    )


class BaseEmbedderConfig(BaseModel):
    """嵌入通用配置。"""

    backend: str = Field(default="bge-zh")
    model: str | None = Field(default=None, description="模型名")
    base_url: str | None = Field(default=None, description="OpenAI 兼容端点")
    embedding_dims: int | None = Field(default=None, description="嵌入维度, None=自动检测")
    api_key: str | None = Field(default=None)


class BaseLLMConfig(BaseModel):
    """LLM 通用配置。"""

    backend: str = Field(default="", description="provider 后端标识, 如 openai/ollama/anthropic/dashscope")
    model: str | None = Field(default=None, description="模型名")
    base_url: str | None = Field(default=None, description="OpenAI 兼容端点, 如 http://localhost:7521/v1")
    temperature: float = Field(default=0.1, description="采样温度")
    api_key: str | None = Field(default=None, description="API key")
    max_tokens: int = Field(default=2000, description="最大 token 数")
    top_p: float = Field(default=0.1, description="核采样参数")


class BaseRerankerConfig(BaseModel):
    """重排器通用配置。"""

    backend: str = Field(default="noop")
    top_k: int | None = Field(default=None, description="重排后返回数量")


class BaseVectorStoreConfig(BaseModel):
    """向量存储通用配置。"""

    backend: str = Field(default="qdrant", description="向量后端")
    embedding_model_dims: int = Field(default=512, description="嵌入维度")
    # Qdrant 专用配置 (统一配置真源, 消除工厂方法中散落的 os.getenv)
    qdrant_path: str | None = Field(default=None, description="Qdrant 本地嵌入路径")
    qdrant_host: str | None = Field(default=None, description="Qdrant 远程 host")
    qdrant_port: int = Field(default=6333, description="Qdrant 远程端口")
    qdrant_url: str | None = Field(default=None, description="Qdrant 完整 URL")
    qdrant_api_key: str | None = Field(default=None, description="Qdrant Cloud API key")
    chroma_persist_path: str | None = Field(default=None, description="Chroma 持久化路径")


class BaseGraphStoreConfig(BaseModel):
    """图存储通用配置。"""

    backend: str = Field(default="sqlite")


class BaseKeywordIndexConfig(BaseModel):
    """关键词索引通用配置。"""

    backend: str = Field(default="sqlite_bm25")


class BaseEntityExtractorConfig(BaseModel):
    """实体抽取通用配置。"""

    backend: str = Field(default="regex")


class MemoryConfig(BaseSettings):
    """SeptMuse 记忆系统配置 (顶层组合)。"""

    model_config = SettingsConfigDict(
        env_prefix="SEPTMUSE_",
        env_nested_delimiter="__",
        yaml_file=["./septmuse.yaml", "~/.septmuse/config.yaml"],
        extra="ignore",
    )

    database: DatabaseConfig = Field(default_factory=DatabaseConfig, description="数据库配置")
    vector_store: BaseVectorStoreConfig = Field(default_factory=BaseVectorStoreConfig, description="向量存储配置")
    keyword_index: BaseKeywordIndexConfig = Field(default_factory=BaseKeywordIndexConfig, description="关键词索引配置")
    graph_store: BaseGraphStoreConfig = Field(default_factory=BaseGraphStoreConfig, description="图存储配置")
    llm: BaseLLMConfig | None = Field(default=None, description="LLM 配置; None=verbatim 模式")
    embedder: BaseEmbedderConfig = Field(default_factory=BaseEmbedderConfig, description="嵌入配置")
    reranker: BaseRerankerConfig = Field(default_factory=BaseRerankerConfig, description="重排器配置")
    entity_extractor: BaseEntityExtractorConfig = Field(
        default_factory=BaseEntityExtractorConfig, description="实体抽取配置"
    )

    top_k: int = Field(default=7, description="默认检索 top_k")
    threshold: float = Field(default=0.1, description="默认相似阈值")
    search_recipe: str = Field(default="HYBRID_RRF", description="检索配方名")
    infer: bool = Field(default=False, description="是否 LLM 抽取事实; False=原文存")
    forgetting_half_life_days: float = Field(default=7.0, description="遗忘曲线半衰期 (天); None=原公式, inf=永不衰减")

    # ── 便捷属性: 委托给子配置, 供 Memory facade 和 services 层使用 ──

    @property
    def db_path(self) -> str | Path | None:
        return self.database.db_path

    @property
    def db_url(self) -> str | None:
        return self.database.db_url

    @property
    def llm_provider(self) -> str | None:
        if self.llm is None:
            return None
        return self.llm.backend or None

    @property
    def llm_model(self) -> str | None:
        if self.llm is None:
            return None
        return self.llm.model

    @property
    def llm_base_url(self) -> str | None:
        if self.llm is None:
            return None
        return getattr(self.llm, "base_url", None) or getattr(self.llm, "host", None)

    @property
    def embedder_backend(self) -> str:
        return self.embedder.backend

    @property
    def embedder_model(self) -> str | None:
        return self.embedder.model

    @property
    def embedder_dims(self) -> int | None:
        return self.embedder.embedding_dims

    @property
    def embedder_base_url(self) -> str | None:
        return getattr(self.embedder, "base_url", None)

    @property
    def embedder_model_cache_dir(self) -> str | None:
        return getattr(self.embedder, "model_cache_dir", None)

    @property
    def reranker_backend(self) -> str:
        return self.reranker.backend

    @classmethod
    def settings_customise_sources(
        cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings
    ):
        """配置源优先级: init kwargs > YAML > 代码默认。"""
        # 从 init kwargs 提取 _yaml_file 运行时覆盖 (pydantic-settings 不自动处理)
        init_kwargs = getattr(init_settings, "init_kwargs", None) or {}
        yaml_file_override = init_kwargs.pop("_yaml_file", None)

        # 展开 ~ 为真实 home 目录 (跨平台: Linux/macOS ~/.septmuse → Windows %USERPROFILE%\.septmuse)
        def _expand_yaml_files(files):
            if files is None:
                return None
            if isinstance(files, str):
                return str(Path(files).expanduser())
            return [str(Path(f).expanduser()) for f in files]

        if yaml_file_override is not None:
            yaml_file = _expand_yaml_files(yaml_file_override)
        else:
            yaml_file = _expand_yaml_files(settings_cls.model_config.get("yaml_file"))
        yaml_source = YamlConfigSettingsSource(settings_cls, yaml_file=yaml_file)
        return (init_settings, yaml_source)

    @model_validator(mode="before")
    @classmethod
    def _flat_env_aliases(cls, data):
        """旧版扁平环境变量 → 嵌套字段（向后兼容）。"""
        if not isinstance(data, dict):
            return data

        # 嵌套环境变量: SEPTMUSE_DATABASE__DB_PATH → database.db_path
        prefix = "SEPTMUSE_"
        delimiter = "__"
        for key, val in os.environ.items():
            if not key.startswith(prefix) or delimiter not in key:
                continue
            # 去掉前缀，按 __ 分割
            parts = key[len(prefix) :].split(delimiter)
            if len(parts) < 2:
                continue
            section = parts[0].lower()
            field = delimiter.join(parts[1:]).lower()
            # 映射到已知子配置 (保持与模型字段名一致)
            section_data = data.get(section)
            if isinstance(section_data, dict):
                section_data[field] = val
            elif section_data is None or isinstance(section_data, str):
                data[section] = {field: val}

        aliases = {
            "SEPTMUSE_DB_URL": ("database", "db_url"),
            "SEPTMUSE_DB_PATH": ("database", "db_path"),
            "SEPTMUSE_EMBEDDER": ("embedder", "backend"),
            "SEPTMUSE_EMBEDDER_MODEL": ("embedder", "model"),
            "SEPTMUSE_EMBEDDER_BASE_URL": ("embedder", "base_url"),
            "SEPTMUSE_EMBEDDER_DIMS": ("embedder", "embedding_dims"),
            "SEPTMUSE_EMBEDDING_DIMS": ("vector_store", "embedding_model_dims"),
            "SEPTMUSE_VECTOR_BACKEND": ("vector_store", "backend"),
            "SEPTMUSE_KEYWORD_BACKEND": ("keyword_index", "backend"),
            "SEPTMUSE_GRAPH_BACKEND": ("graph_store", "backend"),
            "SEPTMUSE_RERANKER": ("reranker", "backend"),
            "SEPTMUSE_ENTITY_EXTRACTOR": ("entity_extractor", "backend"),
            "SEPTMUSE_LLM": ("llm", "backend"),
            "SEPTMUSE_LLM_MODEL": ("llm", "model"),
            "SEPTMUSE_LLM_BASE_URL": ("llm", "base_url"),
            "SEPTMUSE_QDRANT_PATH": ("vector_store", "qdrant_path"),
            "SEPTMUSE_QDRANT_HOST": ("vector_store", "qdrant_host"),
            "SEPTMUSE_QDRANT_PORT": ("vector_store", "qdrant_port"),
            "SEPTMUSE_QDRANT_URL": ("vector_store", "qdrant_url"),
            "SEPTMUSE_QDRANT_API_KEY": ("vector_store", "qdrant_api_key"),
            "SEPTMUSE_CHROMA_PERSIST_PATH": ("vector_store", "chroma_persist_path"),
        }
        for env_name, (section, field) in aliases.items():
            val = os.getenv(env_name)
            if not val:
                continue
            section_data = data.get(section)
            if isinstance(section_data, dict):
                section_data[field] = val
            elif section_data is None or isinstance(section_data, str):
                data[section] = {field: val}
            # else: 已构造的子配置实例 (来自 init kwargs 如 default_config) → 不动
        # 顶层标量 env 变量 (非嵌套)
        top_level_env = {"SEPTMUSE_INFER": "infer"}
        for env_name, field_name in top_level_env.items():
            val = os.getenv(env_name)
            if not val:
                continue
            data[field_name] = val.lower() in ("true", "1", "yes")
        # 旧版扁平 init kwargs → 嵌套字段 (向后兼容)
        flat_kwargs = {
            "db_path": ("database", "db_path"),
            "llm_provider": ("llm", "backend"),
            "llm_model": ("llm", "model"),
            "llm_base_url": ("llm", "base_url"),
            "embedder_backend": ("embedder", "backend"),
            "embedder_model": ("embedder", "model"),
            "embedder_base_url": ("embedder", "base_url"),
            "embedder_dims": ("embedder", "embedding_dims"),
        }
        for kwarg_name, (section, field) in flat_kwargs.items():
            if kwarg_name not in data:
                continue
            val = data.pop(kwarg_name)
            section_data = data.get(section)
            if isinstance(section_data, dict):
                section_data[field] = val
            elif section_data is None or isinstance(section_data, str):
                data[section] = {field: val}
            # else: 已构造的子配置实例 → 不动
        return data

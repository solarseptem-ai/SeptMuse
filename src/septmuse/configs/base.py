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
"""SeptMuse 记忆配置 — 顶层组合 (借鉴 mem0 configs/base.py MemoryConfig)。

顶层 MemoryConfig 只组合子配置, 不含具体后端参数。
每个子系统一个子目录, 每个后端一个 Config 类 (configs/<subsystem>/<backend>.py)。

用法:
    from septmuse.configs import MemoryConfig, default_config

    config = default_config()
    config.database.db_path
    config.embedder.backend
    config.llm
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict, YamlConfigSettingsSource

from septmuse.configs.database import DatabaseConfig
from septmuse.configs.embeddings.base import BaseEmbedderConfig
from septmuse.configs.extraction.base import BaseEntityExtractorConfig
from septmuse.configs.graph_stores.base import BaseGraphStoreConfig
from septmuse.configs.keyword_index.base import BaseKeywordIndexConfig
from septmuse.configs.llms.base import BaseLLMConfig
from septmuse.configs.rerankers.base import BaseRerankerConfig
from septmuse.configs.vector_stores.base import BaseVectorStoreConfig


class MemoryConfig(BaseSettings):
    """SeptMuse 记忆系统配置 (顶层组合, 借鉴 mem0 MemoryConfig)。

    每个子系统一个子配置, 通过 default_config() 从环境变量组装。
    零配置默认: SQLite + HashEmbedder + RegexExtractor + NoopReranker + verbatim。

    配置优先级: init kwargs > 环境变量 > YAML > 代码默认。
    旧版扁平环境变量 (如 SEPTMUSE_EMBEDDER) 通过 _flat_env_aliases 兼容。
    """

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

    top_k: int = Field(default=5, description="默认检索 top_k")
    threshold: float = Field(default=0.1, description="默认相似阈值")
    search_recipe: str = Field(default="HYBRID_RRF", description="检索配方名")
    infer: bool = Field(default=False, description="是否 LLM 抽取事实; False=原文存")

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
        """配置源优先级: init kwargs > YAML > 代码默认。

        不使用默认 EnvSettingsSource: 它会把 SEPTMUSE_EMBEDDER=onnx 这样的扁平
        环境变量直接映射到 embedder 字段并尝试 JSON 解析, 导致 SettingsError。
        环境变量解析交给 _flat_env_aliases before-validator 手动处理, 保持旧版
        扁平 env 向后兼容。
        """
        # 从 init kwargs 提取 _yaml_file 运行时覆盖 (pydantic-settings 不自动处理)
        init_kwargs = getattr(init_settings, "init_kwargs", None) or {}
        yaml_file_override = init_kwargs.pop("_yaml_file", None)
        yaml_source = YamlConfigSettingsSource(settings_cls, yaml_file=yaml_file_override)
        return (init_settings, yaml_source)

    @model_validator(mode="before")
    @classmethod
    def _flat_env_aliases(cls, data):
        """旧版扁平环境变量 → 嵌套字段（向后兼容）。

        pydantic-settings 用 env_nested_delimiter='__' 解析嵌套环境变量,
        但旧版用户习惯用 SEPTMUSE_EMBEDDER=onnx 这样的扁平形式。
        本 validator 在校验前把扁平 env 映射到嵌套子配置的 backend 字段。
        """
        if not isinstance(data, dict):
            return data
        aliases = {
            "SEPTMUSE_DB_URL": ("database", "db_url"),
            "SEPTMUSE_DB_PATH": ("database", "db_path"),
            "SEPTMUSE_EMBEDDER": ("embedder", "backend"),
            "SEPTMUSE_VECTOR_BACKEND": ("vector_store", "backend"),
            "SEPTMUSE_KEYWORD_BACKEND": ("keyword_index", "backend"),
            "SEPTMUSE_GRAPH_BACKEND": ("graph_store", "backend"),
            "SEPTMUSE_RERANKER": ("reranker", "backend"),
            "SEPTMUSE_ENTITY_EXTRACTOR": ("entity_extractor", "backend"),
            "SEPTMUSE_LLM": ("llm", "backend"),
        }
        for env_name, (section, field) in aliases.items():
            val = os.getenv(env_name)
            if not val:
                continue
            section_data = data.get(section)
            if isinstance(section_data, dict):
                # 来自 YAML 或 nested env 的 dict → 更新 backend
                section_data[field] = val
            elif section_data is None or isinstance(section_data, str):
                # None=未设置 / str=flat env 直接映射的原始值 → 用 dict 替换
                data[section] = {field: val}
            # else: 已构造的子配置实例 (来自 init kwargs 如 default_config) → 不动
        # 旧版扁平 init kwargs → 嵌套字段 (向后兼容, db_path=... → database.db_path)
        flat_kwargs = {"db_path": ("database", "db_path")}
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

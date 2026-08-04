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
"""SeptMuse 配置。

顶层组合:
    base.py         MemoryConfig (组合所有子配置)
    defaults.py     default_config() 从环境变量组装
    enums.py        后端枚举 (VectorBackend / EmbedderBackend / ...)
    database.py     DatabaseConfig

子目录 (每类一个, 通用基类 + per-backend 子类):
    vector_stores/  向量存储 (sqlite/qdrant/chroma/pgvector)
    keyword_index/  关键词索引 (sqlite_bm25/rank_bm25)
    graph_stores/   图存储 (sqlite/age/neo4j)
    llms/           LLM (openai/ollama/anthropic/dashscope)
    embeddings/     嵌入 (hash/onnx/openai)
    rerankers/      重排器 (noop/mmr/cross_encoder/llm)
    extraction/     实体抽取 (regex/spacy)

用法:
    from septmuse.configs import MemoryConfig, default_config

    config = default_config()
    config.database.db_path
    config.embedder.backend
    config.llm
"""

from __future__ import annotations

from septmuse.configs.base import MemoryConfig
from septmuse.configs.database import DatabaseConfig
from septmuse.configs.defaults import default_config
from septmuse.configs.enums import (
    EmbedderBackend,
    EntityExtractorBackend,
    GraphBackend,
    KeywordBackend,
    LLMBackend,
    RerankerBackend,
    VectorBackend,
)

__all__ = [
    "DatabaseConfig",
    "EmbedderBackend",
    "EntityExtractorBackend",
    "GraphBackend",
    "KeywordBackend",
    "LLMBackend",
    "MemoryConfig",
    "RerankerBackend",
    "VectorBackend",
    "default_config",
]

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
"""后端枚举 (借鉴 mem0 configs/enums.py)。

每个子系统的后端选择枚举, 用于配置中指明使用哪个实现。
"""

from __future__ import annotations

from enum import Enum


class VectorBackend(str, Enum):
    """向量存储后端。"""

    SQLITE = "sqlite"
    QDRANT = "qdrant"
    CHROMA = "chroma"
    PGVECTOR = "pgvector"


class KeywordBackend(str, Enum):
    """关键词索引后端。"""

    SQLITE_BM25 = "sqlite_bm25"
    RANK_BM25 = "rank_bm25"
    NONE = "none"


class GraphBackend(str, Enum):
    """图存储后端。"""

    SQLITE = "sqlite"
    AGE = "age"
    NEO4J = "neo4j"


class EmbedderBackend(str, Enum):
    """嵌入模型后端。"""

    HASH = "hash"
    ONNX = "onnx"
    ONNX_ZH = "onnx-zh"
    AUTO = "auto"
    ST = "st"
    OPENAI = "openai"
    MOCK = "mock"
    OLLAMA = "ollama"
    TOGETHER = "together"
    LMSTUDIO = "lmstudio"
    AZURE_OPENAI = "azure_openai"
    GEMINI = "gemini"
    VERTEXAI = "vertexai"
    HUGGINGFACE = "huggingface"
    AWS_BEDROCK = "aws_bedrock"
    FASTEMBED = "fastembed"
    LANGCHAIN = "langchain"


class LLMBackend(str, Enum):
    """LLM provider 后端。"""

    OPENAI = "openai"
    OLLAMA = "ollama"
    ANTHROPIC = "anthropic"
    DASHSCOPE = "dashscope"


class RerankerBackend(str, Enum):
    """重排器后端。"""

    NOOP = "noop"
    MMR = "mmr"
    CROSS_ENCODER = "cross_encoder"
    LLM = "llm"


class EntityExtractorBackend(str, Enum):
    """实体抽取后端。"""

    REGEX = "regex"
    SPACY = "spacy"
    NONE = "none"

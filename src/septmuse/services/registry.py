"""声明式后端注册表 — 8 能力 × 多后端的 manifest。

每条 BackendEntry 声明: 模块路径 / 类名 / config 类路径 / 外部依赖。
零依赖后端 (deps=()) 保证零配置可用; config_cls 用字符串路径延迟 import。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BackendEntry:
    """单个后端的声明式描述。

    Attributes:
        module: 完整 Python 模块路径 (如 "septmuse.storage.vector_stores.sqlite_vec")。
        cls: 模块内的类名或函数名; 空串表示 "none" 后端 (resolve 返回 None)。
        config_cls: 对应 config 类的完整路径字符串; None 表示无 config (如 search_recipe)。
        deps: 外部依赖库名元组; 空元组 () 表示零依赖。
    """

    module: str
    cls: str
    config_cls: str | None
    deps: tuple[str, ...]


# 8 能力的后端注册表。每个能力映射 后端名 -> BackendEntry。
BACKEND_MANIFEST: dict[str, dict[str, BackendEntry]] = {
    "vector_store": {
        "sqlite": BackendEntry(
            module="septmuse.storage.vector_stores.sqlite_vec",
            cls="SQLiteVectorStore",
            config_cls="septmuse.configs.vector_stores.sqlite.SQLiteVectorConfig",
            deps=(),
        ),
        "qdrant": BackendEntry(
            module="septmuse.storage.vector_stores.qdrant",
            cls="QdrantVectorStore",
            config_cls="septmuse.configs.vector_stores.qdrant.QdrantVectorConfig",
            deps=("qdrant_client",),
        ),
        "chroma": BackendEntry(
            module="septmuse.storage.vector_stores.chroma",
            cls="ChromaVectorStore",
            config_cls="septmuse.configs.vector_stores.chroma.ChromaVectorConfig",
            deps=("chromadb",),
        ),
        "pgvector": BackendEntry(
            module="septmuse.storage.vector_stores.pgvector",
            cls="PGVectorStore",
            config_cls="septmuse.configs.vector_stores.pgvector.PgVectorConfig",
            deps=("psycopg",),
        ),
    },
    "embedder": {
        "hash": BackendEntry(
            module="septmuse.embedders.hash",
            cls="HashEmbedder",
            config_cls="septmuse.configs.embeddings.hash.HashEmbedderConfig",
            deps=(),
        ),
        "onnx": BackendEntry(
            module="septmuse.embedders.onnx",
            cls="OnnxEmbedder",
            config_cls="septmuse.configs.embeddings.onnx.OnnxEmbedderConfig",
            deps=("onnxruntime",),
        ),
        "onnx-zh": BackendEntry(
            module="septmuse.embedders.onnx",
            cls="OnnxEmbedder",
            config_cls="septmuse.configs.embeddings.onnx.OnnxEmbedderConfig",
            deps=("onnxruntime",),
        ),
        "auto": BackendEntry(
            module="septmuse.embedders.auto",
            cls="AutoOnnxEmbedder",
            config_cls="septmuse.configs.embeddings.onnx.OnnxEmbedderConfig",
            deps=("onnxruntime",),
        ),
        "openai": BackendEntry(
            module="septmuse.embedders.openai",
            cls="OpenAIEmbedder",
            config_cls="septmuse.configs.embeddings.openai.OpenAIEmbedderConfig",
            deps=("openai",),
        ),
        "st": BackendEntry(
            module="septmuse.embedders.sentence_transformers",
            cls="SentenceTransformerEmbedder",
            config_cls="septmuse.configs.embeddings.base.BaseEmbedderConfig",
            deps=("sentence_transformers",),
        ),
    },
    "llm": {
        "openai": BackendEntry(
            module="septmuse.llms.openai",
            cls="OpenAILLM",
            config_cls="septmuse.configs.llms.openai.OpenAILLMConfig",
            deps=("openai",),
        ),
        "ollama": BackendEntry(
            module="septmuse.llms.ollama",
            cls="OllamaLLM",
            config_cls="septmuse.configs.llms.ollama.OllamaLLMConfig",
            deps=("ollama",),
        ),
        "anthropic": BackendEntry(
            module="septmuse.llms.anthropic",
            cls="AnthropicLLM",
            config_cls="septmuse.configs.llms.anthropic.AnthropicLLMConfig",
            deps=("anthropic",),
        ),
        "dashscope": BackendEntry(
            module="septmuse.llms.dashscope",
            cls="DashScopeLLM",
            config_cls="septmuse.configs.llms.dashscope.DashScopeLLMConfig",
            deps=("dashscope",),
        ),
        "litellm": BackendEntry(
            module="septmuse.llms.litellm",
            cls="LitellmLLM",
            config_cls="septmuse.configs.llms.litellm.LitellmLLMConfig",
            deps=("litellm",),
        ),
        "groq": BackendEntry(
            module="septmuse.llms.groq",
            cls="GroqLLM",
            config_cls="septmuse.configs.llms.groq.GroqLLMConfig",
            deps=("groq",),
        ),
        "gemini": BackendEntry(
            module="septmuse.llms.gemini",
            cls="GeminiLLM",
            config_cls="septmuse.configs.llms.gemini.GeminiLLMConfig",
            deps=("google-generativeai",),
        ),
        "deepseek": BackendEntry(
            module="septmuse.llms.deepseek",
            cls="DeepSeekLLM",
            config_cls="septmuse.configs.llms.deepseek.DeepSeekLLMConfig",
            deps=("openai",),
        ),
    },
    "reranker": {
        "noop": BackendEntry(
            module="septmuse.rerankers.noop",
            cls="NoopReranker",
            config_cls="septmuse.configs.rerankers.noop.NoopRerankerConfig",
            deps=(),
        ),
        "mmr": BackendEntry(
            module="septmuse.rerankers.mmr",
            cls="MMRReranker",
            config_cls="septmuse.configs.rerankers.mmr.MMRRerankerConfig",
            deps=(),
        ),
        "cross_encoder": BackendEntry(
            module="septmuse.rerankers.cross_encoder",
            cls="CrossEncoderReranker",
            config_cls="septmuse.configs.rerankers.cross_encoder.CrossEncoderRerankerConfig",
            deps=("onnxruntime",),
        ),
        "llm": BackendEntry(
            module="septmuse.rerankers.llm",
            cls="LLMReranker",
            config_cls="septmuse.configs.rerankers.llm.LLMRerankerConfig",
            deps=(),
        ),
        "batch_llm": BackendEntry(
            module="septmuse.rerankers.batch_llm",
            cls="BatchLLMReranker",
            config_cls="septmuse.configs.rerankers.batch_llm.BatchLLMRerankerConfig",
            deps=(),
        ),
        "cohere": BackendEntry(
            module="septmuse.rerankers.cohere",
            cls="CohereReranker",
            config_cls="septmuse.configs.rerankers.cohere.CohereRerankerConfig",
            deps=("cohere",),
        ),
    },
    "entity_extractor": {
        "regex": BackendEntry(
            module="septmuse.extraction.entity",
            cls="RegexEntityExtractor",
            config_cls="septmuse.configs.extraction.regex.RegexExtractorConfig",
            deps=(),
        ),
        "spacy": BackendEntry(
            module="septmuse.extraction.entity",
            cls="SpacyEntityExtractor",
            config_cls="septmuse.configs.extraction.spacy.SpacyExtractorConfig",
            deps=("spacy",),
        ),
        "none": BackendEntry(
            module="",
            cls="",
            config_cls=None,
            deps=(),
        ),
    },
    "keyword_index": {
        "sqlite_bm25": BackendEntry(
            module="septmuse.storage.keyword_stores.sqlite_bm25",
            cls="SQLiteBM25Index",
            config_cls="septmuse.configs.keyword_index.sqlite_bm25.SQLiteBM25Config",
            deps=(),
        ),
        "rank_bm25": BackendEntry(
            module="septmuse.storage.keyword_stores.rank_bm25",
            cls="RankBM25Index",
            config_cls="septmuse.configs.keyword_index.rank_bm25.RankBM25Config",
            deps=("rank_bm25",),
        ),
        "none": BackendEntry(
            module="",
            cls="",
            config_cls=None,
            deps=(),
        ),
    },
    "graph_store": {
        "sqlite": BackendEntry(
            module="septmuse.storage.graph_stores.sqlite",
            cls="SQLiteGraphStore",
            config_cls="septmuse.configs.graph_stores.sqlite.SQLiteGraphConfig",
            deps=(),
        ),
        "age": BackendEntry(
            module="septmuse.storage.graph_stores.age",
            cls="AGEGraphStore",
            config_cls="septmuse.configs.graph_stores.age.AgeGraphConfig",
            deps=("psycopg",),
        ),
        "neo4j": BackendEntry(
            module="septmuse.storage.graph_stores.neo4j",
            cls="Neo4jGraphStore",
            config_cls="septmuse.configs.graph_stores.neo4j.Neo4jGraphConfig",
            deps=("neo4j",),
        ),
    },
    "search_recipe": {
        "HYBRID_RRF": BackendEntry(
            module="septmuse.retrieval.recipes",
            cls="get_recipe",
            config_cls=None,
            deps=(),
        ),
        "HYBRID_RRF_ENTITY": BackendEntry(
            module="septmuse.retrieval.recipes",
            cls="get_recipe",
            config_cls=None,
            deps=(),
        ),
        "HYBRID_RRF_CROSS_ENCODER": BackendEntry(
            module="septmuse.retrieval.recipes",
            cls="get_recipe",
            config_cls=None,
            deps=(),
        ),
        "HYBRID_RRF_MMR": BackendEntry(
            module="septmuse.retrieval.recipes",
            cls="get_recipe",
            config_cls=None,
            deps=(),
        ),
        "GRAPH_BFS": BackendEntry(
            module="septmuse.retrieval.recipes",
            cls="get_recipe",
            config_cls=None,
            deps=(),
        ),
        "PROGRESSIVE": BackendEntry(
            module="septmuse.retrieval.recipes",
            cls="get_recipe",
            config_cls=None,
            deps=(),
        ),
        "FORGETTING": BackendEntry(
            module="septmuse.retrieval.recipes",
            cls="get_recipe",
            config_cls=None,
            deps=(),
        ),
    },
}

# 代码默认后端。空串表示该能力默认不创建 (如 llm 需显式配置)。
_DEFAULTS: dict[str, str] = {
    "vector_store": "sqlite",
    "embedder": "hash",
    "llm": "",
    "reranker": "noop",
    "entity_extractor": "regex",
    "keyword_index": "sqlite_bm25",
    "graph_store": "sqlite",
    "search_recipe": "HYBRID_RRF",
}

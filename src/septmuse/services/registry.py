"""声明式后端注册表 — 8 能力 × 多后端的 manifest。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BackendEntry:
    """单个后端的声明式描述。"""

    module: str
    cls: str
    deps: tuple[str, ...]
    config_cls: str | None = None


# 8 能力的后端注册表。每个能力映射 后端名 -> BackendEntry。
BACKEND_MANIFEST: dict[str, dict[str, BackendEntry]] = {
    "vector_store": {
        "sqlite": BackendEntry(
            module="septmuse.storage.vector_stores.sqlalchemy_vec",
            cls="SQLAlchemyVectorStore",
            deps=(),
        ),
        "qdrant": BackendEntry(
            module="septmuse.storage.vector_stores.qdrant",
            cls="QdrantVectorStore",
            deps=("qdrant_client",),
        ),
        "chroma": BackendEntry(
            module="septmuse.storage.vector_stores.chroma",
            cls="ChromaVectorStore",
            deps=("chromadb",),
        ),
        "pgvector": BackendEntry(
            module="septmuse.storage.vector_stores.pgvector_store",
            cls="PgvectorVectorStore",
            deps=("psycopg2",),
        ),
    },
    "embedder": {
        "hash": BackendEntry(
            module="septmuse.embedders.hash",
            cls="HashEmbedder",
            deps=(),
        ),
        "onnx": BackendEntry(
            module="septmuse.embedders.onnx",
            cls="OnnxEmbedder",
            deps=("onnxruntime",),
        ),
        "onnx-zh": BackendEntry(
            module="septmuse.embedders.onnx",
            cls="OnnxEmbedder",
            deps=("onnxruntime",),
        ),
        "bge-zh": BackendEntry(
            module="septmuse.embedders.onnx",
            cls="OnnxEmbedder",
            deps=("onnxruntime",),
        ),
        "auto": BackendEntry(
            module="septmuse.embedders.auto",
            cls="AutoOnnxEmbedder",
            deps=("onnxruntime",),
        ),
        "openai": BackendEntry(
            module="septmuse.embedders.openai",
            cls="OpenAIEmbedder",
            deps=("openai",),
        ),
        "st": BackendEntry(
            module="septmuse.embedders.sentence_transformers",
            cls="SentenceTransformerEmbedder",
            deps=("sentence_transformers",),
        ),
        "mock": BackendEntry(
            module="septmuse.embedders.mock",
            cls="MockEmbedder",
            deps=(),
        ),
        "ollama": BackendEntry(
            module="septmuse.embedders.ollama",
            cls="OllamaEmbedder",
            deps=("ollama",),
        ),
        "together": BackendEntry(
            module="septmuse.embedders.together",
            cls="TogetherEmbedder",
            deps=("openai",),
        ),
        "lmstudio": BackendEntry(
            module="septmuse.embedders.lmstudio",
            cls="LMStudioEmbedder",
            deps=("openai",),
        ),
        "azure_openai": BackendEntry(
            module="septmuse.embedders.azure_openai",
            cls="AzureOpenAIEmbedder",
            deps=("openai",),
        ),
        "gemini": BackendEntry(
            module="septmuse.embedders.gemini",
            cls="GeminiEmbedder",
            deps=("google-genai",),
        ),
        "vertexai": BackendEntry(
            module="septmuse.embedders.vertexai",
            cls="VertexAIEmbedder",
            deps=("google-cloud-aiplatform",),
        ),
        "huggingface": BackendEntry(
            module="septmuse.embedders.huggingface",
            cls="HuggingFaceEmbedder",
            deps=("sentence_transformers",),
        ),
        "aws_bedrock": BackendEntry(
            module="septmuse.embedders.aws_bedrock",
            cls="AWSBedrockEmbedder",
            deps=("boto3",),
        ),
        "fastembed": BackendEntry(
            module="septmuse.embedders.fastembed",
            cls="FastEmbedEmbedder",
            deps=("fastembed",),
        ),
        "langchain": BackendEntry(
            module="septmuse.embedders.langchain",
            cls="LangchainEmbedder",
            deps=("langchain",),
        ),
    },
    "llm": {
        "openai": BackendEntry(
            module="septmuse.llms.openai",
            cls="OpenAILLM",
            deps=("openai",),
        ),
        "ollama": BackendEntry(
            module="septmuse.llms.ollama",
            cls="OllamaLLM",
            deps=("ollama",),
        ),
        "anthropic": BackendEntry(
            module="septmuse.llms.anthropic",
            cls="AnthropicLLM",
            deps=("anthropic",),
        ),
        "dashscope": BackendEntry(
            module="septmuse.llms.dashscope",
            cls="DashScopeLLM",
            deps=("dashscope",),
        ),
        "litellm": BackendEntry(
            module="septmuse.llms.litellm",
            cls="LitellmLLM",
            deps=("litellm",),
        ),
        "groq": BackendEntry(
            module="septmuse.llms.groq",
            cls="GroqLLM",
            deps=("groq",),
        ),
        "gemini": BackendEntry(
            module="septmuse.llms.gemini",
            cls="GeminiLLM",
            deps=("google-generativeai",),
        ),
        "deepseek": BackendEntry(
            module="septmuse.llms.deepseek",
            cls="DeepSeekLLM",
            deps=("openai",),
        ),
    },
    "reranker": {
        "noop": BackendEntry(
            module="septmuse.rerankers.noop",
            cls="NoopReranker",
            deps=(),
        ),
        "mmr": BackendEntry(
            module="septmuse.rerankers.mmr",
            cls="MMRReranker",
            deps=(),
        ),
        "cross_encoder": BackendEntry(
            module="septmuse.rerankers.cross_encoder",
            cls="CrossEncoderReranker",
            deps=("onnxruntime",),
        ),
        "llm": BackendEntry(
            module="septmuse.rerankers.llm",
            cls="LLMReranker",
            deps=(),
        ),
        "batch_llm": BackendEntry(
            module="septmuse.rerankers.batch_llm",
            cls="BatchLLMReranker",
            deps=(),
        ),
        "cohere": BackendEntry(
            module="septmuse.rerankers.cohere",
            cls="CohereReranker",
            deps=("cohere",),
        ),
        "sentence_transformer": BackendEntry(
            module="septmuse.rerankers.sentence_transformer",
            cls="SentenceTransformerReranker",
            deps=("sentence_transformers",),
        ),
    },
    "entity_extractor": {
        "regex": BackendEntry(
            module="septmuse.extraction.entity",
            cls="RegexEntityExtractor",
            deps=(),
        ),
        "spacy": BackendEntry(
            module="septmuse.extraction.entity",
            cls="SpacyEntityExtractor",
            deps=("spacy",),
        ),
        "none": BackendEntry(
            module="",
            cls="",
            deps=(),
        ),
    },
    "keyword_index": {
        "sqlite_bm25": BackendEntry(
            module="septmuse.storage.keyword_stores.sqlite_bm25",
            cls="SQLiteBM25Index",
            deps=(),
        ),
        "rank_bm25": BackendEntry(
            module="septmuse.storage.keyword_stores.rank_bm25",
            cls="RankBM25Index",
            deps=("rank_bm25",),
        ),
        "none": BackendEntry(
            module="",
            cls="",
            deps=(),
        ),
    },
    "graph_store": {
        "sqlite": BackendEntry(
            module="septmuse.storage.graph_stores.sqlite",
            cls="SQLiteGraphStore",
            deps=(),
        ),
        "age": BackendEntry(
            module="septmuse.storage.graph_stores.age",
            cls="AGEGraphStore",
            deps=("psycopg",),
        ),
        "neo4j": BackendEntry(
            module="septmuse.storage.graph_stores.neo4j",
            cls="Neo4jGraphStore",
            deps=("neo4j",),
        ),
    },
    "search_recipe": {
        "HYBRID_RRF": BackendEntry(
            module="septmuse.retrieval.recipes",
            cls="get_recipe",
            deps=(),
        ),
        "HYBRID_RRF_ENTITY": BackendEntry(
            module="septmuse.retrieval.recipes",
            cls="get_recipe",
            deps=(),
        ),
        "HYBRID_RRF_CROSS_ENCODER": BackendEntry(
            module="septmuse.retrieval.recipes",
            cls="get_recipe",
            deps=(),
        ),
        "HYBRID_RRF_MMR": BackendEntry(
            module="septmuse.retrieval.recipes",
            cls="get_recipe",
            deps=(),
        ),
        "GRAPH_BFS": BackendEntry(
            module="septmuse.retrieval.recipes",
            cls="get_recipe",
            deps=(),
        ),
        "PROGRESSIVE": BackendEntry(
            module="septmuse.retrieval.recipes",
            cls="get_recipe",
            deps=(),
        ),
        "FORGETTING": BackendEntry(
            module="septmuse.retrieval.recipes",
            cls="get_recipe",
            deps=(),
        ),
    },
}

# 代码默认后端。空串表示该能力默认不创建 (如 llm 需显式配置)。
_DEFAULTS: dict[str, str] = {
    "vector_store": "qdrant",
    "embedder": "bge-zh",
    "llm": "",
    "reranker": "noop",
    "entity_extractor": "regex",
    "keyword_index": "sqlite_bm25",
    "graph_store": "sqlite",
    "search_recipe": "HYBRID_RRF",
}

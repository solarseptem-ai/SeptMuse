"""扁平环境变量向后兼容测试。"""

from septmuse.configs.base import MemoryConfig


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
    """SEPTMUSE_LLM=openai 设置 llm.backend。"""
    monkeypatch.setenv("SEPTMUSE_LLM", "openai")
    config = MemoryConfig()
    assert config.llm is not None
    assert config.llm.backend == "openai"


def test_flat_env_entity_extractor(monkeypatch):
    """SEPTMUSE_ENTITY_EXTRACTOR=spacy 覆盖。"""
    monkeypatch.setenv("SEPTMUSE_ENTITY_EXTRACTOR", "spacy")
    config = MemoryConfig()
    assert config.entity_extractor.backend == "spacy"


def test_flat_env_keyword_backend(monkeypatch):
    """SEPTMUSE_KEYWORD_BACKEND=rank_bm25 覆盖。"""
    monkeypatch.setenv("SEPTMUSE_KEYWORD_BACKEND", "rank_bm25")
    config = MemoryConfig()
    assert config.keyword_index.backend == "rank_bm25"


def test_flat_env_graph_backend(monkeypatch):
    """SEPTMUSE_GRAPH_BACKEND=neo4j 覆盖。"""
    monkeypatch.setenv("SEPTMUSE_GRAPH_BACKEND", "neo4j")
    config = MemoryConfig()
    assert config.graph_store.backend == "neo4j"

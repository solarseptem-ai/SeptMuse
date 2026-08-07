"""ServiceProvider 容器单元测试。"""
from unittest.mock import patch

import pytest

from septmuse.services.providers import (
    ALL_PROVIDERS,
    embedder_provider,
    llm_provider,
    search_recipe_provider,
    vector_store_provider,
)


def test_list_backends_returns_all():
    backends = vector_store_provider.list_backends()
    assert "sqlite" in backends
    assert "qdrant" in backends
    assert "chroma" in backends
    assert "pgvector" in backends


def test_default_backend():
    assert vector_store_provider.default_backend() == "qdrant"
    assert embedder_provider.default_backend() == "bge-zh"
    assert llm_provider.default_backend() == ""


def test_is_available_zero_dep():
    assert vector_store_provider.is_available("sqlite") is True
    assert embedder_provider.is_available("hash") is True


def test_is_available_missing_dep():
    with patch("importlib.util.find_spec", return_value=None):
        assert vector_store_provider.is_available("qdrant") is False


def test_available_backends():
    avail = embedder_provider.available_backends()
    assert "hash" in avail


def test_resolve_returns_instance():
    embedder = embedder_provider.resolve("hash")
    assert embedder is not None
    assert hasattr(embedder, "embed")


def test_resolve_default():
    embedder = embedder_provider.resolve()
    assert embedder is not None


def test_resolve_unknown_raises():
    with pytest.raises(ValueError, match="Unknown"):
        vector_store_provider.resolve("nonexistent")


def test_resolve_llm_empty_returns_none():
    result = llm_provider.resolve()
    assert result is None


def test_resolve_search_recipe():
    recipe = search_recipe_provider.resolve("HYBRID_RRF")
    assert recipe is not None


def test_all_providers_has_8():
    assert len(ALL_PROVIDERS) == 8
    for cap in ["vector_store", "embedder", "llm", "reranker",
                "entity_extractor", "keyword_index", "graph_store", "search_recipe"]:
        assert cap in ALL_PROVIDERS

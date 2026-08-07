"""LangchainEmbedder 测试 — 用 FakeEmbeddings 验证桥接。"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest


class FakeEmbeddings:
    """模拟 langchain Embeddings 接口。"""

    def __init__(self, dim: int = 256):
        self._dim = dim

    def embed_query(self, text: str) -> list[float]:
        return [0.1] * self._dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * self._dim for _ in texts]


@pytest.fixture()
def mock_langchain(monkeypatch):
    """注入 mock langchain 模块, 使 FakeEmbeddings 被识别为 Embeddings 实例。"""

    class FakeEmbeddingsBase:
        def __instancecheck__(cls, instance):
            return True

    mock_embeddings = MagicMock()
    mock_embeddings.Embeddings = type("Embeddings", (), {"__instancecheck__": lambda cls, i: True})

    mock_base = MagicMock()
    mock_base.Embeddings = mock_embeddings.Embeddings

    mock_langchain = MagicMock()
    mock_langchain.embeddings = MagicMock()
    mock_langchain.embeddings.base = mock_base

    monkeypatch.setitem(sys.modules, "langchain", mock_langchain)
    monkeypatch.setitem(sys.modules, "langchain.embeddings", mock_langchain.embeddings)
    monkeypatch.setitem(sys.modules, "langchain.embeddings.base", mock_base)


class TestLangchainEmbedder:
    def test_inherits_embedder_abc(self, mock_langchain):
        from septmuse.embedders.base import Embedder
        from septmuse.embedders.langchain import LangchainEmbedder

        emb = LangchainEmbedder(model=FakeEmbeddings())
        assert isinstance(emb, Embedder)

    def test_dimension_from_model(self, mock_langchain):
        from septmuse.embedders.langchain import LangchainEmbedder

        emb = LangchainEmbedder(model=FakeEmbeddings(dim=256), embedding_dims=256)
        assert emb.dimension == 256

    def test_embed(self, mock_langchain):
        from septmuse.embedders.langchain import LangchainEmbedder

        emb = LangchainEmbedder(model=FakeEmbeddings(dim=256))
        vec = emb.embed("hello")
        assert len(vec) == 256

    def test_embed_batch(self, mock_langchain):
        from septmuse.embedders.langchain import LangchainEmbedder

        emb = LangchainEmbedder(model=FakeEmbeddings(dim=256))
        vecs = emb.embed_batch(["hello", "world"])
        assert len(vecs) == 2

    def test_requires_model_instance(self, mock_langchain):
        from septmuse.embedders.langchain import LangchainEmbedder

        with pytest.raises(ValueError, match="model"):
            LangchainEmbedder(model=None)

    def test_embed_batch_empty(self, mock_langchain):
        from septmuse.embedders.langchain import LangchainEmbedder

        emb = LangchainEmbedder(model=FakeEmbeddings())
        assert emb.embed_batch([]) == []

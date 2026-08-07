"""FastEmbedEmbedder 测试 — mock TextEmbedding, 验证 embed/embed_batch/dimension。"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest


@pytest.fixture()
def mock_fastembed(monkeypatch):
    mock_model = MagicMock()
    mock_model.embedding_size = 768
    mock_model.embed.return_value = iter([[0.1] * 768])

    mock_module = MagicMock()
    mock_module.TextEmbedding = MagicMock(return_value=mock_model)

    monkeypatch.setitem(sys.modules, "fastembed", mock_module)
    return mock_model


class TestFastEmbedEmbedder:
    def test_inherits_embedder_abc(self, mock_fastembed):
        from septmuse.embedders.base import Embedder
        from septmuse.embedders.fastembed import FastEmbedEmbedder

        emb = FastEmbedEmbedder()
        assert isinstance(emb, Embedder)

    def test_default_model(self, mock_fastembed):
        from septmuse.embedders.fastembed import FastEmbedEmbedder

        emb = FastEmbedEmbedder()
        assert emb.model == "thenlper/gte-large"

    def test_dimension(self, mock_fastembed):
        from septmuse.embedders.fastembed import FastEmbedEmbedder

        emb = FastEmbedEmbedder()
        assert emb.dimension == 768

    def test_embed(self, mock_fastembed):
        from septmuse.embedders.fastembed import FastEmbedEmbedder

        emb = FastEmbedEmbedder()
        vec = emb.embed("hello")
        assert len(vec) == 768

    def test_embed_batch(self, mock_fastembed):
        mock_fastembed.embed.return_value = iter([[0.1] * 768, [0.2] * 768])
        from septmuse.embedders.fastembed import FastEmbedEmbedder

        emb = FastEmbedEmbedder()
        vecs = emb.embed_batch(["hello", "world"])
        assert len(vecs) == 2

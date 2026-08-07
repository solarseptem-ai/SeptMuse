"""OllamaEmbedder 测试 — mock ollama.Client, 验证 embed/embed_batch/pull/零配置。"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def mock_ollama_client():
    mock_client = MagicMock()
    mock_client.list.return_value = {"models": [{"name": "nomic-embed-text:latest"}]}
    mock_client.embed.return_value = {"embeddings": [[0.1, 0.2, 0.3]]}
    return mock_client


class TestOllamaEmbedder:
    def test_inherits_embedder_abc(self, mock_ollama_client):
        with patch("ollama.Client", return_value=mock_ollama_client):
            from septmuse.embedders.base import Embedder
            from septmuse.embedders.ollama import OllamaEmbedder

            emb = OllamaEmbedder()
            assert isinstance(emb, Embedder)

    def test_default_model(self, mock_ollama_client):
        with patch("ollama.Client", return_value=mock_ollama_client):
            from septmuse.embedders.ollama import OllamaEmbedder

            emb = OllamaEmbedder()
            assert emb.model == "nomic-embed-text"

    def test_default_dimension(self, mock_ollama_client):
        with patch("ollama.Client", return_value=mock_ollama_client):
            from septmuse.embedders.ollama import OllamaEmbedder

            emb = OllamaEmbedder()
            assert emb.dimension == 512

    def test_embed(self, mock_ollama_client):
        with patch("ollama.Client", return_value=mock_ollama_client):
            from septmuse.embedders.ollama import OllamaEmbedder

            emb = OllamaEmbedder()
            vec = emb.embed("hello")
            assert vec == [0.1, 0.2, 0.3]
            mock_ollama_client.embed.assert_called_once_with(model="nomic-embed-text", input="hello")

    def test_embed_batch(self, mock_ollama_client):
        mock_ollama_client.embed.return_value = {"embeddings": [[0.1, 0.2], [0.3, 0.4]]}
        with patch("ollama.Client", return_value=mock_ollama_client):
            from septmuse.embedders.ollama import OllamaEmbedder

            emb = OllamaEmbedder()
            vecs = emb.embed_batch(["hello", "world"])
            assert len(vecs) == 2
            assert vecs[0] == [0.1, 0.2]

    def test_embed_batch_empty(self, mock_ollama_client):
        with patch("ollama.Client", return_value=mock_ollama_client):
            from septmuse.embedders.ollama import OllamaEmbedder

            emb = OllamaEmbedder()
            assert emb.embed_batch([]) == []

    def test_embed_batch_count_mismatch_raises(self, mock_ollama_client):
        mock_ollama_client.embed.return_value = {"embeddings": [[0.1]]}
        with patch("ollama.Client", return_value=mock_ollama_client):
            from septmuse.embedders.ollama import OllamaEmbedder

            emb = OllamaEmbedder()
            with pytest.raises(ValueError, match="embed"):
                emb.embed_batch(["a", "b"])

    def test_ensure_model_exists_pulls(self, mock_ollama_client):
        mock_ollama_client.list.return_value = {"models": []}
        with patch("ollama.Client", return_value=mock_ollama_client):
            from septmuse.embedders.ollama import OllamaEmbedder

            OllamaEmbedder()
            mock_ollama_client.pull.assert_called_once_with("nomic-embed-text")

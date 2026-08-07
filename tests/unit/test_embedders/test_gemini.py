"""GeminiEmbedder 测试 — mock genai.Client, 验证 embed/embed_batch/output_dimensionality。"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def mock_genai():
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.embeddings = [MagicMock(values=[0.1] * 768)]
    mock_client.models.embed_content.return_value = mock_response
    return mock_client


class TestGeminiEmbedder:
    def test_inherits_embedder_abc(self, mock_genai):
        with patch("google.genai.Client", return_value=mock_genai):
            from septmuse.embedders.base import Embedder
            from septmuse.embedders.gemini import GeminiEmbedder

            emb = GeminiEmbedder(api_key="test")
            assert isinstance(emb, Embedder)

    def test_default_model(self, mock_genai):
        with patch("google.genai.Client", return_value=mock_genai):
            from septmuse.embedders.gemini import GeminiEmbedder

            emb = GeminiEmbedder(api_key="test")
            assert emb.model == "models/gemini-embedding-001"

    def test_default_dimension(self, mock_genai):
        with patch("google.genai.Client", return_value=mock_genai):
            from septmuse.embedders.gemini import GeminiEmbedder

            emb = GeminiEmbedder(api_key="test")
            assert emb.dimension == 768

    def test_embed(self, mock_genai):
        with patch("google.genai.Client", return_value=mock_genai):
            from septmuse.embedders.gemini import GeminiEmbedder

            emb = GeminiEmbedder(api_key="test")
            vec = emb.embed("hello")
            assert len(vec) == 768

    def test_embed_batch(self, mock_genai):
        mock_response = MagicMock()
        mock_response.embeddings = [MagicMock(values=[0.1] * 768), MagicMock(values=[0.2] * 768)]
        mock_genai.models.embed_content.return_value = mock_response
        with patch("google.genai.Client", return_value=mock_genai):
            from septmuse.embedders.gemini import GeminiEmbedder

            emb = GeminiEmbedder(api_key="test")
            vecs = emb.embed_batch(["hello", "world"])
            assert len(vecs) == 2

    def test_embed_batch_count_mismatch_raises(self, mock_genai):
        mock_response = MagicMock()
        mock_response.embeddings = [MagicMock(values=[0.1] * 768)]
        mock_genai.models.embed_content.return_value = mock_response
        with patch("google.genai.Client", return_value=mock_genai):
            from septmuse.embedders.gemini import GeminiEmbedder

            emb = GeminiEmbedder(api_key="test")
            with pytest.raises(ValueError, match="embed_batch"):
                emb.embed_batch(["a", "b"])

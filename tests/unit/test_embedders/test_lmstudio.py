"""LMStudioEmbedder 测试 — mock openai.OpenAI, 验证 base_url + 默认值。"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


@pytest.fixture()
def mock_openai(monkeypatch):
    import openai

    mock_client = MagicMock()
    mock_client.embeddings.create.return_value = MagicMock(
        data=[MagicMock(embedding=[0.1] * 1536, index=0)]
    )
    monkeypatch.setattr(openai, "OpenAI", MagicMock(return_value=mock_client))
    return mock_client


class TestLMStudioEmbedder:
    def test_inherits_embedder_abc(self, mock_openai):
        from septmuse.embedders.base import Embedder
        from septmuse.embedders.lmstudio import LMStudioEmbedder

        emb = LMStudioEmbedder()
        assert isinstance(emb, Embedder)

    def test_default_model(self, mock_openai):
        from septmuse.embedders.lmstudio import LMStudioEmbedder

        emb = LMStudioEmbedder()
        assert "nomic-embed" in emb.model

    def test_default_dimension(self, mock_openai):
        from septmuse.embedders.lmstudio import LMStudioEmbedder

        emb = LMStudioEmbedder()
        assert emb.dimension == 1536

    def test_embed(self, mock_openai):
        from septmuse.embedders.lmstudio import LMStudioEmbedder

        emb = LMStudioEmbedder()
        vec = emb.embed("hello")
        assert len(vec) == 1536

    def test_embed_batch(self, mock_openai):
        mock_openai.embeddings.create.return_value = MagicMock(
            data=[MagicMock(embedding=[0.1] * 1536, index=i) for i in range(2)]
        )
        from septmuse.embedders.lmstudio import LMStudioEmbedder

        emb = LMStudioEmbedder()
        vecs = emb.embed_batch(["hello", "world"])
        assert len(vecs) == 2

    def test_default_base_url(self, mock_openai):
        import openai

        from septmuse.embedders.lmstudio import LMStudioEmbedder

        LMStudioEmbedder()
        call_kwargs = openai.OpenAI.call_args.kwargs
        assert call_kwargs["base_url"] == "http://localhost:1234/v1"

    def test_custom_base_url(self, mock_openai):
        import openai

        from septmuse.embedders.lmstudio import LMStudioEmbedder

        LMStudioEmbedder(lmstudio_base_url="http://my-host:8080/v1")
        call_kwargs = openai.OpenAI.call_args.kwargs
        assert call_kwargs["base_url"] == "http://my-host:8080/v1"

"""TogetherEmbedder 测试 — mock openai.OpenAI, 验证 base_url + 默认 model/dims。"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


@pytest.fixture()
def mock_openai(monkeypatch):
    import openai

    mock_client = MagicMock()
    mock_client.embeddings.create.return_value = MagicMock(
        data=[MagicMock(embedding=[0.1] * 1024, index=0)]
    )
    monkeypatch.setattr(openai, "OpenAI", MagicMock(return_value=mock_client))
    return mock_client


class TestTogetherEmbedder:
    def test_inherits_embedder_abc(self, mock_openai):
        from septmuse.embedders.base import Embedder
        from septmuse.embedders.together import TogetherEmbedder

        emb = TogetherEmbedder(api_key="test")
        assert isinstance(emb, Embedder)

    def test_default_model(self, mock_openai):
        from septmuse.embedders.together import TogetherEmbedder

        emb = TogetherEmbedder(api_key="test")
        assert emb.model == "intfloat/multilingual-e5-large-instruct"

    def test_default_dimension(self, mock_openai):
        from septmuse.embedders.together import TogetherEmbedder

        emb = TogetherEmbedder(api_key="test")
        assert emb.dimension == 1024

    def test_embed(self, mock_openai):
        from septmuse.embedders.together import TogetherEmbedder

        emb = TogetherEmbedder(api_key="test")
        vec = emb.embed("hello")
        assert len(vec) == 1024

    def test_embed_batch(self, mock_openai):
        mock_openai.embeddings.create.return_value = MagicMock(
            data=[MagicMock(embedding=[0.1] * 1024, index=i) for i in range(2)]
        )
        from septmuse.embedders.together import TogetherEmbedder

        emb = TogetherEmbedder(api_key="test")
        vecs = emb.embed_batch(["hello", "world"])
        assert len(vecs) == 2

    def test_uses_together_base_url(self, mock_openai, monkeypatch):
        import openai

        monkeypatch.setenv("TOGETHER_API_KEY", "env-key")
        from septmuse.embedders.together import TogetherEmbedder

        TogetherEmbedder()
        call_kwargs = openai.OpenAI.call_args.kwargs
        assert "together.xyz" in call_kwargs.get("base_url", "")

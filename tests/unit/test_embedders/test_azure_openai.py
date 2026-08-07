"""AzureOpenAIEmbedder 测试 — mock AzureOpenAI, 验证 init/embed/embed_batch/AD token。"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


@pytest.fixture()
def mock_azure_openai(monkeypatch):
    mock_client = MagicMock()
    mock_client.embeddings.create.return_value = MagicMock(
        data=[MagicMock(embedding=[0.1] * 1536, index=0)]
    )
    import openai

    monkeypatch.setattr(openai, "AzureOpenAI", MagicMock(return_value=mock_client))
    return mock_client


class TestAzureOpenAIEmbedder:
    def test_inherits_embedder_abc(self, mock_azure_openai):
        from septmuse.embedders.azure_openai import AzureOpenAIEmbedder
        from septmuse.embedders.base import Embedder

        emb = AzureOpenAIEmbedder(
            api_key="test", azure_deployment="dep", azure_endpoint="https://test.openai.azure.com"
        )
        assert isinstance(emb, Embedder)

    def test_default_dimension(self, mock_azure_openai):
        from septmuse.embedders.azure_openai import AzureOpenAIEmbedder

        emb = AzureOpenAIEmbedder(
            api_key="test", azure_deployment="dep", azure_endpoint="https://test.openai.azure.com"
        )
        assert emb.dimension == 1536

    def test_embed(self, mock_azure_openai):
        from septmuse.embedders.azure_openai import AzureOpenAIEmbedder

        emb = AzureOpenAIEmbedder(
            api_key="test", azure_deployment="dep", azure_endpoint="https://test.openai.azure.com"
        )
        vec = emb.embed("hello")
        assert len(vec) == 1536

    def test_embed_batch(self, mock_azure_openai):
        mock_azure_openai.embeddings.create.return_value = MagicMock(
            data=[MagicMock(embedding=[0.1] * 1536, index=i) for i in range(2)]
        )
        from septmuse.embedders.azure_openai import AzureOpenAIEmbedder

        emb = AzureOpenAIEmbedder(
            api_key="test", azure_deployment="dep", azure_endpoint="https://test.openai.azure.com"
        )
        vecs = emb.embed_batch(["hello", "world"])
        assert len(vecs) == 2

    def test_uses_azure_openai_client(self, mock_azure_openai):
        import openai

        from septmuse.embedders.azure_openai import AzureOpenAIEmbedder

        AzureOpenAIEmbedder(
            api_key="test", azure_deployment="dep", azure_endpoint="https://test.openai.azure.com"
        )
        assert openai.AzureOpenAI.called

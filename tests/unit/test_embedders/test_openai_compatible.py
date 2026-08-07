"""_OpenAICompatibleEmbedder 基类测试 — mock client 验证 embed/embed_batch/matryoshka。"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


@pytest.fixture()
def mock_client():
    client = MagicMock()
    client.embeddings = MagicMock()
    client.embeddings.create = MagicMock()
    return client


class TestOpenAICompatibleEmbedder:
    def test_embed_calls_create(self, mock_client):
        from septmuse.embedders._openai_compatible import _OpenAICompatibleEmbedder

        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3], index=0)]
        mock_client.embeddings.create.return_value = mock_response

        emb = _OpenAICompatibleEmbedder(
            client=mock_client, model="text-embedding-3-small", dim=3, pass_dimensions_to_api=False
        )
        vec = emb.embed("hello")
        assert vec == [0.1, 0.2, 0.3]
        mock_client.embeddings.create.assert_called_once()

    def test_embed_replaces_newlines(self, mock_client):
        from septmuse.embedders._openai_compatible import _OpenAICompatibleEmbedder

        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1], index=0)]
        mock_client.embeddings.create.return_value = mock_response

        emb = _OpenAICompatibleEmbedder(
            client=mock_client, model="m", dim=1, pass_dimensions_to_api=False
        )
        emb.embed("hello\nworld")
        call_kwargs = mock_client.embeddings.create.call_args
        assert "\n" not in call_kwargs.kwargs["input"][0]

    def test_embed_passes_dimensions_when_matryoshka(self, mock_client):
        from septmuse.embedders._openai_compatible import _OpenAICompatibleEmbedder

        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1], index=0)]
        mock_client.embeddings.create.return_value = mock_response

        emb = _OpenAICompatibleEmbedder(
            client=mock_client, model="m", dim=512, pass_dimensions_to_api=True
        )
        emb.embed("hello")
        call_kwargs = mock_client.embeddings.create.call_args
        assert call_kwargs.kwargs["dimensions"] == 512

    def test_embed_no_dimensions_when_not_matryoshka(self, mock_client):
        from septmuse.embedders._openai_compatible import _OpenAICompatibleEmbedder

        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1], index=0)]
        mock_client.embeddings.create.return_value = mock_response

        emb = _OpenAICompatibleEmbedder(
            client=mock_client, model="m", dim=1536, pass_dimensions_to_api=False
        )
        emb.embed("hello")
        call_kwargs = mock_client.embeddings.create.call_args
        assert "dimensions" not in call_kwargs.kwargs

    def test_embed_batch_chunks_100(self, mock_client):
        from septmuse.embedders._openai_compatible import _OpenAICompatibleEmbedder

        def fake_create(*args, **kwargs):
            texts = kwargs["input"]
            return MagicMock(data=[MagicMock(embedding=[0.5], index=i) for i in range(len(texts))])

        mock_client.embeddings.create.side_effect = fake_create

        emb = _OpenAICompatibleEmbedder(
            client=mock_client, model="m", dim=1, pass_dimensions_to_api=False
        )
        texts = [f"text{i}" for i in range(250)]
        vecs = emb.embed_batch(texts)
        assert len(vecs) == 250
        assert mock_client.embeddings.create.call_count == 3

    def test_embed_batch_count_mismatch_raises(self, mock_client):
        from septmuse.embedders._openai_compatible import _OpenAICompatibleEmbedder

        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1], index=0)]
        mock_client.embeddings.create.return_value = mock_response

        emb = _OpenAICompatibleEmbedder(
            client=mock_client, model="m", dim=1, pass_dimensions_to_api=False
        )
        with pytest.raises(ValueError, match="embed_batch"):
            emb.embed_batch(["a", "b"])

    def test_dimension_property(self, mock_client):
        from septmuse.embedders._openai_compatible import _OpenAICompatibleEmbedder

        emb = _OpenAICompatibleEmbedder(
            client=mock_client, model="m", dim=768, pass_dimensions_to_api=False
        )
        assert emb.dimension == 768

    def test_inherits_embedder_abc(self, mock_client):
        from septmuse.embedders._openai_compatible import _OpenAICompatibleEmbedder
        from septmuse.embedders.base import Embedder

        emb = _OpenAICompatibleEmbedder(
            client=mock_client, model="m", dim=1, pass_dimensions_to_api=False
        )
        assert isinstance(emb, Embedder)

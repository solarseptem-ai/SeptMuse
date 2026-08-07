"""HuggingFaceEmbedder 测试 — mock 双模式 (TEI server + 本地 ST)。"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


class TestHuggingFaceTEIMode:
    @pytest.fixture()
    def mock_openai(self, monkeypatch):
        import openai

        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = MagicMock(
            data=[MagicMock(embedding=[0.1] * 768, index=0)]
        )
        monkeypatch.setattr(openai, "OpenAI", MagicMock(return_value=mock_client))
        return mock_client

    def test_tei_mode_uses_openai_client(self, mock_openai):
        from septmuse.embedders.huggingface import HuggingFaceEmbedder

        emb = HuggingFaceEmbedder(huggingface_base_url="https://my-tei.server")
        vec = emb.embed("hello")
        assert len(vec) == 768

    def test_tei_mode_default_model(self, mock_openai):
        from septmuse.embedders.huggingface import HuggingFaceEmbedder

        emb = HuggingFaceEmbedder(huggingface_base_url="https://my-tei.server")
        assert emb.model == "tei"

    def test_tei_mode_embed_batch(self, mock_openai):
        mock_openai.embeddings.create.return_value = MagicMock(
            data=[MagicMock(embedding=[0.1] * 768, index=i) for i in range(2)]
        )
        from septmuse.embedders.huggingface import HuggingFaceEmbedder

        emb = HuggingFaceEmbedder(huggingface_base_url="https://my-tei.server")
        vecs = emb.embed_batch(["hello", "world"])
        assert len(vecs) == 2


class TestHuggingFaceLocalMode:
    @pytest.fixture()
    def mock_st(self, monkeypatch):
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.return_value = MagicMock(tolist=lambda: [0.1] * 384)
        mock_module = MagicMock()
        mock_module.SentenceTransformer = MagicMock(return_value=mock_model)
        monkeypatch.setitem(__import__("sys").modules, "sentence_transformers", mock_module)
        return mock_model

    def test_local_mode_uses_sentence_transformer(self, mock_st):
        from septmuse.embedders.huggingface import HuggingFaceEmbedder

        emb = HuggingFaceEmbedder()
        assert emb.dimension == 384

    def test_local_mode_embed(self, mock_st):
        from septmuse.embedders.huggingface import HuggingFaceEmbedder

        emb = HuggingFaceEmbedder()
        vec = emb.embed("hello")
        assert len(vec) == 384

    def test_local_mode_embed_batch(self, mock_st):
        mock_st.encode.return_value = [MagicMock(tolist=lambda: [0.1] * 384)]
        from septmuse.embedders.huggingface import HuggingFaceEmbedder

        emb = HuggingFaceEmbedder()
        vecs = emb.embed_batch(["hello"])
        assert len(vecs) == 1

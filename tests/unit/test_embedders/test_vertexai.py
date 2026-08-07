"""VertexAIEmbedder 测试 — mock TextEmbeddingModel, 验证 memory_action task_type 切换。"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest


@pytest.fixture()
def mock_vertexai(monkeypatch):
    mock_model = MagicMock()
    mock_model.get_embeddings.return_value = [MagicMock(values=[0.1] * 256)]

    mock_input_cls = MagicMock()
    mock_model_cls = MagicMock()
    mock_model_cls.from_pretrained.return_value = mock_model

    mock_lang_models = MagicMock()
    mock_lang_models.TextEmbeddingInput = mock_input_cls
    mock_lang_models.TextEmbeddingModel = mock_model_cls

    monkeypatch.setitem(sys.modules, "vertexai", MagicMock())
    monkeypatch.setitem(sys.modules, "vertexai.language_models", mock_lang_models)
    return mock_model, mock_input_cls


class TestVertexAIEmbedder:
    def test_inherits_embedder_abc(self, mock_vertexai):
        from septmuse.embedders.base import Embedder
        from septmuse.embedders.vertexai import VertexAIEmbedder

        emb = VertexAIEmbedder(vertex_credentials_json="/fake/path.json")
        assert isinstance(emb, Embedder)

    def test_default_model(self, mock_vertexai):
        from septmuse.embedders.vertexai import VertexAIEmbedder

        emb = VertexAIEmbedder(vertex_credentials_json="/fake/path.json")
        assert emb.model == "gemini-embedding-001"

    def test_default_dimension(self, mock_vertexai):
        from septmuse.embedders.vertexai import VertexAIEmbedder

        emb = VertexAIEmbedder(vertex_credentials_json="/fake/path.json")
        assert emb.dimension == 256

    def test_embed_no_memory_action(self, mock_vertexai):
        from septmuse.embedders.vertexai import VertexAIEmbedder

        emb = VertexAIEmbedder(vertex_credentials_json="/fake/path.json")
        vec = emb.embed("hello")
        assert len(vec) == 256

    def test_embed_add_uses_retrieval_document(self, mock_vertexai):
        _, mock_input_cls = mock_vertexai
        from septmuse.embedders.vertexai import VertexAIEmbedder

        emb = VertexAIEmbedder(vertex_credentials_json="/fake/path.json")
        emb.embed("hello", memory_action="add")
        assert mock_input_cls.called
        call_kwargs = mock_input_cls.call_args
        assert call_kwargs.kwargs.get("task_type") == "RETRIEVAL_DOCUMENT"

    def test_embed_search_uses_retrieval_query(self, mock_vertexai):
        _, mock_input_cls = mock_vertexai
        from septmuse.embedders.vertexai import VertexAIEmbedder

        emb = VertexAIEmbedder(vertex_credentials_json="/fake/path.json")
        emb.embed("hello", memory_action="search")
        assert mock_input_cls.called
        call_kwargs = mock_input_cls.call_args
        assert call_kwargs.kwargs.get("task_type") == "RETRIEVAL_QUERY"

    def test_embed_batch(self, mock_vertexai):
        mock_model, _ = mock_vertexai
        mock_model.get_embeddings.return_value = [
            MagicMock(values=[0.1] * 256), MagicMock(values=[0.2] * 256)
        ]
        from septmuse.embedders.vertexai import VertexAIEmbedder

        emb = VertexAIEmbedder(vertex_credentials_json="/fake/path.json")
        vecs = emb.embed_batch(["hello", "world"])
        assert len(vecs) == 2

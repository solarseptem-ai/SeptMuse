"""AWSBedrockEmbedder 测试 — mock boto3.client, 验证 titan/cohere body + L2 归一化。"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def mock_bedrock():
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.get.return_value.read.return_value = json.dumps({"embedding": [3.0, 4.0]})
    mock_client.invoke_model.return_value = mock_response
    return mock_client


class TestAWSBedrockEmbedder:
    def test_inherits_embedder_abc(self, mock_bedrock):
        with patch("boto3.client", return_value=mock_bedrock):
            from septmuse.embedders.aws_bedrock import AWSBedrockEmbedder
            from septmuse.embedders.base import Embedder

            emb = AWSBedrockEmbedder()
            assert isinstance(emb, Embedder)

    def test_default_model(self, mock_bedrock):
        with patch("boto3.client", return_value=mock_bedrock):
            from septmuse.embedders.aws_bedrock import AWSBedrockEmbedder

            emb = AWSBedrockEmbedder()
            assert emb.model == "amazon.titan-embed-text-v1"

    def test_embed_titan(self, mock_bedrock):
        with patch("boto3.client", return_value=mock_bedrock):
            from septmuse.embedders.aws_bedrock import AWSBedrockEmbedder

            emb = AWSBedrockEmbedder()
            vec = emb.embed("hello")
            assert len(vec) == 2
            call_kwargs = mock_bedrock.invoke_model.call_args.kwargs
            body = json.loads(call_kwargs["body"])
            assert body["inputText"] == "hello"

    def test_embed_cohere(self, mock_bedrock):
        mock_response = MagicMock()
        mock_response.get.return_value.read.return_value = json.dumps({"embeddings": [[1.0, 0.0]]})
        mock_bedrock.invoke_model.return_value = mock_response
        with patch("boto3.client", return_value=mock_bedrock):
            from septmuse.embedders.aws_bedrock import AWSBedrockEmbedder

            emb = AWSBedrockEmbedder(model="cohere.embed-multilingual-v3")
            vec = emb.embed("hello")
            assert len(vec) == 2
            call_kwargs = mock_bedrock.invoke_model.call_args.kwargs
            body = json.loads(call_kwargs["body"])
            assert body["input_type"] == "search_document"
            assert body["texts"] == ["hello"]

    def test_embed_l2_normalization(self, mock_bedrock):
        mock_response = MagicMock()
        mock_response.get.return_value.read.return_value = json.dumps({"embedding": [3.0, 4.0]})
        mock_bedrock.invoke_model.return_value = mock_response
        with patch("boto3.client", return_value=mock_bedrock):
            from septmuse.embedders.aws_bedrock import AWSBedrockEmbedder

            emb = AWSBedrockEmbedder()
            vec = emb.embed("hello")
            assert abs(vec[0] ** 2 + vec[1] ** 2 - 1.0) < 1e-6

#  Copyright 2026 The sonhhxg0529 Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""OpenAIEmbedder 单元测试 (mock client, 不调真实 API)。

借鉴 tests/unit/test_rbac_rest_openai.py TestOpenAILLM 模式:
- mock openai.OpenAI 构造函数返回 MagicMock client
- 验证 embed/embed_batch/dimension/matryoshka/零配置
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest


@pytest.fixture()
def mock_openai_client(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Mock openai.OpenAI 构造函数, 返回 mock client。"""
    import openai

    mock_client = MagicMock()
    mock_client.embeddings = MagicMock()
    mock_client.embeddings.create = MagicMock()
    monkeypatch.setattr(openai, "OpenAI", MagicMock(return_value=mock_client))
    return mock_client


@pytest.fixture()
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """清理 OPENAI 环境变量, 避免污染测试。"""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)


class TestOpenAIEmbedderInit:
    def test_inherits_embedder_abc(self, mock_openai_client: MagicMock, clean_env: None) -> None:
        from septmuse.embedders.base import Embedder
        from septmuse.embedders.openai import OpenAIEmbedder

        embedder = OpenAIEmbedder(api_key="sk-test")
        assert isinstance(embedder, Embedder)

    def test_dimension_default(self, mock_openai_client: MagicMock, clean_env: None) -> None:
        from septmuse.embedders.openai import OpenAIEmbedder

        embedder = OpenAIEmbedder(api_key="sk-test")
        assert embedder.dimension == 1536

    def test_dimension_custom(self, mock_openai_client: MagicMock, clean_env: None) -> None:
        from septmuse.embedders.openai import OpenAIEmbedder

        embedder = OpenAIEmbedder(api_key="sk-test", embedding_dims=256)
        assert embedder.dimension == 256

    def test_zero_config_reads_env_key(self, mock_openai_client: MagicMock, monkeypatch: pytest.MonkeyPatch) -> None:
        from septmuse.embedders.openai import OpenAIEmbedder

        monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
        embedder = OpenAIEmbedder()
        assert embedder._api_key == "sk-from-env"

    def test_missing_key_uses_dummy(self, mock_openai_client: MagicMock, clean_env: None) -> None:
        from septmuse.embedders.openai import OpenAIEmbedder

        embedder = OpenAIEmbedder(api_key=None)
        assert embedder is not None
        assert embedder.dimension == 1536

    def test_base_url_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_constructor = MagicMock()
        monkeypatch.setattr("openai.OpenAI", mock_constructor)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("OPENAI_BASE_URL", "https://custom.example.com")
        from septmuse.embedders.openai import OpenAIEmbedder

        OpenAIEmbedder()
        assert mock_constructor.call_args.kwargs.get("base_url") == "https://custom.example.com"

    def test_import_error_without_openai(self, monkeypatch: pytest.MonkeyPatch, clean_env: None) -> None:
        monkeypatch.setitem(sys.modules, "openai", None)
        with pytest.raises(ImportError, match="openai package required"):
            from septmuse.embedders.openai import OpenAIEmbedder

            OpenAIEmbedder(api_key="sk-test")


class TestOpenAIEmbedderEmbed:
    def test_embed_returns_list_of_floats(self, mock_openai_client: MagicMock, clean_env: None) -> None:
        from septmuse.embedders.openai import OpenAIEmbedder

        mock_openai_client.embeddings.create.return_value.data = [MagicMock(embedding=[0.1, 0.2, 0.3], index=0)]
        embedder = OpenAIEmbedder(api_key="sk-test")
        vec = embedder.embed("hello")
        assert isinstance(vec, list)
        assert all(isinstance(x, float) for x in vec)
        assert len(vec) == 3

    def test_embed_replaces_newlines(self, mock_openai_client: MagicMock, clean_env: None) -> None:
        from septmuse.embedders.openai import OpenAIEmbedder

        mock_openai_client.embeddings.create.return_value.data = [MagicMock(embedding=[0.1], index=0)]
        embedder = OpenAIEmbedder(api_key="sk-test")
        embedder.embed("line1\nline2")
        call_kwargs = mock_openai_client.embeddings.create.call_args.kwargs
        input_text = call_kwargs["input"][0]
        assert "\n" not in input_text
        assert " " in input_text

    def test_matryoshka_passes_dimensions(self, mock_openai_client: MagicMock, clean_env: None) -> None:
        from septmuse.embedders.openai import OpenAIEmbedder

        mock_openai_client.embeddings.create.return_value.data = [MagicMock(embedding=[0.1], index=0)]
        embedder = OpenAIEmbedder(api_key="sk-test", embedding_dims=256)
        embedder.embed("text")
        call_kwargs = mock_openai_client.embeddings.create.call_args.kwargs
        assert call_kwargs.get("dimensions") == 256

    def test_non_matryoshka_omits_dimensions(self, mock_openai_client: MagicMock, clean_env: None) -> None:
        from septmuse.embedders.openai import OpenAIEmbedder

        mock_openai_client.embeddings.create.return_value.data = [MagicMock(embedding=[0.1], index=0)]
        embedder = OpenAIEmbedder(api_key="sk-test")
        embedder.embed("text")
        call_kwargs = mock_openai_client.embeddings.create.call_args.kwargs
        assert "dimensions" not in call_kwargs


class TestOpenAIEmbedderBatch:
    def test_embed_batch_empty(self, mock_openai_client: MagicMock, clean_env: None) -> None:
        from septmuse.embedders.openai import OpenAIEmbedder

        embedder = OpenAIEmbedder(api_key="sk-test")
        assert embedder.embed_batch([]) == []

    def test_embed_batch_chunks_100(self, mock_openai_client: MagicMock, clean_env: None) -> None:
        from septmuse.embedders.openai import OpenAIEmbedder

        def mock_create(**kwargs):
            n = len(kwargs["input"])
            return MagicMock(data=[MagicMock(embedding=[0.1], index=i) for i in range(n)])

        mock_openai_client.embeddings.create.side_effect = mock_create
        embedder = OpenAIEmbedder(api_key="sk-test")
        vecs = embedder.embed_batch(["text"] * 150)
        assert len(vecs) == 150
        assert mock_openai_client.embeddings.create.call_count == 2

    def test_embed_batch_sorts_by_index(self, mock_openai_client: MagicMock, clean_env: None) -> None:
        from septmuse.embedders.openai import OpenAIEmbedder

        mock_openai_client.embeddings.create.return_value.data = [
            MagicMock(embedding=[0.3], index=2),
            MagicMock(embedding=[0.1], index=0),
            MagicMock(embedding=[0.2], index=1),
        ]
        embedder = OpenAIEmbedder(api_key="sk-test")
        vecs = embedder.embed_batch(["a", "b", "c"])
        assert vecs[0] == [0.1]
        assert vecs[1] == [0.2]
        assert vecs[2] == [0.3]

    def test_embed_batch_count_mismatch(self, mock_openai_client: MagicMock, clean_env: None) -> None:
        from septmuse.embedders.openai import OpenAIEmbedder

        mock_openai_client.embeddings.create.return_value.data = [
            MagicMock(embedding=[0.1], index=0),
            MagicMock(embedding=[0.2], index=1),
        ]
        embedder = OpenAIEmbedder(api_key="sk-test")
        with pytest.raises(ValueError, match="2 embeddings for 3 texts"):
            embedder.embed_batch(["a", "b", "c"])

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
"""OnnxEmbedder 单元测试。

onnxruntime 未安装时全部 skip (integration marker)。
模型首次使用会从 HuggingFace 下载量化 ONNX (~23MB), 故标记为 integration。
"""

from __future__ import annotations

import math

import pytest

pytestmark = pytest.mark.integration

try:
    import onnxruntime  # noqa: F401
    import tokenizers  # noqa: F401

    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

SKIP_REASON = "onnxruntime + tokenizers 未安装。pip install septmuse[onnx]"


@pytest.fixture(scope="module")
def embedder():
    from septmuse.embedders.onnx import OnnxEmbedder

    return OnnxEmbedder()


@pytest.mark.skipif(not HAS_ONNX, reason=SKIP_REASON)
class TestOnnxEmbedder:
    def test_dimension_is_384(self, embedder) -> None:
        assert embedder.dimension == 384

    def test_embed_returns_normalized_vector(self, embedder) -> None:
        vec = embedder.embed("hello world")
        assert len(vec) == 384
        norm = math.sqrt(sum(v * v for v in vec))
        assert abs(norm - 1.0) < 0.01

    def test_embed_batch(self, embedder) -> None:
        texts = ["hello", "world"]
        vecs = embedder.embed_batch(texts)
        assert len(vecs) == 2
        assert all(len(v) == 384 for v in vecs)

    def test_similar_texts_high_cosine(self, embedder) -> None:
        """语义相似文本余弦相似度高。"""
        v1 = embedder.embed("I love programming")
        v2 = embedder.embed("I enjoy coding")
        dot = sum(a * b for a, b in zip(v1, v2, strict=True))
        assert dot > 0.5

    def test_unrelated_texts_low_cosine(self, embedder) -> None:
        """无关文本余弦相似度低。"""
        v1 = embedder.embed("python programming")
        v2 = embedder.embed("sunny weather today")
        dot = sum(a * b for a, b in zip(v1, v2, strict=True))
        assert dot < 0.5

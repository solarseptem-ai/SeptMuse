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
"""AutoOnnxEmbedder 单元测试。"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration

try:
    import onnxruntime  # noqa: F401

    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

SKIP_REASON = "onnxruntime + tokenizers 未安装。pip install septmuse[onnx]"


@pytest.mark.skipif(not HAS_ONNX, reason=SKIP_REASON)
class TestAutoOnnxEmbedder:
    def test_default_lang_zh_when_no_env(self, monkeypatch) -> None:
        monkeypatch.delenv("SEPTMUSE_LANG", raising=False)
        monkeypatch.setenv("SEPTMUSE_EMBEDDER", "auto")
        from septmuse.embedders.auto import AutoOnnxEmbedder

        emb = AutoOnnxEmbedder()
        assert emb._lang == "zh"
        assert emb.dimension == 384

    def test_lang_en_from_env(self, monkeypatch) -> None:
        monkeypatch.setenv("SEPTMUSE_LANG", "en")
        from septmuse.embedders.auto import AutoOnnxEmbedder

        emb = AutoOnnxEmbedder()
        assert emb._lang == "en"

    def test_detect_from_sample_text(self, monkeypatch) -> None:
        monkeypatch.delenv("SEPTMUSE_LANG", raising=False)
        from septmuse.embedders.auto import AutoOnnxEmbedder

        emb_zh = AutoOnnxEmbedder(sample_text="我喜欢编程")
        assert emb_zh._lang == "zh"

        emb_en = AutoOnnxEmbedder(sample_text="I love programming")
        assert emb_en._lang == "en"

    def test_embed_uses_same_model_for_all(self, monkeypatch) -> None:
        """整个 session 用同一模型 (不 per-query 切换)。"""
        monkeypatch.setenv("SEPTMUSE_LANG", "zh")
        from septmuse.embedders.auto import AutoOnnxEmbedder

        emb = AutoOnnxEmbedder()
        v1 = emb.embed("我喜欢编程")
        v2 = emb.embed("I love programming")
        # 同一模型嵌入, 维度一致, 可比较
        assert len(v1) == len(v2) == 384

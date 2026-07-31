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
"""AutoOnnxEmbedder — init 时语言检测, 自动选 ONNX 嵌入模型。

策略:
1. SEPTMUSE_LANG 环境变量 > sample_text 检测 > 默认 'zh' (中文优先项目)
2. 'zh' → Xenova/paraphrase-multilingual-MiniLM-L12-v2 (多语言, 384 dim)
3. 'en' → Xenova/all-MiniLM-L6-v2 (英文, 384 dim)

关键: 整个 session 用同一个模型 (不同模型投影到不同语义空间,
per-query 切换会破坏向量可比性)。
"""

from __future__ import annotations

import os

from septmuse.core.logging import get_logger
from septmuse.embedders.base import Embedder
from septmuse.embedders.langdetect import detect_language

logger = get_logger(__name__)

ZH_MODEL = "Xenova/paraphrase-multilingual-MiniLM-L12-v2"
EN_MODEL = "Xenova/all-MiniLM-L6-v2"


class AutoOnnxEmbedder(Embedder):
    """语言检测自动选模型 (init 时一次, 不 per-query 切换)。"""

    def __init__(self, *, sample_text: str | None = None, lang: str | None = None) -> None:
        # 1. 显式参数 > 2. 环境变量 > 3. 样本文本检测 > 4. 默认 'zh'
        if lang is not None:
            self._lang = lang
        elif env_lang := os.getenv("SEPTMUSE_LANG"):
            self._lang = env_lang.lower()
        elif sample_text is not None:
            self._lang = detect_language(sample_text)
        else:
            self._lang = "zh"  # 中文优先项目默认

        # 选模型
        model_name = ZH_MODEL if self._lang == "zh" else EN_MODEL
        logger.info("auto_embedder_selecting", lang=self._lang, model=model_name)

        # 委托给 OnnxEmbedder (同一模型, 整个 session 不切换)
        from septmuse.embedders.onnx import OnnxEmbedder

        self._inner = OnnxEmbedder(model_name=model_name)

    @property
    def dimension(self) -> int:
        return self._inner.dimension

    def embed(self, text: str) -> list[float]:
        return self._inner.embed(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return self._inner.embed_batch(texts)

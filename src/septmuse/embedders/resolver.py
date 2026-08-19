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
"""Embedder 解析。"""

from __future__ import annotations

from septmuse.configs.defaults import MemoryConfig
from septmuse.core.logging import get_logger
from septmuse.embedders.base import Embedder

logger = get_logger(__name__)


def resolve_embedder(config: MemoryConfig) -> Embedder:
    """从配置解析并创建 Embedder 实例。"""
    from septmuse.services.providers import embedder_provider

    backend = config.embedder.backend.lower()
    if backend in ("sentence-transformers", "sentence_transformers"):
        backend = "st"
    if backend == "onnx-zh":
        return embedder_provider.resolve(
            backend, config=config.embedder, model_name="Xenova/paraphrase-multilingual-MiniLM-L12-v2"
        )
    if backend == "bge-zh":
        try:
            from septmuse.embedders.onnx import BGE_ZH_MODEL, OnnxEmbedder

            return OnnxEmbedder(model_name=BGE_ZH_MODEL, max_length=512)
        except ImportError:
            logger.warning("bge_zh_fallback_to_hash", reason="onnxruntime/tokenizers not installed, run: pip install septmuse[onnx]")
            from septmuse.embedders.hash import HashEmbedder

            return HashEmbedder()
    return embedder_provider.resolve(backend, config=config.embedder)

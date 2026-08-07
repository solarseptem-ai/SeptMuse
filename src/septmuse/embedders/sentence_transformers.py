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
"""sentence-transformers 嵌入实现 — 可选 embedder (需 torch, 启动慢 ~30s)。

零 API key, 本地模型。首次使用从 HuggingFace 下载模型 (~80MB all-MiniLM-L6-v2)。
推荐改用 OnnxEmbedder (SEPTMUSE_EMBEDDER=onnx): 无 torch, ModelScope 下载, CPU 更快。
"""

from __future__ import annotations

from septmuse.core.logging import get_logger
from septmuse.embedders.base import Embedder

logger = get_logger(__name__)

DEFAULT_MODEL = "all-MiniLM-L6-v2"


class SentenceTransformerEmbedder(Embedder):
    """基于 sentence-transformers 的本地嵌入。"""

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError("sentence-transformers 未安装。请运行: pip install septmuse (默认含此依赖)。") from e

        self.backend_name = "st"
        logger.info("embedder_loading", model=model_name)
        self._model = SentenceTransformer(model_name)
        dim = self._model.get_sentence_embedding_dimension()
        assert dim is not None, "sentence-transformers 未返回嵌入维度"
        self._dim: int = dim
        self._model_name = model_name
        logger.info("embedder_ready", model=model_name, dim=self._dim)

    @property
    def dimension(self) -> int:
        return self._dim

    def _embed(self, text: str, memory_action: str | None = None) -> list[float]:
        # normalize_embeddings=True 使向量归一化, 余弦相似 = 点积
        vec = self._model.encode(text, normalize_embeddings=True)
        return vec.tolist()

    def _embed_batch(self, texts: list[str], memory_action: str | None = None) -> list[list[float]]:
        vecs = self._model.encode(texts, normalize_embeddings=True)
        return [v.tolist() for v in vecs]

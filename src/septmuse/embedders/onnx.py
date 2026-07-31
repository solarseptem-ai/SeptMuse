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
"""ONNX 嵌入实现 — onnxruntime + tokenizers, 无 torch 依赖。

模型: ModelScope Xenova/all-MiniLM-L6-v2 (量化 ONNX, ~23MB, 384 dim)。
首次使用自动从 ModelScope 下载到 ~/.septmuse/models/, 后续直接从缓存加载。

优势 (vs sentence-transformers):
- 无 torch 依赖 (~2GB -> 0)
- 启动快 (<2s vs ~30s)
- CPU 推理快 (<50ms/query)
- Windows 稳定 (无模型缓存不完整问题)
- ModelScope 国内下载快 (vs HuggingFace)
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from septmuse.core.logging import get_logger
from septmuse.embedders.base import Embedder

logger = get_logger(__name__)

DEFAULT_EN_MODEL = "Xenova/all-MiniLM-L6-v2"

_MODEL_FILES = {
    "onnx/model_quantized.onnx",
    "tokenizer.json",
}


def _model_cache_dir(model_name: str) -> Path:
    """模型缓存目录: ~/.septmuse/models/<sanitized_model_name>/"""
    safe = model_name.replace("/", "__")
    base = os.getenv("SEPTMUSE_MODEL_CACHE", str(Path.home() / ".septmuse" / "models"))
    return Path(base) / safe


def _ensure_model_files(model_name: str) -> Path:
    """确保模型文件已下载到本地缓存, 返回缓存目录。"""
    cache_dir = _model_cache_dir(model_name)
    cache_dir.mkdir(parents=True, exist_ok=True)

    missing = [f for f in _MODEL_FILES if not (cache_dir / f).exists()]
    if not missing:
        return cache_dir

    try:
        from modelscope import snapshot_download
    except ImportError as e:
        raise ImportError("modelscope 未安装。请运行: pip install septmuse[onnx]") from e

    logger.info("onnx_model_downloading", model=model_name, files=missing, source="modelscope")
    snapshot_download(
        model_id=model_name,
        local_dir=str(cache_dir),
        allow_patterns=list(missing),
    )
    logger.info("onnx_model_files_cached", model=model_name, cache=str(cache_dir))

    return cache_dir


class OnnxEmbedder(Embedder):
    """基于 ONNX Runtime 的嵌入模型 (无 torch 依赖)。

    默认英文模型: Xenova/all-MiniLM-L6-v2 (384 dim, ~23MB 量化)。
    多语言模型: Xenova/paraphrase-multilingual-MiniLM-L12-v2 (384 dim, ~50MB 量化)。
    模型从 ModelScope 下载 (国内 CDN, 无需 HuggingFace)。
    """

    def __init__(self, model_name: str = DEFAULT_EN_MODEL) -> None:
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as e:
            raise ImportError("onnxruntime + tokenizers 未安装。请运行: pip install septmuse[onnx]") from e

        self._model_name = model_name
        cache_dir = _ensure_model_files(model_name)

        onnx_path = cache_dir / "onnx" / "model_quantized.onnx"
        tokenizer_path = cache_dir / "tokenizer.json"

        logger.info("onnx_embedder_loading", model=model_name, cache=str(cache_dir))
        self._session = ort.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"],
        )
        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self._tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        self._tokenizer.enable_truncation(max_length=256)

        # 从 ONNX 模型输出推断维度
        output_info = self._session.get_outputs()[0]
        self._dim: int = output_info.shape[-1]
        if not isinstance(self._dim, int) or self._dim <= 0:
            self._dim = 384  # 安全回退

        # 记录模型期望的输入名 (可能含/不含 token_type_ids)
        self._input_names = [inp.name for inp in self._session.get_inputs()]

        logger.info("onnx_embedder_ready", model=model_name, dim=self._dim, inputs=self._input_names)

    @property
    def dimension(self) -> int:
        return self._dim

    def embed(self, text: str) -> list[float]:
        encoding = self._tokenizer.encode(text)
        feeds = self._build_feeds(encoding)

        outputs = self._session.run(None, feeds)
        last_hidden = outputs[0]  # [1, seq_len, dim]

        # Mean pooling (考虑 attention_mask)
        mask = np.array(encoding.attention_mask, dtype=np.float32)
        pooled = (last_hidden[0] * mask[:, None]).sum(axis=0) / mask.sum()

        # L2 归一化 (余弦相似 = 点积)
        norm = float(np.linalg.norm(pooled))
        if norm > 0:
            pooled = pooled / norm
        return pooled.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]

    def _build_feeds(self, encoding) -> dict[str, np.ndarray]:
        """构建 ONNX 推理输入 (仅传模型期望的输入)。"""
        feeds: dict[str, np.ndarray] = {
            "input_ids": np.array([encoding.ids], dtype=np.int64),
            "attention_mask": np.array([encoding.attention_mask], dtype=np.int64),
        }
        if "token_type_ids" in self._input_names:
            feeds["token_type_ids"] = np.array([encoding.type_ids], dtype=np.int64)
        return feeds

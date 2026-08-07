#  Copyright 2026 The sonhhxg0529 Authors. All Rights Reserved.
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""ONNX cross-encoder reranker — bge-reranker-v2-m3 量化版。

模型: Xenova/bge-reranker-v2-m3 (ONNX 量化, ~50MB, ModelScope 下载)。
推理: (query, document) pair -> logit -> sigmoid 归一化到 [0,1]。
降级: onnxruntime/tokenizers 未安装或模型加载失败时降级为透传 + warning。
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from septmuse.core.logging import get_logger
from septmuse.rerankers.base import BaseReranker

logger = get_logger(__name__)

_DEFAULT_RERANKER_MODEL = "Xenova/bge-reranker-v2-m3"
_RERANKER_MODEL_FILES = {"onnx/model_quantized.onnx", "tokenizer.json"}


def _reranker_cache_dir(model_name: str) -> Path:
    safe = model_name.replace("/", "__")
    base = os.getenv("SEPTMUSE_MODEL_CACHE", str(Path.home() / ".septmuse" / "models"))
    return Path(base) / safe


def _ensure_reranker_model(model_name: str) -> Path:
    cache_dir = _reranker_cache_dir(model_name)
    cache_dir.mkdir(parents=True, exist_ok=True)
    missing = [f for f in _RERANKER_MODEL_FILES if not (cache_dir / f).exists()]
    if not missing:
        return cache_dir
    try:
        from modelscope import snapshot_download
    except ImportError as e:
        raise ImportError("modelscope 未安装。请运行: pip install septmuse[reranker]") from e
    logger.info("reranker_model_downloading", model=model_name, files=missing, source="modelscope")
    snapshot_download(model_id=model_name, local_dir=str(cache_dir), allow_patterns=list(missing))
    logger.info("reranker_model_cached", model=model_name, cache=str(cache_dir))
    return cache_dir


class CrossEncoderReranker(BaseReranker):
    """ONNX cross-encoder reranker。"""

    def __init__(
        self,
        model_name: str = _DEFAULT_RERANKER_MODEL,
        model_cache_dir: str | None = None,
        batch_size: int = 32,
        max_length: int = 512,
        normalize: bool = True,
        device: str | None = None,
        **kwargs,
    ) -> None:
        self._model_name = model_name
        self._custom_cache_dir = model_cache_dir
        self._batch_size = batch_size
        self._max_length = max_length
        self._normalize = normalize
        self._device = device  # None=自动检测, "cuda"/"cpu" 强制
        self._session = None
        self._tokenizer = None
        self._input_names: list[str] = []
        self._degraded = False
        self._init_attempted = False

    def _init_model(self) -> None:
        if self._init_attempted:
            return
        self._init_attempted = True
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError:
            logger.warning("cross_encoder_reranker_degraded", reason="onnxruntime/tokenizers not installed")
            self._degraded = True
            return
        try:
            cache_dir = Path(self._custom_cache_dir) if self._custom_cache_dir else _ensure_reranker_model(self._model_name)
            onnx_path = cache_dir / "onnx" / "model_quantized.onnx"
            tokenizer_path = cache_dir / "tokenizer.json"
            if not onnx_path.exists() or not tokenizer_path.exists():
                logger.warning("cross_encoder_reranker_degraded", reason="model files not found")
                self._degraded = True
                return
            logger.info("cross_encoder_loading", model=self._model_name, cache=str(cache_dir))
            # device: None=自动检测 cuda 可用性, "cuda"/"cpu" 强制指定
            if self._device is None:
                available = ort.get_available_providers()
                self._device = "cuda" if "CUDAExecutionProvider" in available else "cpu"
            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if self._device == "cuda"
                else ["CPUExecutionProvider"]
            )
            self._session = ort.InferenceSession(str(onnx_path), providers=providers)
            self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
            self._tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
            self._tokenizer.enable_truncation(max_length=self._max_length)
            self._input_names = [inp.name for inp in self._session.get_inputs()]
            logger.info("cross_encoder_ready", model=self._model_name, inputs=self._input_names)
        except Exception as e:
            logger.warning("cross_encoder_reranker_degraded", reason=str(e))
            self._degraded = True

    def _build_batch_feeds(self, encodings) -> dict[str, np.ndarray]:
        """构建批量 ONNX 推理输入 [batch, seq_len]。"""
        feeds: dict[str, np.ndarray] = {
            "input_ids": np.array([enc.ids for enc in encodings], dtype=np.int64),
            "attention_mask": np.array([enc.attention_mask for enc in encodings], dtype=np.int64),
        }
        if "token_type_ids" in self._input_names:
            feeds["token_type_ids"] = np.array([enc.type_ids for enc in encodings], dtype=np.int64)
        return feeds

    def rerank(self, query: str, documents: list[str], *, top_k: int | None = None) -> list[tuple[int, float]]:
        if not documents:
            return []
        self._init_model()
        if self._degraded or self._session is None:
            limit = top_k or len(documents)
            return [(i, 0.5) for i in range(min(limit, len(documents)))]

        scored: list[tuple[int, float]] = []
        for start in range(0, len(documents), self._batch_size):
            batch_docs = documents[start : start + self._batch_size]
            try:
                # 批量编码 (query, doc) pairs + 单次 ONNX forward
                encodings = [self._tokenizer.encode(query, doc) for doc in batch_docs]
                feeds = self._build_batch_feeds(encodings)
                outputs = self._session.run(None, feeds)
                logits = outputs[0]
                if logits.ndim > 1:
                    logits = logits.squeeze(-1)  # [batch, 1] → [batch]
                # normalize=True: sigmoid 归一化到 [0,1]; False: 原始 logit
                scores = 1.0 / (1.0 + np.exp(-logits)) if self._normalize else logits
                for i, score in enumerate(scores):
                    scored.append((start + i, float(score)))
            except Exception as e:
                logger.warning("cross_encoder_batch_failed", batch_start=start, error=str(e))
                for i in range(len(batch_docs)):
                    scored.append((start + i, 0.5))

        scored.sort(key=lambda x: x[1], reverse=True)
        if top_k:
            scored = scored[:top_k]
        return scored

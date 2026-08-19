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
"""确定性哈希嵌入 — 仅测试/离线回退用。"""

from __future__ import annotations

import hashlib
import re

import numpy as np

from septmuse.embedders.base import Embedder


def _tokenize(text: str) -> list[str]:
    """简单分词: 英文按词、中文按字。"""
    return re.findall(r"[a-z0-9]+|[^\s\W]", text.lower())


class HashEmbedder(Embedder):
    """词级 hash 确定性嵌入 (测试/离线用)。"""

    def __init__(self, dim: int = 128) -> None:
        self._dim = dim
        self.backend_name = "hash"

    @property
    def dimension(self) -> int:
        return self._dim

    def _embed(self, text: str, memory_action: str | None = None) -> list[float]:
        vec = np.zeros(self._dim, dtype=np.float32)
        for token in _tokenize(text):
            h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
            vec[h % self._dim] += 1.0
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def _embed_batch(self, texts: list[str], memory_action: str | None = None) -> list[list[float]]:
        return [self.embed(t, memory_action) for t in texts]

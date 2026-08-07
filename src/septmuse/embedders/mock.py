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
"""Mock 嵌入 — 固定向量, 确定性测试用。

区别于 HashEmbedder (哈希向量): MockEmbedder 返回固定向量, 不依赖输入文本,
适合验证流程正确性而非嵌入质量。
"""

from __future__ import annotations

from septmuse.embedders.base import Embedder

FIXED_VECTOR = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


class MockEmbedder(Embedder):
    """固定 10 维向量嵌入 (测试用)。"""

    backend_name = "mock"

    @property
    def dimension(self) -> int:
        return 10

    def _embed(self, text: str, memory_action: str | None = None) -> list[float]:
        return list(FIXED_VECTOR)

    def _embed_batch(self, texts: list[str], memory_action: str | None = None) -> list[list[float]]:
        return [list(FIXED_VECTOR) for _ in texts]

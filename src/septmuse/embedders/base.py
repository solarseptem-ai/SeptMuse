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
"""嵌入模型抽象基类 (借鉴 mem0 embeddings/base.py EmbedderBase 模式)。

所有 embedder 实现此接口, 用于把文本转为向量供相似检索。
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Embedder(ABC):
    """嵌入模型抽象。"""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """返回嵌入维度。"""
        ...

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """嵌入单条文本, 返回归一化向量 (便于余弦点积)。"""
        ...

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """批量嵌入。"""
        ...

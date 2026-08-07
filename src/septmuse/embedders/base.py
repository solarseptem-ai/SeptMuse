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
"""嵌入模型抽象基类。

所有 embedder 实现此接口, 用于把文本转为向量供相似检索。
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod

from septmuse.observability.collector import MetricsCollector


class Embedder(ABC):
    """嵌入模型抽象。"""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """返回嵌入维度。"""
        ...

    @abstractmethod
    def _embed(self, text: str, memory_action: str | None = None) -> list[float]:
        """嵌入单条文本, 返回归一化向量。子类实现。"""
        ...

    def embed(self, text: str, memory_action: str | None = None) -> list[float]:
        """嵌入单条文本（带可观测性埋点）。"""
        collector = MetricsCollector.get()
        start = time.perf_counter()
        try:
            return self._embed(text, memory_action)
        finally:
            collector.observe(
                "embed_duration_seconds",
                time.perf_counter() - start,
                labels={"backend": self._backend_name()},
            )

    def _embed_batch(self, texts: list[str], memory_action: str | None = None) -> list[list[float]]:
        """批量嵌入 — 默认逐条调用 _embed(), 子类可 override 实现真批量推理。"""
        return [self._embed(t, memory_action) for t in texts]

    def embed_batch(self, texts: list[str], memory_action: str | None = None) -> list[list[float]]:
        """批量嵌入（带可观测性埋点）。"""
        collector = MetricsCollector.get()
        start = time.perf_counter()
        try:
            return self._embed_batch(texts, memory_action)
        finally:
            collector.observe(
                "embed_batch_duration_seconds",
                time.perf_counter() - start,
                labels={"backend": self._backend_name()},
            )

    def _backend_name(self) -> str:
        """返回后端名称（用于指标标签）。子类应设置 self.backend_name。"""
        return getattr(self, "backend_name", type(self).__name__.lower())

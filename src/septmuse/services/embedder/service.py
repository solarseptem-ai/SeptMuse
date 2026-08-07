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
"""EmbedderService — 嵌入服务实现。

包装 Embedder ABC, 提供:
- 嵌入模型生命周期管理 (延迟初始化 + 缓存)
- 从环境变量 / config 解析 embedder 后端
- 统一 embed / embed_batch 接口
- 统计信息 (请求次数 / 缓存命中)
"""

from __future__ import annotations

from typing import Any

from septmuse.configs.base import MemoryConfig
from septmuse.configs.defaults import default_config
from septmuse.core.logging import get_logger
from septmuse.embedders.base import Embedder
from septmuse.embedders.resolver import resolve_embedder
from septmuse.services.base import Service

logger = get_logger(__name__)


class EmbedderService(Service):
    """嵌入服务 — 包装 Embedder (借鉴 mem0 EmbedderBase + Langflow 服务模式)。

    职责:
    - 延迟创建 Embedder 实例 (首次 embed 调用时才加载模型)
    - 支持 6 种后端: hash / st / onnx / onnx-zh / auto / openai
    - 支持运行时切换后端 (reconfigure)
    - 统计嵌入请求数和模型信息

    用法:
        svc = EmbedderService()
        vec = svc.embed("hello")
        vecs = svc.embed_batch(["hello", "world"])
    """

    name = "embedder_service"

    def __init__(
        self,
        config: MemoryConfig | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self._config = config or default_config()
        self._embedder = embedder
        self._embed_count: int = 0
        self._batch_count: int = 0
        self.set_ready()
        logger.info(
            "embedder_service_initialized",
            backend=self._config.embedder_backend,
            model=self._config.embedder_model,
        )

    @property
    def embedder(self) -> Embedder:
        """获取 Embedder 实例 (延迟初始化)。"""
        if self._embedder is None:
            self._embedder = self._resolve_embedder()
        return self._embedder

    @property
    def dimension(self) -> int:
        """嵌入向量维度。"""
        return self.embedder.dimension

    @property
    def backend(self) -> str:
        """当前 embedder 后端名称。"""
        return self._config.embedder_backend

    @property
    def model(self) -> str | None:
        """当前 embedder 模型名。"""
        return self._config.embedder_model

    @property
    def stats(self) -> dict[str, Any]:
        """嵌入统计信息。"""
        return {
            "embed_count": self._embed_count,
            "batch_count": self._batch_count,
            "total_requests": self._embed_count + self._batch_count,
            "backend": self.backend,
            "model": self.model,
            "dimension": self.embedder.dimension,
        }

    def embed(self, text: str) -> list[float]:
        """嵌入单条文本, 返回归一化向量。"""
        self._embed_count += 1
        return self.embedder.embed(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """批量嵌入多条文本。"""
        self._batch_count += 1
        return self.embedder.embed_batch(texts)

    def reconfigure(self, config: MemoryConfig | None = None) -> None:
        """重新配置 embedder (如切换后端), 丢弃旧实例下次懒创建。"""
        self._config = config or default_config()
        self._embedder = None
        logger.info("embedder_service_reconfigured", backend=self._config.embedder_backend)

    def _resolve_embedder(self) -> Embedder:
        """从配置解析并创建 Embedder 实例 (集中实现, 消除三处重复)。"""
        return resolve_embedder(self._config)

    async def teardown(self) -> None:
        """释放嵌入模型资源。"""
        logger.info("embedder_service_teardown", stats=self.stats)
        self._embedder = None
        return await super().teardown()

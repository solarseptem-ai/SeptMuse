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
"""LLMService — LLM 服务实现。

包装 LLM ABC, 提供:
- LLM provider 生命周期管理 (延迟初始化 + 缓存)
- 从环境变量 / config 解析 LLM provider
- 统一 complete 接口
- 可用性检查 + 统计信息
- 降级策略 (verbatim 模式时 LLM 不可用)
"""

from __future__ import annotations

from typing import Any

from septmuse.configs.base import MemoryConfig
from septmuse.configs.defaults import default_config
from septmuse.core.logging import get_logger
from septmuse.llms.base import LLM
from septmuse.services.base import Service

logger = get_logger(__name__)


class LLMService(Service):
    """LLM 服务 — 包装 LLM provider (借鉴 Langflow 服务模式)。

    职责:
    - 延迟创建 LLM 实例 (首次 complete 调用时才连接)
    - 支持 5 种 provider: openai / ollama / anthropic / dashscope
    - 支持 verbatim 模式 (LLM 不可用时降级)
    - 统计请求数和错误信息

    用法:
        svc = LLMService()
        if svc.is_available:
            result = svc.complete("你是一个助手", "你好")
        else:
            print("LLM 未配置, 使用 verbatim 模式")
    """

    name = "llm_service"

    def __init__(
        self,
        config: MemoryConfig | None = None,
        llm: LLM | None = None,
    ) -> None:
        self._config = config or default_config()
        self._llm = llm
        self._request_count: int = 0
        self._error_count: int = 0
        self._last_error: str | None = None
        self.set_ready()
        logger.info(
            "llm_service_initialized",
            provider=self._config.llm_provider,
            model=self._config.llm_model,
            available=self.is_available,
        )

    @property
    def llm(self) -> LLM | None:
        """获取 LLM 实例 (延迟初始化, None=verbatim 模式)。"""
        if self._llm is None and self._config.llm_provider:
            self._llm = self._resolve_llm()
        return self._llm

    @property
    def is_available(self) -> bool:
        """LLM 是否可用 (provider 已配置且非 verbatim 模式)。"""
        return self._config.llm_provider is not None

    @property
    def provider(self) -> str | None:
        """当前 LLM provider 名称。"""
        return self._config.llm_provider

    @property
    def model(self) -> str | None:
        """当前 LLM 模型名。"""
        return self._config.llm_model

    @property
    def stats(self) -> dict[str, Any]:
        """LLM 统计信息。"""
        return {
            "request_count": self._request_count,
            "error_count": self._error_count,
            "last_error": self._last_error,
            "provider": self.provider,
            "model": self.model,
            "available": self.is_available,
        }

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """同步补全, 返回 LLM 输出文本。

        Raises:
            RuntimeError: LLM 未配置时调用。
        """
        if self.llm is None:
            raise RuntimeError(
                "LLM not configured. Set SEPTMUSE_LLM env var (openai/ollama/anthropic/dashscope) "
                "or use verbatim mode (infer=False)."
            )
        self._request_count += 1
        try:
            return self.llm.complete(system_prompt, user_prompt)
        except Exception as e:
            self._error_count += 1
            self._last_error = str(e)
            logger.error("llm_complete_failed", error=str(e), provider=self.provider)
            raise

    def reconfigure(self, config: MemoryConfig | None = None) -> None:
        """重新配置 LLM (如切换 provider), 丢弃旧实例下次懒创建。"""
        self._config = config or default_config()
        self._llm = None
        self._last_error = None
        logger.info("llm_service_reconfigured", provider=self._config.llm_provider)

    def _resolve_llm(self) -> LLM | None:
        """从配置解析并创建 LLM 实例 (通过 ServiceProvider, 对齐 llms/__init__.py)。"""
        from septmuse.services.providers import llm_provider

        provider = self._config.llm_provider
        if provider is None:
            return None
        return llm_provider.resolve(provider, config=self._config.llm)

    async def teardown(self) -> None:
        """释放 LLM 资源。"""
        logger.info("llm_service_teardown", stats=self.stats)
        self._llm = None
        return await super().teardown()

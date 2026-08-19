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
"""LLMService — LLM 服务实现。"""

from __future__ import annotations

import time
from typing import Any

from septmuse.configs.base import MemoryConfig
from septmuse.configs.defaults import default_config
from septmuse.core.logging import get_logger
from septmuse.llms.base import LLM
from septmuse.services.base import Service

logger = get_logger(__name__)


class LLMService(Service):
    """LLM 服务 — 包装 LLM provider，带统一重试与熔断。"""

    name = "llm_service"

    # 熔断阈值: 连续失败次数 / 时间窗口(秒) / 冷却时间(秒)
    CIRCUIT_BREAKER_THRESHOLD = 5
    CIRCUIT_BREAKER_WINDOW_SECONDS = 60
    CIRCUIT_BREAKER_COOLDOWN_SECONDS = 30

    # 重试配置
    MAX_RETRIES = 3
    RETRY_BASE_DELAY = 2.0

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
        self._consecutive_errors: int = 0
        self._last_error_time: float | None = None
        self._circuit_open: bool = False
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
            "consecutive_errors": self._consecutive_errors,
            "circuit_open": self._circuit_open,
            "last_error": self._last_error,
            "provider": self.provider,
            "model": self.model,
            "available": self.is_available,
        }

    def _is_circuit_open(self) -> bool:
        """检查熔断器是否处于打开状态，支持自动半开（冷却期后重置）。"""
        if not self._circuit_open:
            return False
        if self._last_error_time is None:
            self._circuit_open = False
            self._consecutive_errors = 0
            return False
        if (time.time() - self._last_error_time) > self.CIRCUIT_BREAKER_COOLDOWN_SECONDS:
            logger.info("llm_circuit_breaker_half_open")
            self._circuit_open = False
            self._consecutive_errors = 0
            return False
        return True

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """同步补全，返回 LLM 输出文本。带指数退避重试与熔断保护。"""
        if self._is_circuit_open():
            raise RuntimeError(
                f"LLM circuit breaker open after {self.CIRCUIT_BREAKER_THRESHOLD} consecutive errors. "
                f"Cooldown: {self.CIRCUIT_BREAKER_COOLDOWN_SECONDS}s."
            )
        if self.llm is None:
            raise RuntimeError(
                "LLM not configured. Set SEPTMUSE_LLM env var (openai/ollama/anthropic/dashscope) "
                "or use verbatim mode (infer=False)."
            )

        self._request_count += 1
        last_exception: Exception | None = None

        for attempt in range(self.MAX_RETRIES):
            try:
                result = self.llm.complete(system_prompt, user_prompt)
                self._consecutive_errors = 0
                return result
            except Exception as e:
                last_exception = e
                if attempt < self.MAX_RETRIES - 1:
                    delay = self.RETRY_BASE_DELAY * (2 ** attempt)
                    logger.warning(
                        "llm_retry",
                        attempt=attempt + 1,
                        max_retries=self.MAX_RETRIES,
                        delay=delay,
                        error=str(e)[:200],
                        provider=self.provider,
                    )
                    time.sleep(delay)

        # 全部重试失败
        self._error_count += 1
        self._consecutive_errors += 1
        self._last_error_time = time.time()
        self._last_error = str(last_exception) if last_exception else None

        if self._consecutive_errors >= self.CIRCUIT_BREAKER_THRESHOLD:
            self._circuit_open = True
            logger.error(
                "llm_circuit_breaker_opened",
                consecutive_errors=self._consecutive_errors,
                threshold=self.CIRCUIT_BREAKER_THRESHOLD,
                provider=self.provider,
            )
        else:
            logger.error(
                "llm_complete_failed",
                error=str(last_exception)[:200],
                provider=self.provider,
                consecutive_errors=self._consecutive_errors,
            )

        raise last_exception or RuntimeError("LLM complete failed after retries")

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

    def healthcheck(self) -> bool:
        """轻量级健康检查，优先使用 provider 原生接口避免消耗模型 token。"""
        if self.llm is None:
            return False
        # 优先尝试 provider 原生轻量检查 (models.list / ping)
        if hasattr(self.llm, "_client"):
            client = self.llm._client
            if hasattr(client, "models") and hasattr(client.models, "list"):
                try:
                    client.models.list()
                    return True
                except Exception:
                    return False
        # Fallback: 发送一个极简 prompt，期望极短响应
        try:
            self.llm.complete("Reply 'ok' only.", "ok")
            return True
        except Exception:
            return False

    async def teardown(self) -> None:
        """释放 LLM 资源。"""
        logger.info("llm_service_teardown", stats=self.stats)
        self._llm = None
        return await super().teardown()

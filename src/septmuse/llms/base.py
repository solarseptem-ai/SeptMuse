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
"""LLM 抽象基类。

所有 LLM 实现此接口, 用于记忆抽取 (infer=True)。
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod

from septmuse.observability.collector import MetricsCollector


class LLM(ABC):
    """LLM 抽象。"""

    @abstractmethod
    def _complete(self, system_prompt: str, user_prompt: str) -> str:
        """同步补全, 返回 LLM 输出文本。子类实现。"""
        ...

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """调用 LLM（带可观测性埋点）。"""
        collector = MetricsCollector.get()
        start = time.perf_counter()
        try:
            result = self._complete(system_prompt, user_prompt)
            collector.inc(
                "llm_calls_total",
                labels={"provider": self._provider_name(), "status": "success"},
            )
            return result
        except Exception:
            collector.inc(
                "llm_calls_total",
                labels={"provider": self._provider_name(), "status": "error"},
            )
            raise
        finally:
            collector.observe(
                "llm_call_duration_seconds",
                time.perf_counter() - start,
                labels={"provider": self._provider_name()},
            )

    def _provider_name(self) -> str:
        """返回 provider 名称（用于指标标签）。"""
        return getattr(self, "provider_name", type(self).__name__.lower())

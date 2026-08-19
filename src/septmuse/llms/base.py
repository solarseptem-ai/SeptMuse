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

import json
import re
import time
from abc import ABC, abstractmethod
from typing import Any

from septmuse.observability.collector import MetricsCollector


def _robust_json_parse(raw: str) -> dict[str, Any]:
    """从 LLM 输出中鲁棒地提取 JSON 对象。

    尝试顺序: 直接解析 → markdown code block → 最外层花括号 → 自动补全缺失的闭合符号。
    """
    raw = raw.strip()

    # 1. 直接解析
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # 2. markdown code block
    code_block = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
    if code_block:
        try:
            return json.loads(code_block.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 3. 最外层 { ... }
    brace_match = re.search(r"(\{.*\})", raw, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 4. 尝试修复常见截断: 补充缺失的 } 或 ]
    for suffix in ["}", "]}", "]]", "}}", "}}]"]:
        try:
            return json.loads(raw + suffix)
        except json.JSONDecodeError:
            continue

    raise json.JSONDecodeError(f"Failed to parse JSON from LLM output: {raw[:200]}", raw, 0)


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

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """结构化 JSON 输出。

        子类可 override 以使用 provider 原生 JSON mode（如 OpenAI response_format）。
        默认 fallback 到文本解析 + _robust_json_parse。
        """
        raw = self.complete(system_prompt, user_prompt)
        return _robust_json_parse(raw)

    def _provider_name(self) -> str:
        """返回 provider 名称（用于指标标签）。"""
        return getattr(self, "provider_name", type(self).__name__.lower())

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
"""LLM 抽象基类 (借鉴 mem0 llms/base.py 模式)。

所有 LLM 实现此接口, 用于记忆抽取 (infer=True)。
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class LLM(ABC):
    """LLM 抽象。"""

    @abstractmethod
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """同步补全, 返回 LLM 输出文本。"""
        ...

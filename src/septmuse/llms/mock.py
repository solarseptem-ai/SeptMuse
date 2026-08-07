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
"""确定性 Mock LLM — 测试用 (零 API key, 零网络)。

基于规则模拟 mem0 FACT_RETRIEVAL_PROMPT 的输出 ({"facts": [...]}):
- name 模式: "my name is X" / "i am X" → "Name is X"
- likes 模式: "i like/love X" → "Likes X"
- is a 模式: "i am a/an X" → "Is a X"

非生产 LLM, 仅验证抽取流水线逻辑。生产用 OpenAI/Anthropic。
"""

from __future__ import annotations

import json
import re

from septmuse.llms.base import LLM


class MockLLM(LLM):
    """测试用确定性 LLM (模拟 mem0 fact 抽取输出)。"""

    provider_name = "mock"

    def _complete(self, system_prompt: str, user_prompt: str) -> str:
        facts: list[str] = []
        text = user_prompt.lower()

        # name 模式 (顺序: name 优先于 is a)
        for pat in [r"(?:my name is|i am|i'm)\s+([a-z]+)"]:
            m = re.search(pat, text)
            if m and not any("name is" in f.lower() for f in facts):
                facts.append(f"Name is {m.group(1).title()}")

        # is a/an 模式
        m = re.search(r"i am (?:a|an)\s+([a-z\s]+?)(?:[,.]|$)", text)
        if m and not any("name is" in f.lower() for f in facts):
            facts.append(f"Is a {m.group(1).strip()}")

        # likes 模式
        m = re.search(r"i (?:like|love|enjoy)\s+([a-z\s]+?)(?:[,.]|$)", text)
        if m:
            facts.append(f"Likes {m.group(1).strip()}")

        # dislikes 模式
        m = re.search(r"i (?:hate|dislike)\s+([a-z\s]+?)(?:[,.]|$)", text)
        if m:
            facts.append(f"Dislikes {m.group(1).strip()}")

        return json.dumps({"facts": facts})

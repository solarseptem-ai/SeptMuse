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
"""消息标准化 — 统一 messages 归一化。"""

from __future__ import annotations

from typing import Any


def normalize_messages(messages: Any) -> list[str]:
    """归一化 messages 为 text 列表 (接受 str 或 List[Dict])。"""
    if isinstance(messages, str):
        return [messages]
    if not isinstance(messages, list):
        return []
    texts: list[str] = []
    for m in messages:
        if isinstance(m, dict):
            content = m.get("content")
            if content:
                texts.append(content)
        elif isinstance(m, str):
            texts.append(m)
    return texts

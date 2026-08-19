"""消息标准化 — 统一 messages 归一化 (消除 3 份拷贝)。

接受 str 或 List[Dict/str], 返回 list[str]。
"""

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

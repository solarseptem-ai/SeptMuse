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
"""写校验 + hash 去重 — 记忆写入前拦截 (架构文档 §5.3)。

写校验:
- entity ID 必须是非空 str, trim, 拒绝内部空格
- 至少一个 session ID (user_id 或 agent_id)
- hash 去重: existing_hashes + seen_hashes 双重去重

SeptMuse 简化:
- SHA-256 hash (比 md5 更安全)
- 内存窗口去重 (DedupWindow, 默认 5min 时间窗)
- ValidationResult 返回 allowed/reason/dedup

详见 docs/specs/agent-memory-architecture.md §5.3 治理。
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

from septmuse.core.logging import get_logger

logger = get_logger(__name__)

# 去重窗口默认 5min
DEFAULT_DEDUP_WINDOW_SECONDS = 300


def compute_hash(text: str) -> str:
    """SHA-256 hash (比 md5 更安全)。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate_entity_id(value: str | int | None, name: str = "user_id") -> str | None:
    """校验 + trim entity ID。

    - None → None (可选字段)
    - 非空 str → trim 后返回
    - 空串/纯空格 → None
    - int → str(int) (兼容)
    - 含内部空格 → raise ValueError
    """
    if value is None:
        return None
    if isinstance(value, int):
        value = str(value)
    if not isinstance(value, str):
        raise ValueError(f"{name} must be str, got {type(value).__name__}")
    trimmed = value.strip()
    if not trimmed:
        return None
    if " " in trimmed:
        raise ValueError(f"{name} must not contain internal spaces: {trimmed!r}")
    return trimmed


@dataclass
class ValidationResult:
    """写校验结果。"""

    allowed: bool
    reason: str = ""
    dedup: bool = False
    text_hash: str = ""


class DedupWindow:
    """时间窗去重 (5min 窗口 + 内存 seen_hashes)。

    用法:
        window = DedupWindow(window_seconds=300)
        if window.is_duplicate(text):
            skip  # 5min 内已写过
        else:
            window.add(text)
            write(text)
    """

    def __init__(self, window_seconds: int = DEFAULT_DEDUP_WINDOW_SECONDS) -> None:
        self.window_seconds = window_seconds
        self._seen: dict[str, float] = {}  # hash → timestamp

    def is_duplicate(self, text: str, *, scope: str | None = None) -> bool:
        """检测文本 hash 是否在时间窗内已见过。

        Args:
            text: 文本
            scope: 作用域 (如 user_id), 相同文本在不同 scope 下不重复 (per-user 去重)
        """
        key = f"{scope}:{text}" if scope else text
        h = compute_hash(key)
        now = time.monotonic()
        # 清理过期
        expired = [k for k, ts in self._seen.items() if now - ts > self.window_seconds]
        for k in expired:
            del self._seen[k]
        return h in self._seen

    def add(self, text: str, *, scope: str | None = None) -> str:
        """记录文本 hash 到时间窗。"""
        key = f"{scope}:{text}" if scope else text
        h = compute_hash(key)
        self._seen[h] = time.monotonic()
        return h

    def clear(self) -> None:
        """清空窗口。"""
        self._seen.clear()


class WriteValidator:
    """写校验器 (参数校验 + hash 去重)。

    用法:
        validator = WriteValidator()
        result = validator.validate(text, user_id="alice")
        if not result.allowed:
            skip  # 校验失败
        if result.dedup:
            skip  # 重复
    """

    def __init__(self, dedup_window: DedupWindow | None = None) -> None:
        self.dedup = dedup_window or DedupWindow()

    def validate(
        self,
        text: str,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
    ) -> ValidationResult:
        """写入前校验 (参数校验 + hash 去重)。

        校验:
        1. text 非空
        2. user_id 或 agent_id 至少一个
        3. entity ID 格式校验
        4. hash 去重 (5min 时间窗, per-user scope)
        """
        # 1. text 非空
        if not text or not text.strip():
            return ValidationResult(allowed=False, reason="text is empty")

        # 2. 至少一个 session ID
        uid = validate_entity_id(user_id, "user_id")
        aid = validate_entity_id(agent_id, "agent_id")
        if uid is None and aid is None:
            return ValidationResult(allowed=False, reason="at least one of user_id/agent_id required")

        # 3. hash 去重 (5min 时间窗, per-user scope)
        scope = uid or aid or "default"
        h = compute_hash(text.strip())
        if self.dedup.is_duplicate(text.strip(), scope=scope):
            return ValidationResult(allowed=False, reason="duplicate (hash seen in window)", dedup=True, text_hash=h)

        # 4. 记录到窗口
        self.dedup.add(text.strip(), scope=scope)
        return ValidationResult(allowed=True, text_hash=h)

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
"""隐私脱敏 — 内容级 PII/secrets 脱敏。"""

from __future__ import annotations

import re

from septmuse.core.logging import get_logger

logger = get_logger(__name__)

# 默认脱敏模式 (对齐 Agent Memory 隐私过滤, 扩展常见密钥前缀)
DEFAULT_PATTERNS: dict[str, tuple[str, str]] = {
    # API keys — 常见 provider 前缀
    "api_key_openai": (r"sk-[A-Za-z0-9]{20,}", "[REDACTED_OPENAI_KEY]"),
    "api_key_aws": (r"AKIA[0-9A-Z]{16}", "[REDACTED_AWS_KEY]"),
    "api_key_github_pat": (r"gh[pousr]_[A-Za-z0-9]{36,}", "[REDACTED_GITHUB_TOKEN]"),
    "api_key_slack": (r"xox[baprs]-[A-Za-z0-9-]{10,}", "[REDACTED_SLACK_TOKEN]"),
    "api_key_generic": (
        r"(?i)(?:api[_-]?key|access[_-]?key|secret[_-]?key)\s*[=:]\s*['\"]?[A-Za-z0-9/+_-]{16,}",
        "[REDACTED_API_KEY]",
    ),
    # Bearer tokens
    "bearer_token": (r"(?i)bearer\s+[A-Za-z0-9._\-+/=]{20,}", "[REDACTED_BEARER_TOKEN]"),
    # 密码赋值
    "password": (r"(?i)(?:password|passwd|pwd)\s*[=:]\s*\S+", "[REDACTED_PASSWORD]"),
    # 通用 token 赋值
    "token_assign": (r"(?i)token\s*[=:]\s*['\"]?[A-Za-z0-9/+_-]{16,}", "[REDACTED_TOKEN]"),
    # 信用卡号
    "credit_card": (r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b", "[REDACTED_CC]"),
    # SSN
    "ssn": (r"\b\d{3}-\d{2}-\d{4}\b", "[REDACTED_SSN]"),
}


class PrivacyFilter:
    """隐私脱敏器。"""

    def __init__(
        self,
        patterns: dict[str, tuple[str, str]] | None = None,
    ) -> None:
        self._patterns = patterns if patterns is not None else DEFAULT_PATTERNS
        self._compiled: list[tuple[re.Pattern[str], str, str]] = []
        for name, (pat, repl) in self._patterns.items():
            self._compiled.append((re.compile(pat), repl, name))

    def redact(self, text: str) -> str:
        """对文本做脱敏，返回替换后的文本。"""
        if not text:
            return text
        result = text
        hits: list[str] = []
        for regex, repl, name in self._compiled:
            new_result, count = regex.subn(repl, result)
            if count > 0:
                hits.append(f"{name}:{count}")
                result = new_result
        if hits:
            logger.debug("privacy_redact", patterns=hits)
        return result

    def has_sensitive(self, text: str) -> bool:
        """检测文本是否包含敏感信息 (不替换, 仅检测)。"""
        if not text:
            return False
        return any(regex.search(text) for regex, _, _ in self._compiled)

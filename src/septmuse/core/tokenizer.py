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
"""统一分词模块 — jieba 中文分词 + 正则降级。

jieba 可用时 (pip install jieba): 中文按词切分
  "我喜欢编程" → ["我", "喜欢", "编程"]          ← 语义完整
  "Alice的工作经历" → ["alice", "的", "工作", "经历"]

jieba 不可用时: 降级到正则按字切分 (行为不变, 中文按单字)
  "我喜欢编程" → ["我", "喜", "欢", "编", "程"]  ← 语义碎片化

SEPTMUSE_TOKENIZER=space 强制正则分词 (禁用 jieba)。
SEPTMUSE_TOKENIZER=jieba 强制 jieba (不可用时降级)。
默认: 自动检测, jieba 可用就用 jieba。
"""

from __future__ import annotations

import os
import re

from septmuse.core.logging import get_logger

logger = get_logger(__name__)

_BACKEND: str | None = None


def _resolve_backend() -> str:
    """解析分词后端 (一次性, 缓存结果)。"""
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND

    pref = os.getenv("SEPTMUSE_TOKENIZER", "auto").lower()

    if pref == "space":
        _BACKEND = "space"
        logger.info("tokenizer_space", reason="SEPTMUSE_TOKENIZER=space")
        return _BACKEND

    if pref in ("jieba", "auto"):
        try:
            import jieba  # noqa: F401

            _BACKEND = "jieba"
            logger.info("tokenizer_jieba", reason="jieba available" if pref == "auto" else "SEPTMUSE_TOKENIZER=jieba")
            return _BACKEND
        except ImportError:
            if pref == "jieba":
                logger.warning("tokenizer_jieba_unavailable_fallback_space", reason="jieba not installed, run: pip install jieba")
            else:
                logger.info("tokenizer_space", reason="jieba not installed, fallback to regex")

    _BACKEND = "space"
    return _BACKEND


def tokenize(text: str) -> list[str]:
    """统一分词 — jieba 可用时按词切分, 否则正则按字切分。

    >>> tokenize("Alice works at Google")
    ['alice', 'works', 'at', 'google']
    >>> tokenize("Alice的工作经历")  # jieba 可用时
    ['alice', '的', '工作', '经历']
    >>> tokenize("Alice的工作经历")  # jieba 不可用时 (按字)
    ['alice', '的', '工', '作', '经', '历']
    """
    backend = _resolve_backend()
    if backend == "jieba":
        import jieba

        return [w for w in jieba.lcut(text.lower()) if w.strip()]
    return [w for w in re.findall(r"[a-z0-9]+|[^\s\W]", text.lower()) if w]

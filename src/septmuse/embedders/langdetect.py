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
"""语言检测 — CJK 字符比例启发式。"""

from __future__ import annotations

CJK_RATIO_THRESHOLD = 0.3


def _is_cjk(char: str) -> bool:
    """判断字符是否为 CJK 汉字 (Unified + Extension A)。"""
    code = ord(char)
    return (
        0x4E00 <= code <= 0x9FFF  # CJK Unified Ideographs
        or 0x3400 <= code <= 0x4DBF  # CJK Extension A
    )


def detect_language(text: str) -> str:
    """检测文本主语言。"""
    if not text:
        return "en"
    total = len(text)
    cjk_count = sum(1 for c in text if _is_cjk(c))
    return "zh" if cjk_count / total > CJK_RATIO_THRESHOLD else "en"

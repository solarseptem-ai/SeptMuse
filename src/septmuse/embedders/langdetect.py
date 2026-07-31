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
"""语言检测 — CJK 字符比例启发式 (纯 Python, <1ms, 无外部依赖)。

检测范围:
- CJK Unified Ideographs (U+4E00-U+9FFF): 常用汉字
- CJK Extension A (U+3400-U+4DBF): 罕用汉字

不检测:
- 平假名 (U+3040-U+309F) / 片假名 (U+30A0-U+30FF): 日文
- 谚文 (U+AC00-U+D7AF): 韩文

策略: CJK 字符占文本总字符数 > 30% → 'zh', 否则 'en'。
默认 'en' (空字符串/纯数字/纯符号)。

注意: 此函数用于 init 时选择嵌入模型, 不用于 per-query 切换
(不同模型投影到不同语义空间, per-query 切换会破坏向量可比性)。
"""

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
    """检测文本主语言: 'zh' (中文) 或 'en' (英文)。

    CJK 字符比例 > 30% → 'zh'; 否则 'en'。
    空字符串 → 'en' (安全默认)。
    """
    if not text:
        return "en"
    total = len(text)
    cjk_count = sum(1 for c in text if _is_cjk(c))
    return "zh" if cjk_count / total > CJK_RATIO_THRESHOLD else "en"

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
"""语言检测单元测试 (CJK 字符比例启发式)。"""

from __future__ import annotations

from septmuse.embedders.langdetect import detect_language


class TestDetectLanguage:
    def test_pure_chinese(self) -> None:
        assert detect_language("我喜欢编程") == "zh"

    def test_pure_english(self) -> None:
        assert detect_language("I love programming") == "en"

    def test_mixed_chinese_dominant(self) -> None:
        assert detect_language("我喜欢用 python 编程，它很强大") == "zh"

    def test_mixed_english_dominant(self) -> None:
        assert detect_language("Python is great, 我用它") == "en"

    def test_empty_string_defaults_en(self) -> None:
        assert detect_language("") == "en"

    def test_numbers_only_defaults_en(self) -> None:
        assert detect_language("12345 67890") == "en"

    def test_japanese_kana_not_counted_as_cjk(self) -> None:
        """平假名/片假名不在 CJK Unified 范围, 不算中文。"""
        assert detect_language("こんにちは") == "en"

    def test_cjk_extension_a_counted(self) -> None:
        """CJK Extension A (U+3400-U+4DBF) 算中文。"""
        assert detect_language("㐀㐁㐂") == "zh"

    def test_threshold_boundary(self) -> None:
        """CJK 比例恰好 30% 判为中文。"""
        # 3 中文字 + 7 英文字 = 30% CJK
        text = "我喜欢它abcdefg"
        assert detect_language(text) == "zh"

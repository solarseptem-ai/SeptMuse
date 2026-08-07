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
"""统一分词模块测试 — jieba 中文分词 + 正则降级。

测试两个后端:
- jieba: "Alice的工作经历" → ["alice", "的", "工作", "经历"] (按词)
- space: "Alice的工作经历" → ["alice", "的", "工", "作", "经", "历"] (按字)
"""

from __future__ import annotations

import pytest

from septmuse.core import tokenizer


@pytest.fixture(autouse=True)
def _reset_backend():
    """每个测试前重置 tokenizer 缓存, 避免环境变量串扰。"""
    tokenizer._BACKEND = None
    yield
    tokenizer._BACKEND = None


class TestTokenizerSpace:
    """正则按字分词 (SEPTMUSE_TOKENIZER=space)。"""

    def test_english_words(self, monkeypatch):
        monkeypatch.setenv("SEPTMUSE_TOKENIZER", "space")
        result = tokenizer.tokenize("hello world")
        assert result == ["hello", "world"]

    def test_chinese_by_char(self, monkeypatch):
        monkeypatch.setenv("SEPTMUSE_TOKENIZER", "space")
        result = tokenizer.tokenize("我喜欢编程")
        assert result == ["我", "喜", "欢", "编", "程"]

    def test_mixed(self, monkeypatch):
        monkeypatch.setenv("SEPTMUSE_TOKENIZER", "space")
        result = tokenizer.tokenize("Alice的工作经历")
        assert result == ["alice", "的", "工", "作", "经", "历"]

    def test_empty(self, monkeypatch):
        monkeypatch.setenv("SEPTMUSE_TOKENIZER", "space")
        assert tokenizer.tokenize("") == []

    def test_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("SEPTMUSE_TOKENIZER", "space")
        result = tokenizer.tokenize("Hello WORLD")
        assert result == ["hello", "world"]


class TestTokenizerJieba:
    """jieba 中文分词 (SEPTMUSE_TOKENIZER=jieba 或 auto)。"""

    def test_english_words(self, monkeypatch):
        monkeypatch.setenv("SEPTMUSE_TOKENIZER", "jieba")
        result = tokenizer.tokenize("hello world")
        assert result == ["hello", "world"]

    def test_chinese_by_word(self, monkeypatch):
        monkeypatch.setenv("SEPTMUSE_TOKENIZER", "jieba")
        result = tokenizer.tokenize("我喜欢编程")
        assert result == ["我", "喜欢", "编程"]

    def test_mixed_by_word(self, monkeypatch):
        monkeypatch.setenv("SEPTMUSE_TOKENIZER", "jieba")
        result = tokenizer.tokenize("Alice的工作经历")
        assert "alice" in result
        assert "工作" in result
        assert "经历" in result
        # jieba 不会把 "工" 和 "作" 拆成单字
        assert "工" not in result

    def test_auto_detects_jieba(self, monkeypatch):
        """auto 模式: jieba 可用时自动使用。"""
        monkeypatch.setenv("SEPTMUSE_TOKENIZER", "auto")
        backend = tokenizer._resolve_backend()
        assert backend == "jieba"

    def test_jieba_unavailable_fallback(self, monkeypatch):
        """jieba 不可用时降级到正则。"""
        import sys

        monkeypatch.setenv("SEPTMUSE_TOKENIZER", "auto")
        monkeypatch.setitem(sys.modules, "jieba", None)
        tokenizer._BACKEND = None
        backend = tokenizer._resolve_backend()
        assert backend == "space"

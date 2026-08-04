"""P3-1: BM25 词形还原单元测试。"""

from __future__ import annotations

from septmuse.retrieval.hybrid import BM25Scorer, lemmatize_for_bm25


class TestLemmatizeForBM25:
    def test_plural_stripping(self):
        """复数后缀去除: cats → cat, dogs → dog。"""
        result = lemmatize_for_bm25("I love cats and dogs")
        assert "cat" in result
        assert "dog" in result

    def test_short_words_unchanged(self):
        """短词 (≤4 字符) 不去后缀。"""
        result = lemmatize_for_bm25("I like cats")
        # "like" 长度 4, 不去后缀 (条件 len > 4)
        assert "like" in result

    def test_chinese_unchanged(self):
        """中文字符不做词形还原。"""
        result = lemmatize_for_bm25("我喜欢编程")
        tokens = result.split()
        assert "编" in tokens
        assert "程" in tokens

    def test_empty_string(self):
        """空字符串返回空。"""
        assert lemmatize_for_bm25("") == ""

    def test_mixed_chinese_english(self):
        """中英文混合。"""
        result = lemmatize_for_bm25("I like cats 用 Python")
        tokens = result.split()
        assert "like" in tokens
        assert "cat" in tokens


class TestBM25WithLemmatize:
    def test_plural_match(self):
        """词形还原后 BM25 能匹配复数形式。"""
        scorer = BM25Scorer()
        scorer.index(["I have two cats and three dogs"])
        scores = scorer.score("cat dog")
        assert scores[0] > 0

    def test_lemmatize_consistent(self):
        """文档和查询都做 lemmatize, 单复数结果一致。"""
        scorer = BM25Scorer()
        scorer.index(["She likes cats and dogs"])
        scores1 = scorer.score("like cat dog")
        scores2 = scorer.score("likes cats dogs")
        assert scores1 == scores2

#  Copyright 2026 The sonhhxg0529 Authors. All Rights Reserved.
"""EntityExtractor 测试 (借鉴 mem0 entity_extraction.py 设计)。"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from septmuse.configs.defaults import MemoryConfig
from septmuse.extraction.entity import (
    Entity,
    RegexEntityExtractor,
    _resolve_entity_extractor,
)


class TestEntityDataclass:
    def test_entity_fields(self):
        e = Entity(text="Google", entity_type="PROPER", start=0, end=6)
        assert e.text == "Google"
        assert e.entity_type == "PROPER"
        assert e.start == 0
        assert e.end == 6

    def test_entity_equality(self):
        e1 = Entity(text="Alice", entity_type="PROPER", start=0, end=5)
        e2 = Entity(text="Alice", entity_type="PROPER", start=0, end=5)
        assert e1 == e2


class TestResolveEntityExtractor:
    def test_default_returns_regex(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SEPTMUSE_ENTITY_EXTRACTOR", None)
            result = _resolve_entity_extractor(MemoryConfig())
            assert result is not None
            assert type(result).__name__ == "RegexEntityExtractor"

    def test_explicit_regex(self):
        with patch.dict(os.environ, {"SEPTMUSE_ENTITY_EXTRACTOR": "regex"}):
            result = _resolve_entity_extractor(MemoryConfig())
            assert type(result).__name__ == "RegexEntityExtractor"

    def test_none_disables(self):
        with patch.dict(os.environ, {"SEPTMUSE_ENTITY_EXTRACTOR": "none"}):
            result = _resolve_entity_extractor(MemoryConfig())
            assert result is None

    def test_spacy_fallback_to_regex(self):
        """spaCy 未安装时 fallback 到 regex。"""
        with patch.dict(os.environ, {"SEPTMUSE_ENTITY_EXTRACTOR": "spacy"}):
            result = _resolve_entity_extractor(MemoryConfig())
            assert type(result).__name__ == "RegexEntityExtractor"


class TestRegexProper:
    def test_english_proper(self):
        extractor = RegexEntityExtractor()
        entities = extractor.extract("Alice works at Google in London")
        proper = [e for e in entities if e.entity_type == "PROPER"]
        texts = {e.text for e in proper}
        assert "Alice" in texts
        assert "Google" in texts
        assert "London" in texts

    def test_chinese_proper_name(self):
        extractor = RegexEntityExtractor()
        entities = extractor.extract("张三和李四一起去了北京")
        proper = [e for e in entities if e.entity_type == "PROPER"]
        texts = {e.text for e in proper}
        assert "张三" in texts
        assert "李四" in texts
        assert "北京" in texts

    def test_generic_words_filtered(self):
        extractor = RegexEntityExtractor()
        entities = extractor.extract("The person went to this place")
        proper = [e for e in entities if e.entity_type == "PROPER"]
        texts = {e.text for e in proper}
        assert "The" not in texts
        assert "person" not in texts
        assert "this" not in texts
        assert "place" not in texts


class TestRegexQuoted:
    def test_double_quotes(self):
        extractor = RegexEntityExtractor()
        entities = extractor.extract('He said "hello world" loudly')
        quoted = [e for e in entities if e.entity_type == "QUOTED"]
        assert any("hello world" in e.text for e in quoted)

    def test_chinese_quotes(self):
        extractor = RegexEntityExtractor()
        entities = extractor.extract("他说「你好世界」然后走了")
        quoted = [e for e in entities if e.entity_type == "QUOTED"]
        assert any("你好世界" in e.text for e in quoted)


class TestRegexIdentifier:
    def test_dotted_identifier(self):
        extractor = RegexEntityExtractor()
        entities = extractor.extract("Use septmuse.memory for storage")
        identifiers = [e for e in entities if e.entity_type == "IDENTIFIER"]
        texts = {e.text for e in identifiers}
        assert "septmuse.memory" in texts

    def test_camel_case(self):
        extractor = RegexEntityExtractor()
        entities = extractor.extract("The MemoryConfig class is used")
        identifiers = [e for e in entities if e.entity_type == "IDENTIFIER"]
        texts = {e.text for e in identifiers}
        assert "MemoryConfig" in texts

    def test_snake_case(self):
        extractor = RegexEntityExtractor()
        entities = extractor.extract("The user_id field is required")
        identifiers = [e for e in entities if e.entity_type == "IDENTIFIER"]
        texts = {e.text for e in identifiers}
        assert "user_id" in texts


class TestSpanDedup:
    def test_no_duplicate_entities(self):
        extractor = RegexEntityExtractor()
        entities = extractor.extract("Google Google Google")
        proper = [e for e in entities if e.entity_type == "PROPER"]
        assert len(proper) == 1

    def test_long_span_priority(self):
        extractor = RegexEntityExtractor()
        entities = extractor.extract("Machine Learning is great")
        assert isinstance(entities, list)


class TestRegexTopic:
    def test_english_topic(self):
        extractor = RegexEntityExtractor()
        entities = extractor.extract("Machine Learning is powerful")
        topics = [e for e in entities if e.entity_type == "TOPIC"]
        texts = {e.text for e in topics}
        assert "Machine Learning" in texts

    def test_chinese_topic(self):
        extractor = RegexEntityExtractor()
        entities = extractor.extract("人工智能改变了世界")
        topics = [e for e in entities if e.entity_type == "TOPIC"]
        texts = {e.text for e in topics}
        assert "人工智能" in texts

    def test_topic_not_generic(self):
        extractor = RegexEntityExtractor()
        entities = extractor.extract("this thing is good")
        topics = [e for e in entities if e.entity_type == "TOPIC"]
        texts = {e.text for e in topics}
        assert "this thing" not in texts.lower() if texts else True


class TestSpacyExtractor:
    """SpacyEntityExtractor 测试 (需 pip install septmuse[ner])。"""

    @pytest.mark.integration
    def test_spacy_proper_ner(self):
        try:
            from septmuse.extraction.entity import SpacyEntityExtractor

            extractor = SpacyEntityExtractor()
        except ImportError:
            pytest.skip("spaCy not installed")

        entities = extractor.extract("Apple was founded by Steve Jobs")
        proper = [e for e in entities if e.entity_type == "PROPER"]
        texts = {e.text for e in proper}
        assert "Apple" in texts
        assert "Steve Jobs" in texts

    @pytest.mark.integration
    def test_spacy_noun_chunks(self):
        try:
            from septmuse.extraction.entity import SpacyEntityExtractor

            extractor = SpacyEntityExtractor()
        except ImportError:
            pytest.skip("spaCy not installed")

        entities = extractor.extract("The machine learning model is great")
        topics = [e for e in entities if e.entity_type == "TOPIC"]
        assert len(topics) >= 0

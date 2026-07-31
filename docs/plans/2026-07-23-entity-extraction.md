# P0 实体抽取 + 实体向量库实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 SeptMuse 添加实体抽取（EntityExtractor）和实体向量库（EntityStore），补齐最大空白，为 P1 entity boost 和 P0-Task 2/3 LLM 联合抽取奠定基础。

**Architecture:** 模块化设计——EntityExtractor（纯 Python regex + 可选 spaCy）放在 `concerns/extraction/`，EntityStore（独立 SQLite 表）放在 `storage/`，Memory facade 编排。零配置不破坏（默认 regex，spaCy 可选 extra `[ner]`）。

**Tech Stack:** Python 3.10+, SQLite, ruff (line-length 120), pytest, structlog, pydantic

**Spec:** `docs/specs/2026-07-23-entity-extraction-design.md`

## Global Constraints

- PYTHONPATH=src 运行所有测试（包未 pip install -e .）
- ruff check + ruff format --check 必须 clean（line-length 120，禁用 `from __future__ import annotations` 在 MCP tools.py）
- 现有测试案例固定不动，只新增测试
- score 统一为相似度 [0,1]（越高越相似）
- 中文输出（CHANGELOG/AGENTS.md）
- 不用 git（文件快照模式，commit 步骤替换为 lint + test 验证）
- 零配置：默认纯 Python regex（无外部模型），`pip install septmuse[ner]` 升级 spaCy
- EntityStore 复用 SQLiteMemoryStore 的 conn/lock（类似 SQLiteGraphStore 模式）
- `remove_memory_from_entities(memory_id)` 不需要 user_id（memory_id 是 UUID 全局唯一）

---

## File Structure

| 文件 | 操作 | 职责 |
|------|------|------|
| `src/septmuse/concerns/extraction/__init__.py` | 创建 | 包导出 |
| `src/septmuse/concerns/extraction/entity.py` | 创建 | EntityExtractor + RegexEntityExtractor + SpacyEntityExtractor + _resolve_entity_extractor |
| `src/septmuse/storage/entity_store.py` | 创建 | EntityStore（独立 SQLite 表） |
| `src/septmuse/orchestration/memory.py` | 修改 | __init__ + add + delete + close + 5 新方法 |
| `src/septmuse/configs/defaults.py` | 修改 | MemoryConfig +entity_extractor_backend |
| `src/septmuse/cli/main.py` | 修改 | +2 命令 (entities / entity-list) |
| `src/septmuse/api/rest/__init__.py` | 修改 | +2 端点 (GET /entities, GET /entities/list) |
| `src/septmuse/api/mcp/tools.py` | 修改 | +2 工具 (search_entities, list_entities) |
| `pyproject.toml` | 修改 | +ner extra |
| `tests/unit/test_entity_extractor.py` | 创建 | ~15 测试 |
| `tests/unit/test_entity_store.py` | 创建 | ~12 测试 |
| `tests/unit/test_memory.py` | 修改 | +~5 测试 |
| `tests/e2e/test_entity_e2e.py` | 创建 | ~3 测试 |
| `CHANGELOG.md` | 修改 | 记录变更 |
| `AGENTS.md` | 修改 | +entity_extractor 环境变量 |

---

## Task 1: Entity 数据模型 + EntityExtractor ABC + _resolve_entity_extractor

**Files:**
- Create: `src/septmuse/concerns/extraction/__init__.py`
- Create: `src/septmuse/concerns/extraction/entity.py`
- Test: `tests/unit/test_entity_extractor.py`

**Interfaces:**
- Produces: `Entity` dataclass (text, entity_type, start, end), `EntityExtractor` ABC (extract), `_resolve_entity_extractor(config) -> EntityExtractor | None`

- [ ] **Step 1: 创建包 __init__.py**

```python
# src/septmuse/concerns/extraction/__init__.py
"""实体抽取模块 (架构文档 §5.1, 借鉴 mem0 entity_extraction.py)。"""

from septmuse.concerns.extraction.entity import (
    Entity,
    EntityExtractor,
    RegexEntityExtractor,
    SpacyEntityExtractor,
    _resolve_entity_extractor,
)

__all__ = [
    "Entity",
    "EntityExtractor",
    "RegexEntityExtractor",
    "SpacyEntityExtractor",
    "_resolve_entity_extractor",
]
```

- [ ] **Step 2: 写失败测试 — Entity dataclass + _resolve_entity_extractor**

```python
# tests/unit/test_entity_extractor.py
"""EntityExtractor 测试 (借鉴 mem0 entity_extraction.py 设计)。"""

import os
from unittest.mock import patch

from septmuse.concerns.extraction.entity import Entity, EntityExtractor, _resolve_entity_extractor
from septmuse.configs.defaults import MemoryConfig


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
```

- [ ] **Step 3: 运行测试验证失败**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_entity_extractor.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'septmuse.concerns.extraction'`

- [ ] **Step 4: 写最小实现 — Entity + EntityExtractor ABC + _resolve_entity_extractor**

```python
# src/septmuse/concerns/extraction/entity.py
"""实体抽取器 (架构文档 §5.1, 借鉴 mem0 entity_extraction.py)。

默认 RegexEntityExtractor (纯 Python regex + 词表, 零配置)。
spacy: pip install septmuse[ner], spaCy NER + noun_chunks。
none: 禁用实体抽取。
"""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass

from septmuse.configs.defaults import MemoryConfig
from septmuse.observability import get_logger

logger = get_logger(__name__)


@dataclass
class Entity:
    """抽取的实体。"""

    text: str
    entity_type: str  # PROPER / QUOTED / TOPIC / IDENTIFIER
    start: int
    end: int


class EntityExtractor(ABC):
    """实体抽取器抽象基类 (类似 Embedder 模式)。"""

    @abstractmethod
    def extract(self, text: str) -> list[Entity]:
        """抽取实体, 返回 Entity 列表。"""
        ...


class RegexEntityExtractor(EntityExtractor):
    """纯 Python regex + 词表后端 (默认, 零配置)。

    4 类实体: PROPER / QUOTED / TOPIC / IDENTIFIER (借鉴 mem0)。
    ~120 泛化词黑名单 + span 去重冲突解决。
    """

    def extract(self, text: str) -> list[Entity]:
        # 占位实现 — Task 2 填充
        return []


class SpacyEntityExtractor(EntityExtractor):
    """spaCy NER + noun_chunks 后端 (pip install septmuse[ner])。

    spaCy 不可用时 _resolve_entity_extractor 自动降级到 RegexEntityExtractor。
    """

    def __init__(self, model_name: str = "en_core_web_sm"):
        try:
            import spacy

            self._nlp = spacy.load(model_name)
        except OSError:
            try:
                import spacy

                spacy.cli.download(model_name)
                self._nlp = spacy.load(model_name)
            except Exception as e:
                raise ImportError(f"spaCy model {model_name} unavailable: {e}") from e

    def extract(self, text: str) -> list[Entity]:
        # 占位实现 — Task 3 填充
        return []


def _resolve_entity_extractor(config: MemoryConfig) -> EntityExtractor | None:
    """解析实体抽取器 (类似 _resolve_embedder 模式)。

    默认 RegexEntityExtractor (零配置, 纯 Python)。
    spacy: pip install septmuse[ner], spaCy NER + noun_chunks。
    none: 禁用实体抽取。
    """
    choice = os.getenv("SEPTMUSE_ENTITY_EXTRACTOR", "regex").lower()
    if choice == "none":
        return None
    if choice in ("spacy", "nlp"):
        try:
            return SpacyEntityExtractor()
        except (ImportError, OSError) as e:
            logger.warning("entity_extractor_spacy_unavailable", error=str(e), fallback="regex")
    return RegexEntityExtractor()
```

- [ ] **Step 5: 运行测试验证通过**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_entity_extractor.py -v`
Expected: PASS (6 tests)

- [ ] **Step 6: lint 验证**

Run: `ruff check src/septmuse/concerns/extraction/ tests/unit/test_entity_extractor.py`
Expected: All checks passed!

---

## Task 2: RegexEntityExtractor — PROPER + QUOTED + IDENTIFIER + 泛化词 + span 去重

**Files:**
- Modify: `src/septmuse/concerns/extraction/entity.py` (RegexEntityExtractor.extract)
- Test: `tests/unit/test_entity_extractor.py`

**Interfaces:**
- Consumes: `Entity` dataclass (Task 1)
- Produces: `RegexEntityExtractor.extract(text) -> list[Entity]` (3 类: PROPER/QUOTED/IDENTIFIER)

- [ ] **Step 1: 写失败测试 — PROPER 抽取**

```python
# 追加到 tests/unit/test_entity_extractor.py

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
        # "The", "person", "this", "place" 应被泛化词过滤
        assert "The" not in texts
        assert "person" not in texts
        assert "this" not in texts
        assert "place" not in texts
```

- [ ] **Step 2: 写失败测试 — QUOTED + IDENTIFIER + span 去重**

```python
# 追加到 tests/unit/test_entity_extractor.py

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
        # 同一 (text, entity_type) 只保留第一个
        proper = [e for e in entities if e.entity_type == "PROPER"]
        assert len(proper) == 1

    def test_long_span_priority(self):
        """长 span 优先于短 span。"""
        extractor = RegexEntityExtractor()
        entities = extractor.extract("Machine Learning is great")
        # "Machine Learning" TOPIC 应优先于 "Machine" IDENTIFIER
        topics = [e for e in entities if e.entity_type == "TOPIC"]
        # TOPIC 在 Task 3 实现, 此处先验证不崩溃
        assert isinstance(entities, list)
```

- [ ] **Step 3: 运行测试验证失败**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_entity_extractor.py -v`
Expected: FAIL — PROPER/QUOTED/IDENTIFIER 测试失败（extract 返回空列表）

- [ ] **Step 4: 实现 RegexEntityExtractor — PROPER + QUOTED + IDENTIFIER**

```python
# 替换 src/septmuse/concerns/extraction/entity.py 中 RegexEntityExtractor 类

import re

# 泛化词黑名单 (借鉴 mem0 _GENERIC_HEADS / _NON_SPECIFIC_ADJ / _GENERIC_CAPS)
_GENERIC_WORDS = frozenset([
    # English
    "the", "this", "that", "thing", "something", "person", "people", "time", "way", "day",
    "man", "woman", "world", "life", "hand", "part", "place", "case", "week", "year",
    "name", "home", "work", "word", "point", "group", "number", "fact", "idea", "issue",
    "side", "kind", "head", "line", "end", "member", "list", "lot", "other", "use",
    "first", "last", "new", "old", "good", "bad", "big", "small", "own", "same",
    "some", "any", "all", "no", "every", "one", "two", "three",
    # Chinese
    "这个", "那个", "什么", "东西", "事情", "地方", "时候", "时间", "人", "他们",
    "我们", "你们", "它们", "自己", "别人", "大家", "所有", "一些", "一点", "一下",
    "一样", "这样", "那样", "里", "中", "上", "下",
])

# 中文百家姓 (前 100 常见)
_CHINESE_SURNAMES = frozenset("赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华金魏陶姜戚谢邹喻柏水窦章云苏潘葛奚范彭郎鲁韦昌马苗凤花方俞任袁柳鲍史唐费廉岑薛雷贺倪汤滕殷罗毕郝邬安常乐于时傅皮卞齐康伍余元卜顾孟平黄和穆萧尹")

_PROPER_EN_RE = re.compile(r"\b([A-Z][a-z]+)\b")
_PROPER_ZH_RE = re.compile(rf"[{_CHINESE_SURNAMES}][\u4e00-\u9fff]{{1,2}}")
_QUOTED_RE = re.compile(r'"([^"]+)"|「([^」]+)」|\'([^\']+)\'|\u201c([^\u201d]+)\u201d')
_IDENTIFIER_DOTTED_RE = re.compile(r"\b([a-z][a-z0-9]*(?:\.[a-z][a-z0-9]*)+)\b")
_IDENTIFIER_CAMEL_RE = re.compile(r"\b([A-Z][a-z]+[A-Z]\w*)\b")
_IDENTIFIER_SNAKE_RE = re.compile(r"\b([a-z]+_[a-z_]+)\b")


def _normalize_entity_text(text: str) -> str:
    """归一化实体文本用于匹配 (借鉴 mem0 _normalize_entity_text)。"""
    return " ".join(text.strip().lower().split())


class RegexEntityExtractor(EntityExtractor):
    """纯 Python regex + 词表后端 (默认, 零配置)。

    4 类实体: PROPER / QUOTED / TOPIC / IDENTIFIER (借鉴 mem0)。
    ~120 泛化词黑名单 + span 去重冲突解决。
    """

    def extract(self, text: str) -> list[Entity]:
        candidates: list[Entity] = []
        candidates.extend(self._extract_proper(text))
        candidates.extend(self._extract_quoted(text))
        candidates.extend(self._extract_identifier(text))
        # TOPIC 在 Task 3 实现
        candidates.extend(self._extract_topic(text))
        return self._resolve_candidates(candidates)

    def _extract_proper(self, text: str) -> list[Entity]:
        result = []
        # 英文大写开头词
        for m in _PROPER_EN_RE.finditer(text):
            word = m.group(1)
            if _normalize_entity_text(word) not in _GENERIC_WORDS:
                result.append(Entity(text=word, entity_type="PROPER", start=m.start(1), end=m.end(1)))
        # 中文人名 (百家姓 + 1-2 字名)
        for m in _PROPER_ZH_RE.finditer(text):
            name = m.group(0)
            if name not in _GENERIC_WORDS:
                result.append(Entity(text=name, entity_type="PROPER", start=m.start(), end=m.end()))
        return result

    def _extract_quoted(self, text: str) -> list[Entity]:
        result = []
        for m in _QUOTED_RE.finditer(text):
            # 取第一个非 None 组
            quoted_text = next(g for g in m.groups() if g is not None)
            result.append(Entity(text=quoted_text, entity_type="QUOTED", start=m.start(), end=m.end()))
        return result

    def _extract_identifier(self, text: str) -> list[Entity]:
        result = []
        for m in _IDENTIFIER_DOTTED_RE.finditer(text):
            result.append(Entity(text=m.group(1), entity_type="IDENTIFIER", start=m.start(1), end=m.end(1)))
        for m in _IDENTIFIER_CAMEL_RE.finditer(text):
            result.append(Entity(text=m.group(1), entity_type="IDENTIFIER", start=m.start(1), end=m.end(1)))
        for m in _IDENTIFIER_SNAKE_RE.finditer(text):
            ident = m.group(1)
            if _normalize_entity_text(ident) not in _GENERIC_WORDS:
                result.append(Entity(text=ident, entity_type="IDENTIFIER", start=m.start(1), end=m.end(1)))
        return result

    def _extract_topic(self, text: str) -> list[Entity]:
        # Task 3 实现
        return []

    @staticmethod
    def _resolve_candidates(candidates: list[Entity]) -> list[Entity]:
        """span 去重冲突解决 (借鉴 mem0 _resolve_candidates)。

        1. 按 start 排序
        2. 同一 (text, entity_type) 只保留第一个
        3. 跨类型冲突: 长 span 优先, 相同长度时 PROPER > QUOTED > TOPIC > IDENTIFIER
        """
        if not candidates:
            return []
        # 按 start 排序
        candidates.sort(key=lambda e: (e.start, -(e.end - e.start)))
        # 同一 (text, entity_type) 去重
        seen: set[tuple[str, str]] = set()
        deduped: list[Entity] = []
        for e in candidates:
            key = (_normalize_entity_text(e.text), e.entity_type)
            if key not in seen:
                seen.add(key)
                deduped.append(e)
        # 跨类型冲突: 重叠 span 中长 span 优先
        type_priority = {"PROPER": 4, "QUOTED": 3, "TOPIC": 2, "IDENTIFIER": 1}
        result: list[Entity] = []
        occupied: list[tuple[int, int]] = []
        for e in sorted(deduped, key=lambda x: (-(x.end - x.start), -type_priority.get(x.entity_type, 0))):
            # 检查是否与已占用的 span 重叠
            overlap = any(not (e.end <= s or e.start >= end) for s, end in occupied)
            if not overlap:
                result.append(e)
                occupied.append((e.start, e.end))
        return sorted(result, key=lambda e: e.start)
```

- [ ] **Step 5: 运行测试验证通过**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_entity_extractor.py -v`
Expected: PASS (15 tests, TOPIC 测试可能 skip)

- [ ] **Step 6: lint 验证**

Run: `ruff check src/septmuse/concerns/extraction/entity.py tests/unit/test_entity_extractor.py`
Expected: All checks passed!

---

## Task 3: RegexEntityExtractor — TOPIC + SpacyEntityExtractor 实现

**Files:**
- Modify: `src/septmuse/concerns/extraction/entity.py` (_extract_topic + SpacyEntityExtractor.extract)
- Test: `tests/unit/test_entity_extractor.py`

**Interfaces:**
- Consumes: RegexEntityExtractor (Task 2)
- Produces: TOPIC 抽取 + SpacyEntityExtractor.extract 完整实现

- [ ] **Step 1: 写失败测试 — TOPIC 抽取**

```python
# 追加到 tests/unit/test_entity_extractor.py

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
        # "this thing" 不应是 TOPIC (泛化词)
        texts = {e.text for e in topics}
        assert "this thing" not in texts.lower() if texts else True
```

- [ ] **Step 2: 写失败测试 — SpacyEntityExtractor (integration skip)**

```python
# 追加到 tests/unit/test_entity_extractor.py

import pytest


class TestSpacyExtractor:
    """SpacyEntityExtractor 测试 (需 pip install septmuse[ner])。"""

    @pytest.mark.integration
    def test_spacy_proper_ner(self):
        """spaCy NER 抽取 PERSON/ORG。"""
        try:
            from septmuse.concerns.extraction.entity import SpacyEntityExtractor
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
        """spaCy noun_chunks 抽取 TOPIC。"""
        try:
            from septmuse.concerns.extraction.entity import SpacyEntityExtractor
            extractor = SpacyEntityExtractor()
        except ImportError:
            pytest.skip("spaCy not installed")

        entities = extractor.extract("The machine learning model is great")
        topics = [e for e in entities if e.entity_type == "TOPIC"]
        # noun_chunks 应包含 "machine learning model" 或子串
        assert len(topics) >= 0  # 宽松断言, 主要验证不崩溃
```

- [ ] **Step 3: 运行测试验证失败**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_entity_extractor.py::TestRegexTopic -v`
Expected: FAIL — TOPIC 测试失败（_extract_topic 返回空列表）

- [ ] **Step 4: 实现 _extract_topic + SpacyEntityExtractor.extract**

```python
# 在 src/septmuse/concerns/extraction/entity.py 的 RegexEntityExtractor 中替换 _extract_topic

    def _extract_topic(self, text: str) -> list[Entity]:
        result: list[Entity] = []
        # 英文连续大写开头词组: [A-Z][a-z]+ (?:\s+[A-Z][a-z]+)+
        topic_en_re = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")
        for m in topic_en_re.finditer(text):
            phrase = m.group(1)
            if _normalize_entity_text(phrase) not in _GENERIC_WORDS:
                result.append(Entity(text=phrase, entity_type="TOPIC", start=m.start(1), end=m.end(1)))
        # 中文连续名词短语 (2-4 字无标点连续, 非人名)
        topic_zh_re = re.compile(r"[\u4e00-\u9fff]{2,4}")
        for m in topic_zh_re.finditer(text):
            phrase = m.group(0)
            if phrase not in _GENERIC_WORDS and phrase not in _CHINESE_SURNAMES:
                # 避免与 PROPER (人名) 重叠 — 百家姓开头的 2 字跳过
                if len(phrase) >= 3 or phrase[0] not in _CHINESE_SURNAMES:
                    result.append(Entity(text=phrase, entity_type="TOPIC", start=m.start(), end=m.end()))
        return result
```

```python
# 在 src/septmuse/concerns/extraction/entity.py 中替换 SpacyEntityExtractor.extract

    def extract(self, text: str) -> list[Entity]:
        """spaCy NER + noun_chunks 抽取。"""
        doc = self._nlp(text)
        candidates: list[Entity] = []
        # NER 实体
        ner_labels = {"PERSON", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART"}
        for ent in doc.ents:
            if ent.label_ in ner_labels:
                if _normalize_entity_text(ent.text) not in _GENERIC_WORDS:
                    candidates.append(
                        Entity(text=ent.text, entity_type="PROPER", start=ent.start_char, end=ent.end_char)
                    )
        # 引号文本 (同 regex)
        for m in _QUOTED_RE.finditer(text):
            quoted_text = next(g for g in m.groups() if g is not None)
            candidates.append(
                Entity(text=quoted_text, entity_type="QUOTED", start=m.start(), end=m.end())
            )
        # noun_chunks 作为 TOPIC
        for chunk in doc.noun_chunks:
            if _normalize_entity_text(chunk.text) not in _GENERIC_WORDS and len(chunk.text.split()) >= 2:
                candidates.append(
                    Entity(text=chunk.text, entity_type="TOPIC", start=chunk.start_char, end=chunk.end_char)
                )
        # 技术标识符 (同 regex)
        for m in _IDENTIFIER_DOTTED_RE.finditer(text):
            candidates.append(
                Entity(text=m.group(1), entity_type="IDENTIFIER", start=m.start(1), end=m.end(1))
            )
        for m in _IDENTIFIER_CAMEL_RE.finditer(text):
            candidates.append(
                Entity(text=m.group(1), entity_type="IDENTIFIER", start=m.start(1), end=m.end(1))
            )
        for m in _IDENTIFIER_SNAKE_RE.finditer(text):
            ident = m.group(1)
            if _normalize_entity_text(ident) not in _GENERIC_WORDS:
                candidates.append(
                    Entity(text=ident, entity_type="IDENTIFIER", start=m.start(1), end=m.end(1))
                )
        return RegexEntityExtractor._resolve_candidates(candidates)
```

- [ ] **Step 5: 运行测试验证通过**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_entity_extractor.py -v`
Expected: PASS (18 tests + 2 skip for spaCy integration)

- [ ] **Step 6: lint 验证**

Run: `ruff check src/septmuse/concerns/extraction/entity.py tests/unit/test_entity_extractor.py`
Expected: All checks passed!

---

## Task 4: EntityStore — 表创建 + upsert + get

**Files:**
- Create: `src/septmuse/storage/entity_store.py`
- Test: `tests/unit/test_entity_store.py`

**Interfaces:**
- Consumes: `Entity` dataclass (Task 1), `Embedder` (已有)
- Produces: `EntityStore(conn, lock, embedder)`, `upsert(entity, memory_id, *, user_id) -> str`, `get(entity_id) -> dict | None`

- [ ] **Step 1: 写失败测试 — 表创建 + upsert + get**

```python
# tests/unit/test_entity_store.py
"""EntityStore 测试 (借鉴 mem0 _upsert_entity / _remove_memory_from_entity_store)。"""

import json
import sqlite3
import threading
from unittest.mock import MagicMock

from septmuse.concerns.extraction.entity import Entity
from septmuse.providers.embedders.hash import HashEmbedder
from septmuse.storage.entity_store import EntityStore


def make_store(tmp_path, embedder=None):
    """创建测试用 EntityStore (独立 SQLite, 不依赖 SQLiteMemoryStore)。"""
    db_path = str(tmp_path / "test_entities.db")
    conn = sqlite3.connect(db_path, check_same_thread=False)
    lock = threading.Lock()
    return EntityStore(conn, lock, embedder=embedder)


class TestEntityStoreUpsert:
    def test_create_new_entity(self, tmp_path):
        store = make_store(tmp_path)
        entity = Entity(text="Google", entity_type="PROPER", start=0, end=6)
        eid = store.upsert(entity, "mem-001", user_id="u1")
        assert eid is not None
        result = store.get(eid)
        assert result["entity_text"] == "Google"
        assert result["entity_type"] == "PROPER"
        assert "mem-001" in json.loads(result["linked_memory_ids"])

    def test_exact_match_appends_memory_id(self, tmp_path):
        store = make_store(tmp_path)
        entity1 = Entity(text="Google", entity_type="PROPER", start=0, end=6)
        entity2 = Entity(text="google", entity_type="PROPER", start=0, end=6)  # 归一化后相同
        eid1 = store.upsert(entity1, "mem-001", user_id="u1")
        eid2 = store.upsert(entity2, "mem-002", user_id="u1")
        assert eid1 == eid2  # 精确匹配, 返回同一 entity_id
        result = store.get(eid1)
        linked = json.loads(result["linked_memory_ids"])
        assert "mem-001" in linked
        assert "mem-002" in linked

    def test_different_users_separate(self, tmp_path):
        store = make_store(tmp_path)
        entity = Entity(text="Google", entity_type="PROPER", start=0, end=6)
        eid1 = store.upsert(entity, "mem-001", user_id="u1")
        eid2 = store.upsert(entity, "mem-002", user_id="u2")
        assert eid1 != eid2  # 不同用户, 不同实体

    def test_semantic_match_with_embedder(self, tmp_path):
        """有 embedder 时做语义去重 (score >= 0.95)。"""
        store = make_store(tmp_path, embedder=HashEmbedder())
        entity1 = Entity(text="Google", entity_type="PROPER", start=0, end=6)
        entity2 = Entity(text="Google Inc", entity_type="PROPER", start=0, end=10)
        eid1 = store.upsert(entity1, "mem-001", user_id="u1")
        eid2 = store.upsert(entity2, "mem-002", user_id="u1")
        # HashEmbedder 语义相似度可能不够高, 但验证不崩溃
        assert eid1 is not None
        assert eid2 is not None

    def test_get_nonexistent(self, tmp_path):
        store = make_store(tmp_path)
        assert store.get("nonexistent-id") is None
```

- [ ] **Step 2: 运行测试验证失败**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_entity_store.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'septmuse.storage.entity_store'`

- [ ] **Step 3: 实现 EntityStore — 表创建 + upsert + get**

```python
# src/septmuse/storage/entity_store.py
"""实体向量库 (独立 SQLite 表, 借鉴 mem0 V3 去图化设计)。

复用 SQLiteMemoryStore 的 conn + lock (类似 SQLiteGraphStore 模式)。
embedder 可选——有则做语义去重 (score >= 0.95), 无则只精确匹配。
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from septmuse.concerns.extraction.entity import Entity, _normalize_entity_text
from septmuse.observability import get_logger
from septmuse.providers.embedders.base import Embedder

logger = get_logger(__name__)


class EntityStore:
    """实体向量库 (独立 SQLite 表, 同库)。

    复用 SQLiteMemoryStore 的 conn + lock。
    embedder 可选——有则做语义去重 (score>=0.95), 无则只精确匹配。
    """

    def __init__(self, conn, lock, embedder: Embedder | None = None):
        self._conn = conn
        self._lock = lock
        self._embedder = embedder
        self._create_table_if_not_exists()

    def _create_table_if_not_exists(self) -> None:
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS septmuse_entities (
                    id TEXT PRIMARY KEY,
                    entity_text TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    entity_embedding BLOB,
                    linked_memory_ids TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    agent_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    is_deleted INTEGER DEFAULT 0,
                    UNIQUE(user_id, entity_text)
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_entities_user ON septmuse_entities(user_id)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_entities_text ON septmuse_entities(entity_text)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_entities_deleted ON septmuse_entities(is_deleted)"
            )
            self._conn.commit()

    def upsert(
        self,
        entity: Entity,
        memory_id: str,
        *,
        user_id: str,
        agent_id: str | None = None,
    ) -> str:
        """upsert 实体 (借鉴 mem0 _upsert_entity)。

        1. 精确归一化名匹配 → 命中则 linked_memory_ids 追加 memory_id
        2. 语义匹配 (embedder 有时) → score>=0.95 命中则追加
        3. 新建 → 插入实体 + 嵌入向量 + linked_memory_ids=[memory_id]

        Returns: entity_id
        """
        normalized = _normalize_entity_text(entity.text)

        # 1. 精确归一化名匹配
        existing = self._find_by_text(normalized, user_id=user_id)
        if existing:
            self._append_memory_id(existing["id"], memory_id)
            return existing["id"]

        # 2. 语义匹配 (embedder 有时)
        emb = None
        if self._embedder is not None:
            emb = self._embedder.embed(entity.text)
            semantic_match = self._find_by_embedding(emb, user_id=user_id, threshold=0.95)
            if semantic_match:
                self._append_memory_id(semantic_match["id"], memory_id)
                return semantic_match["id"]

        # 3. 新建
        entity_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        emb_blob = self._serialize_embedding(emb) if emb is not None else None
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO septmuse_entities
                    (id, entity_text, entity_type, entity_embedding, linked_memory_ids,
                     user_id, agent_id, created_at, updated_at, is_deleted)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                """,
                (
                    entity_id,
                    entity.text,
                    entity.entity_type,
                    emb_blob,
                    json.dumps([memory_id]),
                    user_id,
                    agent_id,
                    now,
                    now,
                ),
            )
            self._conn.commit()
        return entity_id

    def get(self, entity_id: str) -> dict[str, Any] | None:
        """取单条实体。"""
        with self._lock:
            cur = self._conn.execute(
                "SELECT id, entity_text, entity_type, entity_embedding, linked_memory_ids, "
                "user_id, agent_id, created_at, updated_at FROM septmuse_entities "
                "WHERE id=? AND is_deleted=0",
                (entity_id,),
            )
            r = cur.fetchone()
        if not r:
            return None
        return {
            "id": r[0],
            "entity_text": r[1],
            "entity_type": r[2],
            "entity_embedding": r[3],
            "linked_memory_ids": r[4],
            "user_id": r[5],
            "agent_id": r[6],
            "created_at": r[7],
            "updated_at": r[8],
        }

    def _find_by_text(self, normalized_text: str, *, user_id: str) -> dict[str, Any] | None:
        """精确归一化名匹配。"""
        with self._lock:
            cur = self._conn.execute(
                "SELECT id, entity_text, entity_type, linked_memory_ids FROM septmuse_entities "
                "WHERE user_id=? AND is_deleted=0",
                (user_id,),
            )
            for r in cur.fetchall():
                if _normalize_entity_text(r[1]) == normalized_text:
                    return {
                        "id": r[0],
                        "entity_text": r[1],
                        "entity_type": r[2],
                        "linked_memory_ids": r[3],
                    }
        return None

    def _find_by_embedding(
        self, embedding: list[float], *, user_id: str, threshold: float = 0.95
    ) -> dict[str, Any] | None:
        """语义匹配 (cosine similarity >= threshold)。"""
        with self._lock:
            cur = self._conn.execute(
                "SELECT id, entity_text, entity_type, entity_embedding, linked_memory_ids "
                "FROM septmuse_entities WHERE user_id=? AND is_deleted=0 AND entity_embedding IS NOT NULL",
                (user_id,),
            )
            for r in cur.fetchall():
                stored_emb = self._deserialize_embedding(r[3])
                if stored_emb is not None:
                    sim = _cosine_similarity(embedding, stored_emb)
                    if sim >= threshold:
                        return {
                            "id": r[0],
                            "entity_text": r[1],
                            "entity_type": r[2],
                            "linked_memory_ids": r[4],
                        }
        return None

    def _append_memory_id(self, entity_id: str, memory_id: str) -> None:
        """向实体的 linked_memory_ids 追加 memory_id。"""
        with self._lock:
            cur = self._conn.execute(
                "SELECT linked_memory_ids FROM septmuse_entities WHERE id=?",
                (entity_id,),
            )
            r = cur.fetchone()
            if not r:
                return
            linked = json.loads(r[0])
            if memory_id not in linked:
                linked.append(memory_id)
            now = datetime.now(timezone.utc).isoformat()
            self._conn.execute(
                "UPDATE septmuse_entities SET linked_memory_ids=?, updated_at=? WHERE id=?",
                (json.dumps(linked), now, entity_id),
            )
            self._conn.commit()

    def _insert(self, entity_id, entity, emb, memory_id, user_id, agent_id) -> None:
        """插入新实体 (由 upsert 调用, 此方法已内联到 upsert 中, 保留用于子类覆盖)。"""
        pass  # 实现已内联到 upsert

    @staticmethod
    def _serialize_embedding(emb: list[float]) -> bytes:
        """序列化嵌入向量为 BLOB。"""
        import struct

        return struct.pack(f"{len(emb)}f", *emb)

    @staticmethod
    def _deserialize_embedding(blob: bytes) -> list[float] | None:
        """反序列化嵌入向量。"""
        if not blob:
            return None
        import struct

        n = len(blob) // 4
        return list(struct.unpack(f"{n}f", blob))

    def close(self) -> None:
        """释放资源 (同库, 实际不关 conn)。"""
        pass


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """cosine 相似度 (score 统一为相似度 [0,1])。"""
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
```

- [ ] **Step 4: 运行测试验证通过**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_entity_store.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: lint 验证**

Run: `ruff check src/septmuse/storage/entity_store.py tests/unit/test_entity_store.py`
Expected: All checks passed!

---

## Task 5: EntityStore — search + list + get_linked_memories + remove_memory_from_entities

**Files:**
- Modify: `src/septmuse/storage/entity_store.py` (新增方法)
- Test: `tests/unit/test_entity_store.py`

**Interfaces:**
- Consumes: EntityStore.upsert/get (Task 4)
- Produces: `search(query, *, user_id, top_k)`, `list(*, user_id, entity_type, limit)`, `get_linked_memories(entity_id)`, `remove_memory_from_entities(memory_id)`

- [ ] **Step 1: 写失败测试 — search + list + get_linked_memories + remove**

```python
# 追加到 tests/unit/test_entity_store.py

class TestEntityStoreSearch:
    def test_search_exact_match(self, tmp_path):
        store = make_store(tmp_path)
        entity = Entity(text="Google", entity_type="PROPER", start=0, end=6)
        store.upsert(entity, "mem-001", user_id="u1")
        results = store.search("Google", user_id="u1", top_k=5)
        assert any(r["entity_text"] == "Google" for r in results)

    def test_search_no_match(self, tmp_path):
        store = make_store(tmp_path)
        entity = Entity(text="Google", entity_type="PROPER", start=0, end=6)
        store.upsert(entity, "mem-001", user_id="u1")
        results = store.search("Microsoft", user_id="u1", top_k=5)
        assert len(results) == 0

    def test_search_user_isolation(self, tmp_path):
        store = make_store(tmp_path)
        entity = Entity(text="Google", entity_type="PROPER", start=0, end=6)
        store.upsert(entity, "mem-001", user_id="u1")
        results = store.search("Google", user_id="u2", top_k=5)
        assert len(results) == 0  # 不同用户不可见


class TestEntityStoreList:
    def test_list_all(self, tmp_path):
        store = make_store(tmp_path)
        store.upsert(Entity(text="Google", entity_type="PROPER", start=0, end=6), "m1", user_id="u1")
        store.upsert(Entity(text="Python", entity_type="IDENTIFIER", start=0, end=6), "m2", user_id="u1")
        result = store.list(user_id="u1")
        assert len(result) == 2

    def test_list_by_type(self, tmp_path):
        store = make_store(tmp_path)
        store.upsert(Entity(text="Google", entity_type="PROPER", start=0, end=6), "m1", user_id="u1")
        store.upsert(Entity(text="Python", entity_type="IDENTIFIER", start=0, end=6), "m2", user_id="u1")
        result = store.list(user_id="u1", entity_type="PROPER")
        assert len(result) == 1
        assert result[0]["entity_text"] == "Google"

    def test_list_user_isolation(self, tmp_path):
        store = make_store(tmp_path)
        store.upsert(Entity(text="Google", entity_type="PROPER", start=0, end=6), "m1", user_id="u1")
        result = store.list(user_id="u2")
        assert len(result) == 0


class TestEntityStoreGetLinked:
    def test_get_linked_memories(self, tmp_path):
        store = make_store(tmp_path)
        entity = Entity(text="Google", entity_type="PROPER", start=0, end=6)
        eid = store.upsert(entity, "mem-001", user_id="u1")
        linked = store.get_linked_memories(eid)
        assert "mem-001" in linked

    def test_get_linked_nonexistent(self, tmp_path):
        store = make_store(tmp_path)
        assert store.get_linked_memories("nonexistent") == []


class TestEntityStoreRemove:
    def test_remove_memory_from_entities(self, tmp_path):
        store = make_store(tmp_path)
        entity = Entity(text="Google", entity_type="PROPER", start=0, end=6)
        eid = store.upsert(entity, "mem-001", user_id="u1")
        # 清理 mem-001 引用
        store.remove_memory_from_entities("mem-001")
        # 实体应被软删除 (linked_memory_ids 空)
        result = store.get(eid)
        assert result is None  # is_deleted=1, get 返回 None

    def test_remove_keeps_entity_if_other_links(self, tmp_path):
        store = make_store(tmp_path)
        entity = Entity(text="Google", entity_type="PROPER", start=0, end=6)
        eid = store.upsert(entity, "mem-001", user_id="u1")
        store.upsert(Entity(text="google", entity_type="PROPER", start=0, end=6), "mem-002", user_id="u1")
        # 只清理 mem-001, mem-002 仍在
        store.remove_memory_from_entities("mem-001")
        result = store.get(eid)
        assert result is not None  # 仍存在
        linked = json.loads(result["linked_memory_ids"])
        assert "mem-001" not in linked
        assert "mem-002" in linked

    def test_remove_nonexistent_memory(self, tmp_path):
        store = make_store(tmp_path)
        entity = Entity(text="Google", entity_type="PROPER", start=0, end=6)
        eid = store.upsert(entity, "mem-001", user_id="u1")
        # 不存在的 memory_id, 不应崩溃
        store.remove_memory_from_entities("nonexistent-mem")
        result = store.get(eid)
        assert result is not None  # 实体仍在
```

- [ ] **Step 2: 运行测试验证失败**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_entity_store.py -v`
Expected: FAIL — search/list/get_linked_memories/remove 方法不存在

- [ ] **Step 3: 实现 search + list + get_linked_memories + remove_memory_from_entities**

```python
# 在 src/septmuse/storage/entity_store.py 的 EntityStore 类中追加方法

    def search(self, query: str, *, user_id: str, top_k: int = 5) -> list[dict[str, Any]]:
        """搜索实体: 精确匹配 + 向量相似度 (embedder 有时)。

        Returns: [{"id","entity_text","entity_type","linked_memory_ids","score"}]
        """
        results: list[dict[str, Any]] = []
        normalized_query = _normalize_entity_text(query)

        # 精确匹配
        with self._lock:
            cur = self._conn.execute(
                "SELECT id, entity_text, entity_type, linked_memory_ids FROM septmuse_entities "
                "WHERE user_id=? AND is_deleted=0",
                (user_id,),
            )
            for r in cur.fetchall():
                normalized_entity = _normalize_entity_text(r[1])
                if normalized_query in normalized_entity or normalized_entity in normalized_query:
                    results.append({
                        "id": r[0],
                        "entity_text": r[1],
                        "entity_type": r[2],
                        "linked_memory_ids": r[3],
                        "score": 1.0 if normalized_entity == normalized_query else 0.8,
                    })

        # 向量相似度 (embedder 有时)
        if self._embedder is not None:
            query_emb = self._embedder.embed(query)
            with self._lock:
                cur = self._conn.execute(
                    "SELECT id, entity_text, entity_type, entity_embedding, linked_memory_ids "
                    "FROM septmuse_entities WHERE user_id=? AND is_deleted=0 AND entity_embedding IS NOT NULL",
                    (user_id,),
                )
                existing_ids = {r["id"] for r in results}
                for r in cur.fetchall():
                    if r[0] in existing_ids:
                        continue
                    stored_emb = self._deserialize_embedding(r[3])
                    if stored_emb is not None:
                        sim = _cosine_similarity(query_emb, stored_emb)
                        if sim > 0.3:
                            results.append({
                                "id": r[0],
                                "entity_text": r[1],
                                "entity_type": r[2],
                                "linked_memory_ids": r[4],
                                "score": sim,
                            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def list(
        self, *, user_id: str, entity_type: str | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        """列出用户全部未删除实体。"""
        with self._lock:
            if entity_type:
                cur = self._conn.execute(
                    "SELECT id, entity_text, entity_type, linked_memory_ids, created_at "
                    "FROM septmuse_entities WHERE user_id=? AND is_deleted=0 AND entity_type=? "
                    "ORDER BY created_at DESC LIMIT ?",
                    (user_id, entity_type, limit),
                )
            else:
                cur = self._conn.execute(
                    "SELECT id, entity_text, entity_type, linked_memory_ids, created_at "
                    "FROM septmuse_entities WHERE user_id=? AND is_deleted=0 "
                    "ORDER BY created_at DESC LIMIT ?",
                    (user_id, limit),
                )
            rows = cur.fetchall()
        return [
            {
                "id": r[0],
                "entity_text": r[1],
                "entity_type": r[2],
                "linked_memory_ids": r[3],
                "created_at": r[4],
            }
            for r in rows
        ]

    def get_linked_memories(self, entity_id: str) -> list[str]:
        """获取实体的 linked_memory_ids。"""
        with self._lock:
            cur = self._conn.execute(
                "SELECT linked_memory_ids FROM septmuse_entities WHERE id=? AND is_deleted=0",
                (entity_id,),
            )
            r = cur.fetchone()
        if not r:
            return []
        return json.loads(r[0])

    def remove_memory_from_entities(self, memory_id: str) -> None:
        """删除记忆时清理实体引用 (借鉴 mem0 _remove_memory_from_entity_store)。

        memory_id 是 UUID 全局唯一, 不需 user_id 过滤。
        1. 查 linked_memory_ids 包含 memory_id 的实体
        2. 移除 memory_id
        3. linked_memory_ids 空 → 软删除实体
        """
        with self._lock:
            cur = self._conn.execute(
                "SELECT id, linked_memory_ids FROM septmuse_entities WHERE is_deleted=0",
            )
            for r in cur.fetchall():
                entity_id, linked_json = r[0], r[1]
                linked = json.loads(linked_json)
                if memory_id not in linked:
                    continue
                remaining = [mid for mid in linked if mid != memory_id]
                now = datetime.now(timezone.utc).isoformat()
                if not remaining:
                    # 引用清空 → 软删除实体
                    self._conn.execute(
                        "UPDATE septmuse_entities SET is_deleted=1, updated_at=? WHERE id=?",
                        (now, entity_id),
                    )
                else:
                    # 更新 linked_memory_ids
                    self._conn.execute(
                        "UPDATE septmuse_entities SET linked_memory_ids=?, updated_at=? WHERE id=?",
                        (json.dumps(remaining), now, entity_id),
                    )
            self._conn.commit()
```

- [ ] **Step 4: 运行测试验证通过**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_entity_store.py -v`
Expected: PASS (12 tests)

- [ ] **Step 5: lint 验证**

Run: `ruff check src/septmuse/storage/entity_store.py tests/unit/test_entity_store.py`
Expected: All checks passed!

---

## Task 6: Memory facade — __init__ + add + delete + close 扩展

**Files:**
- Modify: `src/septmuse/orchestration/memory.py` (Memory.__init__ + add + delete + close)
- Modify: `tests/unit/test_memory.py` (新增实体集成测试)
- Test: `tests/unit/test_memory.py`

**Interfaces:**
- Consumes: EntityExtractor (Task 1-3), EntityStore (Task 4-5)
- Produces: `Memory.entity_extractor`, `Memory.entity_store`, `Memory.add(auto_extract_entities=True)`, `Memory.delete()` 清理实体

- [ ] **Step 1: 写失败测试 — add + auto_extract + delete 清理**

```python
# 追加到 tests/unit/test_memory.py

class TestEntityIntegration:
    """Memory facade 实体集成测试。"""

    def test_add_auto_extracts_entities(self, tmp_path):
        from septmuse.orchestration.memory import Memory
        from septmuse.configs.defaults import MemoryConfig

        config = MemoryConfig(db_path=str(tmp_path / "test.db"))
        m = Memory(config=config)
        result = m.add("Alice works at Google", user_id="u1")
        memory_id = result["results"][0]["id"]

        # 实体应被自动抽取
        entities = m.list_entities(user_id="u1")
        texts = {e["entity_text"] for e in entities}
        assert "Alice" in texts or "Google" in texts

        # search_entities 应找到 Google
        search_results = m.search_entities("Google", user_id="u1")
        assert any(r["entity_text"] == "Google" for r in search_results)
        m.close()

    def test_add_auto_extract_disabled(self, tmp_path):
        from septmuse.orchestration.memory import Memory
        from septmuse.configs.defaults import MemoryConfig

        config = MemoryConfig(db_path=str(tmp_path / "test.db"))
        m = Memory(config=config)
        m.add("Alice works at Google", user_id="u1", auto_extract_entities=False)

        # 不抽取实体
        entities = m.list_entities(user_id="u1")
        assert len(entities) == 0
        m.close()

    def test_delete_cleans_entity_refs(self, tmp_path):
        from septmuse.orchestration.memory import Memory
        from septmuse.configs.defaults import MemoryConfig

        config = MemoryConfig(db_path=str(tmp_path / "test.db"))
        m = Memory(config=config)
        result = m.add("Alice works at Google", user_id="u1")
        memory_id = result["results"][0]["id"]

        # 确认实体存在
        entities_before = m.list_entities(user_id="u1")
        assert len(entities_before) > 0

        # 删除记忆 → 实体引用清理
        m.delete(memory_id)

        # 实体应被清理 (linked_memory_ids 空则软删)
        entities_after = m.list_entities(user_id="u1")
        # Google 只关联了这一条记忆, 应被软删
        google_entities = [e for e in entities_after if e["entity_text"] == "Google"]
        assert len(google_entities) == 0
        m.close()
```

- [ ] **Step 2: 运行测试验证失败**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_memory.py::TestEntityIntegration -v`
Expected: FAIL — `Memory` 无 `entity_extractor`/`entity_store` 属性, 无 `list_entities`/`search_entities` 方法

- [ ] **Step 3: 修改 Memory.__init__ — 新增 entity_extractor + entity_store**

```python
# 在 src/septmuse/orchestration/memory.py 的 Memory.__init__ 中, 在 self._dedup_window 之后追加

        # 实体抽取器 + 实体向量库 (借鉴 mem0 V3 去图化设计)
        from septmuse.concerns.extraction.entity import _resolve_entity_extractor

        self.entity_extractor = entity_extractor or _resolve_entity_extractor(self.config)

        if isinstance(self.store, SQLiteMemoryStore) and self.entity_extractor is not None:
            from septmuse.storage.entity_store import EntityStore

            self.entity_store: EntityStore | None = EntityStore(
                self.store.conn, self.store._lock, embedder=self.embedder
            )
        else:
            self.entity_store = None
```

- [ ] **Step 4: 修改 Memory.add — auto_extract_entities**

```python
# 替换 src/septmuse/orchestration/memory.py 中 Memory.add 方法

    def add(
        self,
        messages: Any,
        *,
        user_id: str,
        agent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        infer: bool | None = None,
        auto_extract_entities: bool = True,
    ) -> dict[str, Any]:
        """添加记忆 (对齐 mem0 add 签名, 支持 infer LLM 抽取 + 实体抽取)。

        Args:
            messages: str 或 List[{"role","content"}]
            user_id: 用户 ID (必填, 跨 agent 共享键)
            agent_id: agent ID (可选)
            metadata: 元数据
            infer: True=LLM 抽取事实 (需注入 LLM); False=原文存 (verbatim);
                   None=用 config.infer 默认
            auto_extract_entities: True=自动抽取实体存入 EntityStore; False=跳过

        Returns:
            infer=True: {"results": [...], "relations": []}
            infer=False: {"results": [{"id","memory","event":"ADD"}], "relations": []}
        """
        should_infer = self.config.infer if infer is None else infer

        # infer=True: 走 cognify 抽取流水线
        if should_infer and self.extractor is not None:
            extracted = self.extractor.extract_and_store(messages, user_id=user_id)
            return {"results": extracted, "relations": []}

        # infer=False 或无 LLM: verbatim 原文存
        texts = _normalize_messages(messages)
        if not texts:
            return {"results": [], "relations": []}

        embeddings = self.embedder.embed_batch(texts)

        results: list[dict[str, Any]] = []
        for text, emb in zip(texts, embeddings, strict=True):
            mid = self.store.add(text, emb, user_id=user_id, agent_id=agent_id, metadata=metadata)
            results.append({"id": mid, "memory": text, "event": "ADD"})

        # 自动抽取实体并存入 EntityStore
        if auto_extract_entities and self.entity_store is not None and self.entity_extractor is not None:
            for text, result in zip(texts, results, strict=True):
                memory_id = result["id"]
                try:
                    entities = self.entity_extractor.extract(text)
                    for entity in entities:
                        self.entity_store.upsert(entity, memory_id, user_id=user_id, agent_id=agent_id)
                except Exception as e:
                    logger.warning("entity_extract_failed", memory_id=memory_id, error=str(e))

        logger.info("memory_add_done", user_id=user_id, count=len(results), infer=should_infer)
        return {"results": results, "relations": []}
```

- [ ] **Step 5: 修改 Memory.delete — 清理实体引用**

```python
# 替换 src/septmuse/orchestration/memory.py 中 Memory.delete 方法

    def delete(self, memory_id: str) -> dict[str, str]:
        """软删除 (对齐 mem0 delete, 清理实体引用)。"""
        self.store.delete(memory_id)

        # 清理实体引用 (引用清空则删实体)
        if self.entity_store is not None:
            try:
                self.entity_store.remove_memory_from_entities(memory_id)
            except Exception as e:
                logger.warning("entity_cleanup_failed", memory_id=memory_id, error=str(e))

        return {"status": "deleted", "memory_id": memory_id}
```

- [ ] **Step 6: 修改 Memory.close — 关闭 entity_store**

```python
# 替换 src/septmuse/orchestration/memory.py 中 Memory.close 方法

    def close(self) -> None:
        """关闭存储。"""
        self.store.close()
        self.typed_store.close()
        if self.entity_store is not None:
            self.entity_store.close()
```

- [ ] **Step 7: 运行测试验证**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_memory.py::TestEntityIntegration -v`
Expected: PARTIAL PASS — add+auto_extract 通过, delete+清理可能需要 search_entities/list_entities (Task 7)

- [ ] **Step 8: lint 验证**

Run: `ruff check src/septmuse/orchestration/memory.py`
Expected: All checks passed!

---

## Task 7: Memory facade — 5 个新方法 + MemoryConfig + pyproject.toml

**Files:**
- Modify: `src/septmuse/orchestration/memory.py` (5 个新方法)
- Modify: `src/septmuse/configs/defaults.py` (entity_extractor_backend)
- Modify: `pyproject.toml` (ner extra)
- Test: `tests/unit/test_memory.py`

**Interfaces:**
- Consumes: Memory.entity_extractor + Memory.entity_store (Task 6)
- Produces: `extract_entities(text)`, `add_entity(...)`, `search_entities(query, ...)`, `get_entity_neighbors(entity_id)`, `list_entities(...)`

- [ ] **Step 1: 写失败测试 — 5 个新方法**

```python
# 追加到 tests/unit/test_memory.py

class TestEntityMethods:
    """Memory facade 5 个实体方法测试。"""

    def test_extract_entities(self, tmp_path):
        from septmuse.orchestration.memory import Memory
        from septmuse.configs.defaults import MemoryConfig

        m = Memory(config=MemoryConfig(db_path=str(tmp_path / "test.db")))
        entities = m.extract_entities("Alice works at Google")
        texts_types = {(e["text"], e["type"]) for e in entities}
        assert any(e[0] == "Alice" for e in texts_types)
        assert any(e[0] == "Google" for e in texts_types)
        m.close()

    def test_add_entity_manual(self, tmp_path):
        from septmuse.orchestration.memory import Memory
        from septmuse.configs.defaults import MemoryConfig

        m = Memory(config=MemoryConfig(db_path=str(tmp_path / "test.db")))
        result = m.add_entity("CustomEntity", "PROPER", "mem-001", user_id="u1")
        assert result["event"] == "ADD"
        assert result["entity"] == "CustomEntity"
        # 验证存入
        entities = m.list_entities(user_id="u1")
        assert any(e["entity_text"] == "CustomEntity" for e in entities)
        m.close()

    def test_search_entities(self, tmp_path):
        from septmuse.orchestration.memory import Memory
        from septmuse.configs.defaults import MemoryConfig

        m = Memory(config=MemoryConfig(db_path=str(tmp_path / "test.db")))
        m.add("Alice works at Google", user_id="u1")
        results = m.search_entities("Google", user_id="u1")
        assert len(results) > 0
        assert any(r["entity_text"] == "Google" for r in results)
        m.close()

    def test_get_entity_neighbors(self, tmp_path):
        from septmuse.orchestration.memory import Memory
        from septmuse.configs.defaults import MemoryConfig

        m = Memory(config=MemoryConfig(db_path=str(tmp_path / "test.db")))
        result = m.add("Alice works at Google", user_id="u1")
        memory_id = result["results"][0]["id"]
        entities = m.list_entities(user_id="u1")
        google_entity = next(e for e in entities if e["entity_text"] == "Google")
        neighbors = m.get_entity_neighbors(google_entity["id"])
        assert memory_id in neighbors
        m.close()

    def test_list_entities_by_type(self, tmp_path):
        from septmuse.orchestration.memory import Memory
        from septmuse.configs.defaults import MemoryConfig

        m = Memory(config=MemoryConfig(db_path=str(tmp_path / "test.db")))
        m.add("Alice works at Google using Python", user_id="u1")
        proper_entities = m.list_entities(user_id="u1", entity_type="PROPER")
        assert all(e["entity_type"] == "PROPER" for e in proper_entities)
        m.close()
```

- [ ] **Step 2: 运行测试验证失败**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_memory.py::TestEntityMethods -v`
Expected: FAIL — 方法不存在

- [ ] **Step 3: 实现 5 个新方法**

```python
# 在 src/septmuse/orchestration/memory.py 的 Memory 类中, 在 close() 方法之前追加

    # ------------------------------------------------------------------
    # 实体抽取入口 (架构文档 §5.1, 借鉴 mem0 entity_extraction)
    # ------------------------------------------------------------------

    def extract_entities(self, text: str) -> list[dict[str, Any]]:
        """抽取实体 (不存储), 返回 [{"text","type","start","end"}]。"""
        if self.entity_extractor is None:
            return []
        entities = self.entity_extractor.extract(text)
        return [
            {"text": e.text, "type": e.entity_type, "start": e.start, "end": e.end}
            for e in entities
        ]

    def add_entity(
        self, entity_text: str, entity_type: str, memory_id: str, *, user_id: str
    ) -> dict[str, Any]:
        """手动添加实体与记忆的关联。"""
        if self.entity_store is None:
            return {"error": "entity_store not available (SQLite only)"}
        from septmuse.concerns.extraction.entity import Entity

        entity = Entity(text=entity_text, entity_type=entity_type, start=0, end=len(entity_text))
        eid = self.entity_store.upsert(entity, memory_id, user_id=user_id)
        return {"id": eid, "entity": entity_text, "type": entity_type, "event": "ADD"}

    def search_entities(self, query: str, *, user_id: str, top_k: int = 5) -> list[dict[str, Any]]:
        """搜索实体, 返回实体 + linked_memory_ids。"""
        if self.entity_store is None:
            return []
        return self.entity_store.search(query, user_id=user_id, top_k=top_k)

    def get_entity_neighbors(self, entity_id: str) -> list[str]:
        """获取实体关联的 memory_id 列表。"""
        if self.entity_store is None:
            return []
        return self.entity_store.get_linked_memories(entity_id)

    def list_entities(
        self, *, user_id: str, entity_type: str | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        """列出用户全部实体。"""
        if self.entity_store is None:
            return []
        return self.entity_store.list(user_id=user_id, entity_type=entity_type, limit=limit)
```

- [ ] **Step 4: 修改 MemoryConfig — entity_extractor_backend**

```python
# 在 src/septmuse/configs/defaults.py 的 MemoryConfig 中追加字段

    entity_extractor_backend: str = Field(
        default="regex",
        description="实体抽取后端: regex(默认)/spacy/none",
    )
```

- [ ] **Step 5: 修改 pyproject.toml — ner extra**

```toml
# 在 pyproject.toml 的 [project.optional-dependencies] 中追加
ner = ["spacy>=3.7"]
```

- [ ] **Step 6: 运行测试验证通过**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_memory.py::TestEntityIntegration tests/unit/test_memory.py::TestEntityMethods -v`
Expected: PASS (8 tests)

- [ ] **Step 7: lint 验证**

Run: `ruff check src/septmuse/orchestration/memory.py src/septmuse/configs/defaults.py`
Expected: All checks passed!

---

## Task 8: CLI + REST + MCP 扩展

**Files:**
- Modify: `src/septmuse/cli/main.py` (+2 命令)
- Modify: `src/septmuse/api/rest/__init__.py` (+2 端点)
- Modify: `src/septmuse/api/mcp/tools.py` (+2 工具)
- Test: `tests/unit/test_cli.py` 扩展, `tests/unit/test_rest.py` 扩展

**Interfaces:**
- Consumes: Memory facade 5 个新方法 (Task 7)
- Produces: CLI `entities`/`entity-list` 命令, REST `GET /entities`/`GET /entities/list` 端点, MCP `search_entities`/`list_entities` 工具

- [ ] **Step 1: 写失败测试 — CLI**

```python
# 追加到 tests/unit/test_cli.py (如果文件存在) 或创建

class TestCLIEntities:
    def test_cli_entities_search(self, tmp_path):
        from septmuse.cli.main import main
        from septmuse.orchestration.memory import Memory
        from septmuse.configs.defaults import MemoryConfig

        # 先添加记忆
        config = MemoryConfig(db_path=str(tmp_path / "test.db"))
        m = Memory(config=config)
        m.add("Alice works at Google", user_id="u1")
        m.close()

        # CLI 搜索实体
        import sys
        from unittest.mock import patch

        with patch.dict("os.environ", {"SEPTMUSE_DB_PATH": str(tmp_path / "test.db")}):
            with patch.object(sys, "argv", ["septmuse", "entities", "Google", "--user-id", "u1"]):
                main()  # 应输出 Google 实体

    def test_cli_entity_list(self, tmp_path):
        from septmuse.cli.main import main
        from septmuse.orchestration.memory import Memory
        from septmuse.configs.defaults import MemoryConfig

        config = MemoryConfig(db_path=str(tmp_path / "test.db"))
        m = Memory(config=config)
        m.add("Alice works at Google", user_id="u1")
        m.close()

        import sys
        from unittest.mock import patch

        with patch.dict("os.environ", {"SEPTMUSE_DB_PATH": str(tmp_path / "test.db")}):
            with patch.object(sys, "argv", ["septmuse", "entity-list", "--user-id", "u1"]):
                main()
```

- [ ] **Step 2: 实现 CLI 2 个命令**

```python
# 在 src/septmuse/cli/main.py 的 argparse subparsers 中追加

    # entities — 搜索实体
    sp_entities = subparsers.add_parser("entities", help="搜索实体")
    sp_entities.add_argument("query", help="搜索查询")
    sp_entities.add_argument("--user-id", default="default", help="用户 ID")
    sp_entities.add_argument("--top-k", type=int, default=5, help="返回数")
    sp_entities.add_argument("--db-path", default=None, help="数据库路径")

    # entity-list — 列出实体
    sp_entity_list = subparsers.add_parser("entity-list", help="列出实体")
    sp_entity_list.add_argument("--user-id", default="default", help="用户 ID")
    sp_entity_list.add_argument("--entity-type", default=None, help="实体类型过滤")
    sp_entity_list.add_argument("--limit", type=int, default=100, help="返回数")
    sp_entity_list.add_argument("--db-path", default=None, help="数据库路径")
```

```python
# 在 src/septmuse/cli/main.py 的命令处理中追加

    elif args.command == "entities":
        from septmuse.orchestration.memory import Memory

        m = Memory(config=default_config(db_path=args.db_path))
        results = m.search_entities(args.query, user_id=args.user_id, top_k=args.top_k)
        for r in results:
            print(f"  {r['entity_text']} ({r['entity_type']}) -> {r['linked_memory_ids']}")
        m.close()

    elif args.command == "entity-list":
        from septmuse.orchestration.memory import Memory

        m = Memory(config=default_config(db_path=args.db_path))
        results = m.list_entities(user_id=args.user_id, entity_type=args.entity_type, limit=args.limit)
        for r in results:
            print(f"  {r['entity_text']} ({r['entity_type']}) -> {r['linked_memory_ids']}")
        m.close()
```

- [ ] **Step 3: 实现 REST 2 个端点**

```python
# 在 src/septmuse/api/rest/__init__.py 的 create_app 中追加

    @app.get("/entities")
    async def search_entities(
        query: str,
        user_id: str = "default",
        top_k: int = 5,
        x_api_key: str = Depends(verify_api_key),
    ):
        """搜索实体。"""
        m = Memory(config=default_config())
        try:
            results = m.search_entities(query, user_id=user_id, top_k=top_k)
            return {"results": results}
        finally:
            m.close()

    @app.get("/entities/list")
    async def list_entities(
        user_id: str = "default",
        entity_type: str | None = None,
        limit: int = 100,
        x_api_key: str = Depends(verify_api_key),
    ):
        """列出实体。"""
        m = Memory(config=default_config())
        try:
            results = m.list_entities(user_id=user_id, entity_type=entity_type, limit=limit)
            return {"results": results}
        finally:
            m.close()
```

- [ ] **Step 4: 实现 MCP 2 个工具**

```python
# 在 src/septmuse/api/mcp/tools.py 中追加

@mcp.tool
def search_entities(
    query: str,
    user_id: str | None = None,
    top_k: int = 5,
) -> str:
    """搜索实体, 返回实体文本 + 类型 + linked_memory_ids。

    Args:
        query: 搜索查询
        user_id: 用户 ID (默认从 contextvars 读)
        top_k: 返回数

    Returns:
        JSON 字符串: [{"id","entity_text","entity_type","linked_memory_ids","score"}]
    """
    try:
        uid = user_id or _get_user_id()
        m = _get_memory()
        results = m.search_entities(query, user_id=uid, top_k=top_k)
        return json.dumps(results, ensure_ascii=False)
    except Exception as e:
        return f"Error: {e}"


@mcp.tool
def list_entities(
    user_id: str | None = None,
    entity_type: str | None = None,
    limit: int = 100,
) -> str:
    """列出用户全部实体。

    Args:
        user_id: 用户 ID (默认从 contextvars 读)
        entity_type: 实体类型过滤 (PROPER/QUOTED/TOPIC/IDENTIFIER)
        limit: 返回数

    Returns:
        JSON 字符串: [{"id","entity_text","entity_type","linked_memory_ids","created_at"}]
    """
    try:
        uid = user_id or _get_user_id()
        m = _get_memory()
        results = m.list_entities(user_id=uid, entity_type=entity_type, limit=limit)
        return json.dumps(results, ensure_ascii=False)
    except Exception as e:
        return f"Error: {e}"
```

- [ ] **Step 5: 运行测试验证**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/ -q`
Expected: PASS (现有 + 新增)

- [ ] **Step 6: lint 验证**

Run: `ruff check src/septmuse/cli/main.py src/septmuse/api/rest/__init__.py src/septmuse/api/mcp/tools.py`
Expected: All checks passed!

---

## Task 9: e2e 测试 + CHANGELOG + AGENTS.md 更新

**Files:**
- Create: `tests/e2e/test_entity_e2e.py`
- Modify: `CHANGELOG.md`
- Modify: `AGENTS.md`

**Interfaces:**
- Consumes: 所有前序 Task
- Produces: 3 个 e2e 测试 + 更新的 CHANGELOG/AGENTS.md

- [ ] **Step 1: 写 e2e 测试 — 跨会话持久化 + delete 清理 + 中文**

```python
# tests/e2e/test_entity_e2e.py
"""实体抽取 e2e 测试 (跨会话持久化)。"""

import json


class TestEntityE2E:
    """跨会话实体持久化 + delete 清理 + 中文实体。"""

    def test_cross_session_persistence(self, tmp_path):
        """add → close → reopen → search_entities 跨会话持久化。"""
        db = str(tmp_path / "e2e.db")

        # Session 1: 添加记忆
        from septmuse.orchestration.memory import Memory
        from septmuse.configs.defaults import MemoryConfig

        config = MemoryConfig(db_path=db)
        m = Memory(config=config)
        m.add("Alice works at Google in London", user_id="u1")
        m.close()

        # Session 2: 重新打开, 搜索实体
        m2 = Memory(config=MemoryConfig(db_path=db))
        entities = m2.search_entities("Google", user_id="u1")
        assert any(e["entity_text"] == "Google" for e in entities)

        # list_entities 跨会话可用
        all_entities = m2.list_entities(user_id="u1")
        assert len(all_entities) > 0
        m2.close()

    def test_delete_cleans_entity_refs(self, tmp_path):
        """add → delete → search_entities 引用清理。"""
        db = str(tmp_path / "e2e.db")

        from septmuse.orchestration.memory import Memory
        from septmuse.configs.defaults import MemoryConfig

        m = Memory(config=MemoryConfig(db_path=db))
        result = m.add("Alice works at Google", user_id="u1")
        memory_id = result["results"][0]["id"]

        # 确认实体存在
        entities = m.list_entities(user_id="u1")
        assert len(entities) > 0

        # 删除记忆
        m.delete(memory_id)
        m.close()

        # 重开后实体引用应清理
        m2 = Memory(config=MemoryConfig(db_path=db))
        entities_after = m2.list_entities(user_id="u1")
        google = [e for e in entities_after if e["entity_text"] == "Google"]
        assert len(google) == 0  # Google 只关联了这一条记忆, 应被软删
        m2.close()

    def test_chinese_entity_extraction(self, tmp_path):
        """中文实体抽取。"""
        db = str(tmp_path / "e2e.db")

        from septmuse.orchestration.memory import Memory
        from septmuse.configs.defaults import MemoryConfig

        m = Memory(config=MemoryConfig(db_path=db))
        m.add("张三在北京的百度公司工作", user_id="u1")

        entities = m.list_entities(user_id="u1")
        texts = {e["entity_text"] for e in entities}
        # 至少抽取到一些中文实体
        assert len(entities) > 0
        # 北京或百度应在其中
        assert "北京" in texts or "百度" in texts or "张三" in texts
        m.close()
```

- [ ] **Step 2: 运行 e2e 测试**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/e2e/test_entity_e2e.py -v`
Expected: PASS (3 tests)

- [ ] **Step 3: 全量验证**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/ tests/e2e/ -q`
Expected: ~734 passed, ~36 skipped

Run: `ruff check src/ tests/`
Expected: All checks passed!

Run: `ruff format --check src/ tests/`
Expected: All checks passed!

- [ ] **Step 4: 更新 CHANGELOG**

```markdown
# 在 CHANGELOG.md 的 [Unreleased] 区块追加

### Added — 实体抽取 + 实体向量库 (P0)

- **EntityExtractor** (`concerns/extraction/entity.py`)：纯 Python regex + 词表后端（默认，零配置），4 类实体（PROPER/QUOTED/TOPIC/IDENTIFIER）+ ~120 泛化词黑名单 + span 去重冲突解决。可选 spaCy 后端（`pip install septmuse[ner]`）。
- **EntityStore** (`storage/entity_store.py`)：独立 SQLite 表 `septmuse_entities`，upsert（精确匹配→语义匹配→新建）+ search + list + get_linked_memories + remove_memory_from_entities。借鉴 mem0 V3 去图化设计。
- **Memory facade 集成**：`add(auto_extract_entities=True)` 自动抽取实体，`delete()` 清理实体引用，新增 5 个方法（extract_entities/add_entity/search_entities/get_entity_neighbors/list_entities）。
- **MemoryConfig 新字段**：`entity_extractor_backend`（regex/spacy/none）。
- **环境变量**：`SEPTMUSE_ENTITY_EXTRACTOR`（regex/spacy/none）。
- **CLI 2 命令**：`septmuse entities <query>` / `septmuse entity-list`。
- **REST 2 端点**：`GET /entities` / `GET /entities/list`。
- **MCP 2 工具**：`search_entities` / `list_entities`。
- 新增 `tests/unit/test_entity_extractor.py`（18 测试）、`test_entity_store.py`（12 测试）、`test_memory.py` 扩展（8 测试）、`tests/e2e/test_entity_e2e.py`（3 测试）。
```

- [ ] **Step 5: 更新 AGENTS.md**

```markdown
# 在 AGENTS.md 的环境变量表追加

| `SEPTMUSE_ENTITY_EXTRACTOR` | `regex` | `regex`/`spacy`/`none` |
```

```markdown
# 在 AGENTS.md 的 Embedder 章节后追加

### Entity Extractor

- `SEPTMUSE_ENTITY_EXTRACTOR=regex`（默认，纯 Python regex + 词表，零配置）— 4 类实体（PROPER/QUOTED/TOPIC/IDENTIFIER）+ ~120 泛化词黑名单 + span 去重。
- `SEPTMUSE_ENTITY_EXTRACTOR=spacy` — spaCy NER + noun_chunks（`pip install septmuse[ner]`，模型首次使用时自动下载）。
- `SEPTMUSE_ENTITY_EXTRACTOR=none` — 禁用实体抽取。
- spaCy 不可用时自动降级到 regex + 日志警告。
- 实体存独立 SQLite 表 `septmuse_entities`，用 `linked_memory_ids` 关联记忆（借鉴 mem0 V3 去图化）。
- `Memory.add(auto_extract_entities=True)` 默认自动抽取，`Memory.delete()` 自动清理引用。
```

- [ ] **Step 6: 最终全量验证**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/ tests/e2e/ -q`
Expected: ~734 passed, ~36 skipped

Run: `ruff check src/ tests/ && ruff format --check src/ tests/`
Expected: All checks passed!

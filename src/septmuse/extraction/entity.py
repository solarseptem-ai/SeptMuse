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
"""实体抽取器 (架构文档 §5.1, 借鉴 mem0 entity_extraction.py)。

默认 RegexEntityExtractor (纯 Python regex + 词表, 零配置)。
spacy: pip install septmuse[ner], spaCy NER + noun_chunks。
none: 禁用实体抽取。
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from septmuse.configs.defaults import MemoryConfig
from septmuse.core.logging import get_logger

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


# 泛化词黑名单 (借鉴 mem0 _GENERIC_HEADS / _NON_SPECIFIC_ADJ / _GENERIC_CAPS)
_GENERIC_WORDS = frozenset(
    [
        # English
        "the",
        "this",
        "that",
        "thing",
        "something",
        "person",
        "people",
        "time",
        "way",
        "day",
        "man",
        "woman",
        "world",
        "life",
        "hand",
        "part",
        "place",
        "case",
        "week",
        "year",
        "name",
        "home",
        "work",
        "word",
        "point",
        "group",
        "number",
        "fact",
        "idea",
        "issue",
        "side",
        "kind",
        "head",
        "line",
        "end",
        "member",
        "list",
        "lot",
        "other",
        "use",
        "first",
        "last",
        "new",
        "old",
        "good",
        "bad",
        "big",
        "small",
        "own",
        "same",
        "some",
        "any",
        "all",
        "no",
        "every",
        "one",
        "two",
        "three",
        # Chinese
        "这个",
        "那个",
        "什么",
        "东西",
        "事情",
        "地方",
        "时候",
        "时间",
        "人",
        "他们",
        "我们",
        "你们",
        "它们",
        "自己",
        "别人",
        "大家",
        "所有",
        "一些",
        "一点",
        "一下",
        "一样",
        "这样",
        "那样",
        "里",
        "中",
        "上",
        "下",
    ]
)

# 中文百家姓 (前 100 常见)
_CHINESE_SURNAMES = frozenset(
    "赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华金魏陶姜戚谢邹喻柏水窦章云苏潘葛奚范彭郎鲁韦昌马苗凤花方俞任袁柳鲍史唐费廉岑薛雷贺倪汤滕殷罗毕郝邬安常乐于时傅皮卞齐康伍余元卜顾孟平黄穆萧尹"
)

# 常见中国地名 (直辖市/省会/主要城市; 姓氏表无法覆盖 "北京" 等地名)
_CHINESE_PLACES = frozenset(
    [
        "乌鲁木齐",
        "哈尔滨",
        "石家庄",
        "呼和浩特",
        "北京",
        "上海",
        "天津",
        "重庆",
        "南京",
        "苏州",
        "杭州",
        "宁波",
        "合肥",
        "福州",
        "厦门",
        "南昌",
        "济南",
        "青岛",
        "郑州",
        "武汉",
        "长沙",
        "广州",
        "深圳",
        "南宁",
        "海口",
        "成都",
        "贵阳",
        "昆明",
        "西安",
        "兰州",
        "西宁",
        "银川",
        "拉萨",
        "长春",
        "沈阳",
        "大连",
        "太原",
    ]
)

# 中文虚词/停用词 (用于 TOPIC 过滤; 姓氏表和地名表无法覆盖 "和/的/了" 等连接词)
# 仅含几乎绝不出现在名词短语中的字 (纯虚词/介词/数词/安全动词); 避开 "能/会/变/改/想" 等可组名词的字
_CHINESE_STOPWORDS = frozenset(
    "和与及或同跟地得了着过吗呢吧啊呀在从到向对为以于把被给连之这那哪其就是也都还又再已将正才只便"
    "一二三四五六七八九十百千万去来说做看起些个"
)

_PROPER_EN_RE = re.compile(r"\b([A-Z][a-z]+)\b")
_PROPER_ZH_RE = re.compile(rf"[{_CHINESE_SURNAMES}][\u4e00-\u9fff]{{1,2}}?")
_PROPER_ZH_PLACE_RE = re.compile("|".join(sorted(_CHINESE_PLACES, key=len, reverse=True)))
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
        candidates.extend(self._extract_topic(text))
        return self._resolve_candidates(candidates)

    def _extract_proper(self, text: str) -> list[Entity]:
        result = []
        for m in _PROPER_EN_RE.finditer(text):
            word = m.group(1)
            if _normalize_entity_text(word) not in _GENERIC_WORDS:
                result.append(Entity(text=word, entity_type="PROPER", start=m.start(1), end=m.end(1)))
        for m in _PROPER_ZH_RE.finditer(text):
            name = m.group(0)
            if name not in _GENERIC_WORDS:
                result.append(Entity(text=name, entity_type="PROPER", start=m.start(), end=m.end()))
        for m in _PROPER_ZH_PLACE_RE.finditer(text):
            place = m.group(0)
            result.append(Entity(text=place, entity_type="PROPER", start=m.start(), end=m.end()))
        return result

    def _extract_quoted(self, text: str) -> list[Entity]:
        result = []
        for m in _QUOTED_RE.finditer(text):
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
        result: list[Entity] = []
        # 英文连续大写开头词组: [A-Z][a-z]+ (?:\s+[A-Z][a-z]+)+
        topic_en_re = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")
        for m in topic_en_re.finditer(text):
            phrase = m.group(1)
            # 排除全相同词的重复 (如 "Google Google Google"); 至少 2 个不同词
            if _normalize_entity_text(phrase) not in _GENERIC_WORDS and len(set(phrase.split())) >= 2:
                result.append(Entity(text=phrase, entity_type="TOPIC", start=m.start(1), end=m.end(1)))
        # 中文连续名词短语 (2-4 字无标点连续, 非人名/地名)
        topic_zh_re = re.compile(r"[\u4e00-\u9fff]{2,4}")
        for m in topic_zh_re.finditer(text):
            phrase = m.group(0)
            # 排除含虚词/连接词的短语 (如 "张三和李"); 避免与 PROPER (人名) 重叠 — 百家姓开头的 2 字跳过
            if (
                phrase not in _GENERIC_WORDS
                and phrase not in _CHINESE_SURNAMES
                and phrase not in _CHINESE_PLACES
                and not any(ch in _CHINESE_STOPWORDS for ch in phrase)
                and (len(phrase) >= 3 or phrase[0] not in _CHINESE_SURNAMES)
            ):
                result.append(Entity(text=phrase, entity_type="TOPIC", start=m.start(), end=m.end()))
        return result

    @staticmethod
    def _resolve_candidates(candidates: list[Entity]) -> list[Entity]:
        """span 去重冲突解决 (借鉴 mem0 _resolve_candidates)。

        1. 按 start 排序
        2. 同一 (text, entity_type) 只保留第一个
        3. 跨类型冲突: 长 span 优先, 相同长度时 PROPER > QUOTED > TOPIC > IDENTIFIER
        """
        if not candidates:
            return []
        candidates.sort(key=lambda e: (e.start, -(e.end - e.start)))
        seen: set[tuple[str, str]] = set()
        deduped: list[Entity] = []
        for e in candidates:
            key = (_normalize_entity_text(e.text), e.entity_type)
            if key not in seen:
                seen.add(key)
                deduped.append(e)
        type_priority = {"PROPER": 4, "QUOTED": 3, "TOPIC": 2, "IDENTIFIER": 1}
        result: list[Entity] = []
        occupied: list[tuple[int, int]] = []
        for e in sorted(deduped, key=lambda x: (-(x.end - x.start), -type_priority.get(x.entity_type, 0))):
            overlap = any(not (e.end <= s or e.start >= end) for s, end in occupied)
            if not overlap:
                result.append(e)
                occupied.append((e.start, e.end))
        return sorted(result, key=lambda e: e.start)


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
        """spaCy NER + noun_chunks 抽取。"""
        doc = self._nlp(text)
        candidates: list[Entity] = []
        # NER 实体
        ner_labels = {"PERSON", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART"}
        for ent in doc.ents:
            if ent.label_ in ner_labels and _normalize_entity_text(ent.text) not in _GENERIC_WORDS:
                candidates.append(Entity(text=ent.text, entity_type="PROPER", start=ent.start_char, end=ent.end_char))
        # 引号文本 (同 regex)
        for m in _QUOTED_RE.finditer(text):
            quoted_text = next(g for g in m.groups() if g is not None)
            candidates.append(Entity(text=quoted_text, entity_type="QUOTED", start=m.start(), end=m.end()))
        # noun_chunks 作为 TOPIC
        for chunk in doc.noun_chunks:
            if _normalize_entity_text(chunk.text) not in _GENERIC_WORDS and len(chunk.text.split()) >= 2:
                candidates.append(
                    Entity(text=chunk.text, entity_type="TOPIC", start=chunk.start_char, end=chunk.end_char)
                )
        # 技术标识符 (同 regex)
        for m in _IDENTIFIER_DOTTED_RE.finditer(text):
            candidates.append(Entity(text=m.group(1), entity_type="IDENTIFIER", start=m.start(1), end=m.end(1)))
        for m in _IDENTIFIER_CAMEL_RE.finditer(text):
            candidates.append(Entity(text=m.group(1), entity_type="IDENTIFIER", start=m.start(1), end=m.end(1)))
        for m in _IDENTIFIER_SNAKE_RE.finditer(text):
            ident = m.group(1)
            if _normalize_entity_text(ident) not in _GENERIC_WORDS:
                candidates.append(Entity(text=ident, entity_type="IDENTIFIER", start=m.start(1), end=m.end(1)))
        return RegexEntityExtractor._resolve_candidates(candidates)


def _resolve_entity_extractor(config: MemoryConfig) -> EntityExtractor | None:
    """解析实体抽取器 (通过 ServiceProvider, 类似 _resolve_embedder 模式)。

    默认 RegexEntityExtractor (零配置, 纯 Python)。
    spacy: pip install septmuse[ner], spaCy NER + noun_chunks。
    none: 禁用实体抽取。
    """
    from septmuse.services.providers import entity_extractor_provider

    backend = config.entity_extractor.backend.lower()
    if backend == "nlp":
        backend = "spacy"
    try:
        return entity_extractor_provider.resolve(backend, config=config.entity_extractor)
    except (ImportError, OSError) as e:
        if backend == "spacy":
            logger.warning("entity_extractor_spacy_unavailable", error=str(e), fallback="regex")
            return entity_extractor_provider.resolve("regex", config=config.entity_extractor)
        raise

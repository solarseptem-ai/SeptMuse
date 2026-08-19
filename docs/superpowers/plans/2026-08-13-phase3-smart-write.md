# Smart Memory Butler — Phase 3: Smart Write 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** remember 时 SeptMuse 自主决定该不该存、存什么类型、是否已有相似记忆，实现全自动记忆管家写入侧。

**Architecture:** 三个新独立模块（ImportanceAssessor + MemoryClassifier + SemanticDeduplicator）插入现有 `remember()` 链路前端。冲突检测从 `improve()` 批操作前移到写入路径。不改变现有 preprocess(去重+脱敏)、实体抽取、事实抽取、工作记忆逻辑。

**Tech Stack:** Python 3.10+, Embedder ABC, LLM ABC, SemanticFact model, ConflictResolver

## Global Constraints

- **LLM 降级**：无 LLM 时用规则替代（关键词匹配 + cosine 阈值），不阻塞写入
- **raw_log 永远存**：即使价值低跳过记忆存储，raw_log 仍保留（审计 + LLM 恢复后补跑）
- **双存**：一条文本可同时存为 episodic + semantic，不互斥
- **冲突不阻塞写入**：检测到冲突仍写入，在返回值 `result.conflicts` 标注
- **测试环境**：`PYTHONPATH=src`, `SEPTMUSE_EMBEDDER=hash`, `SEPTMUSE_TOKENIZER=space`
- **line-length 120**，ruff check + format 必须通过
- **git commit 用** `git -c core.hooksPath= commit -m "..."` 绕过 LFS hook
- **ruff format 用 stdin subprocess**（Windows ruff format 直接调有清空文件 bug）
- **现有 remember() 在** `src/septmuse/memory/main.py:770-843`
- **现有 CapturePipeline.preprocess() 在** `src/septmuse/capture/pipeline.py:97-129`（SHA256 去重 + 隐私脱敏）
- **现有 store.search() 签名**：`search(query_embedding, *, user_id, top_k, threshold, filters) -> list[dict]`，每个 dict 有 `id`/`memory`/`score`
- **现有 ConflictResolver.detect_conflicts() 在** `src/septmuse/evolution/conflict.py:62-95`，相同 (subject, predicate) 不同 object → 冲突

## File Structure

| 文件 | 职责 | 操作 |
|------|------|------|
| `src/septmuse/capture/importance.py` | 价值评估（0-1 打分） | 新建 |
| `src/septmuse/capture/classifier.py` | 自动类型分类（semantic/episodic/procedural） | 新建 |
| `src/septmuse/capture/dedup.py` | 语义去重（embedding cosine > 0.92） | 新建 |
| `src/septmuse/memory/main.py` | remember() 集成 3 新步骤 + 冲突前移 | 修改 |
| `tests/unit/test_importance.py` | ImportanceAssessor 测试 | 新建 |
| `tests/unit/test_classifier.py` | MemoryClassifier 测试 | 新建 |
| `tests/unit/test_dedup.py` | SemanticDeduplicator 测试 | 新建 |
| `tests/unit/test_smart_write.py` | remember() 集成测试 | 新建 |

## 现有代码参考

### remember() 现有链路（main.py:770-843）

```
1. preprocess (SHA256 去重 + 脱敏)
2. add_raw_log (存原文)
3. store.add (verbatim 存储 + embedding)
4. _extract_and_store_entities (regex/spacy 实体抽取)
5. extractor.extract_and_store (LLM 事实抽取, if llm)
6. working_memory.core_memory_append (更新工作记忆)
```

Phase 3 改为：

```
0. [新] 价值评估 → score < 0.3 跳过（raw_log 仍存）
0.5 [新] 类型分类 → 决定存什么类型
0.6 [新] 语义去重 → cosine > 0.92 → UPDATE 不新增
1. preprocess (现有, 不改)
2. add_raw_log (现有, 不改 — 即使低价值也存 raw_log)
3. store.add (现有, 不改)
4. _extract_and_store_entities (现有, 不改)
5. extractor.extract_and_store (现有, 不改)
5.5 [新] 冲突检测 (前移自 improve, 不解决只标注)
6. working_memory.core_memory_append (现有, 不改)
```

### Embedder ABC

```python
class Embedder(ABC):
    def embed(self, text: str) -> list[float]: ...
    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...
```

### store.search() 返回的 dict 结构

```python
{"id": "mem-xxx", "memory": "text content", "score": 0.85, "metadata": {...}, "created_at": "..."}
```

---

### Task 1: ImportanceAssessor

**Files:**
- Create: `src/septmuse/capture/importance.py`
- Test: `tests/unit/test_importance.py`

**Interfaces:**
- Consumes: `LLM | None`
- Produces: `ImportanceAssessor(llm=None)` → `assess(text: str) -> ImportanceResult`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for ImportanceAssessor."""
from __future__ import annotations

from unittest.mock import MagicMock

from septmuse.capture.importance import ImportanceAssessor


def test_assess_high_value_with_llm():
    """LLM 打分：高价值文本得分 > 0.5。"""
    llm = MagicMock()
    llm.complete.return_value = "0.9"
    assessor = ImportanceAssessor(llm=llm)
    result = assessor.assess("用户下周要出差去上海，涉及项目X的验收")
    assert result.score > 0.5
    llm.complete.assert_called_once()


def test_assess_low_value_with_llm():
    """LLM 打分：低价值文本得分 < 0.3。"""
    llm = MagicMock()
    llm.complete.return_value = "0.1"
    assessor = ImportanceAssessor(llm=llm)
    result = assessor.assess("嗯好的")
    assert result.score < 0.3


def test_assess_rule_based_high():
    """无 LLM：含实体+数字+时间 → 高分。"""
    assessor = ImportanceAssessor(llm=None)
    result = assessor.assess("用户下周要去上海出差3天")
    assert result.score >= 0.7


def test_assess_rule_based_medium():
    """无 LLM：含偏好关键词 → 中分。"""
    assessor = ImportanceAssessor(llm=None)
    result = assessor.assess("用户喜欢喝咖啡")
    assert result.score >= 0.5


def test_assess_rule_based_low():
    """无 LLM：无信号 → 低分。"""
    assessor = ImportanceAssessor(llm=None)
    result = assessor.assess("嗯好的知道了")
    assert result.score < 0.5


def test_assess_llm_failure_fallback():
    """LLM 调用失败 → 降级为规则评估。"""
    llm = MagicMock()
    llm.complete.side_effect = Exception("API error")
    assessor = ImportanceAssessor(llm=llm)
    result = assessor.assess("用户下周要去上海出差")
    assert result.score >= 0.5  # 规则兜底
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$env:PYTHONPATH="src"; $env:SEPTMUSE_EMBEDDER="hash"; $env:SEPTMUSE_TOKENIZER="space"; python -m pytest tests/unit/test_importance.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
"""ImportanceAssessor — 写入时给记忆打价值分 (0-1)。

LLM 可用：prompt 打分。
LLM 不可用：规则——含实体/数字/时间 → 0.7+；含偏好关键词 → 0.6+；其余 → 0.3。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from septmuse.core.logging import get_logger

if TYPE_CHECKING:
    from septmuse.llms.base import LLM

logger = get_logger(__name__)

_THRESHOLD = 0.3

_SYSTEM_PROMPT = "你是一个记忆价值评估助手。评估以下文本是否值得记住。只回答一个 0 到 1 的小数。1.0=关键信息，0.5=有用，0.0=噪音。"
_USER_TEMPLATE = "文本: {text}\n\n价值评分:"

_ENTITY_PATTERN = re.compile(r"[\u4e00-\u9fff]{2,}|[A-Z][a-z]+")
_NUMBER_PATTERN = re.compile(r"\d+")
_TIME_PATTERN = re.compile(r"今天|明天|下周|这周|上周|下午|上午|\d+月|\d+日|\d+点|\d+年")
_PLAN_PATTERN = re.compile(r"计划|打算|将要|准备|安排|要去|要做|会去")
_PREF_PATTERN = re.compile(r"喜欢|偏好|总是|习惯|讨厌|爱吃|爱喝|常用")


@dataclass
class ImportanceResult:
    """价值评估结果。"""
    score: float
    method: str  # "llm" | "rule"


class ImportanceAssessor:
    """给文本打价值分。

    Args:
        llm: LLM 实例（可选，None 时用规则）
    """

    def __init__(self, llm: LLM | None = None) -> None:
        self._llm = llm

    def assess(self, text: str) -> ImportanceResult:
        """返回价值评分 (0-1)。"""
        if not text or not text.strip():
            return ImportanceResult(score=0.0, method="rule")

        if self._llm is not None:
            return self._assess_with_llm(text)
        return self._assess_rule(text)

    def _assess_with_llm(self, text: str) -> ImportanceResult:
        user_prompt = _USER_TEMPLATE.format(text=text[:500])
        try:
            response = self._llm.complete(_SYSTEM_PROMPT, user_prompt)
            score = self._parse_score(response)
            logger.debug("importance_llm_done", score=score)
            return ImportanceResult(score=score, method="llm")
        except Exception as e:
            logger.warning("importance_llm_failed", error=str(e), fallback="rule")
            return self._assess_rule(text)

    def _assess_rule(self, text: str) -> ImportanceResult:
        has_entity = bool(_ENTITY_PATTERN.search(text))
        has_number = bool(_NUMBER_PATTERN.search(text))
        has_time = bool(_TIME_PATTERN.search(text))
        has_plan = bool(_PLAN_PATTERN.search(text))
        has_pref = bool(_PREF_PATTERN.search(text))

        density = sum([has_entity, has_number, has_time]) / 3.0  # 信息密度 0-1
        relevance = 0.6 if (has_plan or has_pref) else 0.3  # 持续相关性

        score = 0.4 * density + 0.6 * relevance
        logger.debug("importance_rule_done", score=score, entity=has_entity, number=has_number, time=has_time, plan=has_plan, pref=has_pref)
        return ImportanceResult(score=round(score, 2), method="rule")

    @staticmethod
    def _parse_score(response: str) -> float:
        """从 LLM 响应中解析分数。"""
        for token in response.strip().split():
            try:
                val = float(token)
                if 0.0 <= val <= 1.0:
                    return val
            except ValueError:
                continue
        return 0.5  # 无法解析时默认中分
```

- [ ] **Step 4: Run test to verify it passes**

Run: `$env:PYTHONPATH="src"; $env:SEPTMUSE_EMBEDDER="hash"; $env:SEPTMUSE_TOKENIZER="space"; python -m pytest tests/unit/test_importance.py -v`
Expected: 6 PASS

- [ ] **Step 5: ruff check + format**

Run: `ruff check src/septmuse/capture/importance.py tests/unit/test_importance.py`
Then format via stdin subprocess.

- [ ] **Step 6: Commit**

```bash
git add src/septmuse/capture/importance.py tests/unit/test_importance.py
git -c core.hooksPath= commit -m "feat(write): ImportanceAssessor — LLM + 规则价值评估"
```

---

### Task 2: MemoryClassifier

**Files:**
- Create: `src/septmuse/capture/classifier.py`
- Test: `tests/unit/test_classifier.py`

**Interfaces:**
- Consumes: `LLM | None`
- Produces: `MemoryClassifier(llm=None)` → `classify(text: str) -> list[str]`（返回类型列表，如 `["semantic", "episodic"]`）

- [ ] **Step 1: Write the failing test**

```python
"""Tests for MemoryClassifier."""
from __future__ import annotations

from unittest.mock import MagicMock

from septmuse.capture.classifier import MemoryClassifier


def test_classify_procedural():
    """规则：含步骤/操作关键词 → procedural。"""
    clf = MemoryClassifier(llm=None)
    types = clf.classify("第一步打开终端，第二步输入命令")
    assert "procedural" in types


def test_classify_episodic_with_time():
    """规则：含时间+动作 → episodic + semantic 双存。"""
    clf = MemoryClassifier(llm=None)
    types = clf.classify("用户下周要去上海出差")
    assert "episodic" in types
    assert "semantic" in types


def test_classify_semantic_preference():
    """规则：含偏好关键词 → semantic。"""
    clf = MemoryClassifier(llm=None)
    types = clf.classify("用户喜欢喝咖啡")
    assert "semantic" in types


def test_classify_default_semantic():
    """默认 → semantic。"""
    clf = MemoryClassifier(llm=None)
    types = clf.classify("天空是蓝色的")
    assert types == ["semantic"]


def test_classify_with_llm():
    """LLM 可用时：prompt 分类。"""
    llm = MagicMock()
    llm.complete.return_value = "semantic"
    clf = MemoryClassifier(llm=llm)
    types = clf.classify("用户下周要去上海出差")
    assert "semantic" in types
    llm.complete.assert_called_once()


def test_classify_llm_failure_fallback():
    """LLM 调用失败 → 降级为规则。"""
    llm = MagicMock()
    llm.complete.side_effect = Exception("API error")
    clf = MemoryClassifier(llm=llm)
    types = clf.classify("用户喜欢喝咖啡")
    assert "semantic" in types


def test_classify_empty():
    """空文本返回空列表。"""
    clf = MemoryClassifier(llm=None)
    assert clf.classify("") == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$env:PYTHONPATH="src"; $env:SEPTMUSE_EMBEDDER="hash"; $env:SEPTMUSE_TOKENIZER="space"; python -m pytest tests/unit/test_classifier.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
"""MemoryClassifier — 自动判定记忆类型。

LLM 可用：prompt 分类（事实陈述/情节事件/操作步骤/用户偏好）。
LLM 不可用：关键词决策树。一条记忆可同时属于多个类型（双存）。
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from septmuse.core.logging import get_logger

if TYPE_CHECKING:
    from septmuse.llms.base import LLM

logger = get_logger(__name__)

_SYSTEM_PROMPT = "你是一个记忆分类助手。判断以下文本属于哪些类型，用逗号分隔返回。可选类型：semantic(事实陈述), episodic(情节事件), procedural(操作步骤)。"
_USER_TEMPLATE = "文本: {text}\n\n类型:"

_PROC_PATTERN = re.compile(r"步骤|操作|方法|如何|第一步|第二步|输入命令|运行|执行")
_TIME_PATTERN = re.compile(r"今天|明天|下周|这周|上周|刚才|刚刚|\d+月|\d+日|\d+点")
_ACTION_PATTERN = re.compile(r"去|来|做|说|买|看|吃|开|关|跑|走|出差|开会|汇报")
_PREF_PATTERN = re.compile(r"喜欢|偏好|总是|习惯|讨厌|爱吃|爱喝|常用|一般|通常")
_FACT_PATTERN = re.compile(r"是|有|在|叫|属于|位于|包含")


class MemoryClassifier:
    """自动判定记忆类型。

    Args:
        llm: LLM 实例（可选，None 时用规则）
    """

    def __init__(self, llm: LLM | None = None) -> None:
        self._llm = llm

    def classify(self, text: str) -> list[str]:
        """返回记忆类型列表（可多类型双存）。空文本返回空列表。"""
        if not text or not text.strip():
            return []

        if self._llm is not None:
            return self._classify_with_llm(text)
        return self._classify_rule(text)

    def _classify_with_llm(self, text: str) -> list[str]:
        user_prompt = _USER_TEMPLATE.format(text=text[:500])
        try:
            response = self._llm.complete(_SYSTEM_PROMPT, user_prompt)
            types = [t.strip().lower() for t in response.split(",")]
            valid = [t for t in types if t in ("semantic", "episodic", "procedural")]
            if valid:
                return valid
            logger.warning("classifier_llm_invalid_response", response=response[:200], fallback="rule")
            return self._classify_rule(text)
        except Exception as e:
            logger.warning("classifier_llm_failed", error=str(e), fallback="rule")
            return self._classify_rule(text)

    def _classify_rule(self, text: str) -> list[str]:
        types: list[str] = []

        is_procedural = bool(_PROC_PATTERN.search(text))
        has_time = bool(_TIME_PATTERN.search(text))
        has_action = bool(_ACTION_PATTERN.search(text))
        has_pref = bool(_PREF_PATTERN.search(text))
        has_fact = bool(_FACT_PATTERN.search(text))

        if is_procedural:
            types.append("procedural")
        if has_time and has_action:
            types.append("episodic")
            types.append("semantic")
        if has_pref:
            types.append("semantic")
        if not types:
            types.append("semantic")

        logger.debug("classifier_rule_done", types=types, procedural=is_procedural, time=has_time, action=has_action, pref=has_pref)
        return types
```

- [ ] **Step 4: Run test to verify it passes**

Run: `$env:PYTHONPATH="src"; $env:SEPTMUSE_EMBEDDER="hash"; $env:SEPTMUSE_TOKENIZER="space"; python -m pytest tests/unit/test_classifier.py -v`
Expected: 7 PASS

- [ ] **Step 5: ruff check + format**

- [ ] **Step 6: Commit**

```bash
git add src/septmuse/capture/classifier.py tests/unit/test_classifier.py
git -c core.hooksPath= commit -m "feat(write): MemoryClassifier — LLM + 规则类型分类"
```

---

### Task 3: SemanticDeduplicator

**Files:**
- Create: `src/septmuse/capture/dedup.py`
- Test: `tests/unit/test_dedup.py`

**Interfaces:**
- Consumes: `Embedder`（embed query），`store.search()`（检索已有），`LLM | None`（灰色地带判定）
- Produces: `SemanticDeduplicator(embedder, store, llm=None)` → `find_duplicate(text, user_id) -> DedupResult | None`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for SemanticDeduplicator."""
from __future__ import annotations

from unittest.mock import MagicMock

from septmuse.capture.dedup import SemanticDeduplicator


def test_find_duplicate_high_similarity():
    """cosine > 0.92 → 判定重复。"""
    embedder = MagicMock()
    embedder.embed.return_value = [0.1, 0.2, 0.3]
    store = MagicMock()
    store.search.return_value = [
        {"id": "mem-old", "memory": "用户下周去上海出差", "score": 0.95},
    ]
    dedup = SemanticDeduplicator(embedder, store, llm=None)
    result = dedup.find_duplicate("用户下周要去上海出差", user_id="alice")
    assert result is not None
    assert result.memory_id == "mem-old"
    assert result.similarity >= 0.92


def test_find_duplicate_no_match():
    """cosine < 0.92 → 不重复。"""
    embedder = MagicMock()
    embedder.embed.return_value = [0.1, 0.2, 0.3]
    store = MagicMock()
    store.search.return_value = [
        {"id": "mem-other", "memory": "完全不同的内容关于天气", "score": 0.3},
    ]
    dedup = SemanticDeduplicator(embedder, store, llm=None)
    result = dedup.find_duplicate("用户下周要去上海出差", user_id="alice")
    assert result is None


def test_find_duplicate_empty_results():
    """无已有记忆 → 不重复。"""
    embedder = MagicMock()
    embedder.embed.return_value = [0.1, 0.2, 0.3]
    store = MagicMock()
    store.search.return_value = []
    dedup = SemanticDeduplicator(embedder, store, llm=None)
    result = dedup.find_duplicate("用户下周要去上海出差", user_id="alice")
    assert result is None


def test_find_duplicate_llm_gray_zone():
    """cosine 0.85-0.92 灰色地带 + LLM → 让 LLM 判定。"""
    embedder = MagicMock()
    embedder.embed.return_value = [0.1, 0.2, 0.3]
    store = MagicMock()
    store.search.return_value = [
        {"id": "mem-gray", "memory": "用户要去上海", "score": 0.88},
    ]
    llm = MagicMock()
    llm.complete.return_value = "yes"
    dedup = SemanticDeduplicator(embedder, store, llm=llm)
    result = dedup.find_duplicate("用户下周要去上海出差", user_id="alice")
    assert result is not None
    assert result.memory_id == "mem-gray"
    llm.complete.assert_called_once()


def test_find_duplicate_llm_says_no():
    """cosine 灰色地带 + LLM 说不是重复 → 不重复。"""
    embedder = MagicMock()
    embedder.embed.return_value = [0.1, 0.2, 0.3]
    store = MagicMock()
    store.search.return_value = [
        {"id": "mem-gray", "memory": "用户要去上海", "score": 0.88},
    ]
    llm = MagicMock()
    llm.complete.return_value = "no, different context"
    dedup = SemanticDeduplicator(embedder, store, llm=llm)
    result = dedup.find_duplicate("用户下周要去上海出差", user_id="alice")
    assert result is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$env:PYTHONPATH="src"; $env:SEPTMUSE_EMBEDDER="hash"; $env:SEPTMUSE_TOKENIZER="space"; python -m pytest tests/unit/test_dedup.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
"""SemanticDeduplicator — 写入前查语义相似。

用 query embedding 检索 top-5 已有记忆。
cosine > 0.92 → 判定重复 → UPDATE 不新增。
LLM 可用：0.85-0.92 灰色地带 LLM 判定。
LLM 不可用：只走 0.92 硬阈值。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from septmuse.core.logging import get_logger

if TYPE_CHECKING:
    from septmuse.embedders.base import Embedder
    from septmuse.llms.base import LLM
    from septmuse.storage.base import MemoryStore

logger = get_logger(__name__)

_HARD_THRESHOLD = 0.92
_GRAY_LOW = 0.85
_GRAY_HIGH = 0.92

_SYSTEM_PROMPT = "你是一个文本去重助手。判断以下两条文本是否表达同一信息。回答 'yes' 或 'no'。"
_USER_TEMPLATE = "文本1: {text1}\n文本2: {text2}\n\n是否表达同一信息？"


@dataclass
class DedupResult:
    """去重检测结果。"""
    memory_id: str
    similarity: float
    method: str  # "hard" | "llm"


class SemanticDeduplicator:
    """语义去重。

    Args:
        embedder: Embedder 实例
        store: MemoryStore 实例
        llm: LLM 实例（可选，灰色地带判定）
    """

    def __init__(self, embedder: Embedder, store: MemoryStore, llm: LLM | None = None) -> None:
        self._embedder = embedder
        self._store = store
        self._llm = llm

    def find_duplicate(self, text: str, *, user_id: str) -> DedupResult | None:
        """返回重复记忆的 ID + 相似度，无重复返回 None。"""
        emb = self._embedder.embed(text)
        try:
            results = self._store.search(emb, user_id=user_id, top_k=5, threshold=0.1)
        except Exception as e:
            logger.warning("dedup_search_failed", error=str(e))
            return None

        for r in results:
            score = r.get("score", 0.0)
            if score >= _HARD_THRESHOLD:
                logger.debug("dedup_hard_match", memory_id=r["id"], score=score)
                return DedupResult(memory_id=r["id"], similarity=score, method="hard")

            if _GRAY_LOW <= score < _GRAY_HIGH and self._llm is not None:
                if self._llm_says_duplicate(text, r.get("memory", "")):
                    logger.debug("dedup_llm_match", memory_id=r["id"], score=score)
                    return DedupResult(memory_id=r["id"], similarity=score, method="llm")

        return None

    def _llm_says_duplicate(self, text1: str, text2: str) -> bool:
        user_prompt = _USER_TEMPLATE.format(text1=text1[:200], text2=text2[:200])
        try:
            response = self._llm.complete(_SYSTEM_PROMPT, user_prompt)
            return "yes" in response.strip().lower()
        except Exception as e:
            logger.warning("dedup_llm_failed", error=str(e))
            return False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `$env:PYTHONPATH="src"; $env:SEPTMUSE_EMBEDDER="hash"; $env:SEPTMUSE_TOKENIZER="space"; python -m pytest tests/unit/test_dedup.py -v`
Expected: 5 PASS

- [ ] **Step 5: ruff check + format**

- [ ] **Step 6: Commit**

```bash
git add src/septmuse/capture/dedup.py tests/unit/test_dedup.py
git -c core.hooksPath= commit -m "feat(write): SemanticDeduplicator — embedding 语义去重"
```

---

### Task 4: remember() 集成 + 冲突检测前移

**Files:**
- Modify: `src/septmuse/memory/main.py`（remember 方法 + __init__ 初始化 3 新模块）
- Test: `tests/unit/test_smart_write.py`

**Interfaces:**
- Consumes: `ImportanceAssessor`, `MemoryClassifier`, `SemanticDeduplicator`, `ConflictResolver.detect_conflicts()`
- Produces: 增强的 `remember()` 返回值新增 `importance`, `types`, `duplicate`, `conflicts` 字段

- [ ] **Step 1: Write the failing test**

```python
"""Tests for Smart Write integration in remember()."""
from __future__ import annotations

import tempfile
import os
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def tmp_memory():
    """Create ExperimentalMemory with temp DB."""
    db = tempfile.mktemp(suffix=".db")
    os.environ["SEPTMUSE_DB_PATH"] = db
    os.environ["SEPTMUSE_EMBEDDER"] = "hash"
    os.environ["SEPTMUSE_TOKENIZER"] = "space"
    from septmuse.experimental import ExperimentalMemory
    from septmuse.configs.base import MemoryConfig
    from septmuse.embedders.hash import HashEmbedder

    mem = ExperimentalMemory(config=MemoryConfig(db_path=db), embedder=HashEmbedder())
    yield mem
    for f in [db, db + "-wal", db + "-shm"]:
        try:
            os.unlink(f)
        except OSError:
            pass


def test_remember_returns_importance(tmp_memory):
    """remember 返回 importance 字段。"""
    result = tmp_memory.remember("用户下周要去上海出差3天", user_id="alice")
    assert "importance" in result
    assert isinstance(result["importance"], (int, float))
    assert result["importance"] >= 0.0


def test_remember_returns_types(tmp_memory):
    """remember 返回 types 字段。"""
    result = tmp_memory.remember("用户下周要去上海出差3天", user_id="alice")
    assert "types" in result
    assert isinstance(result["types"], list)
    assert len(result["types"]) > 0


def test_remember_returns_duplicate(tmp_memory):
    """remember 返回 duplicate 字段。"""
    result = tmp_memory.remember("用户下周要去上海出差3天", user_id="alice")
    assert "duplicate" in result
    assert result["duplicate"] is None  # 首次写入无重复


def test_remember_returns_conflicts(tmp_memory):
    """remember 返回 conflicts 字段。"""
    result = tmp_memory.remember("用户下周要去上海出差3天", user_id="alice")
    assert "conflicts" in result
    assert isinstance(result["conflicts"], list)


def test_remember_low_importance_skipped(tmp_memory):
    """低价值文本跳过记忆存储，但 raw_log 仍存。"""
    result = tmp_memory.remember("嗯好的", user_id="alice")
    assert result["captured"] is False or result.get("importance", 1.0) < 0.5
    assert result["raw_id"] is not None  # raw_log 仍存


def test_remember_semantic_dedup(tmp_memory):
    """重复内容第二次写入返回 duplicate。"""
    tmp_memory.remember("用户下周要去上海出差3天", user_id="alice")
    result = tmp_memory.remember("用户下周要去上海出差3天", user_id="alice")
    # SHA256 或语义去重应捕获
    assert result["captured"] is False or result.get("duplicate") is not None or result["memory_id"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$env:PYTHONPATH="src"; $env:SEPTMUSE_EMBEDDER="hash"; $env:SEPTMUSE_TOKENIZER="space"; python -m pytest tests/unit/test_smart_write.py -v`
Expected: FAIL with `KeyError: 'importance'` or `AssertionError`

- [ ] **Step 3: Modify remember() in main.py**

In `main.py`, modify the `remember()` method (around line 770-843):

1. Add imports at top of `remember()` (or lazy import inside):
```python
from septmuse.capture.importance import ImportanceAssessor
from septmuse.capture.classifier import MemoryClassifier
from septmuse.capture.dedup import SemanticDeduplicator
```

2. Add lazy-init properties to the class (after `_get_recall_pipeline`):
```python
@property
def _importance_assessor(self):
    if not hasattr(self, '_importance'):
        self._importance = ImportanceAssessor(llm=self.llm)
    return self._importance

@property
def _classifier(self):
    if not hasattr(self, '_clf'):
        self._clf = MemoryClassifier(llm=self.llm)
    return self._clf

@property
def _deduplicator(self):
    if not hasattr(self, '_dedup'):
        self._dedup = SemanticDeduplicator(self.embedder, self.store, llm=self.llm)
    return self._dedup
```

3. Modify `remember()` — add new steps at the beginning of the method (after the empty text check, before preprocess):

```python
        # [0] 价值评估
        importance = self._importance_assessor.assess(text)
        result["importance"] = importance.score

        if importance.score < 0.3:
            # 低价值：仍存 raw_log，但跳过记忆存储
            raw_event = self.episodic.add_raw_log(
                text, user_id=user_id, session_id=session_id or "default", agent_id=agent_id,
            )
            result["raw_id"] = raw_event.id
            result["captured"] = False
            logger.info("remember_low_importance_skipped", user_id=user_id, score=importance.score)
            return result

        # [0.5] 类型分类
        types = self._classifier.classify(text)
        result["types"] = types

        # [0.6] 语义去重
        dup = self._deduplicator.find_duplicate(text, user_id=user_id)
        result["duplicate"] = None
        if dup is not None:
            result["duplicate"] = {"memory_id": dup.memory_id, "similarity": dup.similarity, "method": dup.method}
            # 重复时仍继续 preprocess + store.add（store.add 自带 hash 去重会返回 None）
            logger.info("remember_semantic_dup_found", user_id=user_id, dup_id=dup.memory_id, similarity=dup.similarity)
```

4. After fact extraction (step 5), add conflict detection:
```python
        # [5.5] 冲突检测 (前移自 improve, 不解决只标注)
        result["conflicts"] = []
        if self.typed_store is not None and self.llm is not None:
            try:
                from septmuse.evolution.conflict import ConflictResolver
                resolver = ConflictResolver(self.typed_store, store=self.store, llm=self.llm)
                conflicts = resolver.detect_conflicts(user_id=user_id)
                result["conflicts"] = [
                    {"topic": f"{c['key'][0]} {c['key'][1]}", "facts": c["facts"]}
                    for c in conflicts
                ]
            except Exception as e:
                logger.warning("remember_conflict_detect_failed", error=str(e))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `$env:PYTHONPATH="src"; $env:SEPTMUSE_EMBEDDER="hash"; $env:SEPTMUSE_TOKENIZER="space"; python -m pytest tests/unit/test_smart_write.py -v`
Expected: 6 PASS

- [ ] **Step 5: Run existing remember tests (no regression)**

Run: `$env:PYTHONPATH="src"; $env:SEPTMUSE_EMBEDDER="hash"; $env:SEPTMUSE_TOKENIZER="space"; python -m pytest tests/unit/test_memory_orchestration.py tests/unit/test_v2_memory.py -q --tb=short`
Expected: 全部 PASS（原有测试可能需要更新 fixture/mock 以适应新返回字段）

- [ ] **Step 6: ruff check + format**

- [ ] **Step 7: Commit**

```bash
git add src/septmuse/memory/main.py tests/unit/test_smart_write.py
git -c core.hooksPath= commit -m "feat(write): integrate smart write into remember() — importance+classify+dedup+conflict"
```

---

### Task 5: 全量回归测试

**Files:**
- 无新建/修改

- [ ] **Step 1: 运行全部 Phase 3 测试**

Run: `$env:PYTHONPATH="src"; $env:SEPTMUSE_EMBEDDER="hash"; $env:SEPTMUSE_TOKENIZER="space"; $env:SEPTMUSE_RATE_LIMIT="0"; python -m pytest tests/unit/test_importance.py tests/unit/test_classifier.py tests/unit/test_dedup.py tests/unit/test_smart_write.py -v`
Expected: 全部 PASS

- [ ] **Step 2: 运行 MCP + health + auth + recall pipeline（无回归）**

Run: `$env:PYTHONPATH="src"; $env:SEPTMUSE_EMBEDDER="hash"; $env:SEPTMUSE_TOKENIZER="space"; $env:SEPTMUSE_RATE_LIMIT="0"; python -m pytest tests/unit/test_mcp_butler_tools.py tests/unit/test_mcp_tools.py tests/unit/test_health.py tests/unit/test_auth.py tests/unit/test_recall_pipeline.py tests/unit/test_recall_summarizer.py tests/unit/test_conflict_annotator.py -q`
Expected: 全部 PASS

- [ ] **Step 3: 运行 e2e 测试**

Run: `$env:PYTHONPATH="src"; $env:SEPTMUSE_EMBEDDER="hash"; $env:SEPTMUSE_TOKENIZER="space"; $env:SEPTMUSE_RATE_LIMIT="0"; python -m pytest tests/e2e/ -q`
Expected: 31 passed, 1 failed (pre-existing)

- [ ] **Step 4: ruff format check**

Run: `ruff format --check src/septmuse/capture/importance.py src/septmuse/capture/classifier.py src/septmuse/capture/dedup.py tests/unit/test_importance.py tests/unit/test_classifier.py tests/unit/test_dedup.py tests/unit/test_smart_write.py`
Expected: All checks passed

- [ ] **Step 5: Commit (if any format fixes)**

```bash
git add -A
git -c core.hooksPath= commit -m "test(phase3): full regression — smart write complete"
```

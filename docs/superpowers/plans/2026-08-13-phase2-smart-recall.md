# Smart Memory Butler — Phase 2: Smart Recall 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** recall 返回的不再是原始记忆列表，是加工好的即用上下文包——包含 LLM 压缩摘要和冲突标注。

**Architecture:** 新增两个独立模块（RecallSummarizer + ConflictAnnotator），插入 RecallPipeline 现有 7 步链路的 budget 之后、prompt 之前。MCP recall 工具返回结构新增 `summary` 和 `conflicts` 字段。

**Tech Stack:** Python 3.10+, LLM ABC (`septmuse.llms.base.LLM`), SemanticFact model, RecallPipeline

## Global Constraints

- **不使用 `from __future__ import annotations`**（仅 `api/mcp/tools.py` 的约束，新文件可用）
- **LLM 降级**：无 LLM 时 summary 取 top-3 拼接截断 200 字，conflicts 只做规则匹配
- **冲突不全库扫描**：只在 recall 返回的 top-K 结果内找矛盾
- **冲突只标注不解决**：解决留给 `improve()` 或 agent
- **测试环境**：`PYTHONPATH=src`, `SEPTMUSE_EMBEDDER=hash`, `SEPTMUSE_TOKENIZER=space`
- **line-length 120**，ruff check + format 必须通过
- **git commit 用** `git -c core.hooksPath= commit -m "..."` 绕过 LFS hook
- **SemanticFact 字段**：`id`, `subject`, `predicate`, `object`, `context`, `confidence`, `created_at`, `is_deleted`, `tags`
- **recall 返回的 memories 列表元素**：`dict` with keys `id`, `memory`, `score`, `metadata`, `created_at`
- **RecallPipeline.recall() 返回**：`{"memories": list, "injected_prompt": str, "route": dict|None, "strategy": StrategyResult|None}`

## File Structure

| 文件 | 职责 | 操作 |
|------|------|------|
| `src/septmuse/retrieval/recall_summarizer.py` | LLM 压缩 N 条记忆为一段摘要 | 新建 |
| `src/septmuse/retrieval/conflict_annotator.py` | 在检索结果中找矛盾并标注 | 新建 |
| `src/septmuse/retrieval/recall_pipeline.py` | 召回管线 | 修改：插入 summarize + annotate 步骤 |
| `src/septmuse/memory/main.py` | Memory facade | 修改：_get_recall_pipeline 传入新模块 |
| `src/septmuse/api/mcp/tools.py` | MCP recall 工具 | 修改：返回 summary + conflicts |
| `tests/unit/test_recall_summarizer.py` | RecallSummarizer 测试 | 新建 |
| `tests/unit/test_conflict_annotator.py` | ConflictAnnotator 测试 | 新建 |

## 现有代码参考

### RecallPipeline.recall() 返回结构（recall_pipeline.py:120-125）

```python
return {
    "memories": memories,       # list[dict] — 每个有 id/memory/score/metadata/created_at
    "injected_prompt": injected_prompt,  # str — <memory>...</memory>
    "route": route_result,      # dict|None — L0 路由
    "strategy": strategy,        # StrategyResult|None — L2 策略
}
```

### SemanticFact 字段（models/semantic.py:41-78）

```python
class SemanticFact(SQLModel, table=True):
    id: str
    subject: str
    predicate: str
    object: str
    context: str | None
    user_id: str
    confidence: float
    provenance: str
    tags: list[str]
    created_at: datetime
    is_deleted: bool
```

### ConflictResolver.detect_conflicts 规则（evolution/conflict.py:62-95）

相同 (subject, predicate) 不同 object → 冲突。返回 `[{"key": (subject, predicate), "facts": [...]}]`

### LLM ABC（llms/base.py:27-57）

```python
class LLM(ABC):
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        # 带可观测性埋点的模板方法
    def _complete(self, system_prompt: str, user_prompt: str) -> str:
        # 子类实现
```

### TypedMemoryStore.get_all_facts（typed_store.py:339-345）

```python
def get_all_facts(self, *, user_id: str, include_deleted: bool = False) -> list[SemanticFact]:
```

---

### Task 1: RecallSummarizer

**Files:**
- Create: `src/septmuse/retrieval/recall_summarizer.py`
- Test: `tests/unit/test_recall_summarizer.py`

**Interfaces:**
- Consumes: `LLM | None`（可选，降级时 None）
- Produces: `RecallSummarizer(memories: list[dict], llm: LLM | None = None)` → `summarize() -> str`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for RecallSummarizer."""
from __future__ import annotations

from unittest.mock import MagicMock


def test_summarize_with_llm():
    """LLM 可用时：调 LLM 压缩记忆列表为摘要。"""
    llm = MagicMock()
    llm.complete.return_value = "用户计划下周去上海出差，涉及项目X。"
    memories = [
        {"id": "m1", "memory": "用户下周要出差去上海", "score": 0.9},
        {"id": "m2", "memory": "用户参与了项目X", "score": 0.8},
    ]
    from septmuse.retrieval.recall_summarizer import RecallSummarizer

    s = RecallSummarizer(memories, llm=llm)
    result = s.summarize()
    assert "上海" in result
    assert "项目X" in result
    llm.complete.assert_called_once()


def test_summarize_no_llm():
    """LLM 不可用时：取 top-3 拼接截断 200 字。"""
    memories = [
        {"id": "m1", "memory": "用户下周要出差去上海", "score": 0.9},
        {"id": "m2", "memory": "用户参与了项目X", "score": 0.8},
        {"id": "m3", "memory": "用户上周说可能不去上海了", "score": 0.7},
        {"id": "m4", "memory": "无关内容", "score": 0.5},
    ]
    from septmuse.retrieval.recall_summarizer import RecallSummarizer

    s = RecallSummarizer(memories, llm=None)
    result = s.summarize()
    assert "上海" in result
    assert len(result) <= 200
    assert "无关内容" not in result  # top-3 only


def test_summarize_empty_memories():
    """空记忆列表返回空字符串。"""
    from septmuse.retrieval.recall_summarizer import RecallSummarizer

    s = RecallSummarizer([], llm=None)
    assert s.summarize() == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$env:PYTHONPATH="src"; $env:SEPTMUSE_EMBEDDER="hash"; $env:SEPTMUSE_TOKENIZER="space"; python -m pytest tests/unit/test_recall_summarizer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'septmuse.retrieval.recall_summarizer'`

- [ ] **Step 3: Write minimal implementation**

```python
"""RecallSummarizer — LLM 压缩 N 条记忆为一段摘要。

LLM 可用：prompt 压缩，保留关键事实和时间线索。
LLM 不可用：取 top-3 记忆 memory 字段拼接，截断 200 字。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from septmuse.core.logging import get_logger

if TYPE_CHECKING:
    from septmuse.llms.base import LLM

logger = get_logger(__name__)

_MAX_FALLBACK_LEN = 200
_TOP_N_FALLBACK = 3

_SUMMARY_SYSTEM_PROMPT = "你是一个记忆摘要助手。将以下记忆条目压缩为一段简洁摘要，保留关键事实和时间线索。"
_SUMMARY_USER_TEMPLATE = "记忆条目：\n{items}\n\n请压缩为一段摘要："


class RecallSummarizer:
    """压缩记忆列表为摘要。

    Args:
        memories: recall 返回的记忆列表，每个 dict 有 memory/score 字段
        llm: LLM 实例（可选，None 时降级为拼接）
    """

    def __init__(self, memories: list[dict[str, Any]], llm: "LLM | None" = None) -> None:
        self._memories = memories
        self._llm = llm

    def summarize(self) -> str:
        """返回摘要字符串。空记忆返回空字符串。"""
        if not self._memories:
            return ""

        if self._llm is not None:
            return self._summarize_with_llm()
        return self._summarize_fallback()

    def _summarize_with_llm(self) -> str:
        items = "\n".join(
            f"- {m.get('memory', '')}" for m in self._memories
        )
        user_prompt = _SUMMARY_USER_TEMPLATE.format(items=items)
        try:
            result = self._llm.complete(_SUMMARY_SYSTEM_PROMPT, user_prompt)
            logger.debug("recall_summary_llm_done", memories=len(self._memories), length=len(result))
            return result
        except Exception as e:
            logger.warning("recall_summary_llm_failed", error=str(e), fallback="concat")
            return self._summarize_fallback()

    def _summarize_fallback(self) -> str:
        sorted_memories = sorted(self._memories, key=lambda m: m.get("score", 0.0), reverse=True)
        top = sorted_memories[:_TOP_N_FALLBACK]
        texts = [m.get("memory", "") for m in top]
        joined = " ".join(texts)
        return joined[:_MAX_FALLBACK_LEN]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `$env:PYTHONPATH="src"; $env:SEPTMUSE_EMBEDDER="hash"; $env:SEPTMUSE_TOKENIZER="space"; python -m pytest tests/unit/test_recall_summarizer.py -v`
Expected: 3 PASS

- [ ] **Step 5: ruff check + format**

Run: `ruff check src/septmuse/retrieval/recall_summarizer.py tests/unit/test_recall_summarizer.py`
Then: `python -c "import subprocess; r=subprocess.run(['ruff','format','--stdin-filename','src/septmuse/retrieval/recall_summarizer.py'],stdin=open('src/septmuse/retrieval/recall_summarizer.py','r'),capture_output=True,text=True); open('src/septmuse/retrieval/recall_summarizer.py','w').write(r.stdout); r2=subprocess.run(['ruff','format','--stdin-filename','tests/unit/test_recall_summarizer.py'],stdin=open('tests/unit/test_recall_summarizer.py','r'),capture_output=True,text=True); open('tests/unit/test_recall_summarizer.py','w').write(r2.stdout); print('OK')"`

- [ ] **Step 6: Commit**

```bash
git add src/septmuse/retrieval/recall_summarizer.py tests/unit/test_recall_summarizer.py
git -c core.hooksPath= commit -m "feat(recall): RecallSummarizer — LLM 摘要 + 降级拼接"
```

---

### Task 2: ConflictAnnotator

**Files:**
- Create: `src/septmuse/retrieval/conflict_annotator.py`
- Test: `tests/unit/test_conflict_annotator.py`

**Interfaces:**
- Consumes: `LLM | None`（可选，用于 LLM 判定是否真矛盾）
- Produces: `ConflictAnnotator(llm: LLM | None = None)` → `annotate(memories: list[dict], facts: list[SemanticFact]) -> list[dict]`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for ConflictAnnotator."""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock

from septmuse.models.semantic import SemanticFact
from septmuse.retrieval.conflict_annotator import ConflictAnnotator


def _make_fact(subject: str, predicate: str, obj: str, created_at: datetime) -> SemanticFact:
    return SemanticFact(
        subject=subject,
        predicate=predicate,
        object=obj,
        user_id="alice",
        created_at=created_at,
    )


def test_annotate_no_conflicts():
    """无矛盾事实返回空列表。"""
    now = datetime.now(timezone.utc)
    facts = [
        _make_fact("用户", "喜欢", "咖啡", now),
        _make_fact("用户", "住在", "北京", now),
    ]
    memories = [{"id": "m1", "memory": "用户喜欢咖啡", "score": 0.9}]
    annotator = ConflictAnnotator(llm=None)
    result = annotator.annotate(memories, facts)
    assert result == []


def test_annotate_rule_based_conflict():
    """规则检测：相同 subject+predicate 不同 object。"""
    now = datetime.now(timezone.utc)
    old = now - timedelta(days=7)
    facts = [
        _make_fact("用户", "去", "上海", old),
        _make_fact("用户", "去", "北京", now),
    ]
    memories = [
        {"id": "m1", "memory": "用户去了上海", "score": 0.9},
        {"id": "m2", "memory": "用户去了北京", "score": 0.8},
    ]
    annotator = ConflictAnnotator(llm=None)
    result = annotator.annotate(memories, facts)
    assert len(result) == 1
    assert result[0]["topic"] == "用户 去"
    assert "上海" in result[0]["current"]
    assert "北京" in result[0]["contradicting"]
    assert result[0]["suggestion"] != ""


def test_annotate_with_llm():
    """LLM 可用时：候选冲突对喂给 LLM 判定。"""
    now = datetime.now(timezone.utc)
    old = now - timedelta(days=7)
    facts = [
        _make_fact("用户", "去", "上海", old),
        _make_fact("用户", "去", "北京", now),
    ]
    memories = [
        {"id": "m1", "memory": "用户去了上海", "score": 0.9},
        {"id": "m2", "memory": "用户去了北京", "score": 0.8},
    ]
    llm = MagicMock()
    llm.complete.return_value = "yes"
    annotator = ConflictAnnotator(llm=llm)
    result = annotator.annotate(memories, facts)
    assert len(result) == 1
    llm.complete.assert_called_once()


def test_annotate_llm_says_no():
    """LLM 判定不是真矛盾时：不返回冲突。"""
    now = datetime.now(timezone.utc)
    old = now - timedelta(days=7)
    facts = [
        _make_fact("用户", "去", "上海", old),
        _make_fact("用户", "去", "北京", now),
    ]
    memories = [
        {"id": "m1", "memory": "用户去了上海", "score": 0.9},
        {"id": "m2", "memory": "用户去了北京", "score": 0.8},
    ]
    llm = MagicMock()
    llm.complete.return_value = "no, these are about different trips"
    annotator = ConflictAnnotator(llm=llm)
    result = annotator.annotate(memories, facts)
    assert result == []


def test_annotate_empty():
    """空输入返回空列表。"""
    annotator = ConflictAnnotator(llm=None)
    assert annotator.annotate([], []) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$env:PYTHONPATH="src"; $env:SEPTMUSE_EMBEDDER="hash"; $env:SEPTMUSE_TOKENIZER="space"; python -m pytest tests/unit/test_conflict_annotator.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
"""ConflictAnnotator — 在检索结果中找矛盾并标注。

规则：相同 (subject, predicate) 不同 object → 冲突。
LLM 可用时：候选冲突对喂给 LLM 判定是否真矛盾。
LLM 不可用时：只做规则匹配。

只在 recall 返回的 top-K 结果对应的事实里找，不全库扫描。
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any

from septmuse.core.logging import get_logger

if TYPE_CHECKING:
    from septmuse.llms.base import LLM
    from septmuse.models.semantic import SemanticFact

logger = get_logger(__name__)

_CONFLICT_SYSTEM_PROMPT = "你是一个矛盾判定助手。判断以下两条事实是否真的矛盾。回答 'yes' 或 'no'。"
_CONFLICT_USER_TEMPLATE = "事实1: {fact1}\n事实2: {fact2}\n\n这两条事实是否矛盾？"


class ConflictAnnotator:
    """在检索结果中标注矛盾事实。

    Args:
        llm: LLM 实例（可选，None 时只做规则匹配）
    """

    def __init__(self, llm: "LLM | None" = None) -> None:
        self._llm = llm

    def annotate(
        self,
        memories: list[dict[str, Any]],
        facts: list["SemanticFact"],
    ) -> list[dict[str, Any]]:
        """在事实列表中找矛盾并返回冲突标注。

        Args:
            memories: recall 返回的记忆列表（用于确定范围）
            facts: 对应的 SemanticFact 列表（从 typed_store 获取）

        Returns:
            [{"topic": str, "current": str, "contradicting": str,
              "contradicting_memory_id": str, "suggestion": str}]
        """
        if not facts:
            return []

        groups: dict[tuple[str, str], list[Any]] = defaultdict(list)
        for f in facts:
            if getattr(f, "is_deleted", False):
                continue
            key = (f.subject, f.predicate)
            groups[key].append(f)

        conflicts: list[dict[str, Any]] = []
        for (subject, predicate), group_facts in groups.items():
            objects = {f.object for f in group_facts}
            if len(objects) <= 1:
                continue

            group_facts_sorted = sorted(group_facts, key=lambda f: f.created_at, reverse=True)
            latest = group_facts_sorted[0]

            for old_fact in group_facts_sorted[1:]:
                fact1 = f"{latest.subject} {latest.predicate} {latest.object}"
                fact2 = f"{old_fact.subject} {old_fact.predicate} {old_fact.object}"

                if self._llm is not None:
                    if not self._llm_says_conflict(fact1, fact2):
                        continue

                conflicts.append({
                    "topic": f"{subject} {predicate}",
                    "current": fact1,
                    "contradicting": fact2,
                    "contradicting_memory_id": old_fact.id,
                    "suggestion": f"建议询问{subject}的{predicate}是否已变更",
                })

        logger.info("conflicts_annotated", count=len(conflicts))
        return conflicts

    def _llm_says_conflict(self, fact1: str, fact2: str) -> bool:
        """LLM 判定两条事实是否真矛盾。"""
        user_prompt = _CONFLICT_USER_TEMPLATE.format(fact1=fact1, fact2=fact2)
        try:
            response = self._llm.complete(_CONFLICT_SYSTEM_PROMPT, user_prompt)
            return "yes" in response.strip().lower()
        except Exception as e:
            logger.warning("llm_conflict_check_failed", error=str(e))
            return True
```

- [ ] **Step 4: Run test to verify it passes**

Run: `$env:PYTHONPATH="src"; $env:SEPTMUSE_EMBEDDER="hash"; $env:SEPTMUSE_TOKENIZER="space"; python -m pytest tests/unit/test_conflict_annotator.py -v`
Expected: 5 PASS

- [ ] **Step 5: ruff check + format**

Run: `ruff check src/septmuse/retrieval/conflict_annotator.py tests/unit/test_conflict_annotator.py`
Then format via stdin subprocess (same pattern as Task 1).

- [ ] **Step 6: Commit**

```bash
git add src/septmuse/retrieval/conflict_annotator.py tests/unit/test_conflict_annotator.py
git -c core.hooksPath= commit -m "feat(recall): ConflictAnnotator — 规则+LLM 矛盾标注"
```

---

### Task 3: RecallPipeline 集成

**Files:**
- Modify: `src/septmuse/retrieval/recall_pipeline.py`
- Modify: `src/septmuse/memory/main.py`（`_get_recall_pipeline` 传入新模块）
- Test: `tests/unit/test_recall_pipeline.py`（已有，追加测试）

**Interfaces:**
- Consumes: `RecallSummarizer`, `ConflictAnnotator`, `TypedMemoryStore.get_all_facts()`
- Produces: `RecallPipeline.recall()` 返回新增 `summary: str` 和 `conflicts: list`

- [ ] **Step 1: Write the failing test**

追加到 `tests/unit/test_recall_pipeline.py` 末尾：

```python
def test_recall_returns_summary_and_conflicts():
    """recall 返回包含 summary 和 conflicts 字段。"""
    from septmuse.retrieval.recall_pipeline import RecallPipeline

    mock_search = MagicMock()
    mock_search.search.return_value = [
        {"id": "m1", "memory": "用户喜欢咖啡", "score": 0.9, "metadata": {}, "created_at": "2026-01-01T00:00:00Z"},
    ]
    pipeline = RecallPipeline(mock_search)
    result = pipeline.recall("咖啡", user_id="alice")

    assert "summary" in result
    assert "conflicts" in result
    assert isinstance(result["summary"], str)
    assert isinstance(result["conflicts"], list)


def test_recall_summary_fallback_no_llm():
    """无 LLM 时 summary 是降级拼接（非空）。"""
    from septmuse.retrieval.recall_pipeline import RecallPipeline

    mock_search = MagicMock()
    mock_search.search.return_value = [
        {"id": "m1", "memory": "用户喜欢咖啡", "score": 0.9, "metadata": {}, "created_at": "2026-01-01T00:00:00Z"},
    ]
    pipeline = RecallPipeline(mock_search)
    result = pipeline.recall("咖啡", user_id="alice")

    assert result["summary"] != ""
    assert "咖啡" in result["summary"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$env:PYTHONPATH="src"; $env:SEPTMUSE_EMBEDDER="hash"; $env:SEPTMUSE_TOKENIZER="space"; python -m pytest tests/unit/test_recall_pipeline.py::test_recall_returns_summary_and_conflicts -v`
Expected: FAIL with `KeyError: 'summary'` or `AssertionError`

- [ ] **Step 3: Modify RecallPipeline**

在 `recall_pipeline.py` 中：

1. 在 `__init__` 新增两个可选参数 `llm` 和 `typed_store`：

```python
def __init__(
    self,
    search_pipeline: Any,
    meta: Any | None = None,
    token_budget: Any | None = None,
    working_memory: Any | None = None,
    summarizer: Any | None = None,
    typed_store: Any | None = None,
    llm: Any | None = None,
) -> None:
    self._search = search_pipeline
    self._meta = meta
    self._token_budget = token_budget
    self._working_memory = working_memory
    self._summarizer = summarizer
    self._typed_store = typed_store
    self._llm = llm
```

2. 在 budget fit（step [4]）之后、prompt assembly（step [5]）之前插入两个新步骤：

```python
        # [4.5] summarize
        from septmuse.retrieval.recall_summarizer import RecallSummarizer

        summarizer = RecallSummarizer(memories, llm=self._llm)
        summary = summarizer.summarize()

        # [4.6] conflict annotation
        conflicts: list[dict[str, Any]] = []
        if self._typed_store is not None:
            from septmuse.retrieval.conflict_annotator import ConflictAnnotator

            facts = self._typed_store.get_all_facts(user_id=user_id)
            annotator = ConflictAnnotator(llm=self._llm)
            conflicts = annotator.annotate(memories, facts)
```

3. 在返回字典中新增两个字段：

```python
        return {
            "memories": memories,
            "injected_prompt": injected_prompt,
            "route": route_result,
            "strategy": strategy,
            "summary": summary,
            "conflicts": conflicts,
        }
```

- [ ] **Step 4: Modify Memory._get_recall_pipeline**

在 `main.py` 的 `_get_recall_pipeline` 方法中，传入 `llm` 和 `typed_store`：

```python
def _get_recall_pipeline(self):
    if self._recall_pipeline is None:
        self._recall_pipeline = RecallPipeline(
            search_pipeline=self._get_search_pipeline(),
            meta=self.meta,
            token_budget=self.token_budget,
            working_memory=self.working_memory,
            summarizer=None,
            typed_store=self.typed_store,
            llm=self.llm,
        )
    return self._recall_pipeline
```

- [ ] **Step 5: Run test to verify it passes**

Run: `$env:PYTHONPATH="src"; $env:SEPTMUSE_EMBEDDER="hash"; $env:SEPTMUSE_TOKENIZER="space"; python -m pytest tests/unit/test_recall_pipeline.py::test_recall_returns_summary_and_conflicts tests/unit/test_recall_pipeline.py::test_recall_summary_fallback_no_llm -v`
Expected: 2 PASS

- [ ] **Step 6: Run full recall pipeline tests (no regression)**

Run: `$env:PYTHONPATH="src"; $env:SEPTMUSE_EMBEDDER="hash"; $env:SEPTMUSE_TOKENIZER="space"; python -m pytest tests/unit/test_recall_pipeline.py -v`
Expected: 全部 PASS（原有 14 + 新增 2 = 16）

- [ ] **Step 7: ruff check + format**

Run: `ruff check src/septmuse/retrieval/recall_pipeline.py src/septmuse/memory/main.py tests/unit/test_recall_pipeline.py`
Then format via stdin subprocess.

- [ ] **Step 8: Commit**

```bash
git add src/septmuse/retrieval/recall_pipeline.py src/septmuse/memory/main.py tests/unit/test_recall_pipeline.py
git -c core.hooksPath= commit -m "feat(recall): integrate summarize+conflict into RecallPipeline"
```

---

### Task 4: MCP recall 返回 summary + conflicts

**Files:**
- Modify: `src/septmuse/api/mcp/tools.py`（recall 工具的 docstring 更新）
- Test: `tests/unit/test_mcp_butler_tools.py`（追加测试）

**Interfaces:**
- Consumes: RecallPipeline 返回的 `summary` 和 `conflicts` 字段
- Produces: MCP recall 工具返回 JSON 包含 `summary` 和 `conflicts`

- [ ] **Step 1: Write the failing test**

追加到 `tests/unit/test_mcp_butler_tools.py`：

```python
def test_recall_returns_summary_and_conflicts(mock_memory):
    """recall MCP 工具返回包含 summary 和 conflicts。"""
    mock_memory.recall.return_value = {
        "memories": [{"id": "m1", "memory": "hello", "score": 0.9}],
        "injected_prompt": "<memory>hello</memory>",
        "route": None,
        "strategy": None,
        "summary": "用户喜欢咖啡",
        "conflicts": [{"topic": "咖啡偏好", "current": "喜欢", "contradicting": "不喜欢", "contradicting_memory_id": "m2", "suggestion": "建议询问"}],
    }
    with patch("septmuse.api.mcp.tools.get_memory_safe", return_value=mock_memory):
        from septmuse.api.mcp.tools import recall
        import asyncio

        result = asyncio.run(recall("咖啡", user_id="alice"))
        parsed = json.loads(result)
        assert "summary" in parsed
        assert parsed["summary"] == "用户喜欢咖啡"
        assert "conflicts" in parsed
        assert len(parsed["conflicts"]) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$env:PYTHONPATH="src"; $env:SEPTMUSE_EMBEDDER="hash"; $env:SEPTMUSE_TOKENIZER="space"; python -m pytest tests/unit/test_mcp_butler_tools.py::test_recall_returns_summary_and_conflicts -v`
Expected: FAIL — mock 返回了 summary 但旧测试 mock 不含此字段，或 KeyError

- [ ] **Step 3: Update mock_memory fixture and recall docstring**

在 `test_mcp_butler_tools.py` 的 `mock_memory` fixture 中，给 `recall.return_value` 加上 `"summary": ""` 和 `"conflicts": []`：

```python
m.recall.return_value = {
    "memories": [{"id": "mem-001", "memory": "hello", "score": 0.9}],
    "injected_prompt": "<memory>hello</memory>",
    "route": "semantic",
    "strategy": "MAINTAIN",
    "used_tokens": 50,
    "summary": "",
    "conflicts": [],
}
```

在 `tools.py` 的 `recall` 工具 docstring 中，更新 Returns：

```python
    Returns:
        JSON: {memories, injected_prompt, route, strategy, summary, conflicts}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `$env:PYTHONPATH="src"; $env:SEPTMUSE_EMBEDDER="hash"; $env:SEPTMUSE_TOKENIZER="space"; python -m pytest tests/unit/test_mcp_butler_tools.py -v`
Expected: 全部 PASS（原有 + 新增 1）

- [ ] **Step 5: ruff check + format**

Run: `ruff check src/septmuse/api/mcp/tools.py tests/unit/test_mcp_butler_tools.py`
Then format via stdin subprocess.

- [ ] **Step 6: Commit**

```bash
git add src/septmuse/api/mcp/tools.py tests/unit/test_mcp_butler_tools.py
git -c core.hooksPath= commit -m "feat(mcp): recall returns summary + conflicts"
```

---

### Task 5: 全量回归测试

**Files:**
- 无新建/修改

- [ ] **Step 1: 运行全部 Phase 2 测试**

Run: `$env:PYTHONPATH="src"; $env:SEPTMUSE_EMBEDDER="hash"; $env:SEPTMUSE_TOKENIZER="space"; $env:SEPTMUSE_RATE_LIMIT="0"; python -m pytest tests/unit/test_recall_summarizer.py tests/unit/test_conflict_annotator.py tests/unit/test_recall_pipeline.py tests/unit/test_mcp_butler_tools.py -v`
Expected: 全部 PASS

- [ ] **Step 2: 运行 MCP 工具测试 + health + auth（无回归）**

Run: `$env:PYTHONPATH="src"; $env:SEPTMUSE_EMBEDDER="hash"; $env:SEPTMUSE_TOKENIZER="space"; $env:SEPTMUSE_RATE_LIMIT="0"; python -m pytest tests/unit/test_mcp_tools.py tests/unit/test_health.py tests/unit/test_auth.py -q`
Expected: 全部 PASS

- [ ] **Step 3: 运行 e2e 测试**

Run: `$env:PYTHONPATH="src"; $env:SEPTMUSE_EMBEDDER="hash"; $env:SEPTMUSE_TOKENIZER="space"; $env:SEPTMUSE_RATE_LIMIT="0"; python -m pytest tests/e2e/ -q`
Expected: 31 passed, 1 failed (pre-existing test_capture_persists)

- [ ] **Step 4: ruff format check**

Run: `ruff format --check src/septmuse/retrieval/recall_summarizer.py src/septmuse/retrieval/conflict_annotator.py src/septmuse/retrieval/recall_pipeline.py tests/unit/test_recall_summarizer.py tests/unit/test_conflict_annotator.py tests/unit/test_recall_pipeline.py`
Expected: All checks passed

- [ ] **Step 5: Commit (if any format fixes)**

```bash
git add -A
git -c core.hooksPath= commit -m "test(phase2): full regression — smart recall complete"
```

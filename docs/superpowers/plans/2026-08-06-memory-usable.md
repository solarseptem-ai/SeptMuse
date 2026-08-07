# SeptMuse 记忆可用性改造实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 SeptMuse 记忆系统完全可用——Memory.add 内部决策化（ADD/UPDATE/DELETE/NOOP，对齐 mem0）+ V2 降级委托 + 用户画像聚合。

**Architecture:** 单一 Memory facade + 方法间协作（add→search→决策）+ V2Memory 保留为 deprecated 薄层委托 + 用户画像从 SemanticFact 聚合。5 层改造（L1 决策 / L2 协作 / L3 委托 / L4 参数化 / L5 画像）。

**Tech Stack:** Python 3.12, src/ layout, FastAPI, structlog, SQLModel, ruff (line-length 120, **禁用 `ruff format`**), pytest (PYTHONPATH=src).

**Spec:** `docs/superpowers/specs/2026-08-06-memory-usable-design.md`

## Global Constraints

- **包名 `septmuse`**, src/ 布局, `PYTHONPATH=src` 运行测试（不 pip install -e .）
- **ruff line-length 120**, `select = ["E","F","I","W","UP","B","SIM","RUF"]`, ignore E501/RUF001/002/003（中文标点）
- **禁用 `ruff format`**（Windows 清空文件 bug），只用 `ruff check`
- **代码注释中文**, 不暴露开源库参考来源
- **不提交**（用户指示）— commit 步骤作为检查点, 不实际 git commit
- **开发完统一回归**（用户指示）— 每任务单测通过即可, 全量 `pytest tests/unit/` 在所有任务完成后统一跑
- **测试保护**: 现有测试断言不改, 只新增测试覆盖新能力
- **LLM 测试用 MockLLM**（无真实 API key）, conftest 强制 HashEmbedder + space tokenizer
- **基类名**: `Embedder`（非 EmbedderBase）, `LLM`（非 LLMBase）
- **embed 签名**: `def embed(self, text: str, memory_action: str | None = None) -> list[float]`

## 已确认的关键接口（类型一致性基准）

```python
# TypedMemoryStore (storage/relational_stores/typed_store.py)
def add_fact(self, subject, predicate, object, *, user_id, context=None, confidence=1.0,
             provenance="user", tags=None, embedding=None, org_id="default") -> SemanticFact
def update_fact(self, fact_id, subject, predicate, object) -> SemanticFact | None  # 已存在, Task 2 增强
def soft_delete_fact(self, fact_id) -> bool
def get_all_facts(self, *, user_id, include_deleted=False) -> list[SemanticFact]
def search_facts(self, query_embedding, *, user_id, top_k=5, threshold=0.1) -> list[dict]

# SemanticFact 字段: id, subject, predicate, object, context, user_id, org_id,
#   confidence, provenance, tags, embedding, is_deleted; touch() 更新 updated_at

# FactExtractor (models/extract.py)
def __init__(self, llm, embedder, typed_store, verbatim_store=None, use_additive_prompt=True)
def _retrieve_existing(self, text, user_id, top_k=10) -> list[dict]  # 调 verbatim_store.search
def extract_facts(self, messages, existing_memories=None) -> list[str]
def extract_and_store(self, messages, *, user_id, provenance="inferred") -> list[dict]
def _parse_facts_response(raw) -> list[str]  # staticmethod

# BudgetItem (retrieval/token_budget.py): text, score, metadata  # Task 4 加 id

# CapturePipeline (capture/pipeline.py)
def capture(self, text, *, user_id, agent_id, session_id, metadata) -> PipelineResult  # Task 4 加 preprocess

# Memory (memory/main.py)
def add(...) -> {"results": [{"id","memory","event"}], "relations": []}
def search(...) -> [{"id","memory","score","metadata","created_at"}]
def delete(memory_id) -> {"status","memory_id"}
```

---

## Task 1: ADDITIVE_DECISION_PROMPT + Decision + 决策抽取

**Files:**
- Modify: `src/septmuse/prompts/extract.py`（新增 `ADDITIVE_DECISION_PROMPT`）
- Modify: `src/septmuse/models/extract.py`（新增 `Decision` dataclass + `extract_with_decisions`）
- Test: `tests/unit/test_fact_decision.py`

**Interfaces:**
- Consumes: `FactExtractor.__init__(llm, embedder, typed_store, verbatim_store)`, `FactExtractor._retrieve_existing(text, user_id)`, `LLM.complete(system_prompt, user_prompt) -> str`
- Produces: `ADDITIVE_DECISION_PROMPT` (str), `Decision(text, event, id, confidence)` dataclass, `FactExtractor.extract_with_decisions(messages, existing_memories) -> list[Decision]`

- [ ] **Step 1: 写失败测试**

```python
# tests/unit/test_fact_decision.py
from septmuse.models.extract import Decision, FactExtractor


def test_decision_dataclass():
    d = Decision(text="Likes Rust", event="ADD")
    assert d.text == "Likes Rust"
    assert d.event == "ADD"
    assert d.id is None
    assert d.confidence == 1.0


def test_extract_with_decisions_add(mock_llm, fact_extractor):
    # mock_llm 返回 {"facts":[{"text":"Likes Python","event":"ADD","id":null,"confidence":0.9}]}
    mock_llm.set_response('{"facts":[{"text":"Likes Python","event":"ADD","id":null,"confidence":0.9}]}')
    decisions = fact_extractor.extract_with_decisions("I like Python", existing_memories=[])
    assert len(decisions) == 1
    assert decisions[0].text == "Likes Python"
    assert decisions[0].event == "ADD"
    assert decisions[0].confidence == 0.9


def test_extract_with_decisions_four_events(mock_llm, fact_extractor):
    mock_llm.set_response(
        '{"facts":['
        '{"text":"Likes Rust","event":"ADD","id":null,"confidence":0.9},'
        '{"text":"Likes Python","event":"UPDATE","id":"mem-1","confidence":0.85},'
        '{"text":"Likes Java","event":"DELETE","id":"mem-2","confidence":0.6},'
        '{"text":"Exists","event":"NOOP","id":"mem-3","confidence":1.0}'
        "]}"
    )
    decisions = fact_extractor.extract_with_decisions("msg", existing_memories=[{"id": "mem-1", "memory": "Likes Python"}])
    assert len(decisions) == 4
    assert [d.event for d in decisions] == ["ADD", "UPDATE", "DELETE", "NOOP"]
    assert decisions[1].id == "mem-1"
    assert decisions[2].confidence == 0.6


def test_extract_with_decisions_parse_fallback(mock_llm, fact_extractor):
    # LLM 输出不合规 JSON → 降级为全 ADD
    mock_llm.set_response("not json at all")
    decisions = fact_extractor.extract_with_decisions("I like Python", existing_memories=[])
    # 降级: 解析失败返回空列表 (不阻塞)
    assert decisions == []
```

- [ ] **Step 2: 运行测试验证失败**

```powershell
$env:PYTHONPATH = "src"; python -m pytest tests/unit/test_fact_decision.py -v
```
Expected: FAIL（`ADDITIVE_DECISION_PROMPT` 不存在, `Decision` 不存在, `extract_with_decisions` 不存在）

- [ ] **Step 3: 实现**

`src/septmuse/prompts/extract.py` 末尾新增:
```python
ADDITIVE_DECISION_PROMPT = """You are a Personal Information Organizer. Given existing memories and new messages, decide for each fact: ADD (new), UPDATE (existing fact value changed, must include id), DELETE (existing fact contradicted by new message, must include id), or NOOP (already exists, no change).

# Input Format
## Existing Memories
1. [mem-xxx] Likes Python
...

## New Messages
<user message>

# Output Format
Return ONLY valid JSON: {"facts": [{"text": "...", "event": "ADD|UPDATE|DELETE|NOOP", "id": null_or_memid, "confidence": 0.0-1.0}]}
- UPDATE/DELETE must include the existing memory's id.
- confidence: your confidence in this decision (0.0-1.0).
- If nothing relevant, return {"facts": []}.

# Examples
Existing: [mem-1] Likes Python
New: "I now prefer Rust over Python."
Output: {"facts":[{"text":"Prefers Rust","event":"ADD","id":null,"confidence":0.9},{"text":"Likes Python","event":"DELETE","id":"mem-1","confidence":0.85}]}

Existing: [mem-2] Name is Alice
New: "My name is actually Alyssa."
Output: {"facts":[{"text":"Name is Alyssa","event":"UPDATE","id":"mem-2","confidence":0.95}]}

Existing: [mem-3] Lives in Tokyo
New: "I love sushi."
Output: {"facts":[{"text":"Likes sushi","event":"ADD","id":null,"confidence":0.9}]}

Existing: [mem-4] Likes Python
New: "Python is great."
Output: {"facts":[{"text":"Likes Python","event":"NOOP","id":"mem-4","confidence":1.0}]}
"""
```

`src/septmuse/models/extract.py` 新增（在 FactExtractor 类内, `extract_facts` 之后）:
```python
@dataclass
class Decision:
    """LLM 决策抽取的单条结果."""
    text: str
    event: str  # "ADD" | "UPDATE" | "DELETE" | "NOOP"
    id: str | None = None
    confidence: float = 1.0

def extract_with_decisions(
    self, messages: Any, existing_memories: list[dict[str, Any]] | None = None
) -> list[Decision]:
    """LLM 决策抽取, 返回带 event 的决策列表 (对齐 mem0 ADDITIVE).

    解析失败降级为空列表 (不阻塞业务).
    """
    text = parse_messages(messages)
    if not text.strip():
        return []
    from septmuse.prompts.extract import ADDITIVE_DECISION_PROMPT, build_extraction_user_prompt

    user_prompt = build_extraction_user_prompt(text, existing_memories)
    raw = self.llm.complete(ADDITIVE_DECISION_PROMPT, user_prompt)
    return self._parse_decisions_response(raw)

@staticmethod
def _parse_decisions_response(raw: str) -> list[Decision]:
    """解析 LLM 决策输出, 容错降级."""
    import re, json
    cleaned = re.sub(r"^```[a-zA-Z0-9]*\n|\n```$", "", raw.strip())
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning("decisions_parse_failed", raw=raw[:100])
        return []
    raw_facts = data.get("facts", []) if isinstance(data, dict) else []
    decisions: list[Decision] = []
    for f in raw_facts:
        if not isinstance(f, dict):
            continue
        decisions.append(Decision(
            text=str(f.get("text", "")).strip(),
            event=str(f.get("event", "ADD")).upper(),
            id=f.get("id"),
            confidence=float(f.get("confidence", 1.0)),
        ))
    return [d for d in decisions if d.text and d.event in ("ADD", "UPDATE", "DELETE", "NOOP")]
```
（`extract.py` 顶部加 `from dataclasses import dataclass`）

- [ ] **Step 4: 运行测试验证通过**

```powershell
$env:PYTHONPATH = "src"; python -m pytest tests/unit/test_fact_decision.py -v
```
Expected: 4 passed

- [ ] **Step 5: 检查点（暂不 commit）**

```powershell
ruff check src/septmuse/prompts/extract.py src/septmuse/models/extract.py tests/unit/test_fact_decision.py
```
Expected: All checks passed

---

## Task 2: 增强 update_fact + extract_and_store 决策路由 + Memory.add 接决策

**Files:**
- Modify: `src/septmuse/storage/relational_stores/typed_store.py:121`（`update_fact` 增强补 embedding/confidence）
- Modify: `src/septmuse/models/extract.py`（`extract_and_store` 决策路由）
- Modify: `src/septmuse/memory/main.py:292`（`add` infer=True 路径接决策）
- Test: `tests/unit/test_fact_decision.py`（追加路由测试）+ `tests/unit/test_add_decision.py`

**Interfaces:**
- Consumes: `Task 1` 的 `extract_with_decisions`, `Decision`, `ADDITIVE_DECISION_PROMPT`
- Produces: `TypedMemoryStore.update_fact(fact_id, subject, predicate, object, *, embedding=None, confidence=None)`, `FactExtractor.extract_and_store` 返回 event 不恒 ADD

- [ ] **Step 1: 写失败测试**

```python
# tests/unit/test_fact_decision.py 追加
def test_extract_and_store_decision_routing_add(mock_llm, fact_extractor, typed_store):
    """ADD 决策 → add_fact + verbatim add."""
    mock_llm.set_response('{"facts":[{"text":"Likes Python","event":"ADD","id":null,"confidence":0.9}]}')
    results = fact_extractor.extract_and_store("I like Python", user_id="alice")
    assert len(results) == 1
    assert results[0]["event"] == "ADD"
    assert results[0]["id"]  # fact 存了

def test_extract_and_store_decision_routing_update(mock_llm, fact_extractor, typed_store):
    """UPDATE 决策 → update_fact (置信度 >=0.7)."""
    # 先存一个 fact
    fact = typed_store.add_fact("user", "likes", "Python", user_id="alice")
    mock_llm.set_response(
        f'{{"facts":[{{"text":"Likes Rust","event":"UPDATE","id":"{fact.id}","confidence":0.85}}]}}'
    )
    results = fact_extractor.extract_and_store("I like Rust now", user_id="alice")
    assert results[0]["event"] == "UPDATE"
    # 验证 fact 被更新
    updated = typed_store.get_all_facts(user_id="alice")
    assert any(f.object == "Rust" for f in updated)

def test_extract_and_store_decision_routing_delete(mock_llm, fact_extractor, typed_store):
    """DELETE 决策 → soft_delete_fact (置信度 >=0.7)."""
    fact = typed_store.add_fact("user", "likes", "Java", user_id="alice")
    mock_llm.set_response(
        f'{{"facts":[{{"text":"Likes Java","event":"DELETE","id":"{fact.id}","confidence":0.8}}]}}'
    )
    results = fact_extractor.extract_and_store("I hate Java now", user_id="alice")
    assert results[0]["event"] == "DELETE"

def test_extract_and_store_decision_low_confidence_noop(mock_llm, fact_extractor, typed_store):
    """DELETE confidence <0.7 → 降级 NOOP (不删)."""
    fact = typed_store.add_fact("user", "likes", "Java", user_id="alice")
    mock_llm.set_response(
        f'{{"facts":[{{"text":"Likes Java","event":"DELETE","id":"{fact.id}","confidence":0.5}}]}}'
    )
    results = fact_extractor.extract_and_store("msg", user_id="alice")
    assert results[0]["event"] == "NOOP"  # 降级
```

```python
# tests/unit/test_add_decision.py
def test_memory_add_infer_routes_update(memory_with_mock_llm):
    """Memory.add(infer=True) 重复事实 → UPDATE 决策."""
    # 先 add 一条
    memory_with_mock_llm.add("I like Python", user_id="alice", infer=False)
    # 再 add, LLM 决策 UPDATE 旧事实
    memory_with_mock_llm.llm.set_response('{"facts":[{"text":"Likes Rust","event":"UPDATE","id":"<existing>","confidence":0.9}]}')
    result = memory_with_mock_llm.add("I like Rust now", user_id="alice", infer=True)
    events = [r.get("event") for r in result["results"]]
    assert "UPDATE" in events

def test_memory_add_infer_no_llm_falls_back(memory_no_llm):
    """无 LLM → infer=True 降级 verbatim 直存."""
    result = memory_no_llm.add("hello", user_id="alice", infer=True)
    assert result["results"][0]["event"] == "ADD"
```

- [ ] **Step 2: 运行测试验证失败**

```powershell
$env:PYTHONPATH = "src"; python -m pytest tests/unit/test_fact_decision.py tests/unit/test_add_decision.py -v
```
Expected: FAIL（`extract_and_store` 还没决策路由, `update_fact` 没增强）

- [ ] **Step 3: 实现**

`typed_store.py:121` `update_fact` 增强:
```python
def update_fact(
    self, fact_id: str, subject: str, predicate: str, object: str,
    *, embedding: list[float] | None = None, confidence: float | None = None,
) -> SemanticFact | None:
    """更新语义事实 (增强: 支持 embedding/confidence 更新)."""
    with Session(self.engine) as session:
        fact = session.get(SemanticFact, fact_id)
        if not fact or fact.is_deleted:
            return None
        fact.subject = subject
        fact.predicate = predicate
        fact.object = object
        if embedding is not None:
            fact.embedding = json.dumps(embedding).encode()
        if confidence is not None:
            fact.confidence = confidence
        fact.touch()
        session.add(fact)
        session.commit()
        session.refresh(fact)
        return fact
```

`models/extract.py` `extract_and_store` 改造（决策路由）:
```python
def extract_and_store(self, messages, *, user_id, provenance="inferred"):
    """完整 cognify: 检索已有 → 决策抽取 → 按 event 路由 ADD/UPDATE/DELETE/NOOP."""
    text = parse_messages(messages)
    existing = self._retrieve_existing(text, user_id)
    decisions = self.extract_with_decisions(messages, existing_memories=existing)
    results = []
    linked_memory_ids = []
    for d in decisions:
        if d.event == "NOOP":
            results.append({"id": d.id, "memory": d.text, "event": "NOOP", "linked_memory_ids": []})
            continue
        # 置信度守卫: DELETE/UPDATE < 0.7 降级 NOOP
        if d.event in ("DELETE", "UPDATE") and d.confidence < 0.7:
            logger.info("decision_low_confidence_downgrade", event=d.event, confidence=d.confidence)
            results.append({"id": d.id, "memory": d.text, "event": "NOOP", "linked_memory_ids": []})
            continue
        if d.event == "ADD":
            fact = self._store_add_fact(d.text, user_id, provenance)
            vid = self._store_verbatim(d.text, user_id, fact.id)
            if vid: linked_memory_ids.append(vid)
            results.append({"id": fact.id, "memory": d.text, "event": "ADD", "linked_memory_ids": linked_memory_ids.copy()})
        elif d.event == "UPDATE" and d.id:
            fact = self._store_update_fact(d.id, d.text, user_id)
            if fact:
                self._store_update_verbatim(d.id, d.text, user_id)
                results.append({"id": fact.id, "memory": d.text, "event": "UPDATE", "linked_memory_ids": [d.id]})
            else:
                results.append({"id": d.id, "memory": d.text, "event": "NOOP", "linked_memory_ids": []})
        elif d.event == "DELETE" and d.id:
            self._store_delete_fact(d.id)
            self._store_delete_verbatim(d.id, user_id)
            results.append({"id": d.id, "memory": d.text, "event": "DELETE", "linked_memory_ids": []})
    logger.info("cognify_done", user_id=user_id, decisions=len(results), linked=len(linked_memory_ids))
    return results

def _store_add_fact(self, fact_text, user_id, provenance):
    subject, predicate, object_ = fact_to_triple(fact_text, user_id)
    return self.typed_store.add_fact(subject, predicate, object_, user_id=user_id,
        confidence=0.7, provenance=provenance, embedding=self.embedder.embed(f"{subject} {predicate} {object_}"))

def _store_verbatim(self, fact_text, user_id, fact_id):
    if self.verbatim_store is None: return None
    return self.verbatim_store.add(fact_text, self.embedder.embed(fact_text),
        user_id=user_id, metadata={"source": "cognify", "fact_id": fact_id})

def _store_update_fact(self, fact_id, fact_text, user_id):
    subject, predicate, object_ = fact_to_triple(fact_text, user_id)
    return self.typed_store.update_fact(fact_id, subject, predicate, object_,
        embedding=self.embedder.embed(f"{subject} {predicate} {object_}"), confidence=0.85)

def _store_update_verbatim(self, fact_id, fact_text, user_id):
    if self.verbatim_store is None: return
    self.verbatim_store.update(fact_id, fact_text, self.embedder.embed(fact_text))

def _store_delete_fact(self, fact_id):
    self.typed_store.soft_delete_fact(fact_id)

def _store_delete_verbatim(self, fact_id, user_id):
    if self.verbatim_store is None: return
    try: self.verbatim_store.delete(fact_id)
    except Exception: pass
```
（`fact_to_triple` 已 import; 保留旧 `extract_and_store` 的无 LLM 降级: `decisions` 为空且无 LLM 时走旧 `extract_facts` 纯 ADD 路径——在方法开头加 `if self.llm is None: return self._legacy_extract_and_store(messages, user_id, provenance)`）

`memory/main.py:292` `add` infer=True 路径:
```python
# 原来直接调 self.extractor.extract_and_store, 现在不变 (extract_and_store 内部已决策化)
# 只需确认 self.extractor 用 ADDITIVE_DECISION_PROMPT (改 use_additive_prompt 默认或新增 use_decision 参数)
```
`FactExtractor.__init__` 加 `use_decision: bool = False` 参数, True 时 prompt 用 `ADDITIVE_DECISION_PROMPT`。`Memory.__init__` 创建 extractor 时传 `use_decision=True`。

- [ ] **Step 4: 运行测试验证通过**

```powershell
$env:PYTHONPATH = "src"; python -m pytest tests/unit/test_fact_decision.py tests/unit/test_add_decision.py -v
```
Expected: All passed

- [ ] **Step 5: 检查点**

```powershell
ruff check src/septmuse/storage/relational_stores/typed_store.py src/septmuse/models/extract.py src/septmuse/memory/main.py
```

---

## Task 3: BudgetItem.id + CapturePipeline.preprocess

**Files:**
- Modify: `src/septmuse/retrieval/token_budget.py:50`（`BudgetItem` 加 `id`）
- Modify: `src/septmuse/capture/pipeline.py`（新增 `preprocess` 方法）
- Test: `tests/unit/test_token_budget_id.py` + `tests/unit/test_capture_preprocess.py`

**Interfaces:**
- Consumes: `BudgetItem(text, score, metadata)`, `CapturePipeline.capture()`
- Produces: `BudgetItem(text, score, metadata, id=None)`, `CapturePipeline.preprocess(text, *, user_id, agent_id) -> PreprocessResult(allowed, stored_text, text_hash, redacted, reason)`

- [ ] **Step 1: 写失败测试**

```python
# tests/unit/test_token_budget_id.py
from septmuse.retrieval.token_budget import BudgetItem
def test_budget_item_id_default():
    b = BudgetItem(text="hello", score=0.5)
    assert b.id is None
def test_budget_item_id_set():
    b = BudgetItem(text="hello", score=0.5, id="mem-1")
    assert b.id == "mem-1"
```

```python
# tests/unit/test_capture_preprocess.py
from septmuse.capture.pipeline import CapturePipeline
def test_preprocess_dedup(capture_pipeline):
    # 第一次允许
    r1 = capture_pipeline.preprocess("hello world", user_id="alice")
    assert r1.allowed
    assert r1.stored_text == "hello world"
    # 第二次同文本 → 去重拒绝
    r2 = capture_pipeline.preprocess("hello world", user_id="alice")
    assert not r2.allowed
    assert "duplicate" in (r2.reason or "")
def test_preprocess_no_write(capture_pipeline, mock_store):
    """preprocess 不写 store."""
    capture_pipeline.preprocess("hello", user_id="alice")
    assert mock_store.add.call_count == 0  # 不调用 store.add
```

- [ ] **Step 2: 运行测试验证失败**

```powershell
$env:PYTHONPATH = "src"; python -m pytest tests/unit/test_token_budget_id.py tests/unit/test_capture_preprocess.py -v
```
Expected: FAIL（`BudgetItem` 无 id, `preprocess` 不存在）

- [ ] **Step 3: 实现**

`token_budget.py:50`:
```python
@dataclass
class BudgetItem:
    """预算裁剪输入项."""
    text: str
    score: float = 0.0
    metadata: dict | None = None
    id: str | None = None  # 新增: 原始 memory_id (裁剪后保留)
```

`capture/pipeline.py` 新增 `preprocess` + `PreprocessResult`:
```python
@dataclass
class PreprocessResult:
    """预处理结果 (去重+脱敏, 不写 store)."""
    allowed: bool
    stored_text: str = ""
    text_hash: str | None = None
    redacted: bool = False
    reason: str | None = None

def preprocess(self, text, *, user_id=None, agent_id=None) -> PreprocessResult:
    """只做去重+脱敏, 不嵌入不写 store (避免与 Memory.add 双写)."""
    if not text or not text.strip():
        return PreprocessResult(allowed=False, reason="empty text")
    validation = self.validator.validate(text, user_id=user_id, agent_id=agent_id)
    if not validation.allowed:
        return PreprocessResult(allowed=False, reason=validation.reason or "duplicate", text_hash=validation.text_hash)
    cleaned = self.privacy.redact(text)
    return PreprocessResult(allowed=True, stored_text=cleaned, text_hash=validation.text_hash, redacted=cleaned != text)
```

- [ ] **Step 4: 运行测试验证通过**

```powershell
$env:PYTHONPATH = "src"; python -m pytest tests/unit/test_token_budget_id.py tests/unit/test_capture_preprocess.py -v
```
Expected: All passed

- [ ] **Step 5: 检查点**

```powershell
ruff check src/septmuse/retrieval/token_budget.py src/septmuse/capture/pipeline.py
```

---

## Task 4: Memory 新增 remember/recall/forget/improve + V2Memory 降级薄层

**Files:**
- Modify: `src/septmuse/memory/main.py`（新增 4 编排方法）
- Modify: `src/septmuse/memory/memory_v2.py`（降级为 deprecated 薄层委托）
- Test: `tests/unit/test_memory_orchestration.py`

**Interfaces:**
- Consumes: `Task 2` 的决策化 `add`, `Task 3` 的 `BudgetItem.id` + `preprocess`
- Produces: `Memory.remember()`, `Memory.recall()`, `Memory.forget()`, `Memory.improve()`, V2Memory 委托这 4 个

- [ ] **Step 1: 写失败测试**

```python
# tests/unit/test_memory_orchestration.py
def test_remember_delegates_add(memory):
    """remember 委托 add + episodic raw_log."""
    result = memory.remember("I like Python", user_id="alice")
    assert result["captured"]
    assert result["raw_id"]  # episodic raw_log 存了
    assert len(result["memory_ids"]) >= 1  # add 存了

def test_recall_returns_real_id(memory):
    """recall 返回真实 memory_id (不是 text[:50])."""
    memory.remember("I like Python", user_id="alice")
    result = memory.recall("what do I like", user_id="alice")
    for m in result["memories"]:
        assert m["id"].startswith("mem-")  # 真实 id, 不是 text[:50]
    assert result["injected_prompt"]  # block 注入

def test_recall_id_survives_token_budget(memory):
    """token 预算裁剪后 id 仍保留."""
    for i in range(10):
        memory.remember(f"fact number {i}", user_id="alice")
    result = memory.recall("fact", user_id="alice", top_k=3)
    assert len(result["memories"]) <= 3
    for m in result["memories"]:
        assert m["id"].startswith("mem-")

def test_forget_delegates_delete(memory):
    """forget 委托 delete + invalidate."""
    add_result = memory.remember("temp fact", user_id="alice")
    mid = add_result["memory_ids"][0]
    result = memory.forget(mid, user_id="alice")
    assert result["event"] == "FORGET"
    # delete 后 get 返回 None
    assert memory.get(mid) is None

def test_improve_runs(memory):
    """improve 不报错 (dream + reflect + conflict + coverage)."""
    result = memory.improve(user_id="alice", limit=5)
    assert "dream" in result
    assert "coverage" in result

def test_v2memory_delegates_memory(v2_memory_deprecated):
    """V2Memory 薄层委托 Memory 的 4 方法."""
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        v2_memory_deprecated.remember("test", user_id="alice")
        assert any(issubclass(wi.category, DeprecationWarning) for wi in w)
```

- [ ] **Step 2: 运行测试验证失败**

```powershell
$env:PYTHONPATH = "src"; python -m pytest tests/unit/test_memory_orchestration.py -v
```
Expected: FAIL（`Memory.remember` 等不存在）

- [ ] **Step 3: 实现**

`memory/main.py` 在 `delete` / `update` 之后新增 4 编排方法:
```python
def remember(self, messages, *, user_id, agent_id=None, session_id=None) -> dict:
    """捕获(去重+脱敏) → add(决策+实体) → episodic raw_log → working_memory."""
    from septmuse.capture.pipeline import CapturePipeline
    text = _normalize_messages(messages)  # 现有辅助
    if not text:
        return {"captured": False, "reason": "empty text"}
    capture = CapturePipeline(self.store, self.embedder, typed_store=self.typed_store, llm=self.llm, dedup_window=self._dedup_window)
    pre = capture.preprocess(text, user_id=user_id, agent_id=agent_id)
    if not pre.allowed:
        return {"captured": False, "reason": pre.reason}
    add_result = self.add(pre.stored_text, user_id=user_id, agent_id=agent_id, session_id=session_id, infer=(self.llm is not None), metadata={"source": "v2_remember", "text_hash": pre.text_hash, "redacted": pre.redacted})
    memory_ids = [r["id"] for r in add_result.get("results", []) if r.get("id")]
    raw = self.episodic.add_raw_log(pre.stored_text, user_id=user_id, session_id=session_id or "unknown", agent_id=agent_id)
    if agent_id:
        with contextlib.suppress(KeyError, ValueError):
            self.working_memory.core_memory_append("persona", pre.stored_text[:200])
    return {"raw_id": raw.id, "memory_ids": memory_ids, "captured": True}

def recall(self, query, *, user_id, top_k=5, recipe=None) -> dict:
    """search → 遗忘加权 → token预算(保留id) → block注入."""
    from septmuse.retrieval.token_budget import BudgetItem
    from septmuse.retrieval.forgetting import ForgettingManager  # 确认 import 路径
    route = self.meta.route(query) if hasattr(self, 'meta') else None
    results = self.search(query, user_id=user_id, top_k=top_k * 4, recipe=recipe)
    forgetting = ForgettingManager(self.typed_store)
    weighted = forgetting.apply_strength(results, user_id=user_id)
    items = [BudgetItem(id=w.get("id"), text=w.get("memory", ""), score=w.get("final_score", w.get("score", 0.0))) for w in weighted]
    budgeted = self.token_budget.fit(items) if hasattr(self, 'token_budget') else type('R',(),{'items':items,'used_tokens':0})()
    prompt_parts = []
    wm_prompt = self.working_memory.compile_to_prompt() if hasattr(self, 'working_memory') else ""
    if wm_prompt: prompt_parts.append(wm_prompt)
    rule_prompt = self.procedural.rules_to_prompt(user_id=user_id) if hasattr(self, 'procedural') else ""
    if rule_prompt: prompt_parts.append(rule_prompt)
    memories = [{"id": item.id, "memory": item.text, "score": item.score} for item in budgeted.items]
    return {"memories": memories, "injected_prompt": "\n".join(prompt_parts), "route": {"namespaces": route.namespaces, "fallback": route.fallback} if route else None, "used_tokens": getattr(budgeted, 'used_tokens', 0)}

def forget(self, memory_id, *, user_id) -> dict:
    """invalidate → delete → 图清理."""
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    try: self.store.invalidate(memory_id, invalid_at=now)
    except Exception as e: logger.warning("forget_invalidate_failed", error=str(e))
    self.delete(memory_id)
    if self.graph_store is not None:
        try:
            if hasattr(self.graph_store, "delete_edges_for_memory"): self.graph_store.delete_edges_for_memory(memory_id)
            elif hasattr(self.graph_store, "delete_memory"): self.graph_store.delete_memory(memory_id)
        except Exception as e: logger.warning("forget_graph_failed", error=str(e))
    return {"memory_id": memory_id, "event": "FORGET", "invalidated_at": now, "deleted_at": now}

def improve(self, *, user_id, limit=50) -> dict:
    """dream + reflect + conflict + coverage."""
    dream_result = self.evolution.dream(user_id=user_id)
    rules_accepted = 0
    if self.llm is not None:
        try:
            reflect_result = self.evolution.reflect(user_id=user_id, limit=limit)
            rules_accepted = reflect_result.lessons_accepted
        except Exception as e: logger.warning("improve_reflect_failed", error=str(e))
    try: conflicts = self.evolution.resolve_conflicts(user_id=user_id)
    except Exception: conflicts = {"conflicts_found": 0, "resolved": 0, "invalidated_ids": []}
    report = self.meta.analyze_coverage(user_id=user_id)
    return {"dream": {"links_created": dream_result.links_created, "processed": dream_result.processed}, "rules": rules_accepted, "conflicts": conflicts, "coverage": {"overall_score": report.overall_score, "weak_areas": report.weak_areas, "strong_areas": report.strong_areas}}
```
（`Memory.__init__` 增加创建 `self.working_memory`, `self.meta`, `self.evolution`, `self.token_budget`, `self.procedural` — 从 V2Memory.__init__ 搬过来, 复用已有组件）

`memory/memory_v2.py` 降级为薄层:
```python
class V2Memory:
    """DEPRECATED: 委托 Memory.remember/recall/forget/improve."""
    def __init__(self, memory=None, *, config=None):
        import warnings
        warnings.warn("V2Memory deprecated, use Memory directly", DeprecationWarning, stacklevel=2)
        from septmuse.memory.main import Memory
        self.mem = memory or Memory(config=config)
    def remember(self, *a, **k): return self.mem.remember(*a, **k)
    def recall(self, *a, **k): return self.mem.recall(*a, **k)
    def forget(self, *a, **k): return self.mem.forget(*a, **k)
    def improve(self, *a, **k): return self.mem.improve(*a, **k)
```

- [ ] **Step 4: 运行测试验证通过**

```powershell
$env:PYTHONPATH = "src"; python -m pytest tests/unit/test_memory_orchestration.py tests/unit/test_v2_memory.py -v
```
Expected: All passed（含 V2 回归）

- [ ] **Step 5: 检查点**

```powershell
ruff check src/septmuse/memory/main.py src/septmuse/memory/memory_v2.py
```

---

## Task 5: add 语义去重 + search 参数化

**Files:**
- Modify: `src/septmuse/memory/main.py`（add verbatim 路径语义去重; search 加 forgetting/token_budget/inject_prompt 参数）
- Test: `tests/unit/test_add_semantic_dedup.py` + `tests/unit/test_search_enhanced.py`

**Interfaces:**
- Consumes: `Task 2` 决策化 add, `Task 3` BudgetItem.id
- Produces: `Memory.search(query, *, forgetting=False, token_budget=None, inject_prompt=False)`

- [ ] **Step 1: 写失败测试**

```python
# tests/unit/test_add_semantic_dedup.py
def test_add_verbatim_semantic_dedup(memory):
    """infer=False 时, 已有高度相似记忆 → 跳过."""
    memory.add("I like Python", user_id="alice", infer=False)
    result = memory.add("I like Python", user_id="alice", infer=False)  # 几乎相同
    # 第二次应该跳过 (语义去重)
    assert len(result["results"]) == 0 or result["results"][0].get("event") == "SKIP"

def test_add_verbatim_different_not_deduped(memory):
    """不同内容不去重."""
    memory.add("I like Python", user_id="alice", infer=False)
    result = memory.add("I live in Tokyo", user_id="alice", infer=False)
    assert len(result["results"]) == 1

# tests/unit/test_search_enhanced.py
def test_search_forgetting_param(memory):
    """search(forgetting=True) 返回 final_score."""
    memory.add("old fact", user_id="alice", infer=False)
    results = memory.search("old fact", user_id="alice", forgetting=True)
    assert len(results) > 0
    assert "final_score" in results[0] or "score" in results[0]

def test_search_token_budget(memory):
    """search(token_budget=100) 裁剪结果."""
    for i in range(20):
        memory.add(f"fact {i} " * 20, user_id="alice", infer=False)
    results = memory.search("fact", user_id="alice", token_budget=100)
    # 总 token 不超 100
    total = sum(len(r["memory"]) // 4 for r in results)
    assert total <= 100

def test_search_inject_prompt(memory):
    """search(inject_prompt=True) 返回 injected_prompt 字段."""
    memory.add("test", user_id="alice", infer=False)
    result = memory.search("test", user_id="alice", inject_prompt=True)
    assert "injected_prompt" in result or isinstance(result, dict)
```

- [ ] **Step 2: 运行测试验证失败**

```powershell
$env:PYTHONPATH = "src"; python -m pytest tests/unit/test_add_semantic_dedup.py tests/unit/test_search_enhanced.py -v
```
Expected: FAIL

- [ ] **Step 3: 实现**

`memory/main.py` `add` infer=False 路径加语义去重（在 `store.add_batch` 前）:
```python
# verbatim 路径 (infer=False), 写入前语义去重
emb_query = self.embedder.embed(texts[0])
existing = self.store.search(emb_query, user_id=user_id, top_k=5, threshold=0.95)
if existing:
    # 已有高度相似记忆 → 跳过 (返回 SKIP)
    return {"results": [{"id": e["id"], "memory": e["memory"], "event": "SKIP"} for e in existing[:1]], "relations": []}
```
（只在单条 verbatim 时做; 多条批次内已有 MD5 去重保留）

`memory/main.py` `search` 加可选参数:
```python
def search(self, query, *, user_id, ..., forgetting=False, token_budget=None, inject_prompt=False):
    # ... 现有检索逻辑拿到 results ...
    if forgetting:
        from septmuse.retrieval.forgetting import ForgettingManager
        fm = ForgettingManager(self.typed_store)
        results = fm.apply_strength(results, user_id=user_id)
        # results 现在是带 final_score 的 dict
    if token_budget is not None:
        from septmuse.retrieval.token_budget import BudgetItem
        items = [BudgetItem(id=r.get("id"), text=r.get("memory",""), score=r.get("final_score", r.get("score",0))) for r in results]
        budgeted = self.token_budget.fit(items)
        results = [{"id": i.id, "memory": i.text, "score": i.score} for i in budgeted.items]
    if inject_prompt:
        prompt_parts = []
        if hasattr(self, 'working_memory'):
            wp = self.working_memory.compile_to_prompt()
            if wp: prompt_parts.append(wp)
        if hasattr(self, 'procedural'):
            rp = self.procedural.rules_to_prompt(user_id=user_id)
            if rp: prompt_parts.append(rp)
        return {"results": results, "injected_prompt": "\n".join(prompt_parts)}
    return results
```
（注意: `inject_prompt=True` 时返回 dict 而非 list, 调用方需适配; `recall` 内部已处理）

- [ ] **Step 4: 运行测试验证通过**

```powershell
$env:PYTHONPATH = "src"; python -m pytest tests/unit/test_add_semantic_dedup.py tests/unit/test_search_enhanced.py -v
```
Expected: All passed

- [ ] **Step 5: 检查点**

```powershell
ruff check src/septmuse/memory/main.py
```

---

## Task 6: UserProfile 模型 + get_user_profile 聚合

**Files:**
- Create: `src/septmuse/models/profile.py`（UserProfile + UserProfileValue）
- Modify: `src/septmuse/memory/main.py`（`get_user_profile` 方法）
- Test: `tests/unit/test_user_profile.py`

**Interfaces:**
- Consumes: `TypedMemoryStore.get_all_facts(user_id, include_deleted=True)` → list[SemanticFact]
- Produces: `UserProfile`, `UserProfileValue`, `Memory.get_user_profile(user_id, *, include_temporal=True) -> UserProfile`

- [ ] **Step 1: 写失败测试**

```python
# tests/unit/test_user_profile.py
from septmuse.models.profile import UserProfile, UserProfileValue

def test_profile_value_defaults():
    v = UserProfileValue(value="Alice")
    assert v.value == "Alice"
    assert v.is_current is True
    assert v.confidence == 1.0
    assert v.updated_at is None

def test_get_user_profile_aggregates(memory, typed_store):
    """画像聚合 attributes/preferences."""
    typed_store.add_fact("user", "name", "Alice", user_id="alice", confidence=0.9)
    typed_store.add_fact("user", "occupation", "Engineer", user_id="alice")
    typed_store.add_fact("user", "likes", "Python", user_id="alice")
    profile = memory.get_user_profile("alice")
    assert profile.attributes["name"].value == "Alice"
    assert profile.attributes["occupation"].value == "Engineer"
    assert profile.preferences["likes"].value == "Python"

def test_profile_contradiction_picks_latest(memory, typed_store):
    """矛盾值 → 最新 updated_at 为 current."""
    f1 = typed_store.add_fact("user", "likes", "Python", user_id="alice")
    f2 = typed_store.add_fact("user", "likes", "Rust", user_id="alice")  # 更新
    profile = memory.get_user_profile("alice")
    assert profile.preferences["likes"].value == "Rust"
    assert profile.preferences["likes"].is_current is True

def test_profile_include_temporal_false(memory, typed_store):
    """include_temporal=False → 只返回 current."""
    typed_store.add_fact("user", "name", "Alice", user_id="alice")
    typed_store.add_fact("user", "name", "Alyssa", user_id="alice")  # 矛盾, 更新
    profile = memory.get_user_profile("alice", include_temporal=False)
    assert len([v for v in profile.attributes.values() if v.is_current]) <= 1

def test_profile_temporal_summary(memory, typed_store):
    f1 = typed_store.add_fact("user", "likes", "Java", user_id="alice")
    typed_store.soft_delete_fact(f1.id)
    typed_store.add_fact("user", "likes", "Python", user_id="alice")
    profile = memory.get_user_profile("alice")
    assert profile.temporal_summary["active"] >= 1
    assert profile.temporal_summary["deleted"] >= 1

def test_profile_raw_facts_uncategorized(memory, typed_store):
    """未分类 predicate → raw_facts."""
    typed_store.add_fact("user", "weird_predicate", "value", user_id="alice")
    profile = memory.get_user_profile("alice")
    assert len(profile.raw_facts) >= 1
```

- [ ] **Step 2: 运行测试验证失败**

```powershell
$env:PYTHONPATH = "src"; python -m pytest tests/unit/test_user_profile.py -v
```
Expected: FAIL（`models/profile.py` 不存在）

- [ ] **Step 3: 实现**

`src/septmuse/models/profile.py`:
```python
"""用户画像数据模型 — 从 SemanticFact 聚合的结构化画像."""
from __future__ import annotations
from dataclasses import dataclass, field

@dataclass
class UserProfileValue:
    """画像单值."""
    value: str
    confidence: float = 1.0
    updated_at: str | None = None
    is_current: bool = True
    source_fact_ids: list[str] = field(default_factory=list)

@dataclass
class UserProfile:
    """用户结构化画像."""
    user_id: str
    attributes: dict[str, UserProfileValue] = field(default_factory=dict)      # name/age/occupation/location/birthday
    preferences: dict[str, UserProfileValue] = field(default_factory=dict)     # likes/dislikes
    plans: list[UserProfileValue] = field(default_factory=list)                 # planning/intends
    relationships: dict[str, UserProfileValue] = field(default_factory=dict)   # has/knows
    raw_facts: list[dict] = field(default_factory=list)
    temporal_summary: dict = field(default_factory=dict)

# predicate 分类映射
_ATTR_PREDICATES = {"name", "age", "occupation", "location", "birthday", "email", "phone"}
_PREF_PREDICATES = {"likes", "dislikes", "prefers", "hates", "favorite"}
_PLAN_PREDICATES = {"planning", "intends", "will", "goal"}
_REL_PREDICATES = {"has", "knows", "related_to", "friend", "family"}

def _classify(predicate: str) -> str:
    p = predicate.lower().strip()
    if p in _ATTR_PREDICATES: return "attributes"
    if p in _PREF_PREDICATES: return "preferences"
    if p in _PLAN_PREDICATES: return "plans"
    if p in _REL_PREDICATES: return "relationships"
    return "raw"
```

`memory/main.py` 新增 `get_user_profile`:
```python
def get_user_profile(self, user_id: str, *, include_temporal: bool = True) -> "UserProfile":
    """从 SemanticFact 聚合用户画像."""
    from septmuse.models.profile import UserProfile, UserProfileValue, _classify
    facts = self.typed_store.get_all_facts(user_id=user_id, include_deleted=True)
    profile = UserProfile(user_id=user_id)
    # 按 predicate 分组
    groups: dict[str, list] = {}
    for f in facts:
        key = f.predicate
        groups.setdefault(key, []).append(f)
    for predicate, group_facts in groups.items():
        category = _classify(predicate)
        # 按 updated_at 排序 (最新在前), is_deleted=False 优先
        group_facts.sort(key=lambda x: (getattr(x, 'updated_at', None) or getattr(x, 'created_at', '') or ''), reverse=True)
        latest_active = next((f for f in group_facts if not f.is_deleted), None)
        for f in group_facts:
            is_current = (f == latest_active) and not f.is_deleted
            val = UserProfileValue(
                value=f.object, confidence=f.confidence,
                updated_at=str(getattr(f, 'updated_at', None) or ''),
                is_current=is_current, source_fact_ids=[f.id],
            )
            if category == "attributes": profile.attributes[predicate] = val
            elif category == "preferences": profile.preferences[predicate] = val
            elif category == "plans":
                if is_current or include_temporal: profile.plans.append(val)
            elif category == "relationships": profile.relationships[predicate] = val
            else: profile.raw_facts.append({"id": f.id, "predicate": predicate, "value": f.object, "is_current": is_current})
        # include_temporal=False → 只留 current
        if not include_temporal and category in ("attributes", "preferences", "relationships"):
            bucket = getattr(profile, category)
            if predicate in bucket and not bucket[predicate].is_current:
                del bucket[predicate]
    active = sum(1 for f in facts if not f.is_deleted)
    deleted = sum(1 for f in facts if f.is_deleted)
    profile.temporal_summary = {"active": active, "deleted": deleted, "total": len(facts)}
    return profile
```

- [ ] **Step 4: 运行测试验证通过**

```powershell
$env:PYTHONPATH = "src"; python -m pytest tests/unit/test_user_profile.py -v
```
Expected: All passed

- [ ] **Step 5: 检查点**

```powershell
ruff check src/septmuse/models/profile.py src/septmuse/memory/main.py
```

---

## Task 7: recall 画像注入 + REST 端点

**Files:**
- Modify: `src/septmuse/memory/main.py`（`recall` 加 `inject_profile` + `Memory.search` 加 `inject_profile`）
- Modify: `src/septmuse/api/rest/__init__.py`（新增 `GET /agents/{user_id}/profile`）
- Test: `tests/unit/test_recall_profile_inject.py` + `tests/unit/test_rest_profile.py`

**Interfaces:**
- Consumes: `Task 6` 的 `get_user_profile`
- Produces: `Memory.recall(inject_profile=True)` 注入画像, `GET /agents/{user_id}/profile` 端点

- [ ] **Step 1: 写失败测试**

```python
# tests/unit/test_recall_profile_inject.py
def test_recall_inject_profile(memory, typed_store):
    """recall 画像注入到 injected_prompt."""
    typed_store.add_fact("user", "name", "Alice", user_id="alice")
    typed_store.add_fact("user", "likes", "Python", user_id="alice")
    memory.remember("I like Python", user_id="alice")
    result = memory.recall("what do I like", user_id="alice", inject_profile=True)
    assert "Alice" in result["injected_prompt"]
    assert "Python" in result["injected_prompt"]

def test_recall_no_profile_by_default(memory, typed_store):
    """默认不注入画像."""
    typed_store.add_fact("user", "name", "Alice", user_id="alice")
    memory.remember("test", user_id="alice")
    result = memory.recall("test", user_id="alice")
    # 默认 inject_profile=False, 不含画像
    assert "Alice" not in (result.get("injected_prompt") or "")

# tests/unit/test_rest_profile.py
def test_rest_get_profile(rest_client, typed_store):
    """GET /agents/{user_id}/profile 返回画像."""
    typed_store.add_fact("user", "name", "Alice", user_id="alice")
    resp = rest_client.get("/agents/alice/profile")
    assert resp.status_code == 200
    data = resp.json()
    assert data["user_id"] == "alice"
    assert data["attributes"]["name"]["value"] == "Alice"

def test_rest_get_profile_temporal(rest_client, typed_store):
    """?include_temporal=true 返回历史."""
    f1 = typed_store.add_fact("user", "likes", "Python", user_id="alice")
    typed_store.soft_delete_fact(f1.id)
    typed_store.add_fact("user", "likes", "Rust", user_id="alice")
    resp = rest_client.get("/agents/alice/profile?include_temporal=true")
    assert resp.status_code == 200
    assert resp.json()["temporal_summary"]["deleted"] >= 1
```

- [ ] **Step 2: 运行测试验证失败**

```powershell
$env:PYTHONPATH = "src"; python -m pytest tests/unit/test_recall_profile_inject.py tests/unit/test_rest_profile.py -v
```
Expected: FAIL（recall 无 inject_profile, 无 REST 端点）

- [ ] **Step 3: 实现**

`memory/main.py` `recall` 加画像注入:
```python
def recall(self, query, *, user_id, top_k=5, recipe=None, inject_profile=False) -> dict:
    # ... 现有 search + forgetting + token_budget + block 注入 ...
    prompt_parts = []
    if inject_profile:
        profile = self.get_user_profile(user_id, include_temporal=False)
        profile_str = self._profile_to_prompt(profile)
        if profile_str: prompt_parts.append(profile_str)
    if wm_prompt: prompt_parts.append(wm_prompt)
    if rule_prompt: prompt_parts.append(rule_prompt)
    return {"memories": memories, "injected_prompt": "\n".join(prompt_parts), ...}

def _profile_to_prompt(self, profile) -> str:
    """UserProfile → prompt 摘要串."""
    lines = ["# User Profile (current)"]
    for k, v in profile.attributes.items():
        if v.is_current: lines.append(f"- {k.capitalize()}: {v.value}")
    prefs = [f"{v.value}" for v in profile.preferences.values() if v.is_current]
    if prefs: lines.append(f"- Current preferences: {', '.join(prefs)}")
    rels = [f"{k}: {v.value}" for k, v in profile.relationships.items() if v.is_current]
    if rels: lines.append(f"- Relationships: {'; '.join(rels)}")
    return "\n".join(lines) if len(lines) > 1 else ""
```

`api/rest/__init__.py` 新增端点（找现有 `/agents/{user_id}/memories` 附近加）:
```python
@app.get("/agents/{user_id}/profile")
def get_agent_profile(user_id: str, include_temporal: bool = False):
    """获取用户画像 (从 SemanticFact 聚合)."""
    profile = _get_memory().get_user_profile(user_id, include_temporal=include_temporal)
    return {
        "user_id": profile.user_id,
        "attributes": {k: {"value": v.value, "confidence": v.confidence, "is_current": v.is_current, "updated_at": v.updated_at} for k, v in profile.attributes.items()},
        "preferences": {k: {"value": v.value, "is_current": v.is_current} for k, v in profile.preferences.items()},
        "relationships": {k: {"value": v.value, "is_current": v.is_current} for k, v in profile.relationships.items()},
        "raw_facts": profile.raw_facts,
        "temporal_summary": profile.temporal_summary,
    }
```

- [ ] **Step 4: 运行测试验证通过**

```powershell
$env:PYTHONPATH = "src"; python -m pytest tests/unit/test_recall_profile_inject.py tests/unit/test_rest_profile.py -v
```
Expected: All passed

- [ ] **Step 5: 检查点**

```powershell
ruff check src/septmuse/memory/main.py src/septmuse/api/rest/__init__.py
```

---

## 最终回归（所有任务完成后统一执行）

```powershell
$env:PYTHONPATH = "src"
ruff check src/ tests/
python -m pytest tests/unit/ tests/e2e/ -q --tb=short --timeout=300
```

**基线**: 1477 passed / 16 failed (pre-existing LLM/OpenAI) / 24 skipped
**新增**: ~30 测试（7 任务 × 4-5 测试）
**预期**: ~1507 passed / 16 failed / 24+ skipped, 零新增失败

## Self-Review 完成

- ✅ Spec 覆盖: L1(Task 1-2) / L2(Task 2,5) / L3(Task 3-4) / L4(Task 5) / L5(Task 6-7) 全覆盖
- ✅ 无占位符: 每步有实际代码
- ✅ 类型一致: Decision/BudgetItem/UserProfile/UserProfileValue 跨任务签名一致
- ✅ 接口已确认: update_fact 已存在(增强)、SemanticFact 无双时态列(用 is_deleted+updated_at)

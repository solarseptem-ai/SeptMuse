# P0 实施计划: Memory.add 与 mem0 V3 对齐

> 日期: 2026-08-07
> spec: `docs/superpowers/specs/2026-08-07-p0-memory-add-alignment.md`
> 模式: subagent-driven development

## Task 1: add infer=True 决策接入 (linked_memory_ids 透传)

**文件**: `src/septmuse/memory/main.py`
**改动**: `add()` 方法 infer=True 路径

当前:
```python
if should_infer and self.extractor is not None:
    extracted = self.extractor.extract_and_store(messages, user_id=user_id)
    return {"results": extracted, "relations": []}
```

改为:
```python
if should_infer and self.extractor is not None:
    extracted = self.extractor.extract_and_store(messages, user_id=user_id)
    # extracted 已含 event + linked_memory_ids, 直接透传
    self._invalidate_search_cache(user_id)
    return {"results": extracted, "relations": []}
```

**验证**: add(infer=True) 返回的 results 含 linked_memory_ids 字段 (非空时)

**测试**: `tests/unit/test_add_decision_linked.py`
- MockLLM 输出 `{"facts":[{"text":"User likes Python","event":"ADD"}]}` → add 返回 event="ADD" + linked_memory_ids 列表
- MockLLM 输出 `{"facts":[{"text":"User likes Python","event":"NOOP"}]}` → add 返回 event="NOOP"

---

## Task 2: add Phase 0/1 上下文窗口 (last_k_messages)

**文件**: `src/septmuse/models/extract.py` + `src/septmuse/prompts/extract.py`

### 2a: FactExtractor.__init__ 加 episodic_store 参数

```python
class FactExtractor:
    def __init__(self, llm, embedder, typed_store, verbatim_store, *,
                 use_decision=True, episodic_store=None):
        ...
        self.episodic_store = episodic_store  # 可选, EpisodicMemoryStore
```

### 2b: extract_and_store 内部取 last_k_messages

```python
def extract_and_store(self, messages, *, user_id, provenance="inferred"):
    ...
    existing = self._retrieve_existing(text, user_id)
    last_k = self._get_last_k_messages(user_id)  # 新增
    decisions = self.extract_with_decisions(messages, existing_memories=existing, last_k_messages=last_k)
    ...
```

### 2c: _get_last_k_messages 方法

```python
def _get_last_k_messages(self, user_id: str, limit: int = 5) -> list[dict]:
    """取近期 episodic 事件作为对话上下文 (降级: 无 episodic_store 返回空)."""
    if self.episodic_store is None:
        return []
    try:
        events = self.episodic_store.get_timeline(user_id=user_id, limit=limit)
        return [{"role": "assistant", "content": getattr(e, "content", str(e))} for e in events]
    except Exception:
        return []
```

### 2d: extract_with_decisions 传 last_k_messages

```python
def extract_with_decisions(self, messages, existing_memories=None, last_k_messages=None):
    ...
    user_prompt = build_extraction_user_prompt(text, existing_memories, last_k_messages=last_k_messages or [])
    ...
```

### 2e: build_extraction_user_prompt 加 last_k 段落

`prompts/extract.py` 的 `build_extraction_user_prompt` 加:
```python
def build_extraction_user_prompt(text, existing_memories, last_k_messages=None):
    ...
    if last_k_messages:
        k_lines = "\n".join(f"{m['role']}: {m['content']}" for m in last_k_messages)
        sections.append(f"## Last k Messages\n{k_lines}")
    ...
```

### 2f: Memory.__init__ 传 episodic_store 给 FactExtractor

```python
self.extractor = FactExtractor(
    ...,
    episodic_store=self.episodic,  # 已有 self.episodic
)
```

**验证**: FactExtractor 有 episodic_store 时 last_k 非空; 无时降级空列表

**测试**: `tests/unit/test_extract_context.py`
- MockEpisodicStore 返回 3 条事件 → build_extraction_user_prompt 含 "Last k Messages" 段落
- episodic_store=None → last_k_messages=[] (降级不崩)

---

## Task 3: 实体批量链接

**文件**: `src/septmuse/storage/relational_stores/entity_store.py` + `src/septmuse/memory/main.py`

### 3a: EntityStore.upsert_batch 方法

```python
def upsert_batch(self, items, *, user_id, agent_id=None):
    """批量 upsert (对齐 mem0 V3 Phase 7).
    
    Args:
        items: [(Entity, set[memory_id]), ...] 列表
    Returns: [entity_id, ...]
    """
    # 1. 批量 embed
    texts = [e.text for e, _ in items]
    embeddings = self._embedder.embed_batch(texts) if self._embedder else [None]*len(texts)
    
    # 2. 批量精确匹配
    normalized_texts = [_normalize_entity_text(t) for t in texts]
    exact_matches = self._find_by_text_batch(normalized_texts, user_id=user_id)
    
    # 3. 批量语义匹配 (未精确命中的)
    unmatched = [(i, items[i]) for i in range(len(items)) if normalized_texts[i] not in exact_matches]
    semantic_matches = {}
    if unmatched and self._embedder:
        unmatched_embs = [embeddings[i] for i, _ in unmatched]
        semantic_matches = self._find_by_embedding_batch(unmatched_embs, user_id=user_id, threshold=0.95)
    
    # 4. 分流: 命中 → append; 未命中 → insert
    results = []
    to_insert = []
    for i, (entity, memory_ids) in enumerate(items):
        exact = exact_matches.get(normalized_texts[i])
        semantic = semantic_matches.get(i)
        match = exact or semantic
        if match:
            self._append_memory_ids(match["id"], memory_ids)
            results.append(match["id"])
        else:
            to_insert.append((i, entity, memory_ids, embeddings[i]))
    
    if to_insert:
        new_ids = self._batch_insert(to_insert, user_id=user_id, agent_id=agent_id)
        for (idx, entity, mids, emb), eid in zip(to_insert, new_ids):
            results.insert(idx, eid)  # 保持顺序
    
    return results
```

### 3b: _find_by_text_batch

```python
def _find_by_text_batch(self, normalized_texts, *, user_id):
    """SQL IN 批量精确匹配. Returns {normalized_text: entity_dict}."""
    if not normalized_texts:
        return {}
    with Session(self._engine) as session:
        stmt = select(EntityTable).where(
            EntityTable.user_id == user_id,
            EntityTable.is_deleted == 0,
            EntityTable.entity_text.in_(normalized_texts),
        )
        rows = session.exec(stmt).all()
    return {row.entity_text: self._row_to_dict(row) for row in rows}
```

### 3c: _find_by_embedding_batch

```python
def _find_by_embedding_batch(self, embeddings, *, user_id, threshold=0.95):
    """全表扫 + numpy 矩阵余弦. Returns {input_index: entity_dict}."""
    with Session(self._engine) as session:
        stmt = select(EntityTable).where(
            EntityTable.user_id == user_id,
            EntityTable.is_deleted == 0,
        )
        rows = session.exec(stmt).all()
    if not rows:
        return {}
    import numpy as np
    db_embs = np.array([self._deserialize_embedding(r.entity_embedding) for r in rows])
    query_embs = np.array(embeddings)
    # 归一化
    db_norm = db_embs / (np.linalg.norm(db_embs, axis=1, keepdims=True) + 1e-8)
    query_norm = query_embs / (np.linalg.norm(query_embs, axis=1, keepdims=True) + 1e-8)
    sims = query_norm @ db_norm.T  # [n_query, n_db]
    matches = {}
    for i in range(len(embeddings)):
        best = np.argmax(sims[i])
        if sims[i][best] >= threshold:
            matches[i] = self._row_to_dict(rows[best])
    return matches
```

### 3d: _batch_insert / _append_memory_ids / _row_to_dict 辅助

```python
def _batch_insert(self, items, *, user_id, agent_id):
    """批量插入新实体."""
    now = datetime.now(timezone.utc).isoformat()
    new_ids = []
    with Session(self._engine) as session:
        for entity, memory_ids, emb in [(e, m, embs) for _, e, m, embs in items]:
            eid = str(uuid.uuid4())
            emb_blob = self._serialize_embedding(emb) if emb is not None else None
            row = EntityTable(
                id=eid, entity_text=entity.text, entity_type=entity.entity_type,
                entity_embedding=emb_blob, linked_memory_ids=json.dumps(sorted(memory_ids)),
                user_id=user_id, agent_id=agent_id, created_at=now, updated_at=now, is_deleted=0,
            )
            session.add(row)
            new_ids.append(eid)
        session.commit()
    return new_ids

def _append_memory_ids(self, entity_id, memory_ids):
    """批量追加 memory_ids (set → list)."""
    with Session(self._engine) as session:
        row = session.get(EntityTable, entity_id)
        if not row:
            return
        existing = set(json.loads(row.linked_memory_ids))
        existing.update(memory_ids)
        row.linked_memory_ids = json.dumps(sorted(existing))
        row.updated_at = datetime.now(timezone.utc).isoformat()
        session.add(row)
        session.commit()

def _row_to_dict(self, row):
    return {"id": row.id, "entity_text": row.entity_text, "entity_type": row.entity_type, ...}
```

### 3e: _batch_extract_and_store_entities 改用 upsert_batch

`memory/main.py`:
```python
def _batch_extract_and_store_entities(self, pairs, *, user_id, agent_id=None):
    if not pairs or self.entity_extractor is None or self.entity_store is None:
        return
    from septmuse.extraction.entity import Entity
    try:
        # 全局去重
        global_entities = {}
        for text, memory_id in pairs:
            for entity in self.entity_extractor.extract(text):
                key = entity.text.strip().lower()
                if key in global_entities:
                    global_entities[key][1].add(memory_id)
                else:
                    global_entities[key] = (entity, {memory_id})
        
        items = [(e, mids) for (e, mids) in global_entities.values()]
        self.entity_store.upsert_batch(items, user_id=user_id, agent_id=agent_id)
    except Exception as e:
        logger.warning("batch_extract_entities_failed", error=str(e))
```

**验证**: upsert_batch 3 条实体 → 2 精确命中 + 1 新建 = 1 次 embed_batch + 1 次 SQL IN + 1 次 INSERT

**测试**: `tests/unit/test_entity_batch.py`
- 3 条新实体 → upsert_batch 返回 3 个 id, DB 新增 3 行
- 1 条已存在 + 2 条新 → 精确命中追加 memory_id, 2 条新建
- 1 条语义相似 (score>0.95) → 语义命中追加
- 无 embedder → 只精确匹配

---

## Task 4: update 实体重链接

**文件**: `src/septmuse/memory/main.py`

### 4a: Memory.update 加实体重链接

```python
def update(self, memory_id, content, *, metadata=None, user_id="default"):
    # 取旧内容判断 text_changed
    old = self.store.get(memory_id)
    text_changed = old is not None and old.get("memory") != content
    
    emb = self.embedder.embed(content)
    self.store.update(memory_id, content, emb, metadata=metadata)
    
    # 实体重链接 (文本变更时)
    if text_changed and self.entity_store is not None and self.entity_extractor is not None:
        try:
            self.entity_store.remove_memory_from_entities(memory_id)
            self._extract_and_store_entities(content, memory_id, user_id=user_id)
        except Exception as e:
            logger.warning("update_entity_relink_failed", error=str(e))
    
    self._invalidate_search_cache(user_id)
    return {"id": memory_id, "event": "UPDATE"}
```

**注意**: `Memory.update` 当前签名是 `update(self, memory_id, content, *, metadata=None)`,需加 `user_id` 参数。`store.get` 取旧内容。

### 4b: 确认 store.get 返回含 memory 字段

`ORMMemoryStore.get` 返回 `{"id", "memory", "score", "metadata", "created_at"}`,`memory` 字段已有。

**验证**: update "I like Python" → "I like Rust" 后, Python 实体不再 linked, Rust 实体新建 linked

**测试**: `tests/unit/test_update_relink.py`
- add("I like Python") → update(mem_id, "I like Rust") → "Python" 实体 linked_memory_ids 不含 mem_id, "Rust" 实体 linked_memory_ids 含 mem_id
- update 文本不变 → 实体不重链接
- 无 entity_store → 不崩 (降级)

---

## 验证命令

```powershell
$env:PYTHONPATH = "src"
ruff check src/septmuse/memory/main.py src/septmuse/models/extract.py src/septmuse/storage/relational_stores/entity_store.py src/septmuse/prompts/extract.py
python -m pytest tests/unit/test_add_decision_linked.py tests/unit/test_extract_context.py tests/unit/test_entity_batch.py tests/unit/test_update_relink.py -q
python -m pytest tests/unit/test_memory.py tests/unit/test_fact_decision.py tests/unit/test_fact_extraction.py -q  # 回归
```

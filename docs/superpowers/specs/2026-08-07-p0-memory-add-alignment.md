# P0: Memory.add 与 mem0 V3 对齐设计

> 日期: 2026-08-07
> 状态: 待审阅
> 范围: `src/septmuse/memory/main.py` + `src/septmuse/models/extract.py` + `src/septmuse/storage/relational_stores/entity_store.py`

## 1. 背景与动机

当前 SeptMuse `Memory.add` 与 mem0 `Memory.add` 存在 4 项核心差距,导致记忆系统在 LLM 场景下决策断裂、实体链接性能差、update 不清理旧实体。本设计对齐 mem0 V3 的 4 个关键 Phase,让 Memory.add 真正可用。

## 2. 四项差距与改造

### Task 1: add infer=True 决策真正接入

**现状**: `Memory.add(infer=True)` 调 `self.extractor.extract_and_store(messages, user_id=)`,后者已实现决策路由 (ADD/UPDATE/DELETE/NOOP),但 `Memory.add` 内部未传 `session_id`/`agent_id`/`metadata`,且 `extract_and_store` 的 `linked_memory_ids` 在返回中被丢弃。

**改造**:
```python
# memory/main.py add() infer=True 路径
if should_infer and self.extractor is not None:
    extracted = self.extractor.extract_and_store(
        messages,
        user_id=user_id,
        provenance="inferred",
    )
    # extracted 已含 event + linked_memory_ids, 直接透传
    self._invalidate_search_cache(user_id)
    return {"results": extracted, "relations": []}
```

**不变**: `extract_and_store` 内部已完整 (Phase 1 检索 → 决策 → 路由),不改其逻辑。只需确保 `Memory.add` 不丢弃返回的 `linked_memory_ids`。

### Task 2: add Phase 0/1 上下文窗口

**现状**: `FactExtractor._retrieve_existing` 只做向量检索 top-10,不取 `last_k_messages` (近期对话历史)。LLM 无法解析代词/引用 (如 "他"/"那个"/"上次说的")。

**改造**:
```
FactExtractor.__init__ 加 episodic_store 参数 (可选, EpisodicMemoryStore)
_extract_existing 补 last_k_messages:
  1. episodic_store.get_timeline(user_id, limit=5) → 近 5 条 episodic 事件
  2. 传给 build_extraction_user_prompt 的 last_k_messages 参数
```

**降级**: 无 episodic_store 时 `last_k_messages=[]`,不阻塞。

**prompt 改造**: `build_extraction_user_prompt(text, existing_memories, last_k_messages=[])` 加入 last_k 段落。

### Task 3: 实体批量链接

**现状**: `_batch_extract_and_store_entities` 全局去重后**逐个 upsert** (每个实体一次 embed + 一次 DB 查询 + 一次 DB 写入)。100 条记忆 × 3 实体 = 300 次 embed + 300 次 DB round-trip。

**改造** (对齐 mem0 V3 Phase 7):
```
1. 全局去重 (已有): normalized_key → (entity_type, entity_text, set[memory_id])
2. 批量 embed: embedder.embed_batch([entity_texts]) → 一次调用
3. 批量精确匹配: EntityStore._find_by_text_batch(texts, user_id) → 一次 SQL IN 查询
4. 批量语义匹配: EntityStore._find_by_embedding_batch(embeddings, user_id, threshold) → 一次全表扫 + numpy 矩阵余弦
5. 分流: 命中 → 批量 UPDATE linked_memory_ids; 未命中 → 批量 INSERT
```

**新增方法**:
- `EntityStore.upsert_batch(entities, memory_ids, *, user_id, agent_id)` — 批量 upsert 入口
- `EntityStore._find_by_text_batch(normalized_texts, user_id)` — SQL `WHERE entity_text IN (...)`
- `EntityStore._find_by_embedding_batch(embeddings, user_id, threshold)` — 全表扫 + numpy 矩阵余弦
- `EntityStore._batch_append_memory_ids(updates)` — 批量 UPDATE (case when / 逐条)

### Task 4: update 实体重链接

**现状**: `ORMMemoryStore.update` 只更新 content + embedding + metadata,**不清理旧实体也不链接新实体**。文本从 "I like Python" 改成 "I like Rust" 后,旧 "Python" 实体仍指向该 memory_id,新 "Rust" 实体不建立。

**改造** (对齐 mem0 `_update_memory`):
```
Memory.update(memory_id, content, metadata)
  → store.update(memory_id, content, embedding, metadata)
  → if text_changed:
      entity_store.remove_memory_from_entities(memory_id)  # 清旧
      _extract_and_store_entities(content, memory_id, ...)   # 链新
  → _invalidate_search_cache(user_id)
```

**位置**: `Memory.update` (facade 层),非 `ORMMemoryStore.update` (存储层)。存储层只管数据,facade 层管编排。

## 3. 数据流 (改造后)

### add (infer=True, 决策模式)
```
add(messages, user_id, infer=True, agent_id, session_id, metadata)
  FactExtractor._retrieve_existing (向量检索 top-10)
  FactExtractor._get_last_k_messages (episodic timeline top-5)          ← Task 2 新增
  → ADDITIVE_DECISION_PROMPT(text, existing, last_k) → [Decision]
  → 路由: ADD/UPDATE/DELETE/NOOP (置信度守卫 <0.7 → NOOP)
  → 返回 [{id, memory, event, linked_memory_ids}]
  → _invalidate_search_cache
```

### add (infer=False, verbatim 模式)
```
add(messages, user_id, infer=False)
  → embed_batch(texts)
  → store.add_batch (hash 去重 + 批量插入)
  → _batch_extract_and_store_entities (批量 embed + 批量匹配 + 批量 upsert)  ← Task 3 改造
  → _invalidate_search_cache
```

### update (文本变更)
```
update(memory_id, content, metadata)
  → store.update(memory_id, content, embedding, metadata)
  → if text_changed:                                                     ← Task 4 新增
      entity_store.remove_memory_from_entities(memory_id)
      _extract_and_store_entities(content, memory_id, user_id, agent_id)
  → _invalidate_search_cache(user_id)
```

## 4. 不破坏的承诺

- `Memory.add/search/update/delete` 签名**向后兼容** (新增参数有默认值)
- `extract_and_store` 返回结构**兼容** (event 字段已存在,旧消费者按 event=="ADD" 过滤仍工作)
- `EntityStore.upsert` 保留 (向后兼容),新增 `upsert_batch`
- `EntityStore.remove_memory_from_entities` 已存在 (Task 4 复用)
- 现有测试**不改断言绕过缺陷**

## 5. 风险与对策

| 风险 | 对策 |
|------|------|
| 批量语义匹配全表扫慢 (大用户实体多) | EntityStore 默认 SQLite,实体量 <10K 全表扫 <10ms; 超大用户加 LIMIT |
| LLM 决策不稳定 (误 UPDATE/DELETE) | 置信度守卫 <0.7 降级 NOOP (已实现) |
| last_k_messages 取 episodic 失败 | try/except 降级为空列表,不阻塞 |
| update 实体重链接失败 | try/except 吞错,不阻塞 update 主流程 |

## 6. 测试策略

| 测试文件 | 覆盖 |
|----------|------|
| `test_add_decision_linked.py` | add(infer=True) 返回 linked_memory_ids 透传 |
| `test_extract_context.py` | FactExtractor last_k_messages 传入 + 降级 |
| `test_entity_batch.py` | upsert_batch 批量精确+语义匹配+新建 |
| `test_update_relink.py` | update 文本变更时实体清理+重链接 |

## 7. 不在本设计范围

- AsyncMemory 决策接入 (P3)
- `expiration_date` / `attributed_to` / `actor_id` (P1)
- `filters` 统一为 dict 形式 (P1)
- `reset()` 方法 (P2)
- BM25 自适应归一化 (P2)

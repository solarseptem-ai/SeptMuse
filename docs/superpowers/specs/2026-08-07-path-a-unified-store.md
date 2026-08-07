# 路径 A: 统一存储改造设计

> 日期: 2026-08-07
> 状态: 待审阅
> 范围: `services/database/models/memory.py` + `storage/relational_stores/orm_store.py` + `memory/main.py`

## 1. 核心改造

MemoryTable 加 `content_type` + `typed_data` 两列, 成为统一存储. 所有 memory_type 路由写入同一表, search/update/forget 天然覆盖全部类型.

## 2. 表结构变更

### memories 表 (MemoryTable) 新增列

```python
content_type: str = Field(default="verbatim", index=True)
    # "verbatim" | "fact" | "episode" | "rule" | "procedural"
typed_data: str | None = Field(default=None, sa_column=Column("typed_data", Text))
    # JSON: fact={subject,predicate,object,context,confidence,provenance,tags}
    #       episode={event_type,reference_time,observation,thoughts,action,result}
    #       rule={namespace,helpful_count,harmful_count,source_tracing,deprecated,tags}
    #       procedural={namespace,source_tracing}
    #       verbatim=None
```

### 迁移策略: 运行时 ALTER TABLE (对齐现有 _migrate_add_state_columns 模式)

ORMMemoryStore._create_tables() 后加 _migrate_add_content_type().

## 3. add 路由改造

当前 `add(memory_type="fact")` 只写 typed_store. 改为**双写**:

```python
if memory_type == "fact":
    # 1. 写 typed_store (保留, 支持 search_facts 等类型化查询)
    fact = self.semantic.add_fact(...)
    # 2. 写 verbatim store (新增, 统一检索路径)
    typed_data = json.dumps({"subject": subject, "predicate": predicate, "object": object, ...})
    mid = self.store.add(content, embedding, content_type="fact", typed_data=typed_data, ...)
    return {"id": fact.id, "memory_id": mid, "triple": fact.as_triple(), "event": "ADD"}
```

同理 episode/rule/procedural 都双写. content 字段统一存可检索文本 (fact=object, rule=rule, episode=content).

## 4. search 改造

search 已查 memories 表, 双写后天然覆盖所有类型. 新增 content_type 过滤:

```python
def search(self, query, *, content_type=None, ...):
    filters = {"content_type": content_type} if content_type else None
    results = self.store.search(emb, user_id=user_id, filters=filters, ...)
```

## 5. update/forget 改造

已有逻辑操作 memories 表, 双写后天然覆盖所有类型. 无需改动.

## 6. 错误处理改造

- __init__ 吞错保留 (降级设计), 新增 health_check() 方法
- close() 补全 graph_store / working_memory / keyword_index

## 7. 不破坏的承诺

- typed_store 表保留 (向后兼容, search_facts/get_timeline 等仍走 typed_store)
- 现有 add/search/update/delete 签名不变 (新增 content_type/typed_data 有默认值)
- content_type="verbatim" 默认, 旧数据自动兼容

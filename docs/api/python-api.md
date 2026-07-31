# Python API 参考

> 源码：`src/septmuse/orchestration/memory.py`

`Memory` 类是 SeptMuse 的零配置 facade（借鉴 mem0 `Memory`）。`pip install septmuse` 后直接 `from septmuse import Memory` 即可用。

## 构造函数

```python
from septmuse import Memory, MemoryConfig

m = Memory()
# 或自定义配置
m = Memory(config=MemoryConfig(db_path=":memory:"))
# 或依赖注入（测试用）
m = Memory(config=cfg, embedder=emb, store=store, llm=llm)
```

```python
Memory(
    config: MemoryConfig | None = None,
    *,
    embedder: Embedder | None = None,
    store: MemoryStore | None = None,
    graph_store: GraphStore | None = None,
    llm: LLM | None = None,
    entity_extractor: EntityExtractor | None = None,
)
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `config` | `default_config()` | 配置对象（读环境变量） |
| `embedder` | `_resolve_embedder(config)` | 嵌入器（默认 HashEmbedder） |
| `store` | `SQLiteMemoryStore(config.db_path)` | 存储后端 |
| `graph_store` | `SQLiteGraphStore` (复用 store 的 conn) | 图存储（None=图功能不可用） |
| `llm` | `None`（若 `config.llm_provider` 已设则自动创建） | LLM provider |
| `entity_extractor` | `_resolve_entity_extractor(config)` | 实体抽取器 |

---

## 基础 CRUD

### add

```python
m.add(
    messages: str | list[dict],
    *,
    user_id: str,
    agent_id: str | None = None,
    metadata: dict | None = None,
    infer: bool | None = None,
    auto_extract_entities: bool = True,
    valid_at: str | None = None,
) -> dict
```

| 参数 | 说明 |
|------|------|
| `messages` | `str` 或 `list[{"role","content"}]` |
| `user_id` | 用户 ID（必填，跨 agent 共享键） |
| `infer` | `True`=LLM 抽取事实；`False`=原文存；`None`=用 `config.infer` |
| `auto_extract_entities` | `True`=verbatim 模式下自动抽取实体 |
| `valid_at` | 事实开始为真的时间（ISO 8601） |

**返回：**

```python
# infer=False (默认)
{"results": [{"id": "mem_xxx", "memory": "...", "event": "ADD"}], "relations": []}

# infer=True (需 LLM)
{"results": [{"id": "...", "memory": "...", "triple": [...], "event": "ADD"}], "relations": []}
```

---

### search

```python
m.search(
    query: str,
    *,
    user_id: str,
    top_k: int | None = None,
    threshold: float | None = None,
    hybrid: bool = True,
    reranker: str | None = None,
    explain: bool = False,
    recipe: str | None = None,
) -> list[dict]
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `hybrid` | `True` | `True`=BM25+向量 RRF 融合；`False`=纯向量 |
| `reranker` | `None` | `noop`/`mmr`/`cross_encoder`/`llm` |
| `explain` | `False` | `True`=返回 `score_details` |
| `recipe` | `None` | 预置检索配置名（覆盖 hybrid/reranker/explain） |

**recipe 可选值：** `HYBRID_RRF`（默认）/ `HYBRID_RRF_ENTITY` / `HYBRID_RRF_CROSS_ENCODER` / `HYBRID_RRF_MMR` / `GRAPH_BFS` / `PROGRESSIVE` / `FORGETTING`

**返回：**

```python
[
    {
        "id": "mem_xxx",
        "memory": "我喜欢 Python",
        "score": 0.85,           # 统一相似度 [0,1]
        "vector_score": 0.90,
        "bm25_score": 0.80,
        "entity_boost": 0.05,    # 实体提升信号
        "metadata": null,
        "created_at": "2026-07-27T..."
    }
]
```

---

### get_all / get / delete / invalidate / update / get_history

```python
m.get_all(*, user_id: str) -> dict          # {"results": [...]}
m.get(memory_id: str) -> dict | None        # 取单条
m.delete(memory_id: str) -> dict            # 软删除 + 清理实体引用
m.invalidate(memory_id: str, *, invalid_at: str | None = None) -> dict  # 双时态失效
m.update(memory_id: str, data: str | None = None, *, user_id: str | None = None, metadata: dict | None = None) -> dict
m.get_history(memory_id: str) -> list[dict]  # ADD/UPDATE/DELETE 记录
```

**invalidate 返回：**

```python
{"id": "mem_xxx", "invalid_at": "...", "expired_at": "...", "event": "INVALIDATE"}
# 或
{"id": "mem_xxx", "event": "NOT_FOUND"}
```

---

## 工作记忆 Block（对齐 Letta）

```python
m.get_working_memory(agent_id: str) -> WorkingMemory
m.get_blocks(agent_id: str) -> list[dict]
m.update_block(agent_id: str, label: str, value: str) -> dict
m.core_memory_append(agent_id: str, label: str, content: str) -> dict
m.core_memory_replace(agent_id: str, label: str, old_content: str, new_content: str) -> dict
```

---

## 类型化记忆

### 语义事实（SemanticFact）

```python
m.add_fact(subject, predicate, object, *, user_id, context=None, confidence=1.0, provenance="user", tags=None) -> dict
m.update_fact(fact_id, *, subject, predicate, object, user_id) -> dict
m.search_facts(query, *, user_id, top_k=5) -> list[dict]
```

### 情节事件（EpisodicEvent）

```python
m.add_episode(
    content, *, user_id,
    event_type: str = "fact",  # fact | reasoning | raw_log
    session_id: str | None = None,
    observation: str | None = None,   # reasoning 时填
    thoughts: str | None = None,      # reasoning 时填
    action: str | None = None,        # reasoning 时填
    result: str | None = None,         # reasoning 时填
) -> dict
m.update_episode(episode_id, *, content, user_id) -> dict
m.get_timeline(*, user_id, event_type=None, limit=50) -> list[dict]
```

### 程序规则（ProceduralRule）

```python
m.add_rule(rule, *, user_id, namespace="default", source_tracing=None) -> dict
m.update_rule(rule_id, *, rule, user_id) -> dict
m.record_rule_outcome(rule_id, helpful: bool) -> dict   # helpful/harmful 追踪 + 自动退化
m.get_active_rules(*, user_id, namespace="default") -> list[dict]
m.rules_to_prompt(*, user_id, namespace="default") -> str  # 编译为 prompt 注入文本
```

---

## 捕获与检索

### capture

PostToolUse 捕获流水线（SHA256 去重 → 脱敏 → 嵌入 → 双索引）。

```python
m.capture(text, *, user_id, agent_id=None, **kwargs) -> dict
# {"captured": bool, "memory_id": str|None, "deduped": bool, "redacted": bool}
```

### search_hybrid

BM25 + 向量 RRF 融合检索（三信号：向量 + BM25 + entity_boost）。

```python
m.search_hybrid(query, *, user_id, top_k=5, threshold=0.1, explain=False) -> list[dict]
```

### search_at

时态查询：查询某时刻为真的事实。

```python
m.search_at(reference_time, query, *, user_id, top_k=5, threshold=0.1) -> list[dict]
```

过滤条件：`valid_at <= reference_time AND (invalid_at IS NULL OR invalid_at > reference_time)`

`valid_at IS NULL` 的记忆视为"无时间约束"，始终返回（向后兼容）。

### search_interval

时态区间查询：返回 [start, end) 内为真的相关记忆。

```python
m.search_interval(start, end, query, *, user_id, top_k=5, threshold=0.1) -> list[dict]
```

条件：`valid_at <= end AND (invalid_at IS NULL OR invalid_at > start)`

### search_natural

自然语言时态查询（LLM 从查询抽取时间区间）。

```python
m.search_natural(query, *, user_id, top_k=5, threshold=0.1) -> list[dict]
```

流程：LLM 抽取时间（如"上周" → `{start, end}`）→ 有时态过滤 → 无回退普通 search。无 LLM 时回退普通 search。

### compress

消息压缩 Summarizer（对齐 letta Summarizer）。

```python
m.compress(*, user_id, mode="static", buffer_size=20) -> dict
```

| 模式 | 行为 |
|------|------|
| `static` | 固定缓冲区，超限驱逐旧消息 + LLM 摘要（验收：50→20+1） |
| `partial` | 驱逐 30% 旧消息 + LLM 摘要插入 |

**返回：** `{"compressed": bool, "evicted": int, "kept": int, "summary_id": str | None}`

### search_progressive / search_with_strength

```python
m.search_progressive(query, *, user_id, top_k=5, threshold=0.1) -> list[dict]  # 渐进三层 recall→locate→expand
m.search_with_strength(query, *, user_id, top_k=5, threshold=0.1) -> list[dict]  # 遗忘曲线加权 final_score=relevance×strength
```

### apply_token_budget / redact

```python
m.apply_token_budget(texts: list[str], scores=None, budget=2000) -> list[str]  # token 预算裁剪
m.redact(text: str) -> str  # 隐私脱敏
```

---

## 演化（Evolution）

### link_on_add / get_related

Zettelkasten 自动建链接（借鉴 cognee cognify）。

```python
m.link_on_add(memory_id, text, *, user_id) -> list[dict]  # [{"id","source_id","target_id","score"}]
m.get_related(memory_id) -> list[dict]                     # 获取记忆的链接邻居
```

### search_graph / search_graph_fused

BFS 图遍历检索（借鉴 graphiti bfs_search）。

```python
m.search_graph(seed_memory_id, *, max_depth=2, relation=None) -> list[dict]
# [{"id","memory","depth","graph_score"}]  graph_score = 1/2^depth

m.search_graph_fused(query, *, user_id, seed_memory_id, max_depth=2, relation=None, top_k=None) -> list[dict]
# BFS + 向量 RRF 融合 → [{"id","memory","fused_score","vector_score"}]
```

### cognify

构建知识图谱（借鉴 cognee cognify）：存记忆 → 抽三元组 → 存实体/关系 → 建记忆链接。

```python
m.cognify(text, *, user_id, agent_id=None) -> dict
# {"memory_id","triplets","entities","relations","links"}
```

### get_entity_relations

获取实体间关系邻居（双向遍历）。

```python
m.get_entity_relations(entity_name, *, user_id) -> list[dict]
# [{"entity","relation","direction"}]
```

### reflect

Session 反思蒸馏（借鉴 cognee distill）：curator 提取课程 → writer/rejecter 新颖性搜索 + LLM 判定。

```python
m.reflect(*, user_id, limit=20) -> dict
# {"proposed": int, "accepted": int, "rule_ids": [...]}
```

### dream

Dream 整合：空闲期批量建链接（借鉴 ReMe Dream）。

```python
m.dream(*, user_id) -> dict
# {"processed": int, "links_created": int}
```

---

## 共享（Sharing）

```python
m.list_agents(user_id: str) -> list[str]  # 列出该用户的所有 agent
m.is_cross_agent(user_id: str) -> bool    # 检查是否跨 agent 共享
```

---

## 因果 / 复述 / 元认知

### 因果图

```python
m.add_causal_edge(cause_event_id, effect_event_id, *, user_id, relation="causes", confidence=0.5) -> dict
m.find_causes(event_id, *, user_id) -> list[dict]    # [{"path":[...],"confidence","length"}]
m.find_effects(event_id, *, user_id) -> list[dict]
m.counterfactual(cause_event_id, effect_event_id, *, user_id) -> dict
# {"would_still_occur": bool, "confidence": float, "reasoning": str}
```

### 主动复述

```python
m.rehearse(memory_id, *, user_id) -> dict
# {"memory_id","strength","access_count"}

m.find_rehearse_candidates(*, user_id) -> list[dict]
# [{"memory_id","strength","base_value"}]  strength<0.3 且 base_value>0.7
```

### 元认知

```python
m.meta_route(query) -> dict       # L0 路由: {"namespaces":[...],"fallback","scores"}
m.coverage_report(*, user_id) -> dict  # L1 覆盖自描述
# {"overall_score","weak_areas","strong_areas","namespaces":[...],"summary"}
m.adapt_strategy(*, user_id) -> dict    # L2 策略自调
```

---

## 实体（Entity）

```python
m.extract_entities(text) -> list[dict]               # [{"text","type","start","end"}] 不存储
m.add_entity(entity_text, entity_type, memory_id, *, user_id) -> dict
m.search_entities(query, *, user_id, top_k=5) -> list[dict]  # [{"id","entity_text","entity_type","linked_memory_ids","score"}]
m.get_entity_neighbors(entity_id) -> list[str]       # 获取实体关联的 memory_id 列表
m.list_entities(*, user_id, entity_type=None, limit=100) -> list[dict]
```

---

## 生命周期

```python
m.close()  # 关闭存储
```

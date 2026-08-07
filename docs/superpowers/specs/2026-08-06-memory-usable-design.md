# SeptMuse 记忆可用性改造设计

## 元信息

- **日期**: 2026-08-06
- **状态**: 待审阅
- **目标**: 让 SeptMuse 记忆系统完全可用——方法间协作（对齐 mem0 单 facade）+ 用户画像聚合
- **相关**: `docs/specs/2026-08-04-v2-memory-architecture.md`（V2 原架构, 本设计在其上重构协作层）
- **不提交**: 按用户指示, 本 spec 及后续实现暂不 commit

---

## 1. 背景与动机

SeptMuse 记忆系统当前有 `Memory`（CRUD facade, `memory/main.py`）和 `V2Memory`（编排 facade, `memory/memory_v2.py`）两个入口, 但存在根本性协作缺陷, 导致"记忆不可用":

1. **重复/矛盾/过时事实堆积**: `FactExtractor.extract_and_store` 只输出 `event: "ADD"`, `ADDITIVE_EXTRACTION_PROMPT` 只要求输出 `{facts: [...]}`. 用户说"我现在喜欢 Rust 了", 旧事实"喜欢 Python"不会被 UPDATE/DELETE → 矛盾事实共存, 过时事实堆积.
2. **双 facade 平行通路割裂**: V2Memory 绕过 `Memory.add/search/delete` 自己重组数据通路 → 双实例缓存不同步 + recall id bug + 功能遗漏（实体抽取/缓存失效）.
3. **无用户画像聚合**: 只有 `get_all_facts(user_id)` 返回扁平 `list[SemanticFact]`, 无结构化"关于 alice 的画像"视图, 无法直接注入 agent prompt.
4. **高阶能力只绑 V2**: 遗忘曲线/token 预算/block 注入/双时态/图清理只在 V2, 用 Memory 用不到.

**对齐 mem0 的核心**: 单一 facade + `add` 内部调 `search` 检索已有记忆 → LLM 做 ADD/UPDATE/DELETE/NOOP 决策 → 按决策路由. 方法间共享同一套组件实例, 缓存统一管理.

---

## 2. 根因诊断（5 层缺陷）

| 层 | 缺陷 | 现状证据 | 后果 | 对齐 mem0 什么 |
|----|------|----------|------|----------------|
| **L1 决策缺失** | `FactExtractor` 只 ADD, prompt 只输出 facts 列表 | `models/extract.py:208` event 恒 "ADD"; `prompts/extract.py:89` 输出 `{facts:[...]}` | 重复堆积、矛盾共存、过时不更新 | mem0 `ADDITIVE_EXTRACTION_PROMPT` 输出 `{text, event, id}` |
| **L2 add 不调 search** | `Memory.add` 不检索已有做决策 (verbatim 路径仅批次内 MD5 去重) | `main.py:306` store.add_batch | 每次新增, 不知已有相似记忆 | mem0 `_add_to_vector_store` Phase 1 |
| **L3 V2 平行通路** | V2 绕过 Memory 重组数据流 | `memory_v2.py:135` 自己 new HybridRetriever; recall id 用 `item.text[:50]` | 双缓存、id bug、功能遗漏 | mem0 单 facade |
| **L4 能力分散** | 遗忘/token预算/block 只在 V2 | `memory_v2.py:267` forgetting, `:271` token_budget | 用 Memory 用不到这些 | mem0 search 参数化 |
| **L5 无画像聚合** | 只有扁平 `get_all_facts` | `semantic.py:88` 返回 list | 无法结构化呈现用户, 无法注入画像 | mem0 用户记忆 → 画像 |

---

## 3. 目标架构

```
┌──────────────────────────────────────────────────────────────┐
│  Memory (唯一 facade, 对齐 mem0)                               │
│  ├── add()     内部 search → LLM ADD/UPDATE/DELETE 决策 → 路由 │
│  ├── search()  hybrid + reranker + recipe + 可选遗忘/预算/block│
│  ├── update() / delete() / get() / get_all()                  │
│  ├── get_user_profile()  ← L5 用户画像聚合 (新增)              │
│  └── 高阶编排 (薄层, 从 V2 搬入, 委托 self.*):                 │
│      ├── remember()  = add(决策) + episodic + working_mem     │
│      ├── recall()     = search + 遗忘 + 预算 + block + 画像注入 │
│      ├── forget()     = invalidate + delete + 图清理            │
│      └── improve()    = dream + reflect + conflict + 覆盖      │
└──────────────────────────────────────────────────────────────┘
         ↑ 复用同一套组件实例 (store/embedder/llm/retriever)
         ↑ V2Memory 保留为 deprecated 薄兼容层, 委托 Memory
```

- **单一 facade**: Memory 是唯一对外的记忆入口. V2 的 4 个编排方法**降级为 Memory 的方法**, V2Memory 保留为 deprecated 薄层委托.
- **方法间协作**: `add` 内部调 `search` 检索已有记忆 → LLM 做 ADD/UPDATE/DELETE/NOOP 决策 → 按决策路由 add/update/delete.
- **高阶能力可选**: 遗忘加权、token 预算、block 注入、画像注入——作为 Memory 的**可选参数 / 便捷方法**.
- **用户画像**: `get_user_profile` 从扁平事实聚合结构化画像, recall 可注入.

---

## 4. 改造蓝图（5 层, 自底向上）

### L1 — Prompt + FactExtractor 决策化（根基）

**改动 1.1**: `prompts/extract.py` 新增决策版 prompt

```
ADDITIVE_DECISION_PROMPT 输出格式:
{
  "facts": [
    {"text": "Likes Rust now",      "event": "ADD",    "id": null,      "confidence": 0.9},
    {"text": "Likes Python",        "event": "UPDATE", "id": "mem-xxx", "confidence": 0.85},
    {"text": "Likes Java",          "event": "DELETE", "id": "mem-yyy", "confidence": 0.6},
    {"text": "...",                 "event": "NOOP",   "id": "mem-zzz", "confidence": 1.0}
  ]
}
```
`confidence` 为可选字段（LLM 输出, 默认 1.0）, 用于 DELETE/UPDATE 决策的置信度阈值（< 0.7 降级为 NOOP, 避免误删/误改）。

prompt 指令要点（对齐 mem0）:
- 输入: 已有记忆列表 + 新消息
- 对每条事实输出 `event`:
  - `ADD`: 新事实, 已有记忆中没有
  - `UPDATE`: 已有记忆的更新版（同 subject+predicate, 值变了）, 必须带 `id`
  - `DELETE`: 与新消息矛盾的已有记忆, 必须带 `id`
  - `NOOP`: 已存在且无变化, 跳过
- 含 few-shot 示例覆盖 4 种决策
- 返回 ONLY valid JSON

保留 `ADDITIVE_EXTRACTION_PROMPT`（向后兼容, 纯抽取模式 / 无 LLM 降级用）.

**改动 1.2**: `models/extract.py` 新增 `extract_with_decisions` + 增强 `extract_and_store`

```python
@dataclass
class Decision:
    text: str
    event: str  # "ADD" | "UPDATE" | "DELETE" | "NOOP"
    id: str | None = None  # UPDATE/DELETE 必填
    confidence: float = 1.0  # LLM 决策置信度, DELETE/UPDATE < 0.7 降级 NOOP

def extract_with_decisions(self, messages, existing_memories) -> list[Decision]:
    """LLM 决策抽取, 返回带 event 的决策列表."""
    # 用 ADDITIVE_DECISION_PROMPT
    # 解析 {facts: [{text, event, id, confidence}]}, 容错降级为全 ADD
```

`extract_and_store` 增强（不破坏旧返回结构）:
- 内部调 `extract_with_decisions`
- **置信度守卫**: DELETE/UPDATE 的 `confidence < 0.7` 降级为 NOOP（避免误删/误改, 决策日志记降级原因）
- 按 event 路由:
  - `ADD` → `typed_store.add_fact` + `verbatim_store.add`（现有逻辑）
  - `UPDATE` → `typed_store.update_fact`（已存在 typed_store.py:121, 签名 `(fact_id, subject, predicate, object)`, 增强补 embedding/confidence 参数）+ `verbatim_store.update`
  - `DELETE` → `typed_store.soft_delete_fact`（已有, P3-Task 3）+ `verbatim_store.delete` + `verbatim_store.invalidate`（双时态, 保留历史）
  - `NOOP` → 跳过
- 返回 `[{id, memory, triple, event, linked_memory_ids}]`（event 不再恒 ADD）
- 解析失败/无 LLM → 降级为纯 ADD（现有 `extract_facts` 路径）

**改动 1.3**: `Memory.add(infer=True)` 调增强后的 `extract_and_store`（自动走决策路由）

**决策日志可追溯**: store.update/delete 的 history 表记 `event=UPDATE/DELETE`, `reason="llm_decision"` 字段记原因（metadata 里存 LLM 原始决策）.

### L2 — add 内部 search 协作 + 语义去重

**协作（已存在, 补强）**: `FactExtractor._retrieve_existing`（extract.py:144）已调 `verbatim_store.search`. L1 修完决策, add→search 协作自然成立. **无需额外改动 add 显式调 search**——决策由 FactExtractor 内部完成.

**补强 — verbatim 路径语义去重**:
`Memory.add(infer=False)` 当前仅批次内 MD5 去重. 补**批次前 top-K 语义去重**:
```python
# infer=False 路径, 写入前
existing = self.store.search(embed(text), user_id=user_id, top_k=5, threshold=0.95)
if existing:  # 已有高度相似记忆 → 跳过或更新 metadata.hit_count
    skip or merge
```
阈值 0.95（几乎相同才跳过, 避免误杀相似但不同的事实）.

### L3 — V2 降级为 Memory 方法（消除双 facade）

**V2Memory 保留为 deprecated 薄兼容层**:
```python
class V2Memory:  # deprecated, 委托 Memory
    def __init__(self, memory=None, *, config=None):
        warnings.warn("V2Memory deprecated, use Memory", DeprecationWarning)
        self.mem = memory or Memory(config=config)
    def remember(self, *a, **k): return self.mem.remember(*a, **k)
    def recall(self, *a, **k): return self.mem.recall(*a, **k)
    def forget(self, *a, **k): return self.mem.forget(*a, **k)
    def improve(self, *a, **k): return self.mem.improve(*a, **k)
```

**Memory 新增 4 个编排方法**（从 V2 搬入, 委托 self.add/search/delete）:

| 方法 | 委托 | 独有增强（V2 搬入） |
|------|------|----------------------|
| `remember()` | `self.add(infer=llm!=None)` | capture.preprocess(去重+脱敏) + episodic.add_raw_log + working_memory |
| `recall()` | `self.search(recipe=)` | forgetting.apply_strength + token_budget(保留id) + block+rule 注入 + meta.route + 画像注入(L5) |
| `forget()` | `self.delete()` | store.invalidate(双时态) + 图边清理 |
| `improve()` | — | evolution.dream + reflect + conflict + coverage（相对独立, 基本照搬） |

**组件去重**: 删除 V2 重复实例化的 `retrieval/semantic/episodic/procedural`（V2.__init__ 里 self.* 的）, 全部用 `self.mem.*`（即 Memory 的）.

### L4 — 高阶能力参数化（让 Memory.search 也能用）

`Memory.search` 新增可选参数:
```python
def search(self, query, *, user_id, top_k=5,
           forgetting: bool = False,           # 遗忘曲线加权
           token_budget: int | None = None,    # token 预算裁剪 (None=不限)
           inject_prompt: bool = False,        # 返回 block+rule 注入串
           inject_profile: bool = False,       # 返回用户画像摘要 (L5)
           ...):
```
- `recall()` = `search(forgetting=True, token_budget=2000, inject_prompt=True, inject_profile=True)` 的便捷封装
- 不用 V2 的用户也能: `mem.search(query, user_id="alice", forgetting=True)`

### L5 — 用户画像聚合（新增）

**改动 5.1**: 新增 `UserProfile` 数据模型

> **时态适配**: `SemanticFact` 无 `valid_at/invalid_at` 双时态列（只有 `is_deleted` + `touch()` 更新 `updated_at`）. 画像的"当前有效"= `is_deleted=False`, "最新"= `updated_at` 最新. verbatim `MemoryTable` 有双时态列, 但画像聚合从 `SemanticFact` 来.

```python
@dataclass
class UserProfileValue:
    """画像单值 (一个 predicate 的当前/历史值)."""
    value: str
    confidence: float = 1.0
    updated_at: str | None = None      # 最后更新时间 (touch() 维护)
    is_current: bool = True             # 是否当前有效 (is_deleted=False 且同 predicate 最新)
    source_fact_ids: list[str] = None   # 来源 fact (可能多个, 取最新有效)

@dataclass
class UserProfile:
    """用户结构化画像 (从 SemanticFact 聚合)."""
    user_id: str
    attributes: dict[str, UserProfileValue]      # 基本信息: name/age/occupation/location/birthday
    preferences: dict[str, UserProfileValue]     # 偏好: likes/dislikes (predicate→value)
    plans: list[UserProfileValue]                 # 计划/意图 (时序)
    relationships: dict[str, UserProfileValue]   # 关系网
    raw_facts: list[dict]                         # 未分类 predicate 的原始事实 (兜底)
    temporal_summary: dict                        # 时态概览: {active, deleted, total}
```

**改动 5.2**: `Memory.get_user_profile(user_id, *, include_temporal=True)`

聚合逻辑:
1. `typed_store.get_all_facts(user_id, include_deleted=True)` 拿全部 SemanticFact（含已软删除, 用于时态概览）
2. 按 predicate 分类:
   - `name/age/occupation/location/birthday` → `attributes`
   - `likes/dislikes/prefers` → `preferences`
   - `planning/intends` → `plans`（按 `updated_at` 排序）
   - `has/related_to/knows` → `relationships`
   - 其他 → `raw_facts`
3. 每组内**去重定当前值**:
   - 同 predicate 多个值 → 按 `updated_at` 排序, `is_deleted=False` 且最新的 `is_current=True`, 其余 `is_current=False`
   - 矛盾值（同 predicate 不同 object 且都 `is_deleted=False`）→ 取 `updated_at` 最新的为 current, 旧的标记 `is_current=False`（不软删除, 留给 L1 决策处理）
   - `include_temporal=False` 时只返回 `is_current=True` 的（精简画像, 不含历史/已删除）
4. `temporal_summary` = `{active: N(is_deleted=False), deleted: M(is_deleted=True), total: N+M}`

**改动 5.3**: recall 画像注入

`recall(query, user_id, inject_profile=True)`:
- 调 `get_user_profile(user_id)`
- 生成画像摘要串:
  ```
  # User Profile (current)
  - Name: Alice
  - Occupation: Software engineer at Google
  - Current preferences: Rust, vim, Tokyo
  - Relationships: dog named Buddy
  ```
- 注入到 `injected_prompt` 前部（block + 规则之前）

**改动 5.4**: REST 端点 `GET /agents/{user_id}/profile`

返回 `UserProfile` JSON. 支持 `?include_temporal=true/false`.

---

## 5. 数据流（改造后）

### add（决策化, infer=True）
```
add(messages, user_id, infer=True)
  FactExtractor._retrieve_existing (search 已有 top-K)     ← L2 协作
  → ADDITIVE_DECISION_PROMPT → [Decision]                   ← L1 决策
  → 按 event 路由:
      ADD    → typed_store.add_fact + verbatim_store.add
      UPDATE → typed_store.update_fact + verbatim_store.update
      DELETE → typed_store.soft_delete_fact + verbatim_store.delete + invalidate
      NOOP   → skip
  → 实体抽取 + 缓存失效
  → 返回 [{id, memory, event}]  ← event 不再恒 ADD
```

### add（verbatim, infer=False）
```
add(messages, user_id, infer=False)
  批次内 MD5 去重 + 批次前 top-K 语义去重 (>0.95 跳过)   ← L2 补强
  → store.add_batch + 实体抽取 + 缓存失效
  → 返回 [{id, memory, event: "ADD"}]
```

### remember（委托 add）
```
remember(messages, user_id, agent_id, session_id)
  capture.preprocess (去重 DedupWindow + 脱敏 PrivacyFilter)   ← V2 独有, 预处理
  self.add(pre.stored_text, infer=llm!=None)                    ← 委托, 含决策+实体+缓存失效
  self.episodic.add_raw_log                                    ← V2 独有
  self.working_memory.core_memory_append (agent_id 存在时)      ← V2 独有
  返回 {raw_id, memory_ids, captured}
```

### recall（委托 search + 增强 + 画像注入）
```
recall(query, user_id, top_k, recipe)
  self.search(query, recipe=recipe, top_k*4)               ← 委托, 含 hybrid+reranker
  forgetting.apply_strength                                ← V2 独有
  token_budget.fit(保留 id! BudgetItem.id)                 ← L4: BudgetItem 加 id
  meta.route + meta.adapt_strategy (L1 报告存在时)         ← V2 独有
  get_user_profile → 画像摘要 (inject_profile=True 时)      ← L5 新增
  working_memory.compile + procedural.rules_to_prompt      ← V2 独有
  返回 {memories:[{id,memory,score}], injected_prompt, route, strategy, used_tokens}
```

### forget（委托 delete + 增强）
```
forget(memory_id, user_id)
  store.invalidate (双时态失效)     ← V2 独有
  self.delete(memory_id)            ← 委托, 含实体清理+缓存失效
  graph_store 图边清理              ← V2 独有
  返回 {memory_id, event: "FORGET", invalidated_at, deleted_at}
```

### improve（相对独立, 基本照搬）
```
improve(user_id, limit)
  evolution.dream          → 链接生长 (embedding 相似)
  evolution.reflect         → 情节蒸馏规则 (LLM)
  evolution.resolve_conflicts → 冲突解决 (精确+模糊)
  meta.analyze_coverage     → L1 报告
  _persist_coverage         → 持久化标记
  返回 {dream, rules, conflicts, coverage}
```

### get_user_profile（L5 新增）
```
get_user_profile(user_id, include_temporal=True)
  typed_store.get_all_facts(user_id)
  → 按 predicate 分类 (attributes/preferences/plans/relationships/raw)
  → 时态去重 (同 predicate 取最新有效, 矛盾值标记 is_current)
  → temporal_summary 统计
  返回 UserProfile
```

---

## 6. 辅助改造

| 改动 | 位置 | 目的 |
|------|------|------|
| `BudgetItem` 加 `id: str \| None = None` 字段 | `retrieval/token_budget.py` | 修 recall id 丢失 bug |
| `CapturePipeline.preprocess()` 新方法 | `capture/pipeline.py` | 只做去重+脱敏, 不写 store (避免与 mem.add 双写) |
| `ADDITIVE_DECISION_PROMPT` 新增 | `prompts/extract.py` | 决策版 prompt, 输出 event |
| `Decision` dataclass + `extract_with_decisions` | `models/extract.py` | 解析 event 路由 |
| `TypedMemoryStore.update_fact` 增强 | `storage/relational_stores/typed_store.py:121` | 已存在, 补 embedding/confidence 参数 (UPDATE 路由用) |
| `UserProfile` + `UserProfileValue` dataclass | `models/profile.py` (新文件) | 画像数据模型 |
| `Memory.get_user_profile` 方法 | `memory/main.py` | 画像聚合入口 |
| `GET /agents/{user_id}/profile` 端点 | `api/rest/__init__.py` | REST 画像查询 |

---

## 7. 实施顺序（5 阶段, 每阶段可独立验证）

| 阶段 | 内容 | 验证 | 依赖 |
|------|------|------|------|
| **S1 决策根基** | ADDITIVE_DECISION_PROMPT + Decision + extract_with_decisions + TypedMemoryStore.update_fact + Memory.add 接决策 | 新增 test_fact_decision（ADD/UPDATE/DELETE/NOOP 4 场景） | 无 |
| **S2 V2 委托 + 修 bug** | Memory 加 remember/recall/forget/improve + V2Memory 降级薄层 + BudgetItem.id + capture.preprocess | 现有 test_memory + test_v2_memory 全绿, recall 返回真实 id | S1 |
| **S3 add 语义去重** | Memory.add verbatim 路径加批次前 top-K 语义去重 | test_add_dedup 语义级 | S1 |
| **S4 高阶参数化** | Memory.search 加 forgetting/token_budget/inject_prompt 参数 | test_search_enhanced | S2 |
| **S5 用户画像** | UserProfile 模型 + get_user_profile 聚合 + recall 画像注入 + REST 端点 | test_user_profile（聚合/时态去重/矛盾值/REST） | S2 |

---

## 8. 不破坏的承诺

- `Memory.add/search/update/delete` 签名**向后兼容**（新增参数都有默认值）
- `V2Memory` 类**保留**（deprecated 薄层委托）, 老用户代码不破
- `extract_and_store` 旧返回结构**兼容**（event 字段已存在, 值不再恒 ADD, 旧消费者按 event=="ADD" 过滤仍工作）
- `ADDITIVE_EXTRACTION_PROMPT` 保留（纯抽取降级路径用）
- 现有测试**不改断言绕过缺陷**（遵守测试保护规则）, 只新增测试覆盖新能力
- `capture.capture()` 原方法保留（向后兼容）, 新增 `preprocess()` 分离预处理

---

## 9. 风险与对策

| 风险 | 对策 |
|------|------|
| LLM 决策不稳定（误 UPDATE/DELETE） | 决策 prompt 加 few-shot；DELETE 加置信度阈值（LLM 给 confidence < 0.7 降级为 NOOP）；决策日志可追溯（history.reason + metadata.llm_decision） |
| 决策 prompt 输出格式不合规 | 复用 `_parse_facts_response` 容错解析 + 降级为纯 ADD（解析失败不阻塞业务） |
| 无 LLM 时无法决策 | `infer=False` 走 verbatim 直存 + 语义去重（L2）, 不依赖 LLM；`extract_and_store` 无 LLM 降级为 `extract_facts` 纯抽取 |
| V2 API 返回结构微调 | `memory_id` → `memory_ids` 列表, V2Memory 薄层做兼容转换（单条时仍返回 `memory_id`） |
| 画像聚合性能（大用户事实多） | `get_all_facts` 加 limit；画像缓存（用户记忆变更后失效）；默认 `include_temporal=False` 精简版 |
| UPDATE 语义歧义（同 predicate 多值） | 去重定当前值：取 `updated_at` 最新且 `is_deleted=False` 为 current, 旧值标记 `is_current=False` 不删除（保留, 留给 L1 决策处理） |

---

## 10. 测试策略

### 新增测试

| 测试文件 | 覆盖 | 阶段 |
|----------|------|------|
| `test_fact_decision.py` | ADD/UPDATE/DELETE/NOOP 4 决策场景 + 解析容错 + 降级 | S1 |
| `test_add_decision.py` | Memory.add(infer=True) 决策路由（重复事实 UPDATE、矛盾 DELETE） | S1 |
| `test_add_semantic_dedup.py` | verbatim 路径语义去重（>0.95 跳过） | S3 |
| `test_search_enhanced.py` | search(forgetting/token_budget/inject_prompt 参数) | S4 |
| `test_user_profile.py` | 画像聚合 + 时态去重 + 矛盾值 + include_temporal | S5 |
| `test_recall_profile_inject.py` | recall 画像注入到 injected_prompt | S5 |
| `test_rest_profile.py` | GET /agents/{user_id}/profile 端点 | S5 |

### 回归保护

- 现有 `test_memory.py` / `test_v2_memory.py` / `test_fact_extraction.py` / `test_incremental_extraction.py` 全绿（断言不改）
- `test_recall_*` 验证 id 不再是 `text[:50]` 而是真实 memory_id
- V2Memory deprecated 薄层委托后, `test_v2_memory.py` 走 Memory 路径仍通过

### 验证命令

```powershell
$env:PYTHONPATH = "src"
python -m pytest tests/unit/test_fact_decision.py tests/unit/test_add_decision.py -q
python -m pytest tests/unit/test_memory.py tests/unit/test_v2_memory.py -q  # 回归
ruff check src/septmuse/prompts/extract.py src/septmuse/models/extract.py src/septmuse/memory/main.py
```

---

## 11. 不在本设计范围

- **向量数据库子项目 2-4**（FAISS / 托管后端 / 长尾后端）— 独立进行, 不依赖本设计
- **alembic 迁移** — 仍靠运行时 `ALTER TABLE`, update_fact 用 SQLModel 字段更新
- **V2Memory 完全删除** — 仅降级为薄层, 删除留给后续 deprecation 周期
- **跨 user 的画像合并** — 画像按 user_id 隔离, 不做跨用户

# P0 实体抽取 + 实体向量库设计

> 日期：2026-07-23
> 前置文档：`docs/specs/opensource-gap-analysis.md`（差距分析）、`docs/plans/development-roadmap.md`（开发计划 Phase 0）
> 借鉴来源：mem0 `opensource/mem0/mem0/utils/entity_extraction.py` + `opensource/mem0/mem0/memory/main.py` `_upsert_entity` / `_remove_memory_from_entity_store`
> 范围：P0-Task 1（实体抽取模块）+ P0-Task 4（实体向量库），不依赖 LLM

---

## 1. 概述

### 1.1 目标

补齐 SeptMuse 最大空白：从"Zettel 链接"升级到"实体抽取 + 实体向量库"，为 P1-Task 2（entity boost 三信号融合）和 P0-Task 2/3（LLM 联合抽取 + cognify 流水线）奠定基础。

### 1.2 范围

| 包含 | 不包含 |
|------|--------|
| P0-Task 1：EntityExtractor（纯 Python regex + 可选 spaCy） | P0-Task 2：三元组 LLM 联合抽取（依赖 LLM provider，留到 P3-Task 1 后） |
| P0-Task 4：EntityStore（独立 SQLite 表 + 基本查询） | P0-Task 3：cognify 知识图谱构建流水线（依赖 LLM，留到 P3-Task 1 后） |
| Memory facade 集成（add 自动抽取 + delete 清理 + 5 个新方法） | P1-Task 2：entity boost 集成到 search（留到 P1） |

### 1.3 设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| spaCy 依赖 | 可选 extra `[ner]`，默认纯 Python regex | 零配置不破坏，NER 是可选升级 |
| 实体存储 | 独立 SQLite 表 `septmuse_entities` | mem0 V3 去图化模式，简单高效，与 SQLiteMemoryStore 同库 |
| entity boost | 不含（留到 P1-Task 2） | P0 范围清晰，每步可独立验证 |
| LLM 依赖 | 无（P0-Task 1 用 regex/spaCy，不依赖 LLM） | 先做无 LLM 的部分，LLM 联合抽取留到 P3-Task 1 后 |

---

## 2. 整体架构

### 2.1 文件布局

```
src/septmuse/
  concerns/
    extraction/                ← 新目录
      __init__.py
      entity.py                ← EntityExtractor + RegexEntityExtractor + SpacyEntityExtractor
  storage/
    entity_store.py            ← EntityStore（独立 SQLite 表）
  orchestration/
    memory.py                  ← Memory facade 扩展（__init__ + add + delete + 5 新方法）
  configs/
    defaults.py                ← MemoryConfig 扩展（entity_extractor_backend）
```

### 2.2 模块关系

```
写入路径:
  Memory.add(text, auto_extract_entities=True)
    → EntityExtractor.extract(text) → list[Entity]
    → EntityStore.upsert(entity, memory_id)           ← 精确匹配 → 语义匹配 → 新建
    → linked_memory_ids 累积

查询路径:
  Memory.search_entities(query)
    → EntityStore.search(query, top_k) → 实体 + linked_memory_ids

删除路径:
  Memory.delete(memory_id)
    → store.get(memory_id) 获取 user_id
    → EntityStore.remove_memory_from_entities(memory_id)  ← 引用清理，空则删实体
```

### 2.3 设计原则

- **遵循已有模式**：`EntityExtractor` 类似 `Embedder`（可注入、可降级），`EntityStore` 类似 `SQLiteGraphStore`（复用 SQLiteMemoryStore 的 conn/lock），`Memory` facade 编排
- **零配置不破坏**：默认纯 Python regex（无外部模型），`pip install septmuse[ner]` 升级 spaCy
- **独立可测**：EntityExtractor 和 EntityStore 各有独立测试，不依赖 Memory facade

---

## 3. EntityExtractor

### 3.1 数据模型

```python
@dataclass
class Entity:
    text: str           # 实体文本
    entity_type: str    # PROPER / QUOTED / TOPIC / IDENTIFIER
    start: int          # span 起始位置（字符偏移）
    end: int            # span 结束位置（字符偏移）
```

### 3.2 类层次

```python
class EntityExtractor(ABC):
    """实体抽取器抽象基类（类似 Embedder 模式）。"""
    @abstractmethod
    def extract(self, text: str) -> list[Entity]: ...

class RegexEntityExtractor(EntityExtractor):
    """纯 Python regex + 词表后端（默认，零配置）。"""
    # 4 类抽取规则 + ~120 泛化词黑名单 + span 去重

class SpacyEntityExtractor(EntityExtractor):
    """spaCy NER + noun_chunks 后端（pip install septmuse[ner]）。"""
    # spaCy 不可用时自动降级为 RegexEntityExtractor
```

### 3.3 4 类实体抽取规则（借鉴 mem0 `entity_extraction.py`）

| 类型 | RegexEntityExtractor 规则 | SpacyEntityExtractor 规则 | 示例 |
|------|--------------------------|--------------------------|------|
| **PROPER** | 英文大写开头词（`[A-Z][a-z]+`）；中文人名正则（百家姓 + 名） | `ent.label_ in ("PERSON","ORG","GPE","LOC","PRODUCT","EVENT","WORK_OF_ART")` | "Alice", "Google", "北京" |
| **QUOTED** | `"([^"]+)"` / `「([^」]+)」` / `'([^']+)'` / `\u201c([^\u201d]+)\u201d` | 同 regex（spaCy 不抽引号） | `"hello world"` |
| **TOPIC** | 英文连续大写开头词组（`[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+`）；中文连续名词短语（2-4 字无标点连续 `\u4e00-\u9fff{2,4}`） | `doc.noun_chunks` | "Machine Learning", "人工智能" |
| **IDENTIFIER** | `[a-z][a-z0-9]*(\.[a-z][a-z0-9]*)+` / `[A-Z][a-z]+[A-Z]\w*` / `[a-z]+_[a-z_]+` | 同 regex | `septmuse.memory`, `MemoryConfig`, `user_id` |

### 3.4 泛化词黑名单

~120 个泛化词（借鉴 mem0 `_GENERIC_HEADS` / `_NON_SPECIFIC_ADJ` / `_GENERIC_CAPS`）：

- 英文泛化词："the", "this", "that", "thing", "something", "person", "people", "time", "way", "day", "man", "woman", "world", "life", "hand", "part", "place", "case", "week", "year", "name", "home", "work", "word", "point", "group", "number", "fact", "idea", "issue", "side", "kind", "head", "line", "end", "member", "list", "lot", "other", "use", "first", "last", "new", "old", "good", "bad", "big", "small", "own", "same", "own", "some", "any", "all", "no", "every"
- 中文泛化词："这个", "那个", "什么", "东西", "事情", "地方", "时候", "时间", "人", "他们", "我们", "你们", "它们", "自己", "别人", "大家", "所有", "一些", "一点", "一下", "一样", "这样", "那样"

过滤逻辑：抽取后对每个实体 `_normalize_entity_text(text)`（lower + strip），命中黑名单则丢弃。

### 3.5 span 去重冲突解决

借鉴 mem0 `_resolve_candidates`：

1. 按 `start` 排序所有候选 Entity
2. 去重：同一 `(text, entity_type)` 只保留第一个出现的
3. 跨类型冲突：长 span 优先（"machine learning" 的 TOPIC 优先于 "machine" 的 IDENTIFIER），相同长度时 PROPER > QUOTED > TOPIC > IDENTIFIER

### 3.6 后端切换

```python
def _resolve_entity_extractor(config: MemoryConfig) -> EntityExtractor:
    """解析实体抽取器（类似 _resolve_embedder 模式）。

    默认 RegexEntityExtractor（零配置，纯 Python）。
    spacy: pip install septmuse[ner]，spaCy NER + noun_chunks。
    none: 禁用实体抽取。
    """
    choice = os.getenv("SEPTMUSE_ENTITY_EXTRACTOR", "regex").lower()
    if choice == "none":
        return None  # 禁用
    if choice in ("spacy", "nlp"):
        try:
            return SpacyEntityExtractor()
        except (ImportError, OSError) as e:
            logger.warning("entity_extractor_spacy_unavailable", error=str(e), fallback="regex")
    return RegexEntityExtractor()
```

### 3.7 SpacyEntityExtractor 降级逻辑

```python
class SpacyEntityExtractor(EntityExtractor):
    def __init__(self, model_name: str = "en_core_web_sm"):
        try:
            import spacy
            self._nlp = spacy.load(model_name)
        except OSError:
            # 模型未下载，尝试自动下载
            try:
                import spacy
                spacy.cli.download(model_name)
                self._nlp = spacy.load(model_name)
            except Exception:
                raise  # 由 _resolve_entity_extractor 捕获 → fallback regex
```

---

## 4. EntityStore

### 4.1 SQLite 表 schema

```sql
CREATE TABLE IF NOT EXISTS septmuse_entities (
    id TEXT PRIMARY KEY,                -- UUID
    entity_text TEXT NOT NULL,           -- 实体文本
    entity_type TEXT NOT NULL,           -- PROPER / QUOTED / TOPIC / IDENTIFIER
    entity_embedding BLOB,               -- 实体嵌入向量（语义去重用，float32 序列化）
    linked_memory_ids TEXT NOT NULL,     -- JSON array of memory_ids
    user_id TEXT NOT NULL,               -- 租户隔离
    agent_id TEXT,                       -- 租户隔离
    created_at TEXT NOT NULL,            -- ISO UTC
    updated_at TEXT NOT NULL,            -- ISO UTC
    is_deleted INTEGER DEFAULT 0,        -- 软删除
    UNIQUE(user_id, entity_text)         -- 同用户去重（精确匹配约束）
);
CREATE INDEX IF NOT EXISTS idx_entities_user ON septmuse_entities(user_id);
CREATE INDEX IF NOT EXISTS idx_entities_text ON septmuse_entities(entity_text);
CREATE INDEX IF NOT EXISTS idx_entities_deleted ON septmuse_entities(is_deleted);
```

### 4.2 类设计

```python
class EntityStore:
    """实体向量库（独立 SQLite 表，同库，借鉴 mem0 V3 去图化设计）。

    复用 SQLiteMemoryStore 的 conn + lock（类似 SQLiteGraphStore 模式）。
    embedder 可选——有则做语义去重(score>=0.95)，无则只精确匹配。
    """

    def __init__(self, conn, lock, embedder: Embedder | None = None):
        self._conn = conn
        self._lock = lock
        self._embedder = embedder
        self._create_table_if_not_exists()

    def upsert(self, entity: Entity, memory_id: str, *, user_id: str,
               agent_id: str | None = None) -> str:
        """upsert 实体（借鉴 mem0 _upsert_entity）。

        1. 精确归一化名匹配 → 命中则 linked_memory_ids 追加 memory_id
        2. 语义匹配（embedder 有时） → score>=0.95 命中则追加
        3. 新建 → 插入实体 + 嵌入向量 + linked_memory_ids=[memory_id]

        Returns: entity_id
        """

    def search(self, query: str, *, user_id: str, top_k: int = 5) -> list[dict]:
        """搜索实体: 精确匹配 + 向量相似度（embedder 有时）。

        Returns: [{"id","entity_text","entity_type","linked_memory_ids","score"}]
        """

    def get(self, entity_id: str) -> dict | None:
        """取单条实体。"""

    def list(self, *, user_id: str, entity_type: str | None = None,
             limit: int = 100) -> list[dict]:
        """列出用户全部未删除实体。"""

    def get_linked_memories(self, entity_id: str) -> list[str]:
        """获取实体的 linked_memory_ids。"""

    def remove_memory_from_entities(self, memory_id: str) -> None:
        """删除记忆时清理实体引用（借鉴 mem0 _remove_memory_from_entity_store）。

        memory_id 是 UUID 全局唯一，不需 user_id 过滤（全局扫描 linked_memory_ids）。
        1. 查 linked_memory_ids 包含 memory_id 的实体
        2. 移除 memory_id
        3. linked_memory_ids 空 → 软删除实体
        """

    def close(self) -> None:
        """释放资源（同库，实际不关 conn）。"""
```

### 4.3 upsert 逻辑详解

```python
def upsert(self, entity, memory_id, *, user_id, agent_id=None):
    normalized = _normalize_entity_text(entity.text)

    # 1. 精确归一化名匹配
    existing = self._find_by_text(normalized, user_id=user_id)
    if existing:
        self._append_memory_id(existing["id"], memory_id)
        return existing["id"]

    # 2. 语义匹配（embedder 有时）
    if self._embedder is not None:
        emb = self._embedder.embed(entity.text)
        semantic_match = self._find_by_embedding(emb, user_id=user_id, threshold=0.95)
        if semantic_match:
            self._append_memory_id(semantic_match["id"], memory_id)
            return semantic_match["id"]
    else:
        emb = None

    # 3. 新建
    entity_id = str(uuid.uuid4())
    self._insert(entity_id, entity, emb, memory_id, user_id, agent_id)
    return entity_id
```

### 4.4 remove_memory_from_entities 逻辑详解

```python
def remove_memory_from_entities(self, memory_id, *, user_id):
    entities = self._find_entities_containing_memory(memory_id, user_id=user_id)
    for entity in entities:
        linked = json.loads(entity["linked_memory_ids"])
        remaining = [mid for mid in linked if mid != memory_id]
        if not remaining:
            # 引用清空 → 软删除实体
            self._soft_delete(entity["id"])
        else:
            # 更新 linked_memory_ids
            self._update_linked_ids(entity["id"], remaining)
```

---

## 5. Memory facade 集成

### 5.1 __init__ 扩展

```python
# orchestration/memory.py Memory.__init__ 新增

# 实体抽取器（类似 _resolve_embedder 模式）
self.entity_extractor: EntityExtractor | None = (
    entity_extractor or _resolve_entity_extractor(self.config)
)

# 实体向量库（复用 SQLiteMemoryStore 的 conn/lock，类似 SQLiteGraphStore 模式）
if isinstance(self.store, SQLiteMemoryStore) and self.entity_extractor is not None:
    self.entity_store: EntityStore | None = EntityStore(
        self.store.conn, self.store._lock, embedder=self.embedder
    )
else:
    self.entity_store = None  # PGVectorStore 或 entity_extractor=None 时不可用
```

### 5.2 add() 扩展

```python
def add(self, messages, *, user_id, agent_id=None, metadata=None, infer=None,
        auto_extract_entities: bool = True) -> dict:
    # ... 已有 add 逻辑不变（verbatim 模式 / infer 模式）...

    # 自动抽取实体并存入 EntityStore
    if auto_extract_entities and self.entity_store is not None and self.entity_extractor is not None:
        for text, result in zip(texts, results):
            memory_id = result["id"]
            try:
                entities = self.entity_extractor.extract(text)
                for entity in entities:
                    self.entity_store.upsert(entity, memory_id, user_id=user_id, agent_id=agent_id)
            except Exception as e:
                logger.warning("entity_extract_failed", memory_id=memory_id, error=str(e))

    return {"results": results, "relations": []}
```

### 5.3 delete() 扩展

```python
def delete(self, memory_id: str) -> dict[str, str]:
    self.store.delete(memory_id)

    # 清理实体引用（引用清空则删实体）
    # memory_id 是 UUID 全局唯一，remove_memory_from_entities 不需要 user_id
    if self.entity_store is not None:
        try:
            self.entity_store.remove_memory_from_entities(memory_id)
        except Exception as e:
            logger.warning("entity_cleanup_failed", memory_id=memory_id, error=str(e))

    return {"status": "deleted", "memory_id": memory_id}
```

### 5.4 新增 5 个公开方法

```python
def extract_entities(self, text: str) -> list[dict[str, Any]]:
    """抽取实体（不存储），返回 [{"text","type","start","end"}]。"""
    if self.entity_extractor is None:
        return []
    entities = self.entity_extractor.extract(text)
    return [{"text": e.text, "type": e.entity_type, "start": e.start, "end": e.end} for e in entities]

def add_entity(self, entity_text: str, entity_type: str, memory_id: str, *,
               user_id: str) -> dict[str, Any]:
    """手动添加实体与记忆的关联。"""
    if self.entity_store is None:
        return {"error": "entity_store not available (SQLite only)"}
    entity = Entity(text=entity_text, entity_type=entity_type, start=0, end=len(entity_text))
    eid = self.entity_store.upsert(entity, memory_id, user_id=user_id)
    return {"id": eid, "entity": entity_text, "type": entity_type, "event": "ADD"}

def search_entities(self, query: str, *, user_id: str, top_k: int = 5) -> list[dict[str, Any]]:
    """搜索实体，返回实体 + linked_memory_ids。"""
    if self.entity_store is None:
        return []
    return self.entity_store.search(query, user_id=user_id, top_k=top_k)

def get_entity_neighbors(self, entity_id: str) -> list[str]:
    """获取实体关联的 memory_id 列表。"""
    if self.entity_store is None:
        return []
    return self.entity_store.get_linked_memories(entity_id)

def list_entities(self, *, user_id: str, entity_type: str | None = None,
                 limit: int = 100) -> list[dict[str, Any]]:
    """列出用户全部实体。"""
    if self.entity_store is None:
        return []
    return self.entity_store.list(user_id=user_id, entity_type=entity_type, limit=limit)
```

### 5.5 close() 扩展

```python
def close(self) -> None:
    self.store.close()
    self.typed_store.close()
    if self.entity_store is not None:
        self.entity_store.close()
```

---

## 6. MemoryConfig 扩展

```python
# configs/defaults.py MemoryConfig 新增字段

class MemoryConfig(BaseModel):
    # ... 已有字段 ...
    entity_extractor_backend: str = Field(
        default="regex",
        description="实体抽取后端: regex（默认）/ spacy / none",
    )
```

环境变量 `SEPTMUSE_ENTITY_EXTRACTOR` 覆盖 `entity_extractor_backend`（在 `default_config()` 中读取）。

---

## 7. 依赖管理

```toml
# pyproject.toml
[project.optional-dependencies]
# ... 已有 extras ...
ner = ["spacy>=3.7"]
```

- 默认安装 `pip install septmuse` 不含 spaCy，用 RegexEntityExtractor
- `pip install septmuse[ner]` 装 spaCy
- `SEPTMUSE_ENTITY_EXTRACTOR=spacy` 切换到 spaCy 后端
- 模型 `en_core_web_sm` 在 `SpacyEntityExtractor.__init__` 中首次使用时自动 `spacy.cli.download`

---

## 8. 错误处理

借鉴 `record_access` 吞错模式——实体抽取是增强功能，不应阻塞主流程：

| 场景 | 处理 | 实现位置 |
|------|------|----------|
| spaCy `import` 失败 | fallback 到 RegexEntityExtractor + 日志 warn | `_resolve_entity_extractor` |
| spaCy 模型未下载 | 尝试 `spacy.cli.download` → 失败则 fallback regex + 日志 warn | `SpacyEntityExtractor.__init__` |
| `extract(text)` 失败 | 返回空列表 + 日志 warn | `EntityExtractor.extract` try-catch |
| `EntityStore.upsert()` 失败 | 日志 warn + 不阻塞 add | `Memory.add` try-catch |
| `remove_memory_from_entities()` 失败 | 日志 warn + 不阻塞 delete | `Memory.delete` try-catch |
| PGVectorStore 后端 | `entity_store=None`，`add()` 跳过实体抽取 | `Memory.__init__` 条件判断 |

---

## 9. 测试策略

### 9.1 单元测试

| 测试文件 | 测试数 | 覆盖 |
|----------|--------|------|
| `tests/unit/test_entity_extractor.py` | ~15 | RegexEntityExtractor 4 类实体 × 3 + 泛化词过滤 + span 去重 + `_resolve_entity_extractor` 降级 |
| `tests/unit/test_entity_store.py` | ~12 | upsert（精确/语义/新建）+ search + remove + list/get + get_linked_memories |
| `tests/unit/test_memory.py` 扩展 | ~5 | add+auto_extract + add+False + delete+清理 + search_entities + list_entities |
| `tests/unit/test_config.py` 扩展 | ~2 | `entity_extractor_backend` 字段 + 环境变量覆盖 |

### 9.2 e2e 测试

| 测试文件 | 测试数 | 覆盖 |
|----------|--------|------|
| `tests/e2e/test_entity_e2e.py` | ~3 | 跨会话持久化（add→close→reopen→search_entities）+ delete 清理 + 中文实体 |

### 9.3 spaCy 测试

`SpacyEntityExtractor` 测试标记 `@pytest.mark.integration`，无 spaCy 安装时 skip（类似 onnx/auto embedder 测试模式）。

### 9.4 预期测试基线

- 当前：699 passed, 34 skipped
- 新增后：~734 passed, ~36 skipped（+2 skip for spaCy）

---

## 10. 验收标准

1. `ruff check src/ tests/` + `ruff format --check src/ tests/` → clean
2. `PYTHONPATH=src pytest tests/unit/ tests/e2e/ -q` → ~734 passed, ~36 skipped
3. `Memory().add("Alice works at Google in London", user_id="u1")` → `search_entities("Google", user_id="u1")` 返回实体 + linked_memory_ids 含 add 返回的 memory_id
4. `Memory().delete(memory_id)` → `search_entities("Google", user_id="u1")` linked_memory_ids 不再含该 memory_id（引用清理）
5. `Memory().add("我喜欢Python", user_id="u1")` → `list_entities(user_id="u1")` 返回中文实体 "Python"（IDENTIFIER 类型）
6. `SEPTMUSE_ENTITY_EXTRACTOR=none` → `add()` 不抽取实体，`list_entities()` 返回空
7. `auto_extract_entities=False` → `add()` 不抽取实体

---

## 11. API 暴露

### 11.1 CLI 扩展

新增 2 个命令：
- `septmuse entities <query> --user-id <id>` — 搜索实体
- `septmuse entity-list --user-id <id>` — 列出实体

### 11.2 REST 扩展

新增 2 个端点：
- `GET /entities?query=<q>&user_id=<id>&top_k=5` — 搜索实体
- `GET /entities/list?user_id=<id>&entity_type=PROPER&limit=100` — 列出实体

### 11.3 MCP 扩展

新增 2 个工具：
- `search_entities(query: str, user_id: str, top_k: int = 5) -> list[dict]` — 搜索实体
- `list_entities(user_id: str, entity_type: str | None = None, limit: int = 100) -> list[dict]` — 列出实体

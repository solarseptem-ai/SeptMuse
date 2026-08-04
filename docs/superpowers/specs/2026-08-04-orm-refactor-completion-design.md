# ORM 重构收尾设计 — 分阶段删除原生 store

> 日期：2026-08-04
> 状态：已批准
> 前置文档：`docs/superpowers/specs/2026-08-03-orm-store-refactor-design.md`（P1-P4 已完成）
> 阶段：brainstorming → writing-plans

## 1. 背景

P1-P4 已完成 ORMMemoryStore / AsyncORMMemoryStore / VectorStore 工厂 / KeywordIndex 工厂 / RelationalStoreFactory / facade 双路径接入。但设计文档 §3.1 要求"完全删除 SQLiteMemoryStore"，实际实施为双路径保留——零配置走 SQLiteMemoryStore，设 `SEPTMUSE_DB_URL` 走 ORMMemoryStore。

本设计补全剩余工作，分三步完成原生 store 删除。

## 2. 现状分析

### 2.1 已完成（P1-P4）

| 组件 | 状态 |
|------|------|
| `services/database/models/` 包（memory/history/access_log/entity） | ✅ |
| `ORMMemoryStore` sync CRUD | ✅ |
| `AsyncORMMemoryStore` async CRUD | ✅ |
| `DatabaseService` sync + async engine | ✅ |
| `create_vector_store` 方言工厂 | ✅ |
| `create_keyword_index` 方言工厂 | ✅ |
| `RelationalStoreFactory.create()` + `create_async()` | ✅ |
| `Memory._resolve_store()` 双路径 | ✅ |

### 2.2 未完成（本设计目标）

| 组件 | 现状 | 问题 |
|------|------|------|
| `ORMMemoryStore.engine` property | 有 `self._engine` 但未暴露 | facade 无法 duck typing 取 engine |
| `EntityStore` | raw `sqlite3.Connection` + `?` 占位符 | ORMMemoryStore 路径下为 None，cognify/entity 功能断裂 |
| `TypedMemoryStore` | 自建 engine from db_path | 不共享 ORMMemoryStore engine，MySQL/PG 路径下仍连 SQLite |
| `MigrationRunner` | raw conn + PRAGMA/information_schema | 不支持 engine，ORMMemoryStore 路径不执行迁移 |
| `SQLiteMemoryStore` + `AsyncSQLiteMemoryStore` | 保留 | 设计文档要求删除 |
| facade `isinstance` 检查 | `isinstance(self.store, SQLiteMemoryStore)` | ORMMemoryStore 下 graph_store/entity_store 为 None |
| ~1116 现有测试 | 用 `SQLiteMemoryStore(db_path=...)` | 需迁移到 ORMMemoryStore |

### 2.3 模型位置差异

设计文档 §3.2 要求模型移到 `services/database/models/typed.py`，但实际模型已在 `src/septmuse/models/` 目录（episodic.py/semantic.py/procedural.py），且已是 SQLModel table。**本设计不移动模型位置**——已满足 ORM 要求，移动只增加 import churn。

## 3. 设计决策

### 3.1 分阶段执行

采用方案 C（分阶段）：

```
Step 1: 补全 ORMMemoryStore 路径（低风险，可独立交付）
Step 2: 逐个迁移测试文件（中风险，分批回归）
Step 3: 删除原生 store（低风险，测试已迁移完）
```

### 3.2 duck typing 替代 isinstance

facade 不再用 `isinstance(self.store, SQLiteMemoryStore)`，改用 `getattr(self.store, "engine", None)`：
- ORMMemoryStore 有 `engine` property → 走 ORMMemoryStore 路径
- SQLiteMemoryStore 无 `engine` 属性 → 走旧路径

### 3.3 双模式构造（Step 1 兼容）

所有改造组件保留旧构造签名 + 新增 `from_engine` classmethod 或可选 `engine` 参数，确保 Step 1 零测试迁移。

### 3.4 EntityStore 用 SQLModel 重写

EntityStore 的 raw SQL（CREATE TABLE / INSERT / SELECT / UPDATE）改为用 `services/database/models/entity.py` 的 `EntityTable`（SQLModel table）+ `Session(engine)`。语义去重算法不变（embedder 可选，score≥0.95 归一化匹配）。

### 3.5 MigrationRunner 用 SQLAlchemy inspect

`MigrationContext.has_column` 从 `PRAGMA table_info` 改为 `sqlalchemy.inspect(engine).get_columns(table)`，跨方言通用。

## 4. 详细设计

### Step 1: 补全 ORMMemoryStore 路径

#### 4.1.1 ORMMemoryStore 暴露 engine

```python
# storage/relational_stores/orm_store.py
@property
def engine(self) -> Engine:
    return self._engine
```

AsyncORMMemoryStore 加 `async_engine` property 返回 `self._engine`（AsyncEngine）。

#### 4.1.2 EntityStore 双模式

```python
# storage/relational_stores/entity_store.py

class EntityStore:
    def __init__(self, conn, lock, embedder=None):
        """旧构造（SQLiteMemoryStore 路径，向后兼容）。"""
        self._conn = conn
        self._lock = lock
        self._embedder = embedder
        self._engine = None
        self._create_table_if_not_exists()

    @classmethod
    def from_engine(cls, engine, embedder=None):
        """新构造（ORMMemoryStore 路径）。"""
        store = cls.__new__(cls)
        store._conn = None
        store._lock = None
        store._engine = engine
        store._embedder = embedder
        # 用 EntityTable SQLModel 建表
        SQLModel.metadata.create_all(engine)
        return store
```

`from_engine` 路径的 CRUD 用 `Session(engine)` + `select(EntityTable)` 替代 raw SQL。`EntityTable` 已在 `services/database/models/entity.py` 定义（`__tablename__ = "septmuse_entities"`，字段对齐旧 raw SQL）。语义去重逻辑（精确归一化 → 语义匹配 → 新建）不变。

#### 4.1.3 TypedMemoryStore 共享 engine

```python
# storage/relational_stores/typed_store.py

def __init__(self, db_path=None, *, engine=None):
    if engine is not None:
        self.engine = engine
    else:
        if db_path is None:
            db_path = _default_db_path()
        self.db_path = Path(db_path)
        self.engine = create_engine(f"sqlite:///{self.db_path}")
    SQLModel.metadata.create_all(self.engine)
```

facade 中：`TypedMemoryStore(db_path=self.config.db_path, engine=getattr(self.store, "engine", None))`

#### 4.1.4 MigrationRunner 双模式

```python
# storage/migrations/runner.py

class MigrationRunner:
    def __init__(self, conn=None, backend="sqlite", *, engine=None):
        self._engine = engine
        if engine is not None:
            self._conn = None
            self.backend = engine.dialect.name
        else:
            self._conn = conn
            self.backend = backend

    @classmethod
    def from_engine(cls, engine):
        return cls(engine=engine)

    def _has_column(self, table, column):
        if self._engine is not None:
            cols = [c["name"] for c in inspect(self._engine).get_columns(table)]
            return column in cols
        # 旧路径
        ctx = MigrationContext(self._conn, self.backend)
        return ctx.has_column(table, column)
```

#### 4.1.5 facade duck typing

```python
# memory/main.py __init__

engine = getattr(self.store, "engine", None)

if engine is not None:
    # ORMMemoryStore 路径
    from septmuse.storage.relational_stores.entity_store import EntityStore
    self.entity_store = EntityStore.from_engine(engine, self.embedder)
    self.typed_store = TypedMemoryStore(engine=engine)
    # graph_store: SQLite dialect 下从 engine 取 raw connection
    if engine.dialect.name == "sqlite":
        raw_conn = engine.raw_connection()
        self.graph_store = SQLiteGraphStore(raw_conn, threading.Lock())
    else:
        self.graph_store = graph_store  # None 或注入
elif isinstance(self.store, SQLiteMemoryStore):
    # 旧路径不动
    self.graph_store = SQLiteGraphStore(self.store.conn, self.store._lock)
    self.entity_store = EntityStore(self.store.conn, self.store._lock, self.embedder)
    self.typed_store = TypedMemoryStore(db_path=self.config.db_path)
```

AsyncMemory facade 同理，用 `getattr(self.store, "async_engine", None)`。

#### 4.1.6 新增测试

- `tests/unit/test_entity_store_orm.py` — EntityStore.from_engine CRUD + 语义去重
- `tests/unit/test_typed_store_shared_engine.py` — TypedMemoryStore(engine=) 共享验证
- `tests/unit/test_migration_runner_orm.py` — MigrationRunner.from_engine 跨方言
- `tests/unit/test_facade_orm_path.py` — Memory(store=ORMMemoryStore(...)) 完整路径

### Step 2: 测试迁移

#### 4.2.1 批次划分

**批次 1：低层存储测试（~15 文件）**
- `test_sqlite_store.py` / `test_async_sqlite_store.py` / `test_composite_store.py` / `test_entity_store.py` / `test_migrations.py` 等

**批次 2：facade + 中层测试（~20 文件）**
- `test_memory.py` / `test_async_memory.py` / `test_permissions.py` / `test_hybrid_search.py` / `test_cognify.py` 等

**批次 3：高层 + e2e 测试（~10 文件）**
- `test_fact_extraction.py` / `test_update.py` / `test_bitemporal.py` / `tests/e2e/*.py`

#### 4.2.2 迁移规则

- 每批迁移后跑全量 pytest，新增失败数必须为 0
- `SQLiteMemoryStore(db_path=tmp)` → `ORMMemoryStore(engine=create_engine(f"sqlite:///{tmp}"))`
- `EntityStore(conn, lock, embedder)` → `EntityStore.from_engine(engine, embedder)`
- `TypedMemoryStore(db_path=...)` → `TypedMemoryStore(engine=engine)`
- e2e 测试用 `SEPTMUSE_DB_URL=sqlite:///{tmp_path}/test.db` 触发 ORMMemoryStore
- SQLiteMemoryStore 内部实现测试（PRAGMA / raw SQL 行为）标记 `@pytest.mark.legacy` 或删除

### Step 3: 删除原生 store

#### 4.3.1 删除文件

- `storage/relational_stores/store.py`（SQLiteMemoryStore）
- `storage/relational_stores/async_store.py`（AsyncSQLiteMemoryStore）
- `storage/migrations/context.py`（raw conn PRAGMA 路径）

#### 4.3.2 facade 简化

```python
def _resolve_store(self) -> MemoryStore:
    return RelationalStoreFactory.create(self.config)
```

`RelationalStoreFactory.create()` 内部：`db_url` 未设时默认 `sqlite:///~/.septmuse/septmuse.db`。

#### 4.3.3 清理双模式

- 删除 EntityStore 旧 `__init__(conn, lock, embedder)`，只保留 `from_engine`
- 删除 MigrationRunner 旧 `__init__(conn, backend)`，只保留 `from_engine`
- `storage/relational_stores/__init__.py` 删除 SQLiteMemoryStore / AsyncSQLiteMemoryStore 导出

#### 4.3.4 零配置行为验证

- `Memory()` 无参数 → 默认 SQLite → ORMMemoryStore → 全功能
- `SEPTMUSE_DB_URL=mysql://...` → ORMMemoryStore + MySQL → 全功能
- 全量 pytest 通过

## 5. 执行顺序与依赖

```
Step 1 (补全 ORMMemoryStore 路径)
  ← EntityStore.from_engine + TypedMemoryStore(engine=) + MigrationRunner.from_engine + facade duck typing
  ← 零测试迁移，只新增代码
  ← 可独立交付
  ▼
Step 2 (测试迁移)
  ← 批次 1 → 批次 2 → 批次 3
  ← 每批回归验证
  ▼
Step 3 (删除原生 store)
  ← 所有测试迁移完后才删
  ← 零配置行为验证
```

## 6. 验收标准

| Step | 验收标准 | 回归基线 |
|------|---------|---------|
| Step 1 | ruff 全绿 + 新增测试全绿 + 现有测试零退化 | ~1215 passed + 36 skipped + 13 failed |
| Step 2 | 全量 pytest 通过，迁移后总数不变或增加 | ≥ Step 1 基线 |
| Step 3 | 删除后全量 pytest 通过 + `Memory()` 零配置可用 + `SEPTMUSE_DB_URL` 多库可用 | ≥ Step 2 基线 |

## 7. 不做的事（YAGNI）

- 不改 GraphStore ABC / AGE / Neo4j 后端
- 不移动模型位置（已在 `src/septmuse/models/` 且 ORM 化）
- 不改 Block / CausalEdge / MemoryStrength 模型
- 不改 pyproject.toml 依赖（P3-P4 已配好）
- 不重写 EntityStore 语义去重算法（只改存储层 conn → engine）

## 8. 影响范围

### Step 1 修改文件

| 文件 | 改动 |
|------|------|
| `storage/relational_stores/orm_store.py` | 加 `engine` property |
| `storage/relational_stores/async_orm_store.py` | 加 `async_engine` property |
| `storage/relational_stores/entity_store.py` | 加 `from_engine` classmethod + ORM CRUD |
| `storage/relational_stores/typed_store.py` | `__init__` 加可选 `engine` 参数 |
| `storage/migrations/runner.py` | 加 `from_engine` + inspect 路径 |
| `memory/main.py` | isinstance → duck typing |
| `memory/async_main.py` | isinstance → duck typing |
| `storage/relational_stores/__init__.py` | 导出更新 |

### Step 1 新建文件

| 文件 | 内容 |
|------|------|
| `tests/unit/test_entity_store_orm.py` | EntityStore.from_engine 测试 |
| `tests/unit/test_typed_store_shared_engine.py` | TypedMemoryStore 共享 engine 测试 |
| `tests/unit/test_migration_runner_orm.py` | MigrationRunner.from_engine 测试 |
| `tests/unit/test_facade_orm_path.py` | Memory(store=ORMMemoryStore) 完整路径测试 |

### Step 2-3 影响文件

Step 2 逐个修改 ~45 个测试文件。Step 3 删除 3 个文件 + 修改 facade / __init__.py / EntityStore / MigrationRunner。

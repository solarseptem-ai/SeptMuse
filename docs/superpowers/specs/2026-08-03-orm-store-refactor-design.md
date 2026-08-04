# ORMMemoryStore 重构设计 — 多库统一

> 日期：2026-08-03
> 状态：已批准
> 阶段：brainstorming → writing-plans

## 1. 目标

废弃 `SQLiteMemoryStore` + `AsyncSQLiteMemoryStore`（原生 sqlite3/aiosqlite 硬编码），统一改为 `ORMMemoryStore` + `AsyncORMMemoryStore`（SQLModel ORM），一套代码跨 SQLite/MySQL/PostgreSQL。DatabaseService 真正接入主链路。

## 2. 现状问题

| 问题 | 位置 | 说明 |
|------|------|------|
| 原生 sqlite3 连接 | `storage/relational_stores/store.py:75` | `sqlite3.connect(...)` 不走 SQLAlchemy |
| 原生 aiosqlite 连接 | `storage/relational_stores/async_store.py:53` | `aiosqlite.connect(...)` 不走 SQLAlchemy |
| `?` 占位符 | 所有 SQL 语句 | MySQL/PG 用 `%s` 或 `:name` |
| SQLite 方言 DDL | `PRAGMA`、`INTEGER`、`TEXT` | MySQL/PG 用 `BOOLEAN`/`TINYINT`/`TIMESTAMP` |
| MigrationRunner | `migrations/runner.py` | 写死 `"sqlite"` 方言 |
| 双写组件 | `SQLiteVectorStore` + `SQLiteBM25Index` | MySQL/PG 需 pgvector/FTS |
| DatabaseService 孤儿 | `services/database/service.py` | 建好 engine 但没人用 |

## 3. 设计决策

### 3.1 统一 ORMMemoryStore，废弃原生 store

- `SQLiteMemoryStore` + `AsyncSQLiteMemoryStore` **完全删除**
- 新建 `ORMMemoryStore`（sync）+ `AsyncORMMemoryStore`（async），实现 `MemoryStore` ABC
- 所有路径统一走 ORMMemoryStore，dialect 只影响 VectorStore/KeywordIndex 子组件
- 零配置：`db_url` 默认 `sqlite:///~/.septmuse/septmuse.db`，走 ORMMemoryStore
- 多库：`SEPTMUSE_DB_URL=mysql://...` 或 `postgresql://...`，走 ORMMemoryStore

### 3.2 models 目录包

```
services/database/
├── models/
│   ├── __init__.py          # 导出所有表类
│   ├── memory.py            # MemoryTable
│   ├── history.py           # HistoryTable
│   ├── access_log.py        # AccessLogTable
│   ├── entity.py            # EntityTable + EntityRelationTable
│   └── typed.py             # EpisodicEvent + SemanticFact + ProceduralRule（从 typed_store.py 移入）
├── service.py
├── factory.py
└── __init__.py
```

### 3.3 metadata 列命名

`metadata` 是 SQLModel/SQLAlchemy 保留字。Python 属性名用 `metadata_`，`Field(alias="metadata")` 映射到数据库列名 `metadata`。ORM 层写 `mem.metadata_`，返回的 dict key 仍是 `"metadata"`。

### 3.4 双写组件方言工厂

**VectorStore**：

| dialect | 实现 | 向量类型 | 检索方式 |
|---------|------|----------|----------|
| sqlite | `SQLiteVectorStore`（现有） | JSON | numpy 余弦 |
| postgresql | `PgvectorStore`（新建） | `VECTOR(384)` | `<=>` 余弦距离 |
| mysql | `MySQLVectorStore`（新建） | JSON | numpy 余弦（无原生向量） |

**KeywordIndex**：

| dialect | 实现 | 索引类型 | 检索方式 |
|---------|------|----------|----------|
| sqlite | `SQLiteBM25Index`（现有） | BM25 | rank_bm25 |
| postgresql | `PostgresFTSIndex`（新建） | tsvector + GIN | `ts_rank` |
| mysql | `MySQLFulltextIndex`（新建） | FULLTEXT INDEX | `MATCH AGAINST` |

### 3.5 错误处理

- ORMMemoryStore CRUD：`session.commit()` 包 try/except，失败 `rollback()` + 抛 `StorageError`
- 向量检索：返回空列表不报错；pgvector 扩展未安装时降级为 JSON + numpy + 日志警告
- 关键词检索：返回空字典不报错；PG tsvector 未建索引时降级为 LIKE + 日志警告
- 双写一致性：memories 表 commit 后双写 vector/keyword，双写失败只记日志不回滚主表（现有策略不变）

### 3.6 向后兼容

- `SQLiteMemoryStore` + `AsyncSQLiteMemoryStore` **删除**
- 现有测试改为构造 `ORMMemoryStore(sqlite engine)`，逐个修复失败的测试
- `Memory` facade 的 `_resolve_store()` 改为 `RelationalStoreFactory.create(config)`
- `SEPTMUSE_ORM` 环境变量不再需要（唯一路径，无需切换）

### 3.7 MigrationRunner 改造

现有 `MigrationRunner`（`storage/migrations/runner.py`）写死 `"sqlite"` 方言，用 `PRAGMA table_info` 检测列。改造方案：

- **SQLite 路径**：保持 `PRAGMA table_info` 检测（现有逻辑不变）
- **MySQL/PG 路径**：改为用 SQLAlchemy `inspect(engine).get_columns(table)` 检测列是否存在，跨方言通用
- **MigrationRunner 构造改为 `MigrationRunner(engine, dialect)`**：dialect 从 DatabaseService 传入
- **迁移内容不变**：state 列、session_id 列、temporal 列等 ALTER TABLE 仍然幂等，SQLAlchemy 的 `inspect` 替代 `PRAGMA` 检测

### 3.8 typed_store / entity_store 改造

- **`TypedMemoryStore`**（`typed_store.py`）：现有用 `SQLModel.metadata.create_all(self.conn)` 建表。改为从 ORMMemoryStore 拿 engine，用 `SQLModel.metadata.create_all(engine)` 建表。表定义（EpisodicEvent/SemanticFact/ProceduralRule）移到 `services/database/models/` 目录包中。
- **`EntityStore`**（`entity_store.py`）：现有用原生 sqlite3 连接。改为从 ORMMemoryStore 拿 engine，CRUD 改用 SQLModel `select()` / `session.add()`。EntityTable 定义已在 `models/entity.py`。

### 3.9 DatabaseService async engine 支持

现有 `DatabaseService` 只创建 sync `Engine`。补充 async 支持：

- **新增 `get_async_engine()` 方法**：用 `create_async_engine(self.database_url, ...)` 创建 `AsyncEngine`
- **url 自动加 async driver**：`sqlite://` → `sqlite+aiosqlite://`；`mysql://` → `mysql+aiomysql://`；`postgresql://` → `postgresql+psycopg://`（已有 `_resolve_db_url` 逻辑，补充 async 变体）
- **`AsyncORMMemoryStore` 构造时**：`__init__(async_engine, vector_store, keyword_index)` + `async_sessionmaker`
- **DatabaseService 保留两个 engine**：sync + async，按需创建（懒加载 async engine，避免不用 async 时无谓开销）

## 4. 架构图

```
┌─────────────────────────────────────────────────────────┐
│  Memory (facade)                                        │
│  _resolve_store() → RelationalStoreFactory.create()     │
└──────────────┬──────────────────────────────────────────┘
               │
       ┌───────▼────────┐
       │ DatabaseService │  (engine 管理 + 方言检测)
       │  sync: Engine   │  async: AsyncEngine
       └───────┬──────────┘
               │
    ┌──────────▼───────────┐
    │ ORMMemoryStore (sync) │  AsyncORMMemoryStore (async)
    │ MemoryStore ABC 实现  │  MemoryStore ABC 实现
    │ CRUD 全用 SQLModel    │  async_sessionmaker
    └──┬────────────┬───────┘
       │            │
   ┌───▼───┐   ┌───▼────────┐
   │Vector │   │ KeywordIdx  │
   │Store  │   │  Base       │
   └───┬───┘   └───┬────────┘
       │            │
  ┌────┼────┐  ┌───┼────┐
  │    │    │  │   │    │
 SQLite PG  MySQL SQLite PG MySQL
 JSON  pgv  JSON BM25  FTS FULL
```

## 5. 执行阶段

| 阶段 | 内容 | 风险 | 测试 |
|------|------|------|------|
| P1 | models/ 目录包补全所有列 → ORMMemoryStore sync CRUD → 测试 | 中 | 新建 test_orm_memory_store.py + 现有测试改 ORMMemoryStore |
| P2 | AsyncORMMemoryStore async CRUD → 测试 | 中 | 新建 test_async_orm_memory_store.py + 现有 async 测试改 |
| P3 | VectorStore 方言工厂（SQLite 现有 / PG pgvector / MySQL JSON） | 高 | 新建 test_vector_factory.py |
| P4 | KeywordIndex 方言工厂 + Memory facade 接入 + 清理原生 store | 高 | 新建 test_keyword_factory.py + 全量回归 |

## 6. 影响范围

### 删除文件
- `storage/relational_stores/store.py`（SQLiteMemoryStore）
- `storage/relational_stores/async_store.py`（AsyncSQLiteMemoryStore）

### 新建文件
- `services/database/models/__init__.py`
- `services/database/models/memory.py`
- `services/database/models/history.py`
- `services/database/models/access_log.py`
- `services/database/models/entity.py`
- `services/database/models/typed.py`（EpisodicEvent/SemanticFact/ProceduralRule，从 typed_store.py 移入）
- `storage/relational_stores/orm_store.py`（ORMMemoryStore sync）
- `storage/relational_stores/async_orm_store.py`（AsyncORMMemoryStore async）
- `storage/vector_stores/factory.py`（create_vector_store）
- `storage/vector_stores/pgvector_store.py`（PgvectorStore）
- `storage/vector_stores/mysql_vector_store.py`（MySQLVectorStore）
- `storage/keyword_stores/factory.py`（create_keyword_index）
- `storage/keyword_stores/postgres_fts.py`（PostgresFTSIndex）
- `storage/keyword_stores/mysql_fulltext.py`（MySQLFulltextIndex）
- `storage/relational_stores/factory.py`（RelationalStoreFactory）
- `tests/unit/test_orm_memory_store.py`（新建）
- `tests/unit/test_async_orm_memory_store.py`（新建）
- `tests/unit/test_vector_factory.py`
- `tests/unit/test_keyword_factory.py`

### 修改文件
- `services/database/service.py`（补 async engine 支持 + `get_async_engine()` + 懒加载）
- `services/database/factory.py`
- `services/database/models.py` → 删除（拆成 models/ 包）
- `storage/relational_stores/__init__.py`（导出 ORMMemoryStore/AsyncORMMemoryStore）
- `storage/relational_stores/typed_store.py`（改用 engine + 表定义移到 models/）
- `storage/relational_stores/entity_store.py`（改用 engine + SQLModel CRUD）
- `storage/migrations/runner.py`（MigrationRunner 改为 engine + dialect 构造，用 SQLAlchemy inspect 替代 PRAGMA）
- `memory/main.py`（_resolve_store 改为工厂）
- `memory/async_main.py`（_resolve_store 改为工厂）
- `pyproject.toml`（加 pgvector 依赖可选）
- 现有测试文件（SQLiteMemoryStore → ORMMemoryStore）

## 7. 验证标准

- ruff check src/ tests/ 全绿
- pytest tests/unit/ tests/e2e/ 全绿（基线 1116+ passed + 36 skipped，允许新增测试增加数量）
- 现有 14 failed（API key 相关）不变
- `SEPTMUSE_DB_URL=sqlite://` 零配置可用
- `SEPTMUSE_DB_URL=mysql://user:pass@host:3306/septmuse` 可用（需装 mysql extras）
- `SEPTMUSE_DB_URL=postgresql://user:pass@host:5432/septmuse` 可用（需装 pg extras）

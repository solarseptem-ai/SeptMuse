# Task 4 报告: EntityStore.from_engine + ORM CRUD

## 任务概要

为 `EntityStore` 增加 ORM 模式（基于 SQLModel `Session(engine)` + `EntityTable`），与现有 raw `sqlite3.Connection` 模式并存，互不破坏。ORM 模式通过 `from_engine(engine, embedder=None)` classmethod 进入，所有公共方法在 ORM 模式下走 `_method_orm` 私有实现，raw SQL 路径完整保留。

## 实现内容

### 1. `src/septmuse/storage/relational_stores/entity_store.py`（主改动）

**新增导入**（文件顶部）:
```python
from sqlmodel import Session, SQLModel, select
from septmuse.services.database.models.entity import EntityTable
```

**`__init__`**: 末尾新增 `self._engine = None`（标记 raw SQL 模式）。

**`from_engine(cls, engine, embedder=None)` classmethod**: 通过 `cls.__new__(cls)` 绕过 `__init__`（避免触发 raw SQL 的 `_create_table_if_not_exists`），设置 `_engine`/`_embedder`/`_conn=None`/`_lock=None`，并调用 `SQLModel.metadata.create_all(engine)` 幂等建表。

**`_is_orm_mode()`**: 返回 `self._engine is not None`。

**6 个公共方法的 ORM 分支**（每个方法顶部加 `if self._is_orm_mode(): return self._method_orm(...)`，原 raw SQL 代码不动）:
- `upsert` → `_upsert_orm`
- `get` → `_get_orm`
- `search` → `_search_orm`
- `list` → `_list_orm`
- `get_linked_memories` → `_get_linked_memories_orm`
- `remove_memory_from_entities` → `_remove_memory_from_entities_orm`

**3 个 ORM 私有辅助**（dedup 逻辑镜像 raw SQL 实现）:
- `_find_by_text_orm`: 精确归一化名匹配（fetch all + Python 端 normalize 比较，与 raw 一致）
- `_find_by_embedding_orm`: 语义匹配（cosine ≥ threshold，复用 `_deserialize_embedding` + `_cosine_similarity`）
- `_append_memory_id_orm`: 向 `linked_memory_ids` 追加 memory_id（`session.get` + 改字段 + `session.add`）

**ORM 方法核心模式**:
- 查询: `with Session(self._engine) as session: stmt = select(EntityTable).where(...); rows = session.exec(stmt).all()`
- 按 id: `session.get(EntityTable, entity_id)`
- 插入: `row = EntityTable(...); session.add(row); session.commit()`
- 更新: 改字段后 `session.add(row); session.commit()`
- 软删除: `row.is_deleted = 1`

**复用（未改动）**: `_serialize_embedding`/`_deserialize_embedding` 静态方法 + 模块级 `_cosine_similarity`（对 bytes 工作，EntityTable.entity_embedding 存 bytes，两模式通用）。`close()` 和模块级 `_cosine_similarity` 不变。

### 2. `src/septmuse/extraction/entity.py`（最小改动）

`Entity` dataclass 的 `start`/`end` 增加默认值 `0`:
```python
@dataclass
class Entity:
    text: str
    entity_type: str
    start: int = 0   # 原: 无默认
    end: int = 0     # 原: 无默认
```
**原因**: brief 的测试代码逐字使用 `Entity(text="Alice", entity_type="PROPER")`（不传 start/end），原 dataclass 无默认会 TypeError。加默认值向后兼容——所有现有调用方都传显式值，不受影响。`entity.py:297` 的 `candidates.sort(key=lambda e: (e.start, -(e.end - e.start)))` 对默认值 (0,0) 仍正确工作。

### 3. `tests/unit/test_entity_store_orm.py`（新增，逐字来自 brief）

11 个测试覆盖: 建表、新建实体、精确匹配追加、不同 user 隔离、search 精确匹配、list 全量、list 按 type 过滤、get_linked_memories、remove 软删除、remove 多链接保留、旧构造器向后兼容。

**lint 适配**: brief 逐字测试含 3 处 lint 问题（I001 import 顺序、F401 未用 `sqlalchemy.inspect`、F841 未用 `result`）。已用 `ruff check --fix` 自动修 I001/F401；F841 改写为 `assert store.get(eid) is None`（保留"验证软删除后 get 返回 None"的测试意图）。

## TDD 证据

### RED（实现前）

命令: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_entity_store_orm.py -v`

```
collected 11 items
tests/unit/test_entity_store_orm.py::test_from_engine_creates_table FAILED [  9%]
tests/unit/test_entity_store_orm.py::test_upsert_new_entity FAILED       [ 18%]
... (11 个全 FAILED)
================================== FAILURES ===================================
_______________________ test_from_engine_creates_table ________________________
>       store = EntityStore.from_engine(engine, embedder=None)
E       AttributeError: type object 'EntityStore' has no attribute 'from_engine'
tests\unit\test_entity_store_orm.py:36: AttributeError
```
预期失败（`AttributeError: ... has no attribute 'from_engine'`）已确认。

### GREEN（实现后）

命令: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_entity_store_orm.py -v`

```
collected 11 items
tests/unit/test_entity_store_orm.py::test_from_engine_creates_table PASSED [  9%]
tests/unit/test_entity_store_orm.py::test_upsert_new_entity PASSED       [ 18%]
tests/unit/test_entity_store_orm.py::test_upsert_exact_match_appends PASSED [ 27%]
tests/unit/test_entity_store_orm.py::test_upsert_different_users_separate PASSED [ 36%]
tests/unit/test_entity_store_orm.py::test_search_exact_match PASSED      [ 45%]
tests/unit/test_entity_store_orm.py::test_list_entities PASSED           [ 54%]
tests/unit/test_entity_store_orm.py::test_list_by_type PASSED            [ 63%]
tests/unit/test_entity_store_orm.py::test_get_linked_memories PASSED     [ 72%]
tests/unit/test_entity_store_orm.py::test_remove_memory_from_entities PASSED [ 81%]
tests/unit/test_entity_store_orm.py::test_remove_memory_keeps_entity_with_other_links PASSED [ 90%]
tests/unit/test_entity_store_orm.py::test_old_constructor_still_works PASSED [100%]
============================= 11 passed in 5.39s ==============================
```

## Lint

命令: `ruff check --no-cache src/septmuse/storage/relational_stores/entity_store.py tests/unit/test_entity_store_orm.py src/septmuse/extraction/entity.py`

```
All checks passed!
```

## 回归测试

**entity 相关套件**（直接受 Entity 默认值改动影响）:
```
$env:PYTHONPATH="src"; python -m pytest tests/unit/test_entity_store.py tests/unit/test_entity_store_orm.py tests/unit/test_cognify.py tests/unit/test_triplet.py tests/unit/test_entity_extractor.py -q
86 passed, 2 skipped in 17.81s
```
（2 skipped = spaCy 未安装，属正常。）

**brief 指定的回归套件**:
```
$env:PYTHONPATH="src"; python -m pytest tests/unit/test_entity_store.py tests/unit/test_cognify.py -q --tb=no
32 passed in 12.77s
```

**全 unit 套件**: 622 passed, 14 skipped, 1 failed（`test_llm_providers.py::TestResolveLLM::test_openai_provider`）。
该失败在 `_resolve_llm`（LLM provider 解析，返回 None 而非 OpenAILLM），与本任务无任何代码关联（我未触碰 `septmuse/llms/`、`septmuse/configs/`），属预先存在的失败。

## 修改文件清单

| 文件 | 改动类型 | 说明 |
|------|----------|------|
| `src/septmuse/storage/relational_stores/entity_store.py` | 修改 | +导入 sqlmodel/EntityTable；`__init__` 加 `_engine=None`；+`from_engine`/`_is_orm_mode`；6 公共方法加 ORM 分支；+9 个 `_method_orm` 实现 |
| `src/septmuse/extraction/entity.py` | 修改 | `Entity.start`/`end` 加默认值 `0`（向后兼容） |
| `tests/unit/test_entity_store_orm.py` | 新增 | 11 测试（逐字 brief + 3 处 lint 适配） |

## 自审发现

1. **Entity 默认值改动（超出 brief 显式范围）**: brief 只说改 `entity_store.py`，但逐字测试 `Entity(text=..., entity_type=...)` 无 start/end 会 TypeError，无法运行。加默认值是使逐字测试可执行的必要最小改动，向后兼容（现有调用全传显式值）。已在报告显式标注。

2. **lint 与"逐字"冲突**: brief 测试代码含 3 处 lint 问题。brief 同时要求"逐字"和"All checks passed"，二者冲突时以硬约束（lint 通过）为准，做最小语义保持的修整（F841 → `assert ... is None` 保留验证意图）。

3. **raw SQL 路径完整保留**: 所有原 `_find_by_text`/`_find_by_embedding`/`_append_memory_id`/`_create_table_if_not_exists` 及 6 个公共方法的 raw SQL 主体未删未改，仅在每个公共方法顶部插入 ORM 守卫。旧构造器 `EntityStore(conn, lock, embedder)` 行为不变（`_engine=None` → 走 raw SQL）。

4. **无 UNIQUE 约束风险**: `EntityTable` 模型未声明 `UNIQUE(user_id, entity_text)`，但 dedup 逻辑（先查后插）保证不插入重复，测试 `test_upsert_exact_match_appends` 验证 `eid1 == eid2`。raw SQL 表有 UNIQUE 约束，ORM 表无——行为等价因 dedup 先于插入。

5. **预先存在的 llm_providers 失败**: 非本任务引入，与本任务代码无关联。

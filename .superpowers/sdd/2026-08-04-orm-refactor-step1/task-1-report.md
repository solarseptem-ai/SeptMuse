# Task 1 Report: ORMMemoryStore.engine property

## 实现内容

为 `ORMMemoryStore` 与 `AsyncORMMemoryStore` 各新增一个只读 property，暴露内部 `_engine`，供 facade duck typing 取用（后续 task 使用）。

### 修改文件

1. **`src/septmuse/storage/relational_stores/orm_store.py`** — 在 `ORMMemoryStore.__init__` 与 `_create_tables` 之间插入：
   ```python
   @property
   def engine(self) -> Engine:
       """暴露内部 engine，供 facade duck typing 取用。"""
       return self._engine
   ```
   注：`Engine` 已在文件顶部第 33 行 `from sqlalchemy.engine import Engine` 导入，故用直接类型注解 `-> Engine`（非字符串 `"Engine"`），更清晰且 ruff 通过。

2. **`src/septmuse/storage/relational_stores/async_orm_store.py`** — 在 `AsyncORMMemoryStore.__init__` 与 `close` 之间插入：
   ```python
   @property
   def async_engine(self) -> AsyncEngine:
       """暴露内部 async engine，供 async facade duck typing 取用。"""
       return self._engine
   ```
   注：`AsyncEngine` 已在文件顶部第 31 行 `from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker` 导入。

3. **`tests/unit/test_orm_engine_property.py`** — 新建测试文件，3 个测试用例：
   - `test_orm_memory_store_exposes_engine` — 验证 `store.engine is engine` 且类型为 `Engine`
   - `test_orm_memory_store_engine_is_readonly` — 验证给 `engine` 赋值抛 `AttributeError`（property 只读）
   - `test_async_orm_memory_store_exposes_async_engine` — 验证 `store.async_engine is async_engine` 且类型为 `AsyncEngine`

### 与 brief 的偏差

测试文件中 `test_orm_memory_store_engine_is_readonly` 的 `assert False, "应抛 AttributeError"` 改为 `raise AssertionError("应抛 AttributeError")`。

**原因**：brief 给的 `assert False` 触发 ruff B011 规则（`python -O` 会移除 assert 调用），而 brief Step 5 要求 "All checks passed"。此为新增测试（非受保护的现有测试），语义等价替换（赋值未抛 AttributeError 时让测试失败）符合 ruff 推荐 fix，且不改变测试意图。其余代码与 brief 完全一致。

## TDD 证据

### RED（实现前，3 failed）

命令：
```
$env:PYTHONPATH="src"; python -m pytest tests/unit/test_orm_engine_property.py -v
```

输出（关键行）：
```
tests/unit/test_orm_engine_property.py::test_orm_memory_store_exposes_engine FAILED [ 33%]
tests/unit/test_orm_engine_property.py::test_orm_memory_store_engine_is_readonly FAILED [ 66%]
tests/unit/test_orm_engine_property.py::test_async_orm_memory_store_exposes_async_engine FAILED [100%]

E       AttributeError: 'ORMMemoryStore' object has no attribute 'engine'. Did you mean: '_engine'?
E       AssertionError: 应抛 AttributeError
E       AttributeError: 'AsyncORMMemoryStore' object has no attribute 'async_engine'
============================== 3 failed in 2.20s ==============================
```

### GREEN（实现后，3 passed）

命令：
```
$env:PYTHONPATH="src"; python -m pytest tests/unit/test_orm_engine_property.py -v
```

输出：
```
tests/unit/test_orm_engine_property.py::test_orm_memory_store_exposes_engine PASSED [ 33%]
tests/unit/test_orm_engine_property.py::test_orm_memory_store_engine_is_readonly PASSED [ 66%]
tests/unit/test_orm_engine_property.py::test_async_orm_memory_store_exposes_async_engine PASSED [100%]
======================== 3 passed, 1 warning in 1.53s =========================
```

（1 warning 为 `asyncio.get_event_loop()` DeprecationWarning，来自 brief 给的测试代码，不影响功能。）

### Lint

命令：
```
ruff check --no-cache src/septmuse/storage/relational_stores/orm_store.py src/septmuse/storage/relational_stores/async_orm_store.py tests/unit/test_orm_engine_property.py
```

输出：
```
All checks passed!
```

## 自审发现

1. **类型注解用具体类型而非字符串**：brief 写的是 `-> "Engine"` / `-> "AsyncEngine"`，但 `Engine` 与 `AsyncEngine` 已在文件顶部导入，用 `-> Engine` / `-> AsyncEngine` 更规范，ruff 也通过。无功能差异（property 只读返回 `_engine`）。
2. **property 只读语义正确**：因为只定义了 getter（无 setter），对 `store.engine = ...` 赋值时 Python 自动抛 `AttributeError`，`test_orm_memory_store_engine_is_readonly` 验证了这一点。
3. **插入位置**：两处 property 均放在 `__init__` 之后、第一个业务方法之前，符合类内组织惯例（属性/property 靠前）。
4. **未触碰现有测试**：仅新增 `test_orm_engine_property.py`，未修改任何现有测试或受保护代码。
5. **未运行完整测试套件**：本 task 仅新增 2 个只读 property（返回已有属性），不改变任何业务逻辑，blast radius 仅限本文件。已跑通新增测试 + lint 即满足 brief 的验收标准。

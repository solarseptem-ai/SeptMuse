# mem0 风格 filters dict + session_id 全入口暴露 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 与 mem0 API 对齐：FiltersParser 全量操作符 + filters dict 在 store/facade/REST/CLI/MCP 全入口暴露

**Architecture:** 新建 FiltersParser 解析 mem0 风格 filters dict → SQL WHERE 子句；store 层 search/get_all/get/delete 加 filters 参数；facade/REST/CLI/MCP 透传 filters + session_id 暴露

**Tech Stack:** SQLite json_extract、aiosqlite、FastAPI、argparse、FastMCP

## 全局约束

- **PYTHONPATH=src** 运行所有测试（PowerShell: `$env:PYTHONPATH="src"`）
- **ruff line-length=120**，只用 `ruff check --no-cache`（禁用 ruff format）
- **不是 git 仓库**，无 commit 步骤
- **代码注释用中文**
- **现有测试固定不动**，仅新增测试
- **pytest 基线**：1076 passed + 36 skipped + 23 failed（不退化）
- 工作目录：E:\sonhhxg0529\vibe_coding_project\solarseptem-ai\solarseptem-ai-platform\SeptMuse

## 文件结构

**新建：**
- `src/septmuse/storage/filters.py` — FiltersParser
- `tests/unit/test_filters.py` — FiltersParser 测试

**修改：**
- `src/septmuse/storage/sqlite/store.py` — search/get_all/get/delete 加 filters
- `src/septmuse/storage/async_sqlite/store.py` — 同上
- `src/septmuse/memory/main.py` — search/get_all 加 filters
- `src/septmuse/memory/async_main.py` — search/get_all 加 filters
- `src/septmuse/api/rest/__init__.py` — GET /memories 加 session_id + SearchRequest 加 filters
- `src/septmuse/cli/main.py` — search/dump 加 --session-id
- `src/septmuse/api/mcp/tools.py` — search_memory 加 filters

---

## Task 1: FiltersParser + 测试

**Files:**
- Create: `src/septmuse/storage/filters.py`
- Test: `tests/unit/test_filters.py`

**Interfaces:**
- Produces: `FiltersParser().parse(filters: dict, backend: str = "sqlite") -> tuple[str, list]`

- [ ] **Step 1: 写失败测试**

```python
# tests/unit/test_filters.py
"""FiltersParser 测试 — mem0 风格 filters dict → SQL WHERE 子句。"""
import pytest

from septmuse.storage.filters import FiltersParser


@pytest.fixture
def parser():
    return FiltersParser()


def test_empty_filters(parser):
    """空 filters → 空 WHERE 子句。"""
    clause, params = parser.parse({})
    assert clause == ""
    assert params == []


def test_none_filters(parser):
    """None filters → 空 WHERE 子句。"""
    clause, params = parser.parse(None)
    assert clause == ""
    assert params == []


def test_direct_value_exact_match(parser):
    """直接值 → 精确匹配。"""
    clause, params = parser.parse({"user_id": "alice"})
    assert "user_id = ?" in clause
    assert params == ["alice"]


def test_eq_operator(parser):
    """eq 操作符。"""
    clause, params = parser.parse({"category": {"eq": "work"}})
    assert "json_extract(metadata, '$.category') = ?" in clause
    assert params == ["work"]


def test_ne_operator(parser):
    """ne 操作符。"""
    clause, params = parser.parse({"category": {"ne": "work"}})
    assert "json_extract(metadata, '$.category') <> ?" in clause
    assert params == ["work"]


def test_gt_gte_lt_lte_operators(parser):
    """数值比较操作符。"""
    for op, sql_op in [("gt", ">"), ("gte", ">="), ("lt", "<"), ("lte", "<=")]:
        clause, params = parser.parse({"priority": {op: 5}})
        assert f"json_extract(metadata, '$.priority') {sql_op} ?" in clause
        assert params == [5]


def test_in_operator(parser):
    """in 操作符。"""
    clause, params = parser.parse({"tag": {"in": ["urgent", "bug"]}})
    assert "IN (?, ?)" in clause
    assert params == ["urgent", "bug"]


def test_nin_operator(parser):
    """nin 操作符。"""
    clause, params = parser.parse({"tag": {"nin": ["archived"]}})
    assert "NOT IN (?)" in clause
    assert params == ["archived"]


def test_contains_operator(parser):
    """contains 操作符。"""
    clause, params = parser.parse({"content": {"contains": "Python"}})
    assert "LIKE" in clause
    assert params == ["Python"]


def test_icontains_operator(parser):
    """icontains 操作符（不区分大小写）。"""
    clause, params = parser.parse({"content": {"icontains": "python"}})
    assert "LOWER" in clause
    assert params == ["python"]


def test_wildcard(parser):
    """通配符 * → IS NOT NULL。"""
    clause, params = parser.parse({"category": "*"})
    assert "IS NOT NULL" in clause
    assert params == []


def test_and_logical(parser):
    """AND 逻辑运算。"""
    clause, params = parser.parse({"AND": [{"user_id": "alice"}, {"session_id": "s1"}]})
    assert "AND" in clause
    assert "user_id = ?" in clause
    assert "session_id = ?" in clause
    assert params == ["alice", "s1"]


def test_or_logical(parser):
    """OR 逻辑运算。"""
    clause, params = parser.parse({"OR": [{"agent_id": "a1"}, {"agent_id": "a2"}]})
    assert "OR" in clause
    assert params == ["a1", "a2"]


def test_not_logical(parser):
    """NOT 逻辑运算。"""
    clause, params = parser.parse({"NOT": [{"state": "deleted"}]})
    assert "NOT" in clause
    assert "state = ?" in clause
    assert params == ["deleted"]


def test_run_id_maps_to_session_id(parser):
    """run_id 映射到 session_id（mem0 兼容）。"""
    clause, params = parser.parse({"run_id": "sess-123"})
    assert "session_id = ?" in clause
    assert params == ["sess-123"]


def test_entity_field_vs_metadata(parser):
    """实体字段直接引用列名，metadata 字段用 json_extract。"""
    clause, _ = parser.parse({"user_id": "alice", "category": "work"})
    assert "user_id = ?" in clause
    assert "json_extract(metadata, '$.category')" in clause


def test_postgres_backend(parser):
    """PG 后端用 metadata->>'key' 替代 json_extract。"""
    clause, _ = parser.parse({"category": "work"}, backend="postgres")
    assert "metadata->>'category'" in clause
    assert "json_extract" not in clause


def test_nested_logical(parser):
    """嵌套逻辑运算：AND(OR(...), NOT(...))。"""
    clause, params = parser.parse({
        "AND": [
            {"OR": [{"user_id": "a"}, {"user_id": "b"}]},
            {"NOT": [{"state": "deleted"}]},
        ]
    })
    assert "OR" in clause
    assert "NOT" in clause
    assert params == ["a", "b", "deleted"]
```

- [ ] **Step 2: 运行确认失败**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_filters.py -v`
Expected: FAIL — `No module named 'septmuse.storage.filters'`

- [ ] **Step 3: 写 FiltersParser**

```python
# src/septmuse/storage/filters.py
"""mem0 风格 filters dict 解析器 — filters dict → SQL WHERE 子句 + 参数。

支持:
- 直接值: {"key": "value"} → key = ?
- 操作符: eq/ne/gt/gte/lt/lte/in/nin/contains/icontains
- 通配符: {"key": "*"} → key IS NOT NULL
- 逻辑运算: AND/OR/NOT
- 实体字段: user_id/agent_id/session_id/run_id(state
- metadata 字段: 任意 key → json_extract(metadata, '$.key')

run_id 映射到 session_id（mem0 兼容）。
"""
from __future__ import annotations

from typing import Any

# memories 表的列名（直接引用，不走 json_extract）
_ENTITY_KEYS = {"user_id", "agent_id", "session_id", "run_id", "state"}

# 逻辑运算符
_LOGICAL_OPS = {"AND", "OR", "NOT"}

# 比较操作符 → SQL 操作符
_COMPARE_OPS = {
    "eq": "=",
    "ne": "<>",
    "gt": ">",
    "gte": ">=",
    "lt": "<",
    "lte": "<=",
}


class FiltersParser:
    """解析 mem0 风格 filters dict → SQL WHERE 子句 + 参数列表。"""

    def parse(self, filters: dict[str, Any] | None, backend: str = "sqlite") -> tuple[str, list[Any]]:
        """解析 filters → (where_clause, params)。

        Args:
            filters: mem0 风格 filters dict
            backend: "sqlite" 或 "postgres"

        Returns:
            (where_clause, params): WHERE 子句和参数列表。空 filters 返回 ("", [])。
        """
        if not filters:
            return "", []
        return self._parse_dict(filters, backend)

    def _parse_dict(self, filters: dict[str, Any], backend: str) -> tuple[str, list[Any]]:
        parts: list[str] = []
        params: list[Any] = []
        for key, value in filters.items():
            if key in _LOGICAL_OPS:
                clause, p = self._parse_logical(key, value, backend)
                parts.append(clause)
                params.extend(p)
            elif key in _ENTITY_KEYS:
                actual_key = "session_id" if key == "run_id" else key
                clause, p = self._parse_field(actual_key, value, backend, is_entity=True)
                parts.append(clause)
                params.extend(p)
            else:
                clause, p = self._parse_field(key, value, backend, is_entity=False)
                parts.append(clause)
                params.extend(p)
        return " AND ".join(parts), params

    def _parse_field(self, key: str, value: Any, backend: str, is_entity: bool) -> tuple[str, list[Any]]:
        """解析单个字段条件。"""
        col = key if is_entity else self._metadata_col(key, backend)

        # 通配符 * → IS NOT NULL
        if value == "*":
            return f"{col} IS NOT NULL", []

        # 操作符 dict
        if isinstance(value, dict):
            return self._parse_operator(col, value)

        # 直接值 → 精确匹配
        return f"{col} = ?", [value]

    def _metadata_col(self, key: str, backend: str) -> str:
        """生成 metadata 列引用。"""
        if backend == "postgres":
            return f"metadata->>'{key}'"
        return f"json_extract(metadata, '$.{key}')"

    def _parse_operator(self, col: str, value: dict[str, Any]) -> tuple[str, list[Any]]:
        """解析操作符 dict。"""
        if len(value) != 1:
            raise ValueError(f"操作符 dict 只能有一个 key, got: {list(value.keys())}")

        op = list(value.keys())[0]
        val = value[op]

        if op in _COMPARE_OPS:
            return f"{col} {_COMPARE_OPS[op]} ?", [val]
        elif op == "in":
            if not isinstance(val, list) or not val:
                raise ValueError(f"in 操作符需要非空列表, got: {val}")
            placeholders = ", ".join("?" * len(val))
            return f"{col} IN ({placeholders})", list(val)
        elif op == "nin":
            if not isinstance(val, list) or not val:
                raise ValueError(f"nin 操作符需要非空列表, got: {val}")
            placeholders = ", ".join("?" * len(val))
            return f"{col} NOT IN ({placeholders})", list(val)
        elif op == "contains":
            return f"{col} LIKE '%' || ? || '%'", [val]
        elif op == "icontains":
            return f"LOWER({col}) LIKE LOWER('%' || ? || '%')", [val]
        else:
            raise ValueError(f"不支持的操作符: {op}")

    def _parse_logical(self, op: str, conditions: list[dict], backend: str) -> tuple[str, list[Any]]:
        """解析逻辑运算（AND/OR/NOT）。"""
        if not isinstance(conditions, list):
            raise ValueError(f"{op} 需要列表值, got: {type(conditions)}")

        parts: list[str] = []
        params: list[Any] = []
        for cond in conditions:
            clause, p = self._parse_dict(cond, backend)
            parts.append(f"({clause})")
            params.extend(p)

        if op == "NOT":
            inner = " AND ".join(parts)
            return f"NOT ({inner})", params
        else:
            joined = f" {op} ".join(parts)
            return f"({joined})", params
```

- [ ] **Step 4: 运行测试确认通过**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_filters.py -v`
Expected: PASS（18 测试全通过）

- [ ] **Step 5: ruff 检查**

Run: `ruff check --no-cache src/septmuse/storage/filters.py tests/unit/test_filters.py`
Expected: All checks passed!

---

## Task 2: Store 层集成（SQLiteMemoryStore + AsyncSQLiteMemoryStore）

**Files:**
- Modify: `src/septmuse/storage/sqlite/store.py`
- Modify: `src/septmuse/storage/async_sqlite/store.py`

**Interfaces:**
- Consumes: Task 1 的 `FiltersParser`
- Produces: store 层 `search(filters=...)` / `get_all(filters=...)` / `get(filters=...)` / `delete(filters=...)` 支持

- [ ] **Step 1: 修改 SQLiteMemoryStore.search 加 filters 参数**

在 `src/septmuse/storage/sqlite/store.py` 中，找到 `search` 方法签名：

```python
    def search(
        self,
        query_embedding: list[float],
        *,
        user_id: str,
        session_id: str | None = None,
        top_k: int = 5,
        threshold: float = 0.1,
    ) -> list[dict[str, Any]]:
```

改为（加 `filters` 参数）：

```python
    def search(
        self,
        query_embedding: list[float],
        *,
        user_id: str,
        session_id: str | None = None,
        filters: dict[str, Any] | None = None,
        top_k: int = 5,
        threshold: float = 0.1,
    ) -> list[dict[str, Any]]:
```

在方法体中，找到构建 SQL WHERE 的部分：

```python
        with self._lock:
            sql = "SELECT id, content, embedding, metadata, created_at FROM memories WHERE user_id=? AND is_deleted=0"
            params: list[Any] = [user_id]
            if session_id is not None:
                sql += " AND session_id = ?"
                params.append(session_id)
            cur = self.conn.execute(sql, params)
```

替换为（加 filters 解析 + 直接参数覆盖逻辑）：

```python
        with self._lock:
            sql = "SELECT id, content, embedding, metadata, created_at FROM memories WHERE user_id=? AND is_deleted=0"
            params: list[Any] = [user_id]
            if session_id is not None:
                sql += " AND session_id = ?"
                params.append(session_id)
            # mem0 风格 filters dict
            if filters:
                from septmuse.storage.filters import FiltersParser
                clean_filters = filters.copy()
                # 直接参数覆盖 filters 中的同名 key
                if session_id is not None:
                    clean_filters.pop("session_id", None)
                    clean_filters.pop("run_id", None)
                clause, fparams = FiltersParser().parse(clean_filters, "sqlite")
                if clause:
                    sql += f" AND {clause}"
                    params.extend(fparams)
            cur = self.conn.execute(sql, params)
```

- [ ] **Step 2: 修改 SQLiteMemoryStore.get_all 加 filters 参数**

找到 `get_all` 方法签名：

```python
    def get_all(self, *, user_id: str, session_id: str | None = None) -> list[dict[str, Any]]:
```

改为：

```python
    def get_all(
        self, *, user_id: str, session_id: str | None = None, filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
```

在方法体中，找到构建 SQL WHERE 的部分：

```python
        with self._lock:
            sql = "SELECT id, content, metadata, created_at, updated_at FROM memories WHERE user_id=? AND is_deleted=0"
            params: list[Any] = [user_id]
            if session_id is not None:
                sql += " AND session_id = ?"
                params.append(session_id)
            cur = self.conn.execute(sql, params)
```

替换为（加 filters 解析）：

```python
        with self._lock:
            sql = "SELECT id, content, metadata, created_at, updated_at FROM memories WHERE user_id=? AND is_deleted=0"
            params: list[Any] = [user_id]
            if session_id is not None:
                sql += " AND session_id = ?"
                params.append(session_id)
            if filters:
                from septmuse.storage.filters import FiltersParser
                clean_filters = filters.copy()
                if session_id is not None:
                    clean_filters.pop("session_id", None)
                    clean_filters.pop("run_id", None)
                clause, fparams = FiltersParser().parse(clean_filters, "sqlite")
                if clause:
                    sql += f" AND {clause}"
                    params.extend(fparams)
            cur = self.conn.execute(sql, params)
```

- [ ] **Step 3: 修改 AsyncSQLiteMemoryStore.search 加 filters 参数**

在 `src/septmuse/storage/async_sqlite/store.py` 中，找到 `search` 方法签名：

```python
    async def search(
        self,
        query_embedding: list[float],
        *,
        user_id: str,
        session_id: str | None = None,
        top_k: int = 5,
        threshold: float = 0.1,
    ) -> list[dict[str, Any]]:
```

改为（加 `filters` 参数）：

```python
    async def search(
        self,
        query_embedding: list[float],
        *,
        user_id: str,
        session_id: str | None = None,
        filters: dict[str, Any] | None = None,
        top_k: int = 5,
        threshold: float = 0.1,
    ) -> list[dict[str, Any]]:
```

在方法体中，找到构建 SQL WHERE 的部分（当前有 session_id 分支）：

```python
        if session_id:
            cursor = await conn.execute(
                """SELECT id, content, metadata, created_at, embedding FROM memories
                   WHERE user_id=? AND is_deleted=0 AND (state='active' OR state IS NULL) AND session_id=?""",
                (user_id, session_id),
            )
        else:
            cursor = await conn.execute(
                """SELECT id, content, metadata, created_at, embedding FROM memories
                   WHERE user_id=? AND is_deleted=0 AND (state='active' OR state IS NULL)""",
                (user_id,),
            )
```

替换为（统一构建 WHERE 子句）：

```python
        sql = """SELECT id, content, metadata, created_at, embedding FROM memories
                 WHERE user_id=? AND is_deleted=0 AND (state='active' OR state IS NULL)"""
        params_list: list[Any] = [user_id]
        if session_id:
            sql += " AND session_id = ?"
            params_list.append(session_id)
        if filters:
            from septmuse.storage.filters import FiltersParser
            clean_filters = filters.copy()
            if session_id:
                clean_filters.pop("session_id", None)
                clean_filters.pop("run_id", None)
            clause, fparams = FiltersParser().parse(clean_filters, "sqlite")
            if clause:
                sql += f" AND {clause}"
                params_list.extend(fparams)
        cursor = await conn.execute(sql, params_list)
```

- [ ] **Step 4: 修改 AsyncSQLiteMemoryStore.get_all 加 filters 参数**

找到 `get_all` 方法签名：

```python
    async def get_all(self, *, user_id: str, session_id: str | None = None) -> list[dict[str, Any]]:
```

改为：

```python
    async def get_all(
        self, *, user_id: str, session_id: str | None = None, filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
```

在方法体中，找到构建 SQL WHERE 的部分（当前有 session_id 分支）：

```python
        if session_id:
            cursor = await conn.execute(
                """SELECT id, content, metadata, created_at, updated_at FROM memories
                   WHERE user_id=? AND is_deleted=0 AND session_id=? AND (state='active' OR state IS NULL)""",
                (user_id, session_id),
            )
        else:
            cursor = await conn.execute(
                """SELECT id, content, metadata, created_at, updated_at FROM memories
                   WHERE user_id=? AND is_deleted=0 AND (state='active' OR state IS NULL)""",
                (user_id,),
            )
```

替换为（统一构建 WHERE 子句）：

```python
        sql = """SELECT id, content, metadata, created_at, updated_at FROM memories
                 WHERE user_id=? AND is_deleted=0 AND (state='active' OR state IS NULL)"""
        params_list: list[Any] = [user_id]
        if session_id:
            sql += " AND session_id = ?"
            params_list.append(session_id)
        if filters:
            from septmuse.storage.filters import FiltersParser
            clean_filters = filters.copy()
            if session_id:
                clean_filters.pop("session_id", None)
                clean_filters.pop("run_id", None)
            clause, fparams = FiltersParser().parse(clean_filters, "sqlite")
            if clause:
                sql += f" AND {clause}"
                params_list.extend(fparams)
        cursor = await conn.execute(sql, params_list)
```

- [ ] **Step 5: 在 store.py 顶部加 Any import（如缺）**

确认 `src/septmuse/storage/sqlite/store.py` 顶部有 `from typing import Any`。如缺则加。

- [ ] **Step 6: 运行 store 测试确认不破坏**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_composite_store.py tests/unit/test_async_sqlite_store.py tests/unit/test_async_memory.py -v`
Expected: PASS（全通过）

- [ ] **Step 7: ruff 检查**

Run: `ruff check --no-cache src/septmuse/storage/sqlite/store.py src/septmuse/storage/async_sqlite/store.py`
Expected: All checks passed!

---

## Task 3: Facade + REST + CLI 集成

**Files:**
- Modify: `src/septmuse/memory/main.py`
- Modify: `src/septmuse/memory/async_main.py`
- Modify: `src/septmuse/api/rest/__init__.py`
- Modify: `src/septmuse/cli/main.py`

**Interfaces:**
- Consumes: Task 2 的 store 层 `search(filters=...)` / `get_all(filters=...)`
- Produces: facade `search(filters=...)` + REST `GET /memories?session_id=` + `POST /memories/search` filters 字段 + CLI `--session-id`

- [ ] **Step 1: 修改 Memory.search 加 filters 参数**

在 `src/septmuse/memory/main.py` 中，找到 `search` 方法。在签名中加 `filters: dict[str, Any] | None = None` 参数（在 `session_id` 后面），并在调用 `self.store.search` 时透传 `filters=filters`。

找到类似这样的代码：

```python
    def search(self, query, *, user_id, session_id=None, ...):
        ...
        results = self.store.search(emb, user_id=user_id, session_id=session_id, top_k=top_k, threshold=threshold)
```

改为在签名加 `filters` 参数，在调用加 `filters=filters`。

**注意**：`Memory.search` 可能有多个重载或较长签名。仔细读文件找到实际的 search 方法签名和调用 store.search 的行，加 `filters` 参数。

- [ ] **Step 2: 修改 Memory.get_all 加 filters 参数**

同上，找到 `get_all` 方法，加 `filters` 参数并透传。

- [ ] **Step 3: 修改 AsyncMemory.search 加 filters 参数**

在 `src/septmuse/memory/async_main.py` 中，找到 `search` 方法：

```python
    async def search(
        self, query: str, *, user_id: str, session_id: str | None = None,
        top_k: int = 5, threshold: float = 0.1,
    ) -> list[dict[str, Any]]:
        """异步检索记忆。"""
        emb = await asyncio.to_thread(self.embedder.embed, query)
        return await self.store.search(
            emb, user_id=user_id, session_id=session_id, top_k=top_k, threshold=threshold
        )
```

改为：

```python
    async def search(
        self, query: str, *, user_id: str, session_id: str | None = None,
        filters: dict[str, Any] | None = None,
        top_k: int = 5, threshold: float = 0.1,
    ) -> list[dict[str, Any]]:
        """异步检索记忆。"""
        emb = await asyncio.to_thread(self.embedder.embed, query)
        return await self.store.search(
            emb, user_id=user_id, session_id=session_id, filters=filters, top_k=top_k, threshold=threshold
        )
```

- [ ] **Step 4: 修改 AsyncMemory.get_all 加 filters 参数**

找到 `get_all` 方法：

```python
    async def get_all(self, *, user_id: str) -> list[dict[str, Any]]:
        """异步列出全部。"""
        return await self.store.get_all(user_id=user_id)
```

改为：

```python
    async def get_all(
        self, *, user_id: str, session_id: str | None = None, filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """异步列出全部。"""
        return await self.store.get_all(user_id=user_id, session_id=session_id, filters=filters)
```

- [ ] **Step 5: 修改 REST SearchRequest 加 filters 字段**

在 `src/septmuse/api/rest/__init__.py` 中，找到 `SearchRequest` 类：

```python
class SearchRequest(BaseModel):
    query: str = Field(description="查询文本")
    user_id: str = Field(description="用户 ID")
    session_id: str | None = Field(default=None, description="会话 ID (对齐 mem0 run_id)")
    top_k: int = Field(default=5, description="返回数")
    threshold: float = Field(default=0.1, description="相似阈值")
    reranker: str | None = Field(default=None, description="reranker: noop/mmr/cross_encoder/llm")
    explain: bool = Field(default=False, description="返回 score_details")
```

在 `session_id` 后面加一行：

```python
    filters: dict[str, Any] | None = Field(default=None, description="mem0 风格 filters dict")
```

- [ ] **Step 6: 修改 REST GET /memories 加 session_id 参数**

找到 `list_memories` 端点：

```python
    @app.get("/memories")
    async def list_memories(
        user_id: str = Query(..., description="用户 ID"),
        app_id: str | None = None,
    ) -> dict[str, Any]:
```

改为（加 `session_id` 参数）：

```python
    @app.get("/memories")
    async def list_memories(
        user_id: str = Query(..., description="用户 ID"),
        session_id: str | None = Query(default=None, description="会话 ID"),
        app_id: str | None = None,
    ) -> dict[str, Any]:
```

在方法体中，找到 `results = await app.state.async_memory.get_all(user_id=user_id)`，改为：

```python
        results = await app.state.async_memory.get_all(user_id=user_id, session_id=session_id)
```

- [ ] **Step 7: 修改 REST POST /memories/search 透传 filters**

找到 `search_memories` 端点中的 async 分支：

```python
        else:
            # 基础检索，用 async
            results = await app.state.async_memory.search(
                req.query,
                user_id=req.user_id,
                session_id=req.session_id,
                top_k=req.top_k,
                threshold=req.threshold,
            )
```

改为（加 `filters=req.filters`）：

```python
        else:
            # 基础检索，用 async
            results = await app.state.async_memory.search(
                req.query,
                user_id=req.user_id,
                session_id=req.session_id,
                filters=req.filters,
                top_k=req.top_k,
                threshold=req.threshold,
            )
```

- [ ] **Step 8: 修改 CLI search 命令加 --session-id**

在 `src/septmuse/cli/main.py` 中，找到 search 子命令的 argparse 定义，加 `--session-id` 参数：

```python
    p_search.add_argument("--session-id", default=None, help="会话 ID（仅搜该会话的记忆）")
```

在 search 命令处理函数中，找到调用 `memory.search(...)` 的地方，加 `session_id=args.session_id`。

- [ ] **Step 9: 修改 CLI dump 命令加 --session-id**

找到 dump 子命令的 argparse 定义，加 `--session-id` 参数：

```python
    p_dump.add_argument("--session-id", default=None, help="会话 ID（仅导出该会话的记忆）")
```

在 dump 命令处理函数中，找到调用 `memory.get_all(...)` 的地方，加 `session_id=args.session_id`。

- [ ] **Step 10: 运行 async memory 测试确认不破坏**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_async_memory.py -v`
Expected: PASS（5 测试全通过）

- [ ] **Step 11: 运行 REST 测试确认不破坏**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_api_permission_integration.py tests/unit/test_rbac_rest_openai.py -q --tb=line 2>&1 | Select-Object -Last 5`
Expected: 37 passed

- [ ] **Step 12: ruff 检查**

Run: `ruff check --no-cache src/septmuse/memory/main.py src/septmuse/memory/async_main.py src/septmuse/api/rest/__init__.py src/septmuse/cli/main.py`
Expected: All checks passed!

---

## Task 4: MCP 集成 + 全量验证

**Files:**
- Modify: `src/septmuse/api/mcp/tools.py`

**Interfaces:**
- Consumes: Task 3 的 facade `search(filters=...)`
- Produces: MCP `search_memory` 工具加 `filters` 参数

- [ ] **Step 1: 修改 MCP search_memory 工具加 filters 参数**

在 `src/septmuse/api/mcp/tools.py` 中，找到 `search_memory` 工具函数。读文件了解其签名（注意：MCP tools.py 禁止 `from __future__ import annotations`，工具签名必须用具体类型）。

在 `search_memory` 函数签名中加 `filters: dict[str, str] | None = None` 参数（用 `dict[str, str]` 而非 `dict[str, Any]`，因为 FastMCP 的类型解析不支持 Any）。

在函数体中，找到调用 `memory.search(...)` 的地方，加 `filters=filters`。

**注意**：MCP tools.py 中的函数签名必须用具体类型（不能用 `Any`），`dict[str, str]` 是安全的。如果 FastMCP 不支持 `dict[str, str] | None`，用 `dict | None = None`。

- [ ] **Step 2: 全量 ruff**

Run: `ruff check --no-cache src/ tests/`
Expected: All checks passed!

- [ ] **Step 3: 全量 pytest**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/ tests/e2e/ -q --tb=line 2>&1 | Select-Object -Last 5`
Expected: 失败不超过 23（基线），passed 不低于 1076 + 18 个 filters 测试

- [ ] **Step 4: REST GET /memories?session_id= 验证**

Run: `$env:PYTHONPATH="src"; python -c "import os; os.environ['SEPTMUSE_DB_PATH']=os.path.join(os.environ['TEMP'],'f.db'); from fastapi.testclient import TestClient; from septmuse.api.rest import create_app; c=TestClient(create_app()); r=c.post('/memories',json={'content':'test','user_id':'a','session_id':'s1'}); mid=r.json()['results'][0]['id']; r2=c.get('/memories?user_id=a&session_id=s1'); assert len(r2.json()['results'])==1, r2.json(); r3=c.get('/memories?user_id=a&session_id=s2'); assert len(r3.json()['results'])==0; print('OK session_id filter works')"`
Expected: `OK session_id filter works`

- [ ] **Step 5: CLI --session-id 验证**

Run: `$env:PYTHONPATH="src"; python -m septmuse.cli.main add "test session" --user alice --session-id sess-1 2>&1; python -m septmuse.cli.main search "test" --user alice --session-id sess-1 2>&1`
Expected: 能搜到 sess-1 的记忆

- [ ] **Step 6: AsyncMemory filters 验证**

Run: `$env:PYTHONPATH="src"; python -c "import asyncio; from septmuse.memory.async_main import AsyncMemory; from septmuse.embedders.hash import HashEmbedder; import tempfile, os; db=os.path.join(tempfile.mkdtemp(),'t.db'); from septmuse.storage.async_sqlite.store import AsyncSQLiteMemoryStore; s=AsyncSQLiteMemoryStore(db_path=db); m=AsyncMemory(embedder=HashEmbedder(), store=s); asyncio.run(m.add('hello', user_id='a', session_id='s1')); r=asyncio.run(m.search('hello', user_id='a', filters={'session_id':'s1'})); assert len(r)==1; r2=asyncio.run(m.search('hello', user_id='a', filters={'session_id':'s2'})); assert len(r2)==0; print('OK filters work'); asyncio.run(m.close())"`
Expected: `OK filters work`

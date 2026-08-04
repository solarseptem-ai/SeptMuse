# ORMMemoryStore P1 实施计划 — models/ 包 + sync CRUD

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建 models/ 目录包（6 文件）+ ORMMemoryStore（sync 完整 CRUD），用 SQLModel ORM 跨方言，不破坏现有 1116 测试基线。

**Architecture:** 新建 `services/database/models/` 包替代 `models.py`，新建 `ORMMemoryStore` 实现 `MemoryStore` ABC。纯新增代码，现有 `SQLiteMemoryStore` 保留不动（P4 才删除）。ORMMemoryStore 从 DatabaseService 拿 engine，CRUD 全用 SQLModel `select()` / `session.add()`。

**Tech Stack:** SQLModel, SQLAlchemy 2.0, SQLite（测试用 `sqlite://` 内存库）

## Global Constraints

- PYTHONPATH=src 运行所有 pytest（包未 pip install -e .）
- ruff line-length 120，select=["E","F","I","W","UP","B","SIM","RUF"]，ignore=["E501","RUF001","RUF002","RUF003"]
- **禁止 `ruff format <file>`**（Windows 上清空文件 bug）——只用 `ruff check --fix` 或 `ruff check --no-cache`
- 代码注释用中文
- 不用 git（文件快照模式）——无 commit 步骤
- 现有测试固定不动，仅新增测试
- metadata 列命名：Python 属性 `metadata_json`，`sa_column=Column("metadata", Text)` 映射数据库列名 `metadata`
- score 统一为相似度 [0,1]，越高越相似

## 文件结构

### 新建文件
| 文件 | 职责 |
|------|------|
| `src/septmuse/services/database/models/__init__.py` | 导出所有表类 |
| `src/septmuse/services/database/models/memory.py` | MemoryTable |
| `src/septmuse/services/database/models/history.py` | HistoryTable |
| `src/septmuse/services/database/models/access_log.py` | AccessLogTable |
| `src/septmuse/services/database/models/entity.py` | EntityTable + EntityRelationTable |
| `src/septmuse/storage/relational_stores/orm_store.py` | ORMMemoryStore（sync） |
| `tests/unit/test_orm_memory_store.py` | ORMMemoryStore 测试 |

### 修改文件
| 文件 | 改动 |
|------|------|
| `src/septmuse/services/database/models.py` | 删除（内容移到 models/ 包） |
| `src/septmuse/services/database/service.py` | import 路径更新 |
| `src/septmuse/storage/relational_stores/__init__.py` | 导出 ORMMemoryStore |

---

### Task 1: models/ 目录包 — 所有表定义

**Files:**
- Create: `src/septmuse/services/database/models/__init__.py`
- Create: `src/septmuse/services/database/models/memory.py`
- Create: `src/septmuse/services/database/models/history.py`
- Create: `src/septmuse/services/database/models/access_log.py`
- Create: `src/septmuse/services/database/models/entity.py`
- Delete: `src/septmuse/services/database/models.py`（内容移到包中）
- Test: `tests/unit/test_models_package.py`

**Interfaces:**
- Produces: `MemoryTable`, `HistoryTable`, `AccessLogTable`, `EntityTable`, `EntityRelationTable` — 后续 Task 的 ORMMemoryStore 依赖这些表类

- [ ] **Step 1: 写 models/memory.py**

```python
#  Copyright 2026 The sonhhxg0529 Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""MemoryTable — memories 表定义（跨方言 DDL）。"""

from __future__ import annotations

from sqlalchemy import Column, Text
from sqlmodel import Field, SQLModel


class MemoryTable(SQLModel, table=True):
    """memories 表 — 记忆主表。"""

    __tablename__ = "memories"

    id: str = Field(primary_key=True)
    user_id: str = Field(index=True)
    agent_id: str | None = None
    session_id: str | None = None
    content: str
    embedding: str | None = None  # JSON list[float]，跨方言通用
    # metadata 是 Python 保留名, 用 sa_column 映射到数据库列名
    metadata_json: str = Field(default="{}", sa_column=Column("metadata", Text))
    created_at: str | None = None
    updated_at: str | None = None
    is_deleted: int = Field(default=0)
    state: str = Field(default="active")
    app_id: str | None = None
    archived_at: str | None = None
    deleted_at: str | None = None
    valid_at: str | None = None
    invalid_at: str | None = None
    expired_at: str | None = None
```

- [ ] **Step 2: 写 models/history.py**

```python
#  Copyright 2026 The sonhhxg0529 Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""HistoryTable — history 表定义（记忆变更历史）。"""

from __future__ import annotations

from sqlmodel import Field, SQLModel


class HistoryTable(SQLModel, table=True):
    """history 表 — 记忆变更历史。"""

    __tablename__ = "history"

    id: str = Field(primary_key=True)
    memory_id: str | None = None
    old_memory: str | None = None
    new_memory: str | None = None
    event: str | None = None
    created_at: str | None = None
    is_deleted: int = Field(default=0)
```

- [ ] **Step 3: 写 models/access_log.py**

```python
#  Copyright 2026 The sonhhxg0529 Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""AccessLogTable — memory_access_logs 表定义（访问审计日志）。"""

from __future__ import annotations

from sqlalchemy import Column, Text
from sqlmodel import Field, SQLModel


class AccessLogTable(SQLModel, table=True):
    """memory_access_logs 表 — 访问审计日志。"""

    __tablename__ = "memory_access_logs"

    id: str = Field(primary_key=True)
    memory_id: str = Field(index=True)
    app_id: str | None = None
    access_type: str
    metadata_json: str | None = Field(default=None, sa_column=Column("metadata", Text))
    accessed_at: str
```

- [ ] **Step 4: 写 models/entity.py**

```python
#  Copyright 2026 The sonhhxg0529 Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""EntityTable + EntityRelationTable — 实体 + 实体关系表定义。"""

from __future__ import annotations

from sqlmodel import Field, SQLModel


class EntityTable(SQLModel, table=True):
    """septmuse_entities 表 — 实体存储（借鉴 mem0 V3 去图化）。"""

    __tablename__ = "septmuse_entities"

    id: str = Field(primary_key=True)
    entity_text: str
    entity_type: str
    entity_embedding: bytes | None = None  # BLOB 序列化向量
    linked_memory_ids: str  # JSON list[str]
    user_id: str = Field(index=True)
    agent_id: str | None = None
    created_at: str
    updated_at: str
    is_deleted: int = Field(default=0)


class EntityRelationTable(SQLModel, table=True):
    """entity_relations 表 — 实体间关系边（借鉴 graphiti）。"""

    __tablename__ = "entity_relations"

    id: str = Field(primary_key=True)
    source_entity: str = Field(index=True)
    relation: str
    target_entity: str = Field(index=True)
    user_id: str = Field(index=True)
    created_at: str | None = None
```

- [ ] **Step 5: 写 models/__init__.py**

```python
#  Copyright 2026 The sonhhxg0529 Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""数据库表定义包 — 跨方言 DDL（SQLite / MySQL / PostgreSQL）。

SQLModel.metadata.create_all(engine) 会根据 engine dialect 自动生成对应方言的 DDL。
"""

from septmuse.services.database.models.access_log import AccessLogTable
from septmuse.services.database.models.entity import EntityRelationTable, EntityTable
from septmuse.services.database.models.history import HistoryTable
from septmuse.services.database.models.memory import MemoryTable

__all__ = [
    "AccessLogTable",
    "EntityRelationTable",
    "EntityTable",
    "HistoryTable",
    "MemoryTable",
]
```

- [ ] **Step 6: 删除旧 models.py**

删除 `src/septmuse/services/database/models.py`（内容已移到 models/ 包）。

- [ ] **Step 7: 更新 service.py 的 import**

修改 `src/septmuse/services/database/service.py` 第 40 行：

```python
# 旧:
from septmuse.services.database.models import AccessLogTable, HistoryTable, MemoryTable

# 新:
from septmuse.services.database.models import AccessLogTable, HistoryTable, MemoryTable
```

（import 路径不变——`services.database.models` 现在是包不是模块，Python 自动解析到 `models/__init__.py`）

- [ ] **Step 8: 写测试**

创建 `tests/unit/test_models_package.py`：

```python
"""models/ 包测试 — 验证表定义可建表、列完整。"""

import pytest
from sqlalchemy import create_engine, inspect
from sqlmodel import SQLModel

from septmuse.services.database.models import (
    AccessLogTable,
    EntityRelationTable,
    EntityTable,
    HistoryTable,
    MemoryTable,
)


@pytest.fixture
def engine():
    eng = create_engine("sqlite://", echo=False)
    SQLModel.metadata.create_all(eng)
    return eng


def test_memory_table_columns(engine):
    """MemoryTable 有全部 16 列。"""
    cols = {c["name"] for c in inspect(engine).get_columns("memories")}
    expected = {
        "id", "user_id", "agent_id", "session_id", "content", "embedding",
        "metadata", "created_at", "updated_at", "is_deleted", "state",
        "app_id", "archived_at", "deleted_at", "valid_at", "invalid_at", "expired_at",
    }
    assert expected.issubset(cols), f"缺失列: {expected - cols}"


def test_history_table_columns(engine):
    """HistoryTable 有全部 7 列。"""
    cols = {c["name"] for c in inspect(engine).get_columns("history")}
    expected = {"id", "memory_id", "old_memory", "new_memory", "event", "created_at", "is_deleted"}
    assert expected.issubset(cols), f"缺失列: {expected - cols}"


def test_access_log_table_columns(engine):
    """AccessLogTable 有全部 6 列。"""
    cols = {c["name"] for c in inspect(engine).get_columns("memory_access_logs")}
    expected = {"id", "memory_id", "app_id", "access_type", "metadata", "accessed_at"}
    assert expected.issubset(cols), f"缺失列: {expected - cols}"


def test_entity_table_columns(engine):
    """EntityTable 有全部 10 列。"""
    cols = {c["name"] for c in inspect(engine).get_columns("septmuse_entities")}
    expected = {
        "id", "entity_text", "entity_type", "entity_embedding",
        "linked_memory_ids", "user_id", "agent_id", "created_at", "updated_at", "is_deleted",
    }
    assert expected.issubset(cols), f"缺失列: {expected - cols}"


def test_entity_relation_table_columns(engine):
    """EntityRelationTable 有全部 6 列。"""
    cols = {c["name"] for c in inspect(engine).get_columns("entity_relations")}
    expected = {"id", "source_entity", "relation", "target_entity", "user_id", "created_at"}
    assert expected.issubset(cols), f"缺失列: {expected - cols}"


def test_all_tables_registered():
    """5 个表类都注册到 SQLModel.metadata。"""
    table_names = set(SQLModel.metadata.tables.keys())
    assert "memories" in table_names
    assert "history" in table_names
    assert "memory_access_logs" in table_names
    assert "septmuse_entities" in table_names
    assert "entity_relations" in table_names
```

- [ ] **Step 9: 运行测试验证通过**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_models_package.py -v`
Expected: 6 passed

- [ ] **Step 10: ruff 检查**

Run: `ruff check --no-cache src/septmuse/services/database/models/ tests/unit/test_models_package.py`
Expected: All checks passed!

---

### Task 2: ORMMemoryStore 骨架 — __init__ + _create_tables + close

**Files:**
- Create: `src/septmuse/storage/relational_stores/orm_store.py`
- Modify: `src/septmuse/storage/relational_stores/__init__.py`
- Test: `tests/unit/test_orm_memory_store.py`

**Interfaces:**
- Consumes: `MemoryTable`, `HistoryTable`, `AccessLogTable` from Task 1
- Consumes: `MemoryStore` ABC from `storage/base.py`
- Produces: `ORMMemoryStore` 类，`__init__(engine, vector_store=None, keyword_index=None)`，后续 Task 在此类上追加方法

- [ ] **Step 1: 写测试 — 建表验证**

创建 `tests/unit/test_orm_memory_store.py`：

```python
"""ORMMemoryStore 测试 — SQLModel ORM 跨方言 CRUD。"""

import json
from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine, inspect
from sqlmodel import SQLModel

from septmuse.storage.relational_stores.orm_store import ORMMemoryStore


@pytest.fixture
def store():
    """内存 SQLite store，测试自动清理。"""
    engine = create_engine("sqlite://", echo=False)
    s = ORMMemoryStore(engine)
    yield s
    s.close()


def test_orm_store_creates_tables(store):
    """ORMMemoryStore 初始化后自动建 3 张表。"""
    tables = set(inspect(store._engine).get_table_names())
    assert "memories" in tables
    assert "history" in tables
    assert "memory_access_logs" in tables
```

- [ ] **Step 2: 运行测试验证失败**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_orm_memory_store.py::test_orm_store_creates_tables -v`
Expected: FAIL with "ImportError" 或 "ModuleNotFoundError"

- [ ] **Step 3: 写 ORMMemoryStore 骨架**

创建 `src/septmuse/storage/relational_stores/orm_store.py`：

```python
#  Copyright 2026 The sonhhxg0529 Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""ORMMemoryStore — SQLModel ORM 跨方言记忆存储。

一套代码跑 SQLite/MySQL/PostgreSQL。从 DatabaseService 拿 engine，
CRUD 全用 SQLModel select() / session.add()。SQLModel.metadata.create_all()
自动生成对应方言的 DDL。

向量以 JSON list[float] 存储, 检索用 numpy 余弦相似 (跨方言通用)。
组合 vector_store + keyword_index 双写 (方言工厂创建, P3/P4 实现)。
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker
from sqlmodel import Session, SQLModel, select

from septmuse.core.logging import get_logger
from septmuse.services.database.models import AccessLogTable, HistoryTable, MemoryTable
from septmuse.storage.base import MemoryStore

logger = get_logger(__name__)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ORMMemoryStore(MemoryStore):
    """SQLModel ORM 记忆存储 — 跨方言 CRUD。

    用法:
        from sqlalchemy import create_engine
        engine = create_engine("sqlite:///test.db")
        store = ORMMemoryStore(engine)
        mid = store.add("hello", [0.1, 0.2], user_id="alice")
    """

    def __init__(
        self,
        engine: Engine,
        vector_store: Any | None = None,
        keyword_index: Any | None = None,
    ) -> None:
        self._engine = engine
        self._session_maker = sessionmaker(engine, expire_on_commit=False)
        self._vector_store = vector_store
        self._keyword_index = keyword_index
        self._create_tables()
        logger.info("orm_store_ready", dialect=engine.dialect.name)

    def _create_tables(self) -> None:
        """建表 — SQLModel.metadata.create_all 跨方言 DDL。"""
        SQLModel.metadata.create_all(self._engine)

    def close(self) -> None:
        """释放引擎资源。"""
        self._engine.dispose()
```

- [ ] **Step 4: 更新 __init__.py 导出**

修改 `src/septmuse/storage/relational_stores/__init__.py`，在现有导出后追加：

```python
from septmuse.storage.relational_stores.orm_store import ORMMemoryStore
```

并在 `__all__` 列表中加入 `"ORMMemoryStore"`。

- [ ] **Step 5: 运行测试验证通过**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_orm_memory_store.py::test_orm_store_creates_tables -v`
Expected: PASS

- [ ] **Step 6: ruff 检查**

Run: `ruff check --no-cache src/septmuse/storage/relational_stores/orm_store.py tests/unit/test_orm_memory_store.py`
Expected: All checks passed!

---

### Task 3: ORMMemoryStore.add + get

**Files:**
- Modify: `src/septmuse/storage/relational_stores/orm_store.py`
- Test: `tests/unit/test_orm_memory_store.py`

**Interfaces:**
- Produces: `ORMMemoryStore.add(content, embedding, *, user_id, ...) -> str`，`ORMMemoryStore.get(memory_id) -> dict | None`

- [ ] **Step 1: 写测试 — add + get**

追加到 `tests/unit/test_orm_memory_store.py`：

```python
def test_add_returns_memory_id(store):
    """add 返回 'mem-' 前缀的 UUID。"""
    mid = store.add("hello world", [0.1, 0.2, 0.3], user_id="alice")
    assert mid.startswith("mem-")
    assert len(mid) > 10


def test_get_returns_memory(store):
    """get 返回记忆 dict。"""
    mid = store.add("hello world", [0.1, 0.2, 0.3], user_id="alice")
    mem = store.get(mid)
    assert mem is not None
    assert mem["id"] == mid
    assert mem["memory"] == "hello world"
    assert mem["state"] == "active"


def test_get_returns_none_if_not_found(store):
    """get 不存在返回 None。"""
    assert store.get("nonexistent-id") is None


def test_add_with_metadata(store):
    """add 带 metadata 存入。"""
    mid = store.add("test", [0.1], user_id="alice", metadata={"topic": "science"})
    mem = store.get(mid)
    assert mem["metadata"] == {"topic": "science"}


def test_add_with_valid_at(store):
    """add 带 valid_at 双时态。"""
    mid = store.add("earth is round", [0.1], user_id="alice", valid_at="2024-01-01T00:00:00Z")
    mem = store.get(mid)
    assert mem is not None
```

- [ ] **Step 2: 运行测试验证失败**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_orm_memory_store.py::test_add_returns_memory_id -v`
Expected: FAIL with "AttributeError" 或 NotImplementedError

- [ ] **Step 3: 实现 add + get**

在 `ORMMemoryStore` 类中追加（`close` 方法之前）：

```python
    def add(
        self,
        content: str,
        embedding: list[float],
        *,
        user_id: str,
        agent_id: str | None = None,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        valid_at: str | None = None,
    ) -> str:
        """添加记忆, 返回 memory_id。"""
        mid = f"mem-{uuid.uuid4()}"
        now = _utcnow_iso()
        with Session(self._engine) as session:
            mem = MemoryTable(
                id=mid,
                user_id=user_id,
                agent_id=agent_id,
                session_id=session_id,
                content=content,
                embedding=json.dumps(embedding),
                metadata_json=json.dumps(metadata or {}),
                created_at=now,
                updated_at=now,
                valid_at=valid_at,
                is_deleted=0,
                state="active",
            )
            session.add(mem)
            session.add(HistoryTable(
                id=str(uuid.uuid4()),
                memory_id=mid,
                old_memory=None,
                new_memory=content,
                event="ADD",
                created_at=now,
                is_deleted=0,
            ))
            session.commit()
        # 双写: vector_store + keyword_index
        if self._vector_store is not None:
            self._vector_store.insert_vectors([embedding], [mid], [{"user_id": user_id, "session_id": session_id}])
        if self._keyword_index is not None:
            self._keyword_index.add_docs({mid: content})
        logger.info("memory_added", memory_id=mid, user_id=user_id, content_len=len(content))
        return mid

    def get(self, memory_id: str) -> dict[str, Any] | None:
        """取单条, 不存在返回 None。"""
        with Session(self._engine) as session:
            stmt = select(MemoryTable).where(
                MemoryTable.id == memory_id,
                MemoryTable.is_deleted == 0,
            )
            mem = session.exec(stmt).first()
            if mem is None:
                return None
            return {
                "id": mem.id,
                "memory": mem.content,
                "metadata": json.loads(mem.metadata_json) if mem.metadata_json else {},
                "created_at": mem.created_at,
                "state": mem.state or "active",
            }
```

- [ ] **Step 4: 运行测试验证通过**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_orm_memory_store.py -v -k "add or get"`
Expected: 5 passed

- [ ] **Step 5: ruff 检查**

Run: `ruff check --no-cache src/septmuse/storage/relational_stores/orm_store.py`
Expected: All checks passed!

---

### Task 4: ORMMemoryStore.search

**Files:**
- Modify: `src/septmuse/storage/relational_stores/orm_store.py`
- Test: `tests/unit/test_orm_memory_store.py`

**Interfaces:**
- Produces: `ORMMemoryStore.search(query_embedding, *, user_id, ...) -> list[dict]`

- [ ] **Step 1: 写测试 — search**

追加到 `tests/unit/test_orm_memory_store.py`：

```python
def test_search_returns_results(store):
    """search 返回相似度排序结果。"""
    store.add("apple", [1.0, 0.0], user_id="alice")
    store.add("banana", [0.0, 1.0], user_id="alice")
    results = store.search([1.0, 0.0], user_id="alice", top_k=5, threshold=0.0)
    assert len(results) >= 1
    assert results[0]["memory"] == "apple"
    assert results[0]["score"] >= 0.99  # 完全匹配


def test_search_filters_by_user(store):
    """search 按 user_id 隔离。"""
    store.add("alice's memory", [1.0, 0.0], user_id="alice")
    store.add("bob's memory", [1.0, 0.0], user_id="bob")
    results = store.search([1.0, 0.0], user_id="alice", top_k=5, threshold=0.0)
    assert len(results) == 1
    assert results[0]["memory"] == "alice's memory"


def test_search_filters_by_session(store):
    """search 按 session_id 过滤。"""
    store.add("session1", [1.0, 0.0], user_id="alice", session_id="s1")
    store.add("session2", [1.0, 0.0], user_id="alice", session_id="s2")
    results = store.search([1.0, 0.0], user_id="alice", session_id="s1", top_k=5, threshold=0.0)
    assert len(results) == 1
    assert results[0]["memory"] == "session1"


def test_search_threshold_filters(store):
    """search threshold 过滤低相似度。"""
    store.add("orthogonal", [0.0, 1.0], user_id="alice")
    results = store.search([1.0, 0.0], user_id="alice", top_k=5, threshold=0.9)
    assert len(results) == 0


def test_search_excludes_deleted(store):
    """search 排除已删除记忆。"""
    mid = store.add("to delete", [1.0, 0.0], user_id="alice")
    store.delete(mid)
    results = store.search([1.0, 0.0], user_id="alice", top_k=5, threshold=0.0)
    assert len(results) == 0
```

- [ ] **Step 2: 运行测试验证失败**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_orm_memory_store.py -v -k "search"`
Expected: FAIL

- [ ] **Step 3: 实现 search**

在 `ORMMemoryStore` 类中追加（`get` 方法之后）：

```python
    def search(
        self,
        query_embedding: list[float],
        *,
        user_id: str,
        session_id: str | None = None,
        top_k: int = 5,
        threshold: float = 0.1,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """向量检索, 返回 [{"id", "memory", "score", "metadata", "created_at"}]。

        score: 相似度 (越高越相似, 范围 [0, 1])。
        """
        with Session(self._engine) as session:
            stmt = select(MemoryTable).where(
                MemoryTable.user_id == user_id,
                MemoryTable.is_deleted == 0,
            )
            if session_id is not None:
                stmt = stmt.where(MemoryTable.session_id == session_id)
            # filters 字段过滤 (mem0 风格, ORM 原生表达式)
            if filters:
                clean_filters = {k: v for k, v in filters.items() if k not in ("session_id", "run_id")}
                for key, value in clean_filters.items():
                    if hasattr(MemoryTable, key):
                        stmt = stmt.where(getattr(MemoryTable, key) == value)
            rows = session.exec(stmt).all()

        if not rows:
            return []

        q = np.array(query_embedding, dtype=np.float32)
        qnorm = float(np.linalg.norm(q))
        if qnorm > 0:
            q = q / qnorm

        results: list[dict[str, Any]] = []
        for mem in rows:
            if not mem.embedding:
                continue
            emb = np.array(json.loads(mem.embedding), dtype=np.float32)
            score = float(np.dot(q, emb)) if qnorm > 0 else 0.0
            if score >= threshold:
                results.append({
                    "id": mem.id,
                    "memory": mem.content,
                    "score": score,
                    "metadata": json.loads(mem.metadata_json) if mem.metadata_json else {},
                    "created_at": mem.created_at,
                })
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
```

注意：无需额外 import，filters 用 ORM 原生表达式（`hasattr` + `getattr`）。

- [ ] **Step 4: 运行测试验证通过**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_orm_memory_store.py -v -k "search"`
Expected: 5 passed

- [ ] **Step 5: ruff 检查**

Run: `ruff check --no-cache src/septmuse/storage/relational_stores/orm_store.py`
Expected: All checks passed!

---

### Task 5: ORMMemoryStore.get_all + update

**Files:**
- Modify: `src/septmuse/storage/relational_stores/orm_store.py`
- Test: `tests/unit/test_orm_memory_store.py`

- [ ] **Step 1: 写测试 — get_all + update**

追加到 `tests/unit/test_orm_memory_store.py`：

```python
def test_get_all_returns_user_memories(store):
    """get_all 返回用户全部记忆。"""
    store.add("first", [0.1], user_id="alice")
    store.add("second", [0.2], user_id="alice")
    store.add("bob's", [0.3], user_id="bob")
    mems = store.get_all(user_id="alice")
    assert len(mems) == 2
    assert all(m["memory"] in ("first", "second") for m in mems)


def test_get_all_filters_by_session(store):
    """get_all 按 session_id 过滤。"""
    store.add("s1", [0.1], user_id="alice", session_id="s1")
    store.add("s2", [0.1], user_id="alice", session_id="s2")
    mems = store.get_all(user_id="alice", session_id="s1")
    assert len(mems) == 1
    assert mems[0]["memory"] == "s1"


def test_update_changes_content(store):
    """update 修改 content + embedding + metadata。"""
    mid = store.add("old", [0.1, 0.0], user_id="alice")
    ok = store.update(mid, "new content", [0.0, 1.0], metadata={"updated": True})
    assert ok is True
    mem = store.get(mid)
    assert mem["memory"] == "new content"
    assert mem["metadata"] == {"updated": True}


def test_update_returns_false_if_not_found(store):
    """update 不存在返回 False。"""
    assert store.update("nonexistent", "x", [0.1]) is False
```

- [ ] **Step 2: 运行测试验证失败**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_orm_memory_store.py -v -k "get_all or update"`
Expected: FAIL

- [ ] **Step 3: 实现 get_all + update**

在 `ORMMemoryStore` 类中追加（`search` 方法之后）：

```python
    def get_all(
        self, *, user_id: str, session_id: str | None = None, filters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """列出该用户全部未删除记忆。"""
        with Session(self._engine) as session:
            stmt = select(MemoryTable).where(
                MemoryTable.user_id == user_id,
                MemoryTable.is_deleted == 0,
            )
            if session_id is not None:
                stmt = stmt.where(MemoryTable.session_id == session_id)
            rows = session.exec(stmt).all()
        return [
            {
                "id": mem.id,
                "memory": mem.content,
                "metadata": json.loads(mem.metadata_json) if mem.metadata_json else {},
                "created_at": mem.created_at,
                "updated_at": mem.updated_at,
            }
            for mem in rows
        ]

    def update(
        self,
        memory_id: str,
        content: str,
        embedding: list[float],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """更新记忆 content + embedding + metadata, 记录 history。"""
        now = _utcnow_iso()
        with Session(self._engine) as session:
            stmt = select(MemoryTable).where(
                MemoryTable.id == memory_id,
                MemoryTable.is_deleted == 0,
            )
            mem = session.exec(stmt).first()
            if mem is None:
                return False
            old_content = mem.content
            old_meta = json.loads(mem.metadata_json) if mem.metadata_json else {}
            mem.content = content
            mem.embedding = json.dumps(embedding)
            mem.metadata_json = json.dumps(metadata if metadata is not None else old_meta)
            mem.updated_at = now
            session.add(mem)
            session.add(HistoryTable(
                id=str(uuid.uuid4()),
                memory_id=memory_id,
                old_memory=old_content,
                new_memory=content,
                event="UPDATE",
                created_at=now,
                is_deleted=0,
            ))
            session.commit()
        # 双写更新
        if self._vector_store is not None:
            self._vector_store.insert_vectors([embedding], [memory_id])
        if self._keyword_index is not None:
            self._keyword_index.add_docs({memory_id: content})
        logger.info("memory_updated", memory_id=memory_id, content_len=len(content))
        return True
```

- [ ] **Step 4: 运行测试验证通过**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_orm_memory_store.py -v -k "get_all or update"`
Expected: 4 passed

- [ ] **Step 5: ruff 检查**

Run: `ruff check --no-cache src/septmuse/storage/relational_stores/orm_store.py`
Expected: All checks passed!

---

### Task 6: ORMMemoryStore.delete + get_history

**Files:**
- Modify: `src/septmuse/storage/relational_stores/orm_store.py`
- Test: `tests/unit/test_orm_memory_store.py`

- [ ] **Step 1: 写测试 — delete + get_history**

追加到 `tests/unit/test_orm_memory_store.py`：

```python
def test_delete_soft_deletes(store):
    """delete 软删除 (is_deleted=1 + state='deleted')。"""
    mid = store.add("to delete", [0.1], user_id="alice")
    store.delete(mid)
    assert store.get(mid) is None
    # get_all 也排除
    assert store.get_all(user_id="alice") == []


def test_delete_records_history(store):
    """delete 记录 DELETE 事件到 history。"""
    mid = store.add("to delete", [0.1], user_id="alice")
    store.delete(mid)
    history = store.get_history(mid)
    events = [h["event"] for h in history]
    assert "ADD" in events
    assert "DELETE" in events


def test_get_history_returns_chronological(store):
    """get_history 返回时间顺序。"""
    mid = store.add("original", [0.1], user_id="alice")
    store.update(mid, "updated", [0.2])
    history = store.get_history(mid)
    assert len(history) >= 2
    assert history[0]["event"] == "ADD"
    assert history[1]["event"] == "UPDATE"
```

- [ ] **Step 2: 运行测试验证失败**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_orm_memory_store.py -v -k "delete or history"`
Expected: FAIL

- [ ] **Step 3: 实现 delete + get_history**

在 `ORMMemoryStore` 类中追加（`update` 方法之后）：

```python
    def delete(self, memory_id: str) -> None:
        """软删除 (标记 is_deleted + state='deleted' + history 记录)。"""
        now = _utcnow_iso()
        with Session(self._engine) as session:
            stmt = select(MemoryTable).where(MemoryTable.id == memory_id)
            mem = session.exec(stmt).first()
            if mem is None:
                return
            mem.is_deleted = 1
            mem.state = "deleted"
            mem.deleted_at = now
            mem.updated_at = now
            session.add(mem)
            session.add(HistoryTable(
                id=str(uuid.uuid4()),
                memory_id=memory_id,
                old_memory=mem.content,
                new_memory=None,
                event="DELETE",
                created_at=now,
                is_deleted=1,
            ))
            session.commit()
        # 双写清理
        if self._vector_store is not None:
            self._vector_store.delete_vector(memory_id)
        if self._keyword_index is not None:
            self._keyword_index.delete_docs([memory_id])
        logger.info("memory_deleted", memory_id=memory_id)

    def get_history(self, memory_id: str) -> list[dict[str, Any]]:
        """获取记忆变更历史 (ADD/UPDATE/DELETE 记录)。"""
        with Session(self._engine) as session:
            stmt = select(HistoryTable).where(
                HistoryTable.memory_id == memory_id
            ).order_by(HistoryTable.created_at)
            rows = session.exec(stmt).all()
        return [
            {
                "id": h.id,
                "memory_id": h.memory_id,
                "old_memory": h.old_memory,
                "new_memory": h.new_memory,
                "event": h.event,
                "created_at": h.created_at,
                "is_deleted": bool(h.is_deleted),
            }
            for h in rows
        ]
```

- [ ] **Step 4: 运行测试验证通过**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_orm_memory_store.py -v -k "delete or history"`
Expected: 3 passed

- [ ] **Step 5: ruff 检查**

Run: `ruff check --no-cache src/septmuse/storage/relational_stores/orm_store.py`
Expected: All checks passed!

---

### Task 7: ORMMemoryStore._record_access_log + get_access_logs

**Files:**
- Modify: `src/septmuse/storage/relational_stores/orm_store.py`
- Test: `tests/unit/test_orm_memory_store.py`

- [ ] **Step 1: 写测试 — access logs**

追加到 `tests/unit/test_orm_memory_store.py`：

```python
def test_record_access_log(store):
    """_record_access_log 记录访问日志。"""
    mid = store.add("test", [0.1], user_id="alice")
    log_id = store._record_access_log(mid, app_id="app1", access_type="read")
    assert log_id is not None


def test_get_access_logs(store):
    """get_access_logs 返回访问日志。"""
    mid = store.add("test", [0.1], user_id="alice")
    store._record_access_log(mid, app_id="app1", access_type="read")
    store._record_access_log(mid, app_id="app2", access_type="write")
    logs = store.get_access_logs(mid)
    assert len(logs) == 2
    assert all("access_type" in log for log in logs)


def test_get_access_logs_empty(store):
    """get_access_logs 无日志返回空列表。"""
    mid = store.add("test", [0.1], user_id="alice")
    logs = store.get_access_logs(mid)
    assert logs == []
```

- [ ] **Step 2: 运行测试验证失败**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_orm_memory_store.py -v -k "access_log"`
Expected: FAIL

- [ ] **Step 3: 实现 _record_access_log + get_access_logs**

在 `ORMMemoryStore` 类中追加（`get_history` 方法之后）：

```python
    def _record_access_log(
        self,
        memory_id: str,
        app_id: str | None,
        access_type: str,
        metadata: dict[str, Any] | None = None,
    ) -> str | None:
        """记录记忆访问日志（审计用）。"""
        log_id = str(uuid.uuid4())
        now = _utcnow_iso()
        with Session(self._engine) as session:
            session.add(AccessLogTable(
                id=log_id,
                memory_id=memory_id,
                app_id=app_id,
                access_type=access_type,
                metadata_json=json.dumps(metadata) if metadata else None,
                accessed_at=now,
            ))
            session.commit()
        return log_id

    def get_access_logs(self, memory_id: str, limit: int = 100) -> list[dict[str, Any]]:
        """查询记忆访问日志（按 accessed_at 降序）。"""
        with Session(self._engine) as session:
            stmt = select(AccessLogTable).where(
                AccessLogTable.memory_id == memory_id
            ).order_by(AccessLogTable.accessed_at.desc()).limit(limit)
            rows = session.exec(stmt).all()
        return [
            {
                "id": log.id,
                "memory_id": log.memory_id,
                "app_id": log.app_id,
                "access_type": log.access_type,
                "metadata": json.loads(log.metadata_json) if log.metadata_json else None,
                "accessed_at": log.accessed_at,
            }
            for log in rows
        ]
```

- [ ] **Step 4: 运行测试验证通过**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_orm_memory_store.py -v -k "access_log"`
Expected: 3 passed

- [ ] **Step 5: ruff 检查**

Run: `ruff check --no-cache src/septmuse/storage/relational_stores/orm_store.py`
Expected: All checks passed!

---

### Task 8: ORMMemoryStore.invalidate + get_temporal_valid + get_temporal_interval

**Files:**
- Modify: `src/septmuse/storage/relational_stores/orm_store.py`
- Test: `tests/unit/test_orm_memory_store.py`

- [ ] **Step 1: 写测试 — temporal**

追加到 `tests/unit/test_orm_memory_store.py`：

```python
def test_invalidate_sets_invalid_at(store):
    """invalidate 标记事实不再为真。"""
    mid = store.add("earth is flat", [0.1], user_id="alice", valid_at="2020-01-01T00:00:00Z")
    result = store.invalidate(mid, invalid_at="2024-01-01T00:00:00Z")
    assert result["event"] == "INVALIDATE"
    assert result["invalid_at"] == "2024-01-01T00:00:00Z"


def test_invalidate_not_found(store):
    """invalidate 不存在返回 NOT_FOUND。"""
    result = store.invalidate("nonexistent")
    assert result["event"] == "NOT_FOUND"


def test_get_temporal_valid(store):
    """get_temporal_valid 查询某时刻为真的记忆。"""
    store.add("valid fact", [0.1], user_id="alice", valid_at="2020-01-01T00:00:00Z")
    store.add("future fact", [0.1], user_id="alice", valid_at="2025-01-01T00:00:00Z")
    # 2023 年时, valid fact 为真, future fact 还没开始
    results = store.get_temporal_valid("2023-06-01T00:00:00Z", user_id="alice")
    memories = [r["memory"] for r in results]
    assert "valid fact" in memories
    assert "future fact" not in memories


def test_get_temporal_valid_excludes_invalidated(store):
    """get_temporal_valid 排除已 invalidate 的记忆。"""
    mid = store.add("old fact", [0.1], user_id="alice", valid_at="2020-01-01T00:00:00Z")
    store.invalidate(mid, invalid_at="2023-01-01T00:00:00Z")
    # 2024 年时, old fact 已失效
    results = store.get_temporal_valid("2024-06-01T00:00:00Z", user_id="alice")
    memories = [r["memory"] for r in results]
    assert "old fact" not in memories
```

- [ ] **Step 2: 运行测试验证失败**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_orm_memory_store.py -v -k "temporal or invalidate"`
Expected: FAIL

- [ ] **Step 3: 实现 temporal 方法**

在 `ORMMemoryStore` 类中追加（`get_access_logs` 方法之后）。注意需要 `from sqlalchemy import or_`：

```python
    def invalidate(self, memory_id: str, *, invalid_at: str | None = None) -> dict[str, Any]:
        """标记事实不再为真（设置 invalid_at + expired_at, 不删除记忆）。"""
        existing = self.get(memory_id)
        if existing is None:
            return {"id": memory_id, "event": "NOT_FOUND"}
        inv = invalid_at or _utcnow_iso()
        exp = _utcnow_iso()
        now = _utcnow_iso()
        with Session(self._engine) as session:
            stmt = select(MemoryTable).where(MemoryTable.id == memory_id)
            mem = session.exec(stmt).first()
            if mem is None:
                return {"id": memory_id, "event": "NOT_FOUND"}
            mem.invalid_at = inv
            mem.expired_at = exp
            mem.updated_at = now
            session.add(mem)
            session.add(HistoryTable(
                id=str(uuid.uuid4()),
                memory_id=memory_id,
                old_memory=mem.content,
                new_memory=None,
                event="INVALIDATE",
                created_at=now,
                is_deleted=0,
            ))
            session.commit()
        logger.info("memory_invalidated", memory_id=memory_id, invalid_at=inv)
        return {"id": memory_id, "invalid_at": inv, "expired_at": exp, "event": "INVALIDATE"}

    def get_temporal_valid(
        self, reference_time: str, *, user_id: str, session_id: str | None = None
    ) -> list[dict[str, Any]]:
        """查询某时刻为真的记忆 (valid_at <= t AND (invalid_at IS NULL OR invalid_at > t))。"""
        from sqlalchemy import or_
        with Session(self._engine) as session:
            stmt = select(MemoryTable).where(
                MemoryTable.user_id == user_id,
                MemoryTable.is_deleted == 0,
                or_(
                    MemoryTable.valid_at.is_(None),  # 无时间约束, 始终返回
                    MemoryTable.valid_at <= reference_time,
                ),
                or_(
                    MemoryTable.invalid_at.is_(None),  # 未失效
                    MemoryTable.invalid_at > reference_time,
                ),
            )
            if session_id is not None:
                stmt = stmt.where(MemoryTable.session_id == session_id)
            rows = session.exec(stmt).all()
        return [
            {
                "id": mem.id,
                "memory": mem.content,
                "metadata": json.loads(mem.metadata_json) if mem.metadata_json else {},
                "created_at": mem.created_at,
                "valid_at": mem.valid_at,
                "invalid_at": mem.invalid_at,
            }
            for mem in rows
        ]

    def get_temporal_interval(
        self, start: str, end: str, *, user_id: str, session_id: str | None = None
    ) -> list[dict[str, Any]]:
        """查询时间区间 [start, end) 内为真的记忆。"""
        from sqlalchemy import or_
        with Session(self._engine) as session:
            stmt = select(MemoryTable).where(
                MemoryTable.user_id == user_id,
                MemoryTable.is_deleted == 0,
                or_(
                    MemoryTable.valid_at.is_(None),
                    MemoryTable.valid_at <= end,
                ),
                or_(
                    MemoryTable.invalid_at.is_(None),
                    MemoryTable.invalid_at > start,
                ),
            )
            if session_id is not None:
                stmt = stmt.where(MemoryTable.session_id == session_id)
            rows = session.exec(stmt).all()
        return [
            {
                "id": mem.id,
                "memory": mem.content,
                "metadata": json.loads(mem.metadata_json) if mem.metadata_json else {},
                "created_at": mem.created_at,
                "valid_at": mem.valid_at,
                "invalid_at": mem.invalid_at,
            }
            for mem in rows
        ]
```

- [ ] **Step 4: 运行测试验证通过**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_orm_memory_store.py -v -k "temporal or invalidate"`
Expected: 4 passed

- [ ] **Step 5: ruff 检查**

Run: `ruff check --no-cache src/septmuse/storage/relational_stores/orm_store.py`
Expected: All checks passed!

---

### Task 9: ORMMemoryStore.keyword_search + list_agents + list_users + get_shared_memories

**Files:**
- Modify: `src/septmuse/storage/relational_stores/orm_store.py`
- Test: `tests/unit/test_orm_memory_store.py`

- [ ] **Step 1: 写测试 — keyword_search + 关系查询**

追加到 `tests/unit/test_orm_memory_store.py`：

```python
def test_keyword_search_without_index_returns_empty(store):
    """无 keyword_index 时 keyword_search 返回空。"""
    store.add("hello world", [0.1], user_id="alice")
    results = store.keyword_search("hello", user_id="alice")
    assert results == []


def test_list_agents(store):
    """list_agents 返回用户的 agent_id 去重列表。"""
    store.add("m1", [0.1], user_id="alice", agent_id="agent1")
    store.add("m2", [0.1], user_id="alice", agent_id="agent2")
    store.add("m3", [0.1], user_id="alice", agent_id="agent1")  # 重复
    store.add("m4", [0.1], user_id="alice")  # None, 排除
    agents = store.list_agents("alice")
    assert set(agents) == {"agent1", "agent2"}


def test_list_users(store):
    """list_users 返回 agent 的 user_id 去重列表。"""
    store.add("m1", [0.1], user_id="alice", agent_id="agent1")
    store.add("m2", [0.1], user_id="bob", agent_id="agent1")
    users = store.list_users("agent1")
    assert set(users) == {"alice", "bob"}


def test_get_shared_memories(store):
    """get_shared_memories 返回跨 agent 共享记忆。"""
    store.add("shared1", [0.1], user_id="alice", agent_id="agent1")
    store.add("shared2", [0.1], user_id="alice", agent_id="agent2")
    results = store.get_shared_memories("alice", limit=100)
    assert len(results) == 2
```

- [ ] **Step 2: 运行测试验证失败**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_orm_memory_store.py -v -k "keyword_search or list_agents or list_users or shared"`
Expected: FAIL

- [ ] **Step 3: 实现 keyword_search + 关系查询**

在 `ORMMemoryStore` 类中追加（`get_temporal_interval` 方法之后）：

```python
    def keyword_search(
        self, query: str, *, user_id: str, session_id: str | None = None, top_k: int = 5
    ) -> list[dict[str, Any]]:
        """关键词检索 (BM25)。无 keyword_index 时返回空。"""
        if self._keyword_index is None:
            return []
        scores = self._keyword_index.retrieve(query, top_k * 2)
        if not scores:
            return []
        results: list[dict[str, Any]] = []
        with Session(self._engine) as session:
            for doc_id, score in scores.items():
                stmt = select(MemoryTable).where(
                    MemoryTable.id == doc_id,
                    MemoryTable.user_id == user_id,
                    MemoryTable.is_deleted == 0,
                )
                if session_id is not None:
                    stmt = stmt.where(MemoryTable.session_id == session_id)
                mem = session.exec(stmt).first()
                if mem is not None:
                    results.append({
                        "id": mem.id,
                        "memory": mem.content,
                        "score": float(score),
                        "metadata": json.loads(mem.metadata_json) if mem.metadata_json else {},
                        "created_at": mem.created_at,
                    })
        return results[:top_k]

    def list_agents(self, user_id: str) -> list[str]:
        """列出该用户的所有 agent_id (去重, 排除 NULL)。"""
        with Session(self._engine) as session:
            stmt = select(MemoryTable.agent_id).where(
                MemoryTable.user_id == user_id,
                MemoryTable.is_deleted == 0,
                MemoryTable.agent_id.isnot(None),
            ).distinct()
            return list(session.exec(stmt).all())

    def list_users(self, agent_id: str) -> list[str]:
        """列出该 agent 的所有 user_id (去重)。"""
        with Session(self._engine) as session:
            stmt = select(MemoryTable.user_id).where(
                MemoryTable.agent_id == agent_id,
                MemoryTable.is_deleted == 0,
            ).distinct()
            return list(session.exec(stmt).all())

    def get_shared_memories(self, user_id: str, limit: int = 100) -> list[dict[str, Any]]:
        """获取跨 agent 共享的记忆 (不限 agent_id, 按 created_at 降序)。"""
        from sqlalchemy import desc
        with Session(self._engine) as session:
            stmt = select(MemoryTable).where(
                MemoryTable.user_id == user_id,
                MemoryTable.is_deleted == 0,
            ).order_by(desc(MemoryTable.created_at)).limit(limit)
            rows = session.exec(stmt).all()
        return [
            {
                "id": mem.id,
                "memory": mem.content,
                "metadata": json.loads(mem.metadata_json) if mem.metadata_json else {},
                "created_at": mem.created_at,
                "agent_id": mem.agent_id,
            }
            for mem in rows
        ]
```

- [ ] **Step 4: 运行测试验证通过**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_orm_memory_store.py -v -k "keyword_search or list_agents or list_users or shared"`
Expected: 4 passed

- [ ] **Step 5: ruff 检查**

Run: `ruff check --no-cache src/septmuse/storage/relational_stores/orm_store.py`
Expected: All checks passed!

---

### Task 10: 全量回归 + ruff

**Files:**
- 无新增/修改

- [ ] **Step 1: ruff 全量检查**

Run: `ruff check --no-cache src/ tests/unit/test_orm_memory_store.py tests/unit/test_models_package.py`
Expected: All checks passed!

- [ ] **Step 2: ORMMemoryStore 全量测试**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_orm_memory_store.py tests/unit/test_models_package.py -v`
Expected: 全部 passed（~30 测试）

- [ ] **Step 3: 现有测试全量回归**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/ tests/e2e/ -q --tb=line`
Expected: 1116+ passed + 36 skipped + 14 failed（基线不变，新增 ~30 测试 = ~1146 passed）

- [ ] **Step 4: 验证零配置可用**

Run: `$env:PYTHONPATH="src"; python -c "from septmuse.storage.relational_stores.orm_store import ORMMemoryStore; from sqlalchemy import create_engine; s = ORMMemoryStore(create_engine('sqlite://')); print('OK'); s.close()"`
Expected: 打印 `OK`

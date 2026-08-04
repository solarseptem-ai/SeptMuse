# ORMMemoryStore P2 实施计划 — AsyncORMMemoryStore + DatabaseService async engine

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建 AsyncORMMemoryStore（async 完整 CRUD）+ DatabaseService async engine 支持，用 SQLModel async + AsyncEngine，不破坏现有测试基线。

**Architecture:** AsyncORMMemoryStore 是 ORMMemoryStore 的 async 对偶——`AsyncSession` 替代 `Session`，`await session.exec()` 替代 `session.exec()`，双写用 `asyncio.to_thread`。DatabaseService 补 `get_async_engine()` 懒加载。

**Tech Stack:** SQLModel.ext.asyncio (AsyncSession, async_sessionmaker), SQLAlchemy 2.0 AsyncEngine, aiosqlite

## Global Constraints

- PYTHONPATH=src 运行所有 pytest
- ruff line-length 120
- **禁止 `ruff format`** — 只用 `ruff check --no-cache`
- 代码注释用中文
- 不用 git commit
- `pytest_asyncio_mode = "auto"` — async 测试无需 `@pytest.mark.asyncio`
- AsyncMemoryStore ABC 在 `storage/async_base.py`，8 个 abstractmethod: add/search/get_all/get/delete/update/get_history/close
- AsyncSession 用 `from sqlmodel.ext.asyncio import AsyncSession, async_sessionmaker`
- AsyncEngine 用 `from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine`
- 测试用 `sqlite+aiosqlite://` 内存库
- score 统一为相似度 [0,1]

## 文件结构

### 新建文件
| 文件 | 职责 |
|------|------|
| `src/septmuse/storage/relational_stores/async_orm_store.py` | AsyncORMMemoryStore |
| `tests/unit/test_async_orm_memory_store.py` | AsyncORMMemoryStore 测试 |

### 修改文件
| 文件 | 改动 |
|------|------|
| `src/septmuse/services/database/service.py` | 补 `get_async_engine()` + `_resolve_async_db_url()` + 懒加载 |
| `src/septmuse/storage/relational_stores/__init__.py` | 导出 AsyncORMMemoryStore |

---

### Task 1: DatabaseService async engine 支持

**Files:**
- Modify: `src/septmuse/services/database/service.py`
- Test: `tests/unit/test_database_service.py`（追加测试）

**Interfaces:**
- Produces: `DatabaseService.get_async_engine() -> AsyncEngine`，`DatabaseService._resolve_async_db_url() -> str`

- [ ] **Step 1: 读当前 service.py 确认结构**

Run: `Read src/septmuse/services/database/service.py`
确认现有 `__init__`、`_resolve_db_url`、`get_engine` 的位置。

- [ ] **Step 2: 追加 async 测试**

追加到 `tests/unit/test_database_service.py`：

```python
@pytest.mark.asyncio
async def test_database_service_async_engine():
    """get_async_engine 返回 AsyncEngine，懒加载。"""
    import os
    os.environ["SEPTMUSE_DB_URL"] = "sqlite://"
    try:
        svc = DatabaseService()
        ae = svc.get_async_engine()
        from sqlalchemy.ext.asyncio import AsyncEngine
        assert isinstance(ae, AsyncEngine)
        # 懒加载: 第二次返回同一实例
        assert svc.get_async_engine() is ae
    finally:
        del os.environ["SEPTMUSE_DB_URL"]


def test_resolve_async_db_url_adds_driver():
    """_resolve_async_db_url 自动加 async driver。"""
    import os
    os.environ["SEPTMUSE_DB_URL"] = "sqlite:///test.db"
    try:
        svc = DatabaseService()
        async_url = svc._resolve_async_db_url()
        assert "aiosqlite" in async_url
    finally:
        del os.environ["SEPTMUSE_DB_URL"]
```

- [ ] **Step 3: 运行测试验证失败**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_database_service.py::test_database_service_async_engine -v`
Expected: FAIL with AttributeError

- [ ] **Step 4: 实现 get_async_engine + _resolve_async_db_url**

在 `service.py` 的 `DatabaseService` 类中追加（`get_session_maker` 方法之后）。在文件顶部加 `from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine`：

```python
    def _resolve_async_db_url(self) -> str:
        """解析 async db_url — 自动加 async driver。"""
        url = self.database_url
        # 已有 async driver, 不重复加
        if "+aiosqlite" in url or "+aiomysql" in url or "+asyncpg" in url or "+psycopg" in url:
            return url
        # 加 async driver
        if url.startswith("sqlite://"):
            return url.replace("sqlite://", "sqlite+aiosqlite://", 1)
        if url.startswith("mysql://"):
            return url.replace("mysql://", "mysql+aiomysql://", 1)
        if url.startswith("mysql+pymysql://"):
            return url.replace("mysql+pymysql://", "mysql+aiomysql://", 1)
        if url.startswith("postgresql://"):
            return url.replace("postgresql://", "postgresql+asyncpg://", 1)
        if url.startswith("postgresql+psycopg://"):
            return url.replace("postgresql+psycopg://", "postgresql+asyncpg://", 1)
        return url

    def get_async_engine(self) -> AsyncEngine:
        """返回 async engine（懒加载, 首次调用时创建）。"""
        if self._async_engine is None:
            async_url = self._resolve_async_db_url()
            self._async_engine = create_async_engine(async_url, echo=False)
            logger.info("async_engine_created", url=self._safe_url())
        return self._async_engine
```

在 `__init__` 方法中追加 `self._async_engine: AsyncEngine | None = None`（在 `self.session_maker = ...` 之后）。

- [ ] **Step 5: 运行测试验证通过**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_database_service.py -v`
Expected: 9 passed（7 existing + 2 new）

- [ ] **Step 6: ruff 检查**

Run: `ruff check --no-cache src/septmuse/services/database/service.py tests/unit/test_database_service.py`
Expected: All checks passed!

---

### Task 2: AsyncORMMemoryStore 骨架 + add/get

**Files:**
- Create: `src/septmuse/storage/relational_stores/async_orm_store.py`
- Modify: `src/septmuse/storage/relational_stores/__init__.py`
- Test: `tests/unit/test_async_orm_memory_store.py`

**Interfaces:**
- Consumes: `MemoryTable`, `HistoryTable`, `AccessLogTable` from P1
- Consumes: `AsyncMemoryStore` ABC from `storage/async_base.py`
- Produces: `AsyncORMMemoryStore` 类

- [ ] **Step 1: 写测试 — 骨架 + add + get**

创建 `tests/unit/test_async_orm_memory_store.py`：

```python
"""AsyncORMMemoryStore 测试 — SQLModel async ORM CRUD。"""

import json

import pytest
from sqlalchemy import inspect
from sqlalchemy.ext.asyncio import create_async_engine

from septmuse.storage.relational_stores.async_orm_store import AsyncORMMemoryStore


@pytest.fixture
async def store():
    """内存 SQLite async store，测试自动清理。"""
    engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    s = AsyncORMMemoryStore(engine)
    yield s
    await s.close()


async def test_async_store_creates_tables(store):
    """AsyncORMMemoryStore 初始化后自动建表。"""
    async with store._engine.connect() as conn:
        tables = await conn.run_sync(lambda sync_conn: set(inspect(sync_conn).get_table_names()))
    assert "memories" in tables
    assert "history" in tables
    assert "memory_access_logs" in tables


async def test_add_returns_memory_id(store):
    """add 返回 'mem-' 前缀的 UUID。"""
    mid = await store.add("hello world", [0.1, 0.2, 0.3], user_id="alice")
    assert mid.startswith("mem-")


async def test_get_returns_memory(store):
    """get 返回记忆 dict。"""
    mid = await store.add("hello world", [0.1, 0.2, 0.3], user_id="alice")
    mem = await store.get(mid)
    assert mem is not None
    assert mem["id"] == mid
    assert mem["memory"] == "hello world"
    assert mem["state"] == "active"


async def test_get_returns_none_if_not_found(store):
    """get 不存在返回 None。"""
    assert await store.get("nonexistent-id") is None


async def test_add_with_metadata(store):
    """add 带 metadata 存入。"""
    mid = await store.add("test", [0.1], user_id="alice", metadata={"topic": "science"})
    mem = await store.get(mid)
    assert mem["metadata"] == {"topic": "science"}
```

- [ ] **Step 2: 运行测试验证失败**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_async_orm_memory_store.py -v`
Expected: FAIL with ImportError

- [ ] **Step 3: 写 AsyncORMMemoryStore 骨架 + add + get**

创建 `src/septmuse/storage/relational_stores/async_orm_store.py`：

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
"""AsyncORMMemoryStore — SQLModel async ORM 跨方言记忆存储。

ORMMemoryStore 的 async 对偶。用 AsyncSession + async_sessionmaker。
双写 vector_store/keyword_index 用 asyncio.to_thread 包装 sync 调用。
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone
from typing import Any

import numpy as np
from sqlalchemy import desc, or_
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlmodel import select
from sqlmodel.ext.asyncio import AsyncSession, async_sessionmaker

from septmuse.core.logging import get_logger
from septmuse.services.database.models import AccessLogTable, HistoryTable, MemoryTable
from septmuse.storage.async_base import AsyncMemoryStore

logger = get_logger(__name__)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class AsyncORMMemoryStore(AsyncMemoryStore):
    """SQLModel async ORM 记忆存储 — 跨方言 CRUD。

    用法:
        from sqlalchemy.ext.asyncio import create_async_engine
        engine = create_async_engine("sqlite+aiosqlite:///test.db")
        store = AsyncORMMemoryStore(engine)
        mid = await store.add("hello", [0.1, 0.2], user_id="alice")
    """

    def __init__(
        self,
        engine: AsyncEngine,
        vector_store: Any | None = None,
        keyword_index: Any | None = None,
    ) -> None:
        self._engine = engine
        self._session_maker = async_sessionmaker(engine, expire_on_commit=False)
        self._vector_store = vector_store
        self._keyword_index = keyword_index
        # 建表 (sync DDL, 用 run_sync)
        import asyncio as _asyncio
        _asyncio.run(self._create_tables())
        logger.info("async_orm_store_ready", dialect=engine.dialect.name)

    async def _create_tables(self) -> None:
        """建表 — SQLModel.metadata.create_all 跨方言 DDL。"""
        from sqlmodel import SQLModel
        async with self._engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)

    async def close(self) -> None:
        """释放引擎资源。"""
        await self._engine.dispose()

    async def add(
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
        async with AsyncSession(self._engine) as session:
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
            await session.commit()
        # 双写: vector_store + keyword_index (sync, 用 to_thread)
        if self._vector_store is not None:
            await asyncio.to_thread(
                self._vector_store.insert_vectors,
                [embedding], [mid], [{"user_id": user_id, "session_id": session_id}]
            )
        if self._keyword_index is not None:
            await asyncio.to_thread(self._keyword_index.add_docs, {mid: content})
        logger.info("async_memory_added", memory_id=mid, user_id=user_id, content_len=len(content))
        return mid

    async def get(self, memory_id: str) -> dict[str, Any] | None:
        """取单条, 不存在返回 None。"""
        async with AsyncSession(self._engine) as session:
            stmt = select(MemoryTable).where(
                MemoryTable.id == memory_id,
                MemoryTable.is_deleted == 0,
            )
            result = await session.exec(stmt)
            mem = result.first()
            if mem is None:
                return None
            return {
                "id": mem.id,
                "memory": mem.content,
                "metadata": json.loads(mem.metadata_json) if mem.metadata_json else {},
                "created_at": mem.created_at,
                "state": mem.state or "active",
            }

    # 以下方法在后续 Task 实现, 先用 stub
    async def search(self, query_embedding, *, user_id, session_id=None, top_k=5, threshold=0.1, filters=None):
        raise NotImplementedError

    async def get_all(self, *, user_id, session_id=None, filters=None):
        raise NotImplementedError

    async def delete(self, memory_id):
        raise NotImplementedError

    async def update(self, memory_id, content, embedding, *, metadata=None):
        raise NotImplementedError

    async def get_history(self, memory_id):
        raise NotImplementedError
```

**注意**：`__init__` 中 `asyncio.run(self._create_tables())` 在测试环境可能有问题（如果 event loop 已运行）。改用同步建表方式——在 `__init__` 中直接用 sync engine 建表：

```python
    def __init__(
        self,
        engine: AsyncEngine,
        vector_store: Any | None = None,
        keyword_index: Any | None = None,
    ) -> None:
        self._engine = engine
        self._session_maker = async_sessionmaker(engine, expire_on_commit=False)
        self._vector_store = vector_store
        self._keyword_index = keyword_index
        # 建表: 用 sync engine 从 async url 创建临时 sync engine 建表
        from sqlmodel import SQLModel, create_engine
        sync_url = str(engine.url).replace("+aiosqlite", "").replace("+aiomysql", "+pymysql").replace("+asyncpg", "+psycopg2")
        sync_engine = create_engine(sync_url)
        SQLModel.metadata.create_all(sync_engine)
        sync_engine.dispose()
        logger.info("async_orm_store_ready", dialect=engine.dialect.name)
```

- [ ] **Step 4: 更新 __init__.py 导出**

修改 `src/septmuse/storage/relational_stores/__init__.py`，追加：

```python
from septmuse.storage.relational_stores.async_orm_store import AsyncORMMemoryStore
```

并加入 `__all__`。

- [ ] **Step 5: 运行测试验证通过**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_async_orm_memory_store.py -v`
Expected: 5 passed

- [ ] **Step 6: ruff 检查**

Run: `ruff check --no-cache src/septmuse/storage/relational_stores/async_orm_store.py tests/unit/test_async_orm_memory_store.py`
Expected: All checks passed!

---

### Task 3: AsyncORMMemoryStore.search + get_all + update

**Files:**
- Modify: `src/septmuse/storage/relational_stores/async_orm_store.py`
- Test: `tests/unit/test_async_orm_memory_store.py`

- [ ] **Step 1: 写测试**

追加到 `tests/unit/test_async_orm_memory_store.py`：

```python
async def test_search_returns_results(store):
    """search 返回相似度排序结果。"""
    await store.add("apple", [1.0, 0.0], user_id="alice")
    await store.add("banana", [0.0, 1.0], user_id="alice")
    results = await store.search([1.0, 0.0], user_id="alice", top_k=5, threshold=0.0)
    assert len(results) >= 1
    assert results[0]["memory"] == "apple"


async def test_search_filters_by_user(store):
    """search 按 user_id 隔离。"""
    await store.add("alice's memory", [1.0, 0.0], user_id="alice")
    await store.add("bob's memory", [1.0, 0.0], user_id="bob")
    results = await store.search([1.0, 0.0], user_id="alice", top_k=5, threshold=0.0)
    assert len(results) == 1
    assert results[0]["memory"] == "alice's memory"


async def test_search_filters_by_session(store):
    """search 按 session_id 过滤。"""
    await store.add("session1", [1.0, 0.0], user_id="alice", session_id="s1")
    await store.add("session2", [1.0, 0.0], user_id="alice", session_id="s2")
    results = await store.search([1.0, 0.0], user_id="alice", session_id="s1", top_k=5, threshold=0.0)
    assert len(results) == 1
    assert results[0]["memory"] == "session1"


async def test_get_all_returns_user_memories(store):
    """get_all 返回用户全部记忆。"""
    await store.add("first", [0.1], user_id="alice")
    await store.add("second", [0.2], user_id="alice")
    await store.add("bob's", [0.3], user_id="bob")
    mems = await store.get_all(user_id="alice")
    assert len(mems) == 2


async def test_update_changes_content(store):
    """update 修改 content + embedding + metadata。"""
    mid = await store.add("old", [0.1, 0.0], user_id="alice")
    ok = await store.update(mid, "new content", [0.0, 1.0], metadata={"updated": True})
    assert ok is True
    mem = await store.get(mid)
    assert mem["memory"] == "new content"
    assert mem["metadata"] == {"updated": True}


async def test_update_returns_false_if_not_found(store):
    """update 不存在返回 False。"""
    assert await store.update("nonexistent", "x", [0.1]) is False
```

- [ ] **Step 2: 运行测试验证失败**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_async_orm_memory_store.py -v -k "search or get_all or update"`
Expected: FAIL

- [ ] **Step 3: 实现 search + get_all + update**

替换 `async_orm_store.py` 中的 3 个 stub：

```python
    async def search(
        self,
        query_embedding: list[float],
        *,
        user_id: str,
        session_id: str | None = None,
        top_k: int = 5,
        threshold: float = 0.1,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """向量检索, 返回 [{"id", "memory", "score", "metadata", "created_at"}]。"""
        async with AsyncSession(self._engine) as session:
            stmt = select(MemoryTable).where(
                MemoryTable.user_id == user_id,
                MemoryTable.is_deleted == 0,
            )
            if session_id is not None:
                stmt = stmt.where(MemoryTable.session_id == session_id)
            if filters:
                clean_filters = {k: v for k, v in filters.items() if k not in ("session_id", "run_id")}
                for key, value in clean_filters.items():
                    if hasattr(MemoryTable, key):
                        stmt = stmt.where(getattr(MemoryTable, key) == value)
            result = await session.exec(stmt)
            rows = result.all()

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

    async def get_all(
        self, *, user_id: str, session_id: str | None = None, filters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """列出该用户全部未删除记忆。"""
        async with AsyncSession(self._engine) as session:
            stmt = select(MemoryTable).where(
                MemoryTable.user_id == user_id,
                MemoryTable.is_deleted == 0,
            )
            if session_id is not None:
                stmt = stmt.where(MemoryTable.session_id == session_id)
            result = await session.exec(stmt)
            rows = result.all()
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

    async def update(
        self,
        memory_id: str,
        content: str,
        embedding: list[float],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """更新记忆 content + embedding + metadata, 记录 history。"""
        now = _utcnow_iso()
        async with AsyncSession(self._engine) as session:
            stmt = select(MemoryTable).where(
                MemoryTable.id == memory_id,
                MemoryTable.is_deleted == 0,
            )
            result = await session.exec(stmt)
            mem = result.first()
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
            await session.commit()
        if self._vector_store is not None:
            await asyncio.to_thread(self._vector_store.insert_vectors, [embedding], [memory_id])
        if self._keyword_index is not None:
            await asyncio.to_thread(self._keyword_index.add_docs, {memory_id: content})
        logger.info("async_memory_updated", memory_id=memory_id, content_len=len(content))
        return True
```

- [ ] **Step 4: 运行测试验证通过**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_async_orm_memory_store.py -v`
Expected: 11 passed

- [ ] **Step 5: ruff 检查**

Run: `ruff check --no-cache src/septmuse/storage/relational_stores/async_orm_store.py`
Expected: All checks passed!

---

### Task 4: AsyncORMMemoryStore.delete + get_history + access_logs + temporal + keyword_search

**Files:**
- Modify: `src/septmuse/storage/relational_stores/async_orm_store.py`
- Test: `tests/unit/test_async_orm_memory_store.py`

- [ ] **Step 1: 写测试**

追加到 `tests/unit/test_async_orm_memory_store.py`：

```python
async def test_delete_soft_deletes(store):
    """delete 软删除。"""
    mid = await store.add("to delete", [0.1], user_id="alice")
    await store.delete(mid)
    assert await store.get(mid) is None
    assert await store.get_all(user_id="alice") == []


async def test_delete_records_history(store):
    """delete 记录 DELETE 事件。"""
    mid = await store.add("to delete", [0.1], user_id="alice")
    await store.delete(mid)
    history = await store.get_history(mid)
    events = [h["event"] for h in history]
    assert "ADD" in events
    assert "DELETE" in events


async def test_get_history_chronological(store):
    """get_history 返回时间顺序。"""
    mid = await store.add("original", [0.1], user_id="alice")
    await store.update(mid, "updated", [0.2])
    history = await store.get_history(mid)
    assert len(history) >= 2
    assert history[0]["event"] == "ADD"
    assert history[1]["event"] == "UPDATE"


async def test_record_access_log(store):
    """_record_access_log 记录访问日志。"""
    mid = await store.add("test", [0.1], user_id="alice")
    log_id = await store._record_access_log(mid, app_id="app1", access_type="read")
    assert log_id is not None


async def test_get_access_logs(store):
    """get_access_logs 返回访问日志。"""
    mid = await store.add("test", [0.1], user_id="alice")
    await store._record_access_log(mid, app_id="app1", access_type="read")
    await store._record_access_log(mid, app_id="app2", access_type="write")
    logs = await store.get_access_logs(mid)
    assert len(logs) == 2


async def test_invalidate_sets_invalid_at(store):
    """invalidate 标记事实不再为真。"""
    mid = await store.add("earth is flat", [0.1], user_id="alice", valid_at="2020-01-01T00:00:00Z")
    result = await store.invalidate(mid, invalid_at="2024-01-01T00:00:00Z")
    assert result["event"] == "INVALIDATE"


async def test_invalidate_not_found(store):
    """invalidate 不存在返回 NOT_FOUND。"""
    result = await store.invalidate("nonexistent")
    assert result["event"] == "NOT_FOUND"


async def test_get_temporal_valid(store):
    """get_temporal_valid 查询某时刻为真的记忆。"""
    await store.add("valid fact", [0.1], user_id="alice", valid_at="2020-01-01T00:00:00Z")
    await store.add("future fact", [0.1], user_id="alice", valid_at="2025-01-01T00:00:00Z")
    results = await store.get_temporal_valid("2023-06-01T00:00:00Z", user_id="alice")
    memories = [r["memory"] for r in results]
    assert "valid fact" in memories
    assert "future fact" not in memories


async def test_keyword_search_without_index_returns_empty(store):
    """无 keyword_index 时 keyword_search 返回空。"""
    await store.add("hello world", [0.1], user_id="alice")
    results = await store.keyword_search("hello", user_id="alice")
    assert results == []
```

- [ ] **Step 2: 运行测试验证失败**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_async_orm_memory_store.py -v -k "delete or history or access or temporal or invalidate or keyword"`
Expected: FAIL

- [ ] **Step 3: 实现 delete + get_history + _record_access_log + get_access_logs + invalidate + get_temporal_valid + get_temporal_interval + keyword_search**

替换所有剩余 stub。代码是 ORMMemoryStore 的 async 对偶——把 `with Session(self._engine) as session` 换成 `async with AsyncSession(self._engine) as session`，`session.exec` 换成 `await session.exec`，双写用 `await asyncio.to_thread(...)`。参考 `orm_store.py` 的 sync 实现逐一翻译。

- [ ] **Step 4: 运行测试验证通过**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_async_orm_memory_store.py -v`
Expected: 20 passed

- [ ] **Step 5: ruff 检查**

Run: `ruff check --no-cache src/septmuse/storage/relational_stores/async_orm_store.py tests/unit/test_async_orm_memory_store.py`
Expected: All checks passed!

---

### Task 5: 全量回归

- [ ] **Step 1: ruff 全量检查**

Run: `ruff check --no-cache src/septmuse/services/database/ src/septmuse/storage/relational_stores/async_orm_store.py tests/unit/test_async_orm_memory_store.py tests/unit/test_database_service.py`
Expected: All checks passed!

- [ ] **Step 2: 全量测试回归**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/ tests/e2e/ -q --tb=line`
Expected: 1159+ passed + 36 skipped + 13 failed（基线不变，新增 ~22 测试）

- [ ] **Step 3: 验证 async 零配置**

Run: `$env:PYTHONPATH="src"; python -c "import asyncio; from sqlalchemy.ext.asyncio import create_async_engine; from septmuse.storage.relational_stores.async_orm_store import AsyncORMMemoryStore; async def main(): e = create_async_engine('sqlite+aiosqlite://'); s = AsyncORMMemoryStore(e); mid = await s.add('test', [0.1], user_id='u'); print('OK', mid); await s.close(); asyncio.run(main())"`
Expected: 打印 `OK mem-...`

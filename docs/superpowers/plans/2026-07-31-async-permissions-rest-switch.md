# async 权限检查 + REST API 核心端点切换 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 创建 async 权限检查 + 访问日志函数，解锁 REST API 9 个核心端点切换到 AsyncMemory

**Architecture:** 新建 async 权限/日志函数（与 sync 版并存）；AsyncSQLiteMemoryStore 补齐访问日志表 + 方法；AsyncMemory 加 invalidate + search session_id；REST API create_app 持双 memory（AsyncMemory + ExperimentalMemory 共享 DB），9 核心端点切 await + 格式适配，12 实验端点保持 sync

**Tech Stack:** aiosqlite、FastAPI、pytest-asyncio（auto 模式）

## 全局约束

- **PYTHONPATH=src** 运行所有测试（PowerShell: `$env:PYTHONPATH="src"`）
- **ruff line-length=120**，只用 `ruff check --fix`（**禁用 ruff format**）
- **不是 git 仓库**，无 commit 步骤
- **代码注释用中文**，不暴露任何开源库参考来源
- **现有测试固定不动**，仅新增测试
- **pytest 基线**：1050 passed + 36 skipped + 23 failed（不退化）
- **pytest_asyncio_mode = "auto"**：async 测试无需 @pytest.mark.asyncio
- **ruff 缓存问题**：用 `ruff check --no-cache` 避免写入失败
- 工作目录：E:\sonhhxg0529\vibe_coding_project\solarseptem-ai\solarseptem-ai-platform\SeptMuse

## 文件结构

**新建：**
- `src/septmuse/governance/async_permissions.py` — async 权限检查函数
- `src/septmuse/governance/async_access_log.py` — async 访问日志函数
- `tests/unit/test_async_permissions.py` — async 权限检查测试（5 测试）
- `tests/unit/test_async_access_log.py` — async 访问日志测试（3 测试）

**修改：**
- `src/septmuse/storage/async_sqlite/store.py` — 加 memory_access_logs 表 + `_record_access_log` + `get_access_logs`
- `src/septmuse/memory/async_main.py` — 加 `invalidate` 方法 + `search` 加 `session_id` 参数
- `src/septmuse/api/rest/__init__.py` — `create_app` 持双 memory + 9 核心端点切 await + 格式适配

---

## Task 1: async 权限检查函数

**Files:**
- Create: `src/septmuse/governance/async_permissions.py`
- Test: `tests/unit/test_async_permissions.py`

**Interfaces:**
- Consumes: `AsyncMemoryStore`（`storage/async_base.py`），`MemoryState`（`governance/permissions.py`）
- Produces: `async_check_memory_access_permissions(store, memory_id, app_id) -> tuple[bool, str]`

- [ ] **Step 1: 写失败测试**

```python
# tests/unit/test_async_permissions.py
"""async 权限检查函数测试。"""
import pytest

from septmuse.governance.async_permissions import async_check_memory_access_permissions
from septmuse.storage.async_base import AsyncMemoryStore


class _FakeStore(AsyncMemoryStore):
    """测试用假 store。"""

    def __init__(self, mem=None):
        self._mem = mem

    async def add(self, content, embedding, *, user_id, **kwargs):
        pass

    async def search(self, query_embedding, *, user_id, **kwargs):
        return []

    async def get_all(self, *, user_id, **kwargs):
        return []

    async def get(self, memory_id):
        return self._mem

    async def delete(self, memory_id):
        pass

    async def update(self, memory_id, content, embedding, *, metadata=None):
        return True

    async def get_history(self, memory_id):
        return []

    async def close(self):
        pass


async def test_active_memory_allowed():
    """active 记忆 + 无 app_id → 放行。"""
    store = _FakeStore(mem={"id": "m1", "memory": "hello"})
    allowed, reason = await async_check_memory_access_permissions(store, "m1")
    assert allowed is True
    assert "self access" in reason


async def test_not_found_rejected():
    """不存在的记忆 → 拒绝。"""
    store = _FakeStore(mem=None)
    allowed, reason = await async_check_memory_access_permissions(store, "nonexistent")
    assert allowed is False
    assert "not found" in reason


async def test_with_app_id_allowed():
    """active 记忆 + 有 app_id → 放行。"""
    store = _FakeStore(mem={"id": "m1", "memory": "hello"})
    allowed, reason = await async_check_memory_access_permissions(store, "m1", app_id="my-app")
    assert allowed is True
    assert "granted" in reason


async def test_empty_app_id_rejected():
    """active 记忆 + 空 app_id → 拒绝。"""
    store = _FakeStore(mem={"id": "m1", "memory": "hello"})
    allowed, reason = await async_check_memory_access_permissions(store, "m1", app_id="")
    assert allowed is False
    assert "empty" in reason


async def test_none_app_id_allowed():
    """active 记忆 + None app_id → 放行（用户自己访问）。"""
    store = _FakeStore(mem={"id": "m1", "memory": "hello"})
    allowed, reason = await async_check_memory_access_permissions(store, "m1", app_id=None)
    assert allowed is True
```

- [ ] **Step 2: 运行确认失败**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_async_permissions.py -v`
Expected: FAIL — `No module named 'septmuse.governance.async_permissions'`

- [ ] **Step 3: 写 async 权限检查函数**

```python
# src/septmuse/governance/async_permissions.py
"""异步权限检查 — 4 层权限检查（async 版，与 sync 版并存）。

层1: memory 存在 + state=active
层2: 无 app_id → 用户自己访问, 放行
层3: app_id 非空即 active
层4: app 白名单（默认全部可访问）
"""
from __future__ import annotations

from septmuse.governance.permissions import MemoryState
from septmuse.storage.async_base import AsyncMemoryStore


async def async_check_memory_access_permissions(
    store: AsyncMemoryStore,
    memory_id: str,
    app_id: str | None = None,
) -> tuple[bool, str]:
    """4 层权限检查（async 版）。

    Args:
        store: 异步记忆存储后端
        memory_id: 目标记忆 ID
        app_id: 访问方应用 ID; None 表示用户自己访问

    Returns:
        (allowed, reason): True=放行, False=拒绝 + 原因
    """
    # 层1: memory 存在 + state=active
    mem = await store.get(memory_id)
    if not mem:
        return False, "memory not found"
    state = mem.get("state", MemoryState.ACTIVE)
    if state is not None and state != MemoryState.ACTIVE:
        return False, f"memory state is {state} (not active)"

    # 层2: 无 app_id (None) → 用户自己访问, 放行
    if app_id is None:
        return True, "self access"

    # 层3: app_id 非空即 active
    if not app_id.strip():
        return False, "app_id is empty"

    # 层4: app 白名单（默认全部可访问）
    return True, f"app {app_id} access granted"
```

- [ ] **Step 4: 运行测试确认通过**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_async_permissions.py -v`
Expected: PASS（5 测试全通过）

- [ ] **Step 5: ruff 检查**

Run: `ruff check --no-cache src/septmuse/governance/async_permissions.py tests/unit/test_async_permissions.py`
Expected: All checks passed!

---

## Task 2: async 访问日志函数

**Files:**
- Create: `src/septmuse/governance/async_access_log.py`
- Test: `tests/unit/test_async_access_log.py`

**Interfaces:**
- Consumes: `AsyncMemoryStore`（`storage/async_base.py`）
- Produces: `async_record_access(store, memory_id, app_id, access_type, metadata) -> str | None`

- [ ] **Step 1: 写失败测试**

```python
# tests/unit/test_async_access_log.py
"""async 访问日志函数测试。"""
from septmuse.governance.async_access_log import async_record_access
from septmuse.storage.async_base import AsyncMemoryStore


class _FakeStoreWithLog(AsyncMemoryStore):
    """支持访问日志的假 store。"""

    async def add(self, content, embedding, *, user_id, **kwargs):
        pass

    async def search(self, query_embedding, *, user_id, **kwargs):
        return []

    async def get_all(self, *, user_id, **kwargs):
        return []

    async def get(self, memory_id):
        return None

    async def delete(self, memory_id):
        pass

    async def update(self, memory_id, content, embedding, *, metadata=None):
        return True

    async def get_history(self, memory_id):
        return []

    async def close(self):
        pass

    async def _record_access_log(self, memory_id, app_id, access_type, metadata=None):
        return "log-123"


class _FakeStoreNoLog(AsyncMemoryStore):
    """不支持访问日志的假 store。"""

    async def add(self, content, embedding, *, user_id, **kwargs):
        pass

    async def search(self, query_embedding, *, user_id, **kwargs):
        return []

    async def get_all(self, *, user_id, **kwargs):
        return []

    async def get(self, memory_id):
        return None

    async def delete(self, memory_id):
        pass

    async def update(self, memory_id, content, embedding, *, metadata=None):
        return True

    async def get_history(self, memory_id):
        return []

    async def close(self):
        pass


class _FakeStoreRaisesLog(AsyncMemoryStore):
    """_record_access_log 抛异常的假 store。"""

    async def add(self, content, embedding, *, user_id, **kwargs):
        pass

    async def search(self, query_embedding, *, user_id, **kwargs):
        return []

    async def get_all(self, *, user_id, **kwargs):
        return []

    async def get(self, memory_id):
        return None

    async def delete(self, memory_id):
        pass

    async def update(self, memory_id, content, embedding, *, metadata=None):
        return True

    async def get_history(self, memory_id):
        return []

    async def close(self):
        pass

    async def _record_access_log(self, memory_id, app_id, access_type, metadata=None):
        raise RuntimeError("DB locked")


async def test_returns_log_id():
    """支持日志的 store 返回 log_id。"""
    store = _FakeStoreWithLog()
    log_id = await async_record_access(store, "m1", "app1", "get")
    assert log_id == "log-123"


async def test_unsupported_store_returns_none():
    """不支持日志的 store 返回 None。"""
    store = _FakeStoreNoLog()
    log_id = await async_record_access(store, "m1", "app1", "get")
    assert log_id is None


async def test_swallows_exceptions():
    """_record_access_log 抛异常时不传播，返回 None。"""
    store = _FakeStoreRaisesLog()
    log_id = await async_record_access(store, "m1", "app1", "get")
    assert log_id is None
```

- [ ] **Step 2: 运行确认失败**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_async_access_log.py -v`
Expected: FAIL — `No module named 'septmuse.governance.async_access_log'`

- [ ] **Step 3: 写 async 访问日志函数**

```python
# src/septmuse/governance/async_access_log.py
"""异步访问日志（async 版，与 sync 版并存）。

async_record_access: 吞错（日志失败不阻塞业务）。
通过 hasattr 检查 store 是否支持 _record_access_log（向后兼容）。
"""
from __future__ import annotations

from typing import Any

from septmuse.core.logging import get_logger
from septmuse.storage.async_base import AsyncMemoryStore

logger = get_logger(__name__)


async def async_record_access(
    store: AsyncMemoryStore,
    memory_id: str,
    app_id: str | None,
    access_type: str,
    metadata: dict[str, Any] | None = None,
) -> str | None:
    """异步记录记忆访问日志。

    Args:
        store: 异步记忆存储后端（必须支持 _record_access_log 方法）
        memory_id: 被访问的记忆 ID
        app_id: 访问方应用 ID
        access_type: "search" / "get" / "delete" / "list"
        metadata: 额外信息

    Returns:
        log_id 或 None（记录失败时返回 None，不抛异常）
    """
    try:
        if hasattr(store, "_record_access_log"):
            return await store._record_access_log(memory_id, app_id, access_type, metadata)
        logger.warning("async_store_does_not_support_access_log", store=type(store).__name__)
        return None
    except Exception as e:
        logger.warning("async_access_log_failed", error=str(e), memory_id=memory_id)
        return None
```

- [ ] **Step 4: 运行测试确认通过**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_async_access_log.py -v`
Expected: PASS（3 测试全通过）

- [ ] **Step 5: ruff 检查**

Run: `ruff check --no-cache src/septmuse/governance/async_access_log.py tests/unit/test_async_access_log.py`
Expected: All checks passed!

---

## Task 3: AsyncSQLiteMemoryStore 补齐访问日志

**Files:**
- Modify: `src/septmuse/storage/async_sqlite/store.py`
- Test: 无新增（Task 1-2 的测试已覆盖权限/日志函数，这里补齐 store 方法）

**Interfaces:**
- Consumes: Task 1-2 的 async 权限/日志函数会调 `store._record_access_log` 和 `store.get_access_logs`
- Produces: `AsyncSQLiteMemoryStore._record_access_log` + `AsyncSQLiteMemoryStore.get_access_logs`（覆盖 ABC 默认）

- [ ] **Step 1: 修改 _create_tables 加 memory_access_logs 表**

在 `src/septmuse/storage/async_sqlite/store.py` 的 `_create_tables` 方法中，`history` 表的 `CREATE TABLE` 后面加 `memory_access_logs` 表。

找到 `_create_tables` 方法中的 `executescript("""...""")`，在 `CREATE TABLE IF NOT EXISTS history (...)` 后面、`""")` 前面加：

```sql
            CREATE TABLE IF NOT EXISTS memory_access_logs (
                id           TEXT PRIMARY KEY,
                memory_id    TEXT NOT NULL,
                app_id       TEXT,
                access_type  TEXT NOT NULL,
                metadata     TEXT,
                accessed_at  TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_access_logs_memory ON memory_access_logs(memory_id);
```

- [ ] **Step 2: 加 _record_access_log 方法**

在 `AsyncSQLiteMemoryStore` 类中，`close` 方法前面加：

```python
    async def _record_access_log(
        self,
        memory_id: str,
        app_id: str | None,
        access_type: str,
        metadata: dict[str, Any] | None = None,
    ) -> str | None:
        """记录记忆访问日志（审计用）。"""
        import uuid as _uuid
        conn = await self._ensure_conn()
        log_id = str(_uuid.uuid4())
        now = _utcnow_iso()
        await conn.execute(
            "INSERT INTO memory_access_logs (id, memory_id, app_id, access_type, metadata, accessed_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (log_id, memory_id, app_id, access_type, json.dumps(metadata) if metadata else None, now),
        )
        await conn.commit()
        return log_id
```

- [ ] **Step 3: 加 get_access_logs 方法（覆盖 ABC 默认）**

在 `_record_access_log` 方法后面加：

```python
    async def get_access_logs(self, memory_id: str, limit: int = 100) -> list[dict[str, Any]]:
        """查询记忆访问日志（按 accessed_at 降序）。"""
        conn = await self._ensure_conn()
        cursor = await conn.execute(
            "SELECT id, memory_id, app_id, access_type, metadata, accessed_at "
            "FROM memory_access_logs WHERE memory_id=? ORDER BY accessed_at DESC LIMIT ?",
            (memory_id, limit),
        )
        rows = await cursor.fetchall()
        return [
            {
                "id": r[0],
                "memory_id": r[1],
                "app_id": r[2],
                "access_type": r[3],
                "metadata": json.loads(r[4]) if r[4] else None,
                "accessed_at": r[5],
            }
            for r in rows
        ]
```

- [ ] **Step 4: 运行 async store 测试确认不破坏**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_async_sqlite_store.py -v`
Expected: PASS（6 测试全通过）

- [ ] **Step 5: ruff 检查**

Run: `ruff check --no-cache src/septmuse/storage/async_sqlite/store.py`
Expected: All checks passed!

---

## Task 4: AsyncMemory 加 invalidate + search session_id

**Files:**
- Modify: `src/septmuse/memory/async_main.py`
- Test: 无新增（现有 5 测试保持通过）

**Interfaces:**
- Consumes: `AsyncMemoryStore.invalidate`（ABC 默认 raise NotImplementedError），`AsyncMemoryStore.search`（已有 session_id 参数）
- Produces: `AsyncMemory.invalidate(memory_id, invalid_at) -> dict` + `AsyncMemory.search` 加 `session_id` 参数

- [ ] **Step 1: search 方法加 session_id 参数**

在 `src/septmuse/memory/async_main.py` 中，找到 `search` 方法：

```python
    async def search(
        self, query: str, *, user_id: str, top_k: int = 5, threshold: float = 0.1
    ) -> list[dict[str, Any]]:
        """异步检索记忆。"""
        emb = await asyncio.to_thread(self.embedder.embed, query)
        return await self.store.search(emb, user_id=user_id, top_k=top_k, threshold=threshold)
```

改为：

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

- [ ] **Step 2: 加 invalidate 方法**

在 `close` 方法前面加：

```python
    async def invalidate(self, memory_id: str, *, invalid_at: str | None = None) -> dict[str, Any]:
        """异步标记事实不再为真。"""
        return await self.store.invalidate(memory_id, invalid_at=invalid_at)
```

- [ ] **Step 3: AsyncSQLiteMemoryStore 加 invalidate 方法**

在 `src/septmuse/storage/async_sqlite/store.py` 的 `get_access_logs` 方法后面加：

```python
    async def invalidate(self, memory_id: str, *, invalid_at: str | None = None) -> dict[str, Any]:
        """标记事实不再为真（设置 invalid_at + expired_at，不删除记忆）。"""
        from datetime import datetime, timezone
        conn = await self._ensure_conn()
        # 先检查记忆是否存在
        existing = await self.get(memory_id)
        if existing is None:
            return {"id": memory_id, "event": "NOT_FOUND"}
        inv_at = invalid_at or datetime.now(timezone.utc).isoformat()
        exp_at = datetime.now(timezone.utc).isoformat()
        now = datetime.now(timezone.utc).isoformat()
        await conn.execute(
            "UPDATE memories SET invalid_at=?, expired_at=?, updated_at=? WHERE id=?",
            (inv_at, exp_at, now, memory_id),
        )
        await conn.execute(
            """INSERT INTO history (id, memory_id, old_memory, new_memory, event, created_at, is_deleted)
               VALUES (?, ?, ?, ?, ?, ?, 0)""",
            (str(uuid.uuid4()), memory_id, existing.get("memory"), None, "INVALIDATE", now),
        )
        await conn.commit()
        logger.info("async_memory_invalidated", memory_id=memory_id, invalid_at=inv_at)
        return {"id": memory_id, "invalid_at": inv_at, "expired_at": exp_at, "event": "INVALIDATE"}
```

- [ ] **Step 4: 运行 async memory 测试确认不破坏**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_async_memory.py -v`
Expected: PASS（5 测试全通过）

- [ ] **Step 5: ruff 检查**

Run: `ruff check --no-cache src/septmuse/memory/async_main.py src/septmuse/storage/async_sqlite/store.py`
Expected: All checks passed!

---

## Task 5: REST API 核心端点切换 + 全量验证

**Files:**
- Modify: `src/septmuse/api/rest/__init__.py`

**Interfaces:**
- Consumes: Task 1 的 `async_check_memory_access_permissions`，Task 2 的 `async_record_access`，Task 4 的 `AsyncMemory.invalidate` + `AsyncMemory.search(session_id=...)`
- Produces: REST API 9 核心端点用 AsyncMemory，12 实验端点保持 ExperimentalMemory

- [ ] **Step 1: 修改 import**

在 `src/septmuse/api/rest/__init__.py` 的 import 部分，找到：

```python
from septmuse import MemoryConfig
from septmuse.embedders.hash import HashEmbedder
from septmuse.experimental import ExperimentalMemory
from septmuse.governance.access_log import record_access
from septmuse.governance.permissions import check_memory_access_permissions
```

替换为：

```python
from septmuse import MemoryConfig
from septmuse.embedders.hash import HashEmbedder
from septmuse.experimental import ExperimentalMemory
from septmuse.governance.access_log import record_access
from septmuse.governance.async_access_log import async_record_access
from septmuse.governance.async_permissions import async_check_memory_access_permissions
from septmuse.governance.permissions import check_memory_access_permissions
from septmuse.memory.async_main import AsyncMemory
```

- [ ] **Step 2: 修改 register_routes 签名**

找到 `def register_routes(app: FastAPI, memory: ExperimentalMemory) -> None:`，改为：

```python
def register_routes(app: FastAPI, memory: ExperimentalMemory, async_memory: AsyncMemory | None = None) -> None:
```

在函数体开头 `app.state.memory = memory` 后面加：

```python
    app.state.async_memory = async_memory or memory
```

- [ ] **Step 3: 修改 add_memory 端点**

找到 `async def add_memory(req: AddMemoryRequest) -> dict[str, Any]:` 函数，改为：

```python
    @app.post("/memories", status_code=201)
    async def add_memory(req: AddMemoryRequest) -> dict[str, Any]:
        """添加记忆 (架构文档 §11.2)。"""
        if req.memory_type == "semantic":
            parts = req.content.split(None, 2)
            subject = parts[0] if len(parts) > 0 else req.content
            predicate = parts[1] if len(parts) > 1 else "is"
            obj = parts[2] if len(parts) > 2 else ""
            return app.state.memory.add_fact(subject, predicate, obj, user_id=req.user_id)
        elif req.memory_type == "episodic":
            return app.state.memory.add_episode(req.content, user_id=req.user_id)
        elif req.memory_type == "procedural":
            return app.state.memory.add_rule(req.content, user_id=req.user_id)
        else:
            return await app.state.async_memory.add(
                req.content,
                user_id=req.user_id,
                agent_id=req.agent_id,
                session_id=req.session_id,
                infer=req.infer,
                valid_at=req.valid_at,
            )
```

- [ ] **Step 4: 修改 list_memories 端点**

找到 `async def list_memories` 函数，改为：

```python
    @app.get("/memories")
    async def list_memories(
        user_id: str = Query(..., description="用户 ID"),
        app_id: str | None = None,
    ) -> dict[str, Any]:
        """列出记忆 (对齐 mem0 get_all)。"""
        results = await app.state.async_memory.get_all(user_id=user_id)
        store = app.state.async_memory.store
        for r in results:
            await async_record_access(store, r["id"], app_id, "list")
        return {"results": results}
```

- [ ] **Step 5: 修改 get_memory 端点**

找到 `async def get_memory` 函数，改为：

```python
    @app.get("/memories/{memory_id}")
    async def get_memory(memory_id: str, app_id: str | None = None) -> dict[str, Any]:
        """取单条记忆。

        权限层: async_check_memory_access_permissions 校验存在性 + state=active。
        403=存在但非 active (deleted/archived/paused); 404=从未存在。
        """
        store = app.state.async_memory.store
        allowed, reason = await async_check_memory_access_permissions(store, memory_id, app_id)
        if not allowed:
            history = await app.state.async_memory.get_history(memory_id)
            if "not found" in reason and not history:
                raise HTTPException(status_code=404, detail=f"memory {memory_id} not found")
            raise HTTPException(status_code=403, detail=reason)
        await async_record_access(store, memory_id, app_id, "get")
        result = await app.state.async_memory.get(memory_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"memory {memory_id} not found")
        return result
```

- [ ] **Step 6: 修改 update_memory 端点**

找到 `async def update_memory` 函数，改为：

```python
    @app.put("/memories/{memory_id}")
    async def update_memory(memory_id: str, req: UpdateMemoryRequest) -> dict[str, Any]:
        """更新记忆 (对齐 mem0 PUT /memories/{id})。"""
        text = req.text or ""
        success = await app.state.async_memory.update(memory_id, text, metadata=req.metadata)
        if not success:
            raise HTTPException(status_code=404, detail=f"memory {memory_id} not found")
        return {"event": "UPDATE", "memory_id": memory_id, "memory": text}
```

- [ ] **Step 7: 修改 get_history 端点**

找到 `async def get_history` 函数，改为：

```python
    @app.get("/memories/{memory_id}/history")
    async def get_history(memory_id: str) -> list[dict[str, Any]]:
        """获取记忆变更历史 (对齐 mem0 GET /memories/{id}/history)。"""
        return await app.state.async_memory.get_history(memory_id)
```

- [ ] **Step 8: 修改 get_access_logs 端点**

找到 `async def get_access_logs` 函数，改为：

```python
    @app.get("/memories/{memory_id}/access-logs")
    async def get_access_logs(
        memory_id: str,
        limit: int = Query(default=100, ge=1, description="返回日志数上限"),
    ) -> dict[str, Any]:
        """获取记忆访问日志 (审计用, 架构 §11.3)。"""
        store = app.state.async_memory.store
        logs = await store.get_access_logs(memory_id, limit)
        return {"logs": logs}
```

- [ ] **Step 9: 修改 delete_memory 端点**

找到 `async def delete_memory` 函数，改为：

```python
    @app.delete("/memories/{memory_id}")
    async def delete_memory(memory_id: str, app_id: str | None = None) -> dict[str, str]:
        """删除记忆 (软删除)。

        权限层: 校验存在性 + state=active 后才允许删除。
        403=存在但非 active; 404=从未存在。
        """
        store = app.state.async_memory.store
        allowed, reason = await async_check_memory_access_permissions(store, memory_id, app_id)
        if not allowed:
            history = await app.state.async_memory.get_history(memory_id)
            if "not found" in reason and not history:
                raise HTTPException(status_code=404, detail=f"memory {memory_id} not found")
            raise HTTPException(status_code=403, detail=reason)
        await async_record_access(store, memory_id, app_id, "delete")
        await app.state.async_memory.delete(memory_id)
        return {"event": "DELETE", "memory_id": memory_id}
```

- [ ] **Step 10: 修改 invalidate_memory 端点**

找到 `async def invalidate_memory` 函数，改为：

```python
    @app.post("/memories/{memory_id}/invalidate")
    async def invalidate_memory(memory_id: str, req: InvalidateRequest) -> dict[str, Any]:
        """手动标记事实不再为真 (设置 invalid_at + expired_at, 不删除记忆)。"""
        return await app.state.async_memory.invalidate(memory_id, invalid_at=req.invalid_at)
```

- [ ] **Step 11: 修改 search_memories 端点**

找到 `async def search_memories` 函数，改为：

```python
    @app.post("/memories/search")
    async def search_memories(req: SearchRequest) -> dict[str, Any]:
        """统一检索 (元认知路由)。"""
        if req.reranker or req.explain:
            # 高级检索（reranker/explain），回退到 sync ExperimentalMemory
            results = app.state.memory.search(
                req.query,
                user_id=req.user_id,
                session_id=req.session_id,
                top_k=req.top_k,
                threshold=req.threshold,
                reranker=req.reranker,
                explain=req.explain,
            )
        else:
            # 基础检索，用 async
            results = await app.state.async_memory.search(
                req.query,
                user_id=req.user_id,
                session_id=req.session_id,
                top_k=req.top_k,
                threshold=req.threshold,
            )
        return {"results": results}
```

- [ ] **Step 12: 修改 create_app 函数**

找到 `def create_app(memory: ExperimentalMemory | MemoryConfig | None = None) -> FastAPI:` 函数，改为：

```python
def create_app(memory=None) -> FastAPI:
    """创建 FastAPI app (可注入 Memory 实例或 MemoryConfig 便于测试)。

    用法:
        app = create_app()
        # uvicorn septmuse.api.rest:app

    测试:
        app = create_app(MemoryConfig(db_path=str(tmp_path / "rest.db")))
        # TestClient(app).post("/memories", json={...})
    """
    if memory is None:
        config = MemoryConfig()
        sync_memory = ExperimentalMemory(config=config, embedder=HashEmbedder())
        async_memory = AsyncMemory(config=config, embedder=HashEmbedder())
    elif isinstance(memory, MemoryConfig):
        sync_memory = ExperimentalMemory(config=memory, embedder=HashEmbedder())
        async_memory = AsyncMemory(config=memory, embedder=HashEmbedder())
    else:
        # 已有 Memory 实例注入（向后兼容）
        sync_memory = memory
        async_memory = AsyncMemory(config=memory.config, embedder=memory.embedder)

    app = FastAPI(
        title="SeptMuse Memory API",
        description="Agent 记忆系统 REST API (架构文档 §11.2)",
        version="0.1.0",
    )
    from septmuse.api.auth import setup_auth

    setup_auth(app)
    register_routes(app, sync_memory, async_memory)
    return app
```

- [ ] **Step 13: ruff 检查**

Run: `ruff check --no-cache src/septmuse/api/rest/__init__.py`
Expected: All checks passed!

- [ ] **Step 14: REST 权限测试验证**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_api_permission_integration.py tests/unit/test_rbac_rest_openai.py -v --tb=short 2>&1 | Select-Object -Last 20`
Expected: 37 passed（全部通过，不退化）

- [ ] **Step 15: async 测试验证**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_async_permissions.py tests/unit/test_async_access_log.py tests/unit/test_async_store_base.py tests/unit/test_async_sqlite_store.py tests/unit/test_async_memory.py -v`
Expected: PASS（22 测试全通过）

- [ ] **Step 16: 全量 ruff**

Run: `ruff check --no-cache src/ tests/`
Expected: All checks passed!

- [ ] **Step 17: 全量 pytest**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/ tests/e2e/ -q --tb=line 2>&1 | Select-Object -Last 5`
Expected: 失败不超过 23（之前基线），passed 不低于 1050 + 新增 8 个 async 测试

- [ ] **Step 18: AsyncMemory 零配置验证**

Run: `$env:PYTHONPATH="src"; python -c "import asyncio; from septmuse.memory.async_main import AsyncMemory; from septmuse.embedders.hash import HashEmbedder; import tempfile, os; db=os.path.join(tempfile.mkdtemp(), 't.db'); from septmuse.storage.async_sqlite.store import AsyncSQLiteMemoryStore; s=AsyncSQLiteMemoryStore(db_path=db); m=AsyncMemory(embedder=HashEmbedder(), store=s); r=asyncio.run(m.add('hello', user_id='test')); print('OK', r['results'][0]['id']); asyncio.run(m.close())"`
Expected: `OK mem-...`

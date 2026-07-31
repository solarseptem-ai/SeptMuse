# 双时态建模实施计划（P2-Task 1）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现 valid_at/invalid_at/expired_at 三列 + 手动失效 + 时态查询，让 SeptMuse 支持"某时刻为真"的事实检索

**Architecture:** 在 memories 表加 3 个时态列（ALTER TABLE 迁移，复用 _migrate_add_state_columns 模式）。Memory.add() 接受 valid_at，Memory.invalidate() 手动失效，Memory.search_at() 时态过滤 + hybrid 检索。不依赖 LLM，前向兼容 P3 自动矛盾检测。

**Tech Stack:** Python 3.10+ / pytest / ruff / SQLite (ALTER TABLE + PRAGMA table_info)

## Global Constraints

- PYTHONPATH=src 运行所有 pytest 命令（PowerShell: `$env:PYTHONPATH="src"`）
- ruff line-length 120，select=["E","F","I","W","UP","B","SIM","RUF"]，ignore=["E501","RUF001","RUF002","RUF003"]
- **禁止** `ruff format <file>`（Windows 会清空文件），只用 `ruff format --check` 或 `ruff check --fix`
- 不用 git（文件快照模式），每个 Task 完成后更新 `.sdd/progress.md`
- 现有 800 passed + 36 skipped 测试零回归
- score 语义：相似度 [0,1]，越高越相似
- 中文输出（AGENTS.md 强制），代码注释可用英文
- 禁止 `from __future__ import annotations` 在 MCP tools.py（FastMCP 限制）
- e2e 测试用 `tmp_path` 文件 DB（NOT `:memory:`，见 AGENTS.md SQLite quirk）
- `_utcnow_iso()` 已在 store.py:44 定义，返回 ISO 8601 UTC 字符串

---

## File Structure

| 文件 | 职责 | 操作 |
|------|------|------|
| `src/septmuse/storage/sqlite/store.py` | `_migrate_add_temporal_columns()` + `add(valid_at=)` + `get_temporal_valid()` + `invalidate()` | 修改 |
| `src/septmuse/orchestration/memory.py` | `add(valid_at=)` + `invalidate()` + `search_at()` | 修改 |
| `src/septmuse/cli/main.py` | `add --valid-at` + `invalidate` 命令 + `search --at` | 修改 |
| `src/septmuse/api/rest/__init__.py` | `POST /memories` +valid_at + `POST /memories/{id}/invalidate` + `search` +reference_time | 修改 |
| `src/septmuse/api/mcp/tools.py` | `add_memory` +valid_at + `invalidate_memory` + `search_memory` +reference_time | 修改 |
| `tests/unit/test_temporal.py` | ~15 单元测试 | 新建 |
| `tests/e2e/test_temporal_e2e.py` | 3 e2e 测试 | 新建 |
| `CHANGELOG.md` | 变更记录 | 修改 |
| `AGENTS.md` | +双时态章节 | 修改 |

---

## Task 1: Schema 迁移 + Store 方法

**Files:**
- Modify: `src/septmuse/storage/sqlite/store.py`
- Test: `tests/unit/test_temporal.py`

**Interfaces:**
- Produces: `_migrate_add_temporal_columns()`，`add(valid_at=)`，`get_temporal_valid(reference_time, *, user_id)`，`invalidate(memory_id, *, invalid_at)`
- Consumes: `_utcnow_iso()`（已有）

- [ ] **Step 1: Write failing test**

Create `tests/unit/test_temporal.py`:

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
"""双时态建模单元测试 (借鉴 graphiti EntityEdge bitemporal fields)。"""
from __future__ import annotations

import pytest

from septmuse.storage.sqlite.store import SQLiteMemoryStore


class TestTemporalMigration:
    def test_new_db_has_temporal_columns(self, tmp_path):
        """新 DB 直接有 valid_at/invalid_at/expired_at 列。"""
        store = SQLiteMemoryStore(db_path=str(tmp_path / "test.db"))
        cols = {row[1] for row in store.conn.execute("PRAGMA table_info(memories)")}
        assert "valid_at" in cols
        assert "invalid_at" in cols
        assert "expired_at" in cols
        store.close()

    def test_idempotent_migration(self, tmp_path):
        """重复迁移不报错。"""
        store = SQLiteMemoryStore(db_path=str(tmp_path / "test.db"))
        store._migrate_add_temporal_columns()
        store._migrate_add_temporal_columns()
        cols = {row[1] for row in store.conn.execute("PRAGMA table_info(memories)")}
        assert "valid_at" in cols
        store.close()


class TestAddWithValidAt:
    def test_add_with_valid_at(self, tmp_path):
        """add(valid_at=...) 写入 valid_at 列。"""
        store = SQLiteMemoryStore(db_path=str(tmp_path / "test.db"))
        mid = store.add("Alice works at Google", [1.0, 0.0], user_id="u1", valid_at="2024-01-01")
        row = store.conn.execute(
            "SELECT valid_at, invalid_at, expired_at FROM memories WHERE id=?", (mid,)
        ).fetchone()
        assert row[0] == "2024-01-01"
        assert row[1] is None
        assert row[2] is None
        store.close()

    def test_add_without_valid_at(self, tmp_path):
        """不设 valid_at 时列为 NULL (向后兼容)。"""
        store = SQLiteMemoryStore(db_path=str(tmp_path / "test.db"))
        mid = store.add("hello world", [1.0, 0.0], user_id="u1")
        row = store.conn.execute(
            "SELECT valid_at FROM memories WHERE id=?", (mid,)
        ).fetchone()
        assert row[0] is None
        store.close()


class TestGetTemporalValid:
    def test_returns_memories_valid_at_reference_time(self, tmp_path):
        """valid_at <= reference_time 的记忆返回。"""
        store = SQLiteMemoryStore(db_path=str(tmp_path / "test.db"))
        store.add("Alice at Google", [1.0, 0.0], user_id="u1", valid_at="2024-01-01")
        store.add("Alice at Apple", [1.0, 0.0], user_id="u1", valid_at="2025-01-01")
        results = store.get_temporal_valid("2024-06-01", user_id="u1")
        ids = [r["id"] for r in results]
        assert len(results) == 1
        assert "Google" in results[0]["memory"]

    def test_returns_null_valid_at(self, tmp_path):
        """valid_at IS NULL 的记忆始终返回 (向后兼容)。"""
        store = SQLiteMemoryStore(db_path=str(tmp_path / "test.db"))
        store.add("no time constraint", [1.0, 0.0], user_id="u1")
        results = store.get_temporal_valid("2024-06-01", user_id="u1")
        assert len(results) == 1

    def test_excludes_invalidated(self, tmp_path):
        """invalid_at <= reference_time 的记忆不返回。"""
        store = SQLiteMemoryStore(db_path=str(tmp_path / "test.db"))
        mid = store.add("old fact", [1.0, 0.0], user_id="u1", valid_at="2024-01-01")
        store.invalidate(mid, invalid_at="2024-06-01")
        results = store.get_temporal_valid("2024-07-01", user_id="u1")
        assert len(results) == 0

    def test_includes_still_valid(self, tmp_path):
        """invalid_at > reference_time 的记忆返回。"""
        store = SQLiteMemoryStore(db_path=str(tmp_path / "test.db"))
        mid = store.add("current fact", [1.0, 0.0], user_id="u1", valid_at="2024-01-01")
        store.invalidate(mid, invalid_at="2025-01-01")
        results = store.get_temporal_valid("2024-06-01", user_id="u1")
        assert len(results) == 1

    def test_excludes_future_valid_at(self, tmp_path):
        """valid_at > reference_time 的记忆不返回。"""
        store = SQLiteMemoryStore(db_path=str(tmp_path / "test.db"))
        store.add("future fact", [1.0, 0.0], user_id="u1", valid_at="2025-01-01")
        results = store.get_temporal_valid("2024-06-01", user_id="u1")
        assert len(results) == 0


class TestInvalidate:
    def test_invalidate_sets_columns(self, tmp_path):
        """invalidate 后 invalid_at + expired_at 被设置。"""
        store = SQLiteMemoryStore(db_path=str(tmp_path / "test.db"))
        mid = store.add("Alice at Google", [1.0, 0.0], user_id="u1", valid_at="2024-01-01")
        result = store.invalidate(mid, invalid_at="2025-01-01")
        assert result["event"] == "INVALIDATE"
        assert result["invalid_at"] == "2025-01-01"
        row = store.conn.execute(
            "SELECT invalid_at, expired_at FROM memories WHERE id=?", (mid,)
        ).fetchone()
        assert row[0] == "2025-01-01"
        assert row[1] is not None

    def test_invalidate_default_time(self, tmp_path):
        """默认 invalid_at = utc_now()。"""
        store = SQLiteMemoryStore(db_path=str(tmp_path / "test.db"))
        mid = store.add("test", [1.0, 0.0], user_id="u1")
        result = store.invalidate(mid)
        assert result["invalid_at"] is not None
        assert result["expired_at"] is not None

    def test_invalidate_not_found(self, tmp_path):
        """不存在的 memory_id 返回 NOT_FOUND。"""
        store = SQLiteMemoryStore(db_path=str(tmp_path / "test.db"))
        result = store.invalidate("nonexistent-id")
        assert result["event"] == "NOT_FOUND"

    def test_invalidate_does_not_delete(self, tmp_path):
        """invalidate 不删除记忆 (保留历史)。"""
        store = SQLiteMemoryStore(db_path=str(tmp_path / "test.db"))
        mid = store.add("test", [1.0, 0.0], user_id="u1")
        store.invalidate(mid)
        m = store.get(mid)
        assert m is not None
        assert m["id"] == mid
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_temporal.py -q`
Expected: FAIL with `AttributeError: 'SQLiteMemoryStore' object has no attribute '_migrate_add_temporal_columns'`

- [ ] **Step 3: Write implementation**

Read `src/septmuse/storage/sqlite/store.py` first. Make these changes:

**3a. Add `_migrate_add_temporal_columns()` method** after `_migrate_add_state_columns()` (around line 149):

```python
    def _migrate_add_temporal_columns(self) -> None:
        """P2 Task 1 迁移: 加 valid_at/invalid_at/expired_at 列 (借鉴 graphiti EntityEdge)。

        幂等: PRAGMA table_info 检查列存在性。
        """
        with self._lock:
            cols = {row[1] for row in self.conn.execute("PRAGMA table_info(memories)")}
            if "valid_at" not in cols:
                self.conn.execute("ALTER TABLE memories ADD COLUMN valid_at TEXT")
            if "invalid_at" not in cols:
                self.conn.execute("ALTER TABLE memories ADD COLUMN invalid_at TEXT")
            if "expired_at" not in cols:
                self.conn.execute("ALTER TABLE memories ADD COLUMN expired_at TEXT")
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_temporal "
                "ON memories(user_id, valid_at, invalid_at)"
            )
            self.conn.commit()
```

**3b. Call it in `__init__`** — add after `self._create_access_logs_table()` (around line 73):

```python
        self._migrate_add_temporal_columns()
```

**3c. Modify `add()` method** — add `valid_at` parameter to signature and INSERT:

Change the signature from:
```python
    def add(self, content, embedding, *, user_id, agent_id=None, metadata=None) -> str:
```
to:
```python
    def add(self, content, embedding, *, user_id, agent_id=None, metadata=None, valid_at=None) -> str:
```

In the INSERT statement, add `valid_at`:
```python
                self.conn.execute(
                    """
                    INSERT INTO memories
                        (id, user_id, agent_id, content, embedding, metadata, created_at, updated_at, is_deleted, valid_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
                    """,
                    (
                        mid,
                        user_id,
                        agent_id,
                        content,
                        json.dumps(embedding),
                        json.dumps(metadata or {}),
                        now,
                        now,
                        valid_at,
                    ),
                )
```

**3d. Add `get_temporal_valid()` method** after `get_all()` (around line 394):

```python
    def get_temporal_valid(self, reference_time: str, *, user_id: str) -> list[dict[str, Any]]:
        """时态过滤: 返回某时刻为真的全部记忆 (借鉴 graphiti temporal filters)。

        WHERE user_id=? AND is_deleted=0 AND (state='active' OR state IS NULL)
          AND (valid_at IS NULL OR valid_at <= ?)
          AND (invalid_at IS NULL OR invalid_at > ?)
        """
        with self._lock:
            cur = self.conn.execute(
                "SELECT id, content, metadata, created_at, valid_at, invalid_at "
                "FROM memories "
                "WHERE user_id=? AND is_deleted=0 AND (state='active' OR state IS NULL) "
                "AND (valid_at IS NULL OR valid_at <= ?) "
                "AND (invalid_at IS NULL OR invalid_at > ?)",
                (user_id, reference_time, reference_time),
            )
            rows = cur.fetchall()
        return [
            {
                "id": r[0],
                "memory": r[1],
                "metadata": json.loads(r[2]) if r[2] else {},
                "created_at": r[3],
                "valid_at": r[4],
                "invalid_at": r[5],
            }
            for r in rows
        ]
```

**3e. Add `invalidate()` method** after `get_temporal_valid()`:

```python
    def invalidate(self, memory_id: str, *, invalid_at: str | None = None) -> dict[str, Any]:
        """手动标记事实不再为真 (借鉴 graphiti resolve_edge_contradictions)。

        设置 invalid_at + expired_at, 不删除记忆。
        """
        existing = self.get(memory_id)
        if existing is None:
            return {"id": memory_id, "event": "NOT_FOUND"}

        inv_at = invalid_at or _utcnow_iso()
        exp_at = _utcnow_iso()
        now = _utcnow_iso()
        with self._lock:
            try:
                self.conn.execute("BEGIN")
                self.conn.execute(
                    "UPDATE memories SET invalid_at=?, expired_at=?, updated_at=? WHERE id=?",
                    (inv_at, exp_at, now, memory_id),
                )
                self.conn.execute(
                    """
                    INSERT INTO history (id, memory_id, old_memory, new_memory, event, created_at, is_deleted)
                    VALUES (?, ?, ?, ?, ?, ?, 0)
                    """,
                    (str(uuid.uuid4()), memory_id, existing.get("memory"), None, "INVALIDATE", now),
                )
                self.conn.execute("COMMIT")
            except Exception:
                self.conn.execute("ROLLBACK")
                raise
        logger.info("memory_invalidated", memory_id=memory_id, invalid_at=inv_at)
        return {"id": memory_id, "invalid_at": inv_at, "expired_at": exp_at, "event": "INVALIDATE"}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_temporal.py -q`
Expected: PASS (13 tests)

- [ ] **Step 5: Lint + regression**

Run: `ruff check src/septmuse/storage/sqlite/store.py tests/unit/test_temporal.py`
Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/ tests/e2e/ -q`
Expected: 800 + 13 = 813 passed, 36 skipped, ruff clean

- [ ] **Step 6: Update progress**

Append to `.sdd/progress.md`: `Task 1: complete (schema migration + store methods, 13 tests)`

---

## Task 2: Memory Facade (add valid_at + invalidate + search_at)

**Files:**
- Modify: `src/septmuse/orchestration/memory.py`
- Test: `tests/unit/test_temporal.py`

**Interfaces:**
- Consumes: `store.add(valid_at=)`, `store.get_temporal_valid()`, `store.invalidate()` from Task 1
- Produces: `Memory.add(valid_at=)`, `Memory.invalidate(memory_id)`, `Memory.search_at(reference_time, query, user_id)`

- [ ] **Step 1: Write failing test**

Append to `tests/unit/test_temporal.py`:

```python
from septmuse.configs.defaults import MemoryConfig
from septmuse.orchestration.memory import Memory


class TestMemoryAddValidAt:
    def test_add_with_valid_at(self, tmp_path):
        m = Memory(config=MemoryConfig(db_path=str(tmp_path / "test.db")))
        result = m.add("Alice works at Google", user_id="u1", valid_at="2024-01-01")
        assert len(result["results"]) == 1
        mid = result["results"][0]["id"]
        mem = m.get(mid)
        assert mem is not None

    def test_add_without_valid_at(self, tmp_path):
        m = Memory(config=MemoryConfig(db_path=str(tmp_path / "test.db")))
        result = m.add("hello world", user_id="u1")
        assert len(result["results"]) == 1


class TestMemoryInvalidate:
    def test_invalidate_existing(self, tmp_path):
        m = Memory(config=MemoryConfig(db_path=str(tmp_path / "test.db")))
        result = m.add("Alice at Google", user_id="u1", valid_at="2024-01-01")
        mid = result["results"][0]["id"]
        inv = m.invalidate(mid, invalid_at="2025-01-01")
        assert inv["event"] == "INVALIDATE"
        assert inv["invalid_at"] == "2025-01-01"

    def test_invalidate_not_found(self, tmp_path):
        m = Memory(config=MemoryConfig(db_path=str(tmp_path / "test.db")))
        inv = m.invalidate("nonexistent")
        assert inv["event"] == "NOT_FOUND"

    def test_invalidate_default_time(self, tmp_path):
        m = Memory(config=MemoryConfig(db_path=str(tmp_path / "test.db")))
        result = m.add("test", user_id="u1")
        mid = result["results"][0]["id"]
        inv = m.invalidate(mid)
        assert inv["invalid_at"] is not None
        assert inv["expired_at"] is not None


class TestMemorySearchAt:
    def test_search_at_returns_valid_facts(self, tmp_path):
        m = Memory(config=MemoryConfig(db_path=str(tmp_path / "test.db")))
        m.add("Alice works at Google", user_id="u1", valid_at="2024-01-01")
        m.add("Alice works at Apple", user_id="u1", valid_at="2025-01-01")
        results = m.search_at("2024-06-01", "Alice", user_id="u1")
        assert len(results) >= 1
        assert any("Google" in r["memory"] for r in results)

    def test_search_at_excludes_future(self, tmp_path):
        m = Memory(config=MemoryConfig(db_path=str(tmp_path / "test.db")))
        m.add("future fact", user_id="u1", valid_at="2025-01-01")
        results = m.search_at("2024-06-01", "future", user_id="u1")
        assert len(results) == 0

    def test_search_at_includes_null_valid_at(self, tmp_path):
        m = Memory(config=MemoryConfig(db_path=str(tmp_path / "test.db")))
        m.add("no time constraint", user_id="u1")
        results = m.search_at("2024-06-01", "time", user_id="u1")
        assert len(results) >= 1

    def test_search_at_excludes_invalidated(self, tmp_path):
        m = Memory(config=MemoryConfig(db_path=str(tmp_path / "test.db")))
        result = m.add("old fact", user_id="u1", valid_at="2024-01-01")
        mid = result["results"][0]["id"]
        m.invalidate(mid, invalid_at="2024-06-01")
        results = m.search_at("2024-07-01", "old", user_id="u1")
        assert len(results) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_temporal.py::TestMemoryAddValidAt -q`
Expected: FAIL with `TypeError: add() got an unexpected keyword argument 'valid_at'`

- [ ] **Step 3: Write implementation**

Read `src/septmuse/orchestration/memory.py` first. Make these changes:

**3a. Modify `Memory.add()` signature** — add `valid_at` parameter:

Find the `add()` method (around line 169). Change:
```python
    def add(self, messages, *, user_id, agent_id=None, metadata=None, infer=None, auto_extract_entities=True) -> dict[str, Any]:
```
to:
```python
    def add(self, messages, *, user_id, agent_id=None, metadata=None, infer=None, auto_extract_entities=True, valid_at=None) -> dict[str, Any]:
```

In the verbatim path (where `self.store.add()` is called), pass `valid_at`:
```python
            mid = self.store.add(text, emb, user_id=user_id, agent_id=agent_id, metadata=metadata, valid_at=valid_at)
```

**3b. Add `invalidate()` method** — after `delete()` method:

```python
    def invalidate(self, memory_id: str, *, invalid_at: str | None = None) -> dict[str, Any]:
        """手动标记事实不再为真 (借鉴 graphiti resolve_edge_contradictions)。

        设置 invalid_at + expired_at, 不删除记忆。
        invalid_at=None 时用 utc_now()。

        Returns:
            {"id", "invalid_at", "expired_at", "event": "INVALIDATE"}
            {"id", "event": "NOT_FOUND"} 如果记忆不存在
        """
        return self.store.invalidate(memory_id, invalid_at=invalid_at)
```

**3c. Add `search_at()` method** — after `search_hybrid()` method:

```python
    def search_at(
        self,
        reference_time: str,
        query: str,
        *,
        user_id: str,
        top_k: int = 5,
        threshold: float = 0.1,
    ) -> list[dict[str, Any]]:
        """时态查询: 查询某时刻为真的事实 (借鉴 graphiti temporal filters)。

        过滤: valid_at <= reference_time AND (invalid_at IS NULL OR invalid_at > reference_time)
        叠加 hybrid 检索 (向量+BM25+entity boost)。

        valid_at IS NULL 的记忆视为"无时间约束", 始终返回 (向后兼容)。
        """
        # 1. 时态过滤得到有效记忆 ID 集合
        valid_memories = self.store.get_temporal_valid(reference_time, user_id=user_id)
        if not valid_memories:
            return []

        valid_ids = {m["id"] for m in valid_memories}

        # 2. 在全部记忆上做 hybrid 检索, 然后后过滤只保留时态有效的
        results = self.search(query, user_id=user_id, top_k=top_k * 2, threshold=threshold, hybrid=True)
        filtered = [r for r in results if r["id"] in valid_ids]

        # 3. 补充 valid_at 信息到结果
        valid_map = {m["id"]: m for m in valid_memories}
        for r in filtered:
            vm = valid_map.get(r["id"])
            if vm:
                r["valid_at"] = vm.get("valid_at")
                r["invalid_at"] = vm.get("invalid_at")

        logger.info("memory_search_at_done", user_id=user_id, reference_time=reference_time,
                     candidates=len(valid_memories), returned=len(filtered[:top_k]))
        return filtered[:top_k]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_temporal.py -q`
Expected: PASS (23 tests: 13 store + 10 facade)

- [ ] **Step 5: Lint + regression**

Run: `ruff check src/septmuse/orchestration/memory.py tests/unit/test_temporal.py`
Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/ tests/e2e/ -q`
Expected: 813 + 10 = 823 passed, 36 skipped, ruff clean

- [ ] **Step 6: Update progress**

Append to `.sdd/progress.md`: `Task 2: complete (Memory facade add/invalidate/search_at, 10 tests)`

---

## Task 3: CLI + REST + MCP 集成

**Files:**
- Modify: `src/septmuse/cli/main.py`, `src/septmuse/api/rest/__init__.py`, `src/septmuse/api/mcp/tools.py`
- Test: `tests/unit/test_temporal.py`

**Interfaces:**
- Produces: CLI `add --valid-at` + `invalidate` 命令 + `search --at`，REST `POST /memories/{id}/invalidate`，MCP `invalidate_memory` 工具

- [ ] **Step 1: Write failing test**

Append to `tests/unit/test_temporal.py`:

```python
class TestCLIValidAt:
    def test_cli_add_with_valid_at(self):
        from septmuse.cli.main import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["add", "hello", "--user-id", "u1", "--valid-at", "2024-01-01"])
        assert args.valid_at == "2024-01-01"

    def test_cli_invalidate_command(self):
        from septmuse.cli.main import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["invalidate", "mem-123"])
        assert args.memory_id == "mem-123"


class TestRESTInvalidate:
    def test_rest_invalidate(self, tmp_path):
        from fastapi.testclient import TestClient
        from septmuse.api.rest import create_app
        from septmuse.configs.defaults import MemoryConfig

        config = MemoryConfig(db_path=str(tmp_path / "rest.db"))
        app = create_app(config)
        client = TestClient(app)

        resp = client.post("/memories", json={"messages": "hello", "user_id": "u1"})
        mid = resp.json()["results"][0]["id"]

        resp = client.post(f"/memories/{mid}/invalidate", json={"invalid_at": "2025-01-01"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["event"] == "INVALIDATE"
        assert data["invalid_at"] == "2025-01-01"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_temporal.py::TestCLIValidAt tests/unit/test_temporal.py::TestRESTInvalidate -q`
Expected: FAIL

- [ ] **Step 3: Write implementation**

Read each file first, then make minimal changes:

**3a. CLI** (`src/septmuse/cli/main.py`):
- Find the `add` subparser, add: `add_parser.add_argument("--valid-at", default=None, help="事实开始为真的时间 (ISO 8601)")`
- In the add command handler, pass `valid_at=args.valid_at` to `m.add()`
- Add a new `invalidate` subparser: `inv_parser = sub.add_parser("invalidate", help="手动标记事实不再为真")`
- Add: `inv_parser.add_argument("memory_id", help="记忆 ID")`
- Add: `inv_parser.add_argument("--invalid-at", default=None, help="失效时间 (ISO 8601)")`
- Add the invalidate command handler that calls `m.invalidate(args.memory_id, invalid_at=args.invalid_at)`

**3b. REST** (`src/septmuse/api/rest/__init__.py`):
- Add `valid_at: str | None = None` to the add memory request body model
- Pass `valid_at=valid_at` to `m.add()` in the handler
- Add a new endpoint: `POST /memories/{memory_id}/invalidate` with body `{"invalid_at": str | None}` that calls `m.invalidate(memory_id, invalid_at=invalid_at)`

**3c. MCP** (`src/septmuse/api/mcp/tools.py`):
- Add `valid_at: str | None = None` parameter to `add_memory` tool
- Add a new `invalidate_memory` tool: `def invalidate_memory(memory_id: str, invalid_at: str | None = None) -> str:`
- REMEMBER: NO `from __future__ import annotations` in this file

- [ ] **Step 4: Run test to verify it passes**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_temporal.py -q`
Expected: PASS (28 tests)

- [ ] **Step 5: Lint + full regression**

Run: `ruff check src/ tests/`
Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/ tests/e2e/ -q`
Expected: 823 + 5 = 828 passed, 36 skipped, ruff clean

- [ ] **Step 6: Update progress**

Append to `.sdd/progress.md`: `Task 3: complete (CLI+REST+MCP, 5 tests)`

---

## Task 4: e2e Tests + CHANGELOG + AGENTS.md

**Files:**
- Create: `tests/e2e/test_temporal_e2e.py`
- Modify: `CHANGELOG.md`, `AGENTS.md`

- [ ] **Step 1: Write e2e tests**

Create `tests/e2e/test_temporal_e2e.py`:

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
"""双时态 e2e 测试: 跨会话时态持久化 + invalidate + search_at。"""
from __future__ import annotations

from septmuse.configs.defaults import MemoryConfig
from septmuse.orchestration.memory import Memory


def test_cross_session_temporal_search(tmp_path):
    """写入 valid_at → 新实例 search_at → 正确过滤。"""
    db = str(tmp_path / "e2e_temporal.db")

    m1 = Memory(config=MemoryConfig(db_path=db))
    m1.add("Alice works at Google", user_id="u1", valid_at="2024-01-01")
    m1.add("Alice works at Apple", user_id="u1", valid_at="2025-01-01")

    m2 = Memory(config=MemoryConfig(db_path=db))
    results = m2.search_at("2024-06-01", "Alice", user_id="u1")
    assert len(results) >= 1
    assert any("Google" in r["memory"] for r in results)
    assert not any("Apple" in r["memory"] for r in results)

    results = m2.search_at("2025-06-01", "Alice", user_id="u1")
    assert any("Apple" in r["memory"] for r in results)


def test_invalidate_then_search_at(tmp_path):
    """invalidate 后 search_at 返回新事实不返回旧事实。"""
    db = str(tmp_path / "e2e_invalidate.db")
    m = Memory(config=MemoryConfig(db_path=db))

    r1 = m.add("Alice works at Google", user_id="u1", valid_at="2024-01-01")
    mid = r1["results"][0]["id"]
    m.add("Alice works at Apple", user_id="u1", valid_at="2025-01-01")

    # 失效旧事实
    m.invalidate(mid, invalid_at="2025-01-01")

    # 2024 年应该返回 Google
    results = m.search_at("2024-06-01", "Alice", user_id="u1")
    assert any("Google" in r["memory"] for r in results)

    # 2025 年不应该返回 Google (已失效)
    results = m.search_at("2025-06-01", "Alice", user_id="u1")
    assert not any("Google" in r["memory"] for r in results)
    assert any("Apple" in r["memory"] for r in results)


def test_null_valid_at_always_returned(tmp_path):
    """valid_at=None 的记忆在 search_at 中始终返回 (向后兼容)。"""
    db = str(tmp_path / "e2e_null.db")
    m = Memory(config=MemoryConfig(db_path=db))
    m.add("permanent fact no time", user_id="u1")

    results = m.search_at("2024-06-01", "permanent", user_id="u1")
    assert len(results) >= 1
    results = m.search_at("2025-06-01", "permanent", user_id="u1")
    assert len(results) >= 1
```

- [ ] **Step 2: Run e2e tests**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/e2e/test_temporal_e2e.py -q`
Expected: PASS (3 tests)

- [ ] **Step 3: Full test suite + lint**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/ tests/e2e/ -q`
Expected: 828 + 3 = 831 passed, 36 skipped

Run: `ruff check src/ tests/`; `ruff format --check src/ tests/`

- [ ] **Step 4: Update CHANGELOG**

Add to `CHANGELOG.md` `[Unreleased]` → `### Added`:

```markdown
- 双时态建模: valid_at/invalid_at/expired_at 三列 + 手动失效 (原因: 补齐时态能力; 影响: 存储层)
- Memory.search_at(reference_time, query, user_id): 时态查询 (原因: 查询某时刻为真的事实; 影响: Memory facade)
- Memory.invalidate(memory_id): 手动标记事实不再为真 (原因: 矛盾检测降级为手动; 影响: Memory facade)
- Memory.add(valid_at=): 写入时设置事实有效期 (原因: 支持时态建模; 影响: Memory facade)
- CLI add --valid-at / invalidate / search --at (原因: API 一致性; 影响: CLI)
- REST POST /memories/{id}/invalidate (原因: API 一致性; 影响: REST)
- MCP invalidate_memory 工具 (原因: API 一致性; 影响: MCP)
```

- [ ] **Step 5: Update AGENTS.md**

Add after the "### Reranker" section:

```markdown
### Bitemporal (双时态)

- `memories` 表有 `valid_at`/`invalid_at`/`expired_at` 三列（P2-Task 1 迁移）。
- `Memory.add(valid_at="2024-01-01")`：设置事实开始为真的时间。
- `Memory.invalidate(memory_id)`：手动标记事实不再为真（设置 invalid_at + expired_at，不删除）。
- `Memory.search_at(reference_time, query, user_id)`：时态查询，过滤 `valid_at <= time AND (invalid_at IS NULL OR invalid_at > time)`。
- `valid_at=None` 的记忆视为"无时间约束"，search_at 始终返回（向后兼容）。
- LLM 自动矛盾检测留给 P3-Task 3（在 add() 中插入矛盾检测步骤，不改存储层）。
```

- [ ] **Step 6: Update progress**

Append to `.sdd/progress.md`:

```
Task 4: complete (e2e 3 tests + CHANGELOG + AGENTS.md)

## P2-Task 1 Bitemporal Complete: 831 passed, 36 skipped, ZERO REGRESSION from P1 baseline (800)
- Schema: valid_at/invalid_at/expired_at + migration + temporal index
- Store: add(valid_at=) + get_temporal_valid() + invalidate()
- Memory facade: add(valid_at=) + invalidate() + search_at()
- CLI/REST/MCP: valid_at param + invalidate + search --at
- 31 new tests (13 store + 10 facade + 5 cli/rest + 3 e2e)
```

---

## Self-Review

**1. Spec coverage:**
- Section 2 (Schema migration) → Task 1 Step 3a ✅
- Section 2.2 (调用时机) → Task 1 Step 3b ✅
- Section 3.1 (add valid_at) → Task 2 Step 3a ✅
- Section 3.2 (invalidate) → Task 1 Step 3e (store) + Task 2 Step 3b (facade) ✅
- Section 3.3 (search_at) → Task 2 Step 3c ✅
- Section 3.4 (get_temporal_valid) → Task 1 Step 3d ✅
- Section 4 (CLI/REST/MCP) → Task 3 ✅
- Section 5 (Testing) → Tasks 1-4 ✅
- Section 6 (File changes) → All covered ✅

**2. Placeholder scan:** No TBD/TODO in steps. ✅

**3. Type consistency:** `get_temporal_valid` consistent across all tasks. `invalidate` signature consistent (store: `invalidate(memory_id, *, invalid_at=None)`, facade: `invalidate(memory_id, *, invalid_at=None)`). ✅

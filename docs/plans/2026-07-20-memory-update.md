# SeptMuse 记忆更新能力实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现长时记忆 update（对齐 mem0）+ 工作记忆 Block replace 持久化（对齐 Letta + 架构 §11.2），覆盖 Memory facade + REST + CLI 三层。

**Architecture:** store 层加 update 抽象 + SQLite 实现；TypedMemoryStore 加 Block CRUD；WorkingMemory 加 store 参数实现自动持久化（store=None 回退纯内存向后兼容）；Memory facade 包装暴露；REST 加 PUT 端点；CLI 加 update/block 子命令。

**Tech Stack:** Python 3.10+, SQLModel, SQLite, FastAPI, argparse

## Global Constraints

- ruff line-length 120，规则 `E F I W UP B SIM RUF`（忽略 E501 RUF002 RUF003）
- PYTHONPATH=src 运行 pytest（包未 pip install -e .）
- 项目非 git repo，无 commit 步骤，用 "验证" 替代
- HashEmbedder 默认（CLI 测试用，零模型加载）
- 现有测试不退化：test_block.py 的 WorkingMemory 纯内存模式（store=None）必须全绿
- TypedMemoryStore Session 模式：`with Session(self.engine) as session: session.add(); session.commit(); session.refresh()`
- Block 是 `SQLModel, table=True`，表名 `septmuse_blocks`，已有 `id/agent_id/label/value/limit/read_only/tags/created_at/updated_at` 字段
- SQLiteMemoryStore 已有 `history` 表（字段：id/memory_id/old_memory/new_memory/event/created_at/is_deleted）
- Memory facade 签名：`add`/`search`/`get_all`/`get`/`delete` 已有，本次加 `update` + block 方法

---

### Task 1: store 层 — MemoryStore ABC update + SQLiteMemoryStore 实现

**Files:**
- Modify: `src/septmuse/storage/base.py`（MemoryStore ABC 加 update 抽象方法）
- Modify: `src/septmuse/storage/sqlite/store.py`（SQLiteMemoryStore 加 update 实现）
- Test: `tests/unit/test_update.py`（新建）

**Interfaces:**
- Produces: `MemoryStore.update(memory_id: str, content: str, embedding: list[float], *, metadata: dict[str, Any] | None = None) -> bool` — True=成功, False=不存在/已删除

- [ ] **Step 1: 写 update 的失败测试**

创建 `tests/unit/test_update.py`：

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
"""记忆更新能力测试 — 长时记忆 update + Block 持久化。"""

from __future__ import annotations

import json

import numpy as np

from septmuse.providers.embedders.hash import HashEmbedder
from septmuse.storage.sqlite.store import SQLiteMemoryStore


class TestStoreUpdate:
    def test_update_content(self, tmp_path):
        store = SQLiteMemoryStore(db_path=str(tmp_path / "test.db"))
        mid = store.add("旧内容", [1.0, 0.0, 0.0], user_id="alice")
        ok = store.update(mid, "新内容", [0.0, 1.0, 0.0])
        assert ok is True
        result = store.get(mid)
        assert result["memory"] == "新内容"
        store.close()

    def test_update_not_found(self, tmp_path):
        store = SQLiteMemoryStore(db_path=str(tmp_path / "test.db"))
        ok = store.update("nonexistent", "x", [1.0])
        assert ok is False
        store.close()

    def test_update_deleted_returns_false(self, tmp_path):
        store = SQLiteMemoryStore(db_path=str(tmp_path / "test.db"))
        mid = store.add("内容", [1.0], user_id="alice")
        store.delete(mid)
        ok = store.update(mid, "新", [0.0])
        assert ok is False
        store.close()

    def test_update_history_recorded(self, tmp_path):
        store = SQLiteMemoryStore(db_path=str(tmp_path / "test.db"))
        mid = store.add("旧内容", [1.0], user_id="alice")
        store.update(mid, "新内容", [0.0])
        with store._lock:
            cur = store.conn.execute(
                "SELECT event, old_memory, new_memory FROM history WHERE memory_id=? AND event='UPDATE'",
                (mid,),
            )
            rows = cur.fetchall()
        assert len(rows) == 1
        assert rows[0][0] == "UPDATE"
        assert rows[0][1] == "旧内容"
        assert rows[0][2] == "新内容"
        store.close()

    def test_update_metadata_only(self, tmp_path):
        store = SQLiteMemoryStore(db_path=str(tmp_path / "test.db"))
        mid = store.add("内容", [1.0], user_id="alice", metadata={"k": "v1"})
        ok = store.update(mid, "内容", [1.0], metadata={"k": "v2"})
        assert ok is True
        result = store.get(mid)
        assert result["metadata"]["k"] == "v2"
        store.close()
```

- [ ] **Step 2: 运行测试验证失败**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_update.py::TestStoreUpdate -v 2>&1 | Select-Object -Last 10`
Expected: FAIL（`MemoryStore` 无 `update` 方法 → `AttributeError`）

- [ ] **Step 3: MemoryStore ABC 加 update 抽象方法**

在 `src/septmuse/storage/base.py` 的 `MemoryStore` 类，在 `delete` 方法之后、`close` 方法之前加：

```python
    @abstractmethod
    def update(
        self,
        memory_id: str,
        content: str,
        embedding: list[float],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """更新记忆 content + embedding + metadata, 记录 history。

        Returns:
            True = 更新成功; False = memory_id 不存在或已删除
        """
        ...
```

- [ ] **Step 4: SQLiteMemoryStore 加 update 实现**

在 `src/septmuse/storage/sqlite/store.py` 的 `SQLiteMemoryStore` 类，在 `delete` 方法之后、`close` 方法之前加：

```python
    def update(
        self,
        memory_id: str,
        content: str,
        embedding: list[float],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """更新记忆 (对齐 mem0 update — UPDATE + history)。"""
        now = _utcnow_iso()
        with self._lock:
            cur = self.conn.execute(
                "SELECT content, metadata FROM memories WHERE id=? AND is_deleted=0",
                (memory_id,),
            )
            row = cur.fetchone()
            if not row:
                return False
            old_content, old_meta_json = row
            old_meta = json.loads(old_meta_json) if old_meta_json else {}

            try:
                self.conn.execute("BEGIN")
                self.conn.execute(
                    """UPDATE memories
                       SET content=?, embedding=?, metadata=?, updated_at=?
                       WHERE id=? AND is_deleted=0""",
                    (
                        content,
                        json.dumps(embedding),
                        json.dumps(metadata if metadata is not None else old_meta),
                        now,
                        memory_id,
                    ),
                )
                self.conn.execute(
                    """INSERT INTO history (id, memory_id, old_memory, new_memory, event, created_at, is_deleted)
                       VALUES (?, ?, ?, ?, ?, ?, 0)""",
                    (str(uuid.uuid4()), memory_id, old_content, content, "UPDATE", now),
                )
                self.conn.execute("COMMIT")
            except Exception:
                self.conn.execute("ROLLBACK")
                raise
        logger.info("memory_updated", memory_id=memory_id, content_len=len(content))
        return True
```

- [ ] **Step 5: 运行测试验证通过**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_update.py::TestStoreUpdate -v 2>&1 | Select-Object -Last 10`
Expected: 5 passed

- [ ] **Step 6: ruff + 全回归验证**

Run: `$env:PYTHONPATH="src"; ruff check src/septmuse/storage/ tests/unit/test_update.py`
Expected: `All checks passed!`

Run: `$env:PYTHONPATH="src"; python -m pytest tests/ -q --ignore=tests/unit/test_rbac_rest_openai.py -k "not test_mount_routes" 2>&1 | Select-Object -Last 3`
Expected: 507 passed（502 + 5 新增）, 9 skipped, 1 deselected

---

### Task 2: facade 层 — Memory.update 方法

**Files:**
- Modify: `src/septmuse/orchestration/memory.py`（Memory 类加 update 方法）
- Test: `tests/unit/test_update.py`（追加 TestFacadeUpdate）

**Interfaces:**
- Consumes: `MemoryStore.update`（Task 1 产出）
- Produces: `Memory.update(memory_id: str, data: str | None = None, *, user_id: str | None = None, metadata: dict[str, Any] | None = None) -> dict[str, Any]` — 返回 `{"id", "memory", "event": "UPDATE"}` 或 `{"id", "event": "NOT_FOUND"}`

- [ ] **Step 1: 写 facade update 的失败测试**

在 `tests/unit/test_update.py` 末尾追加：

```python
from septmuse import Memory, MemoryConfig
from septmuse.providers.embedders.hash import HashEmbedder


def _make_memory(tmp_path):
    return Memory(
        config=MemoryConfig(db_path=str(tmp_path / "test.db")),
        embedder=HashEmbedder(),
    )


class TestFacadeUpdate:
    def test_update_content(self, tmp_path):
        m = _make_memory(tmp_path)
        result = m.add("旧内容", user_id="alice")
        mid = result["results"][0]["id"]
        updated = m.update(mid, "新内容")
        assert updated["event"] == "UPDATE"
        assert updated["memory"] == "新内容"
        # 验证可检索新内容
        hits = m.search("新内容", user_id="alice")
        assert any(h["memory"] == "新内容" for h in hits)

    def test_update_not_found(self, tmp_path):
        m = _make_memory(tmp_path)
        result = m.update("nonexistent", "x")
        assert result["event"] == "NOT_FOUND"

    def test_update_metadata_only(self, tmp_path):
        m = _make_memory(tmp_path)
        result = m.add("内容", user_id="alice")
        mid = result["results"][0]["id"]
        updated = m.update(mid, metadata={"tag": "important"})
        assert updated["event"] == "UPDATE"
        # 验证 metadata 更新
        item = m.get(mid)
        assert item["metadata"]["tag"] == "important"

    def test_update_re_embedding(self, tmp_path):
        m = _make_memory(tmp_path)
        result = m.add("python programming", user_id="alice")
        mid = result["results"][0]["id"]
        # update 为完全不同的内容
        m.update(mid, "cooking recipes")
        # 旧 query 不应高匹配
        old_hits = m.search("python", user_id="alice", threshold=0.5)
        assert not any(h["id"] == mid for h in old_hits)
```

- [ ] **Step 2: 运行测试验证失败**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_update.py::TestFacadeUpdate -v 2>&1 | Select-Object -Last 10`
Expected: FAIL（`Memory` 无 `update` 方法 → `AttributeError`）

- [ ] **Step 3: 实现 Memory.update**

在 `src/septmuse/orchestration/memory.py` 的 `Memory` 类，在 `delete` 方法之后（约 line 212）加：

```python
    def update(
        self,
        memory_id: str,
        data: str | None = None,
        *,
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """更新记忆 (对齐 mem0 update)。

        Args:
            memory_id: 记忆 ID
            data: 新内容; None=不改 content, 只改 metadata
            user_id: 用户 ID (仅用于日志, 不做归属权验证)
            metadata: 新 metadata; None=不改

        Returns:
            {"id", "memory", "event": "UPDATE"} 或 {"id", "event": "NOT_FOUND"}
        """
        existing = self.store.get(memory_id)
        if existing is None:
            return {"id": memory_id, "event": "NOT_FOUND"}

        new_content = data if data is not None else existing["memory"]
        new_embedding = self.embedder.embed(new_content)

        if metadata is not None:
            merged_meta = {**existing.get("metadata", {}), **metadata}
        else:
            merged_meta = existing.get("metadata", {})

        ok = self.store.update(memory_id, new_content, new_embedding, metadata=merged_meta)
        if not ok:
            return {"id": memory_id, "event": "NOT_FOUND"}
        logger.info("memory_update_done", memory_id=memory_id, user_id=user_id)
        return {"id": memory_id, "memory": new_content, "event": "UPDATE"}
```

- [ ] **Step 4: 运行测试验证通过**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_update.py::TestFacadeUpdate -v 2>&1 | Select-Object -Last 10`
Expected: 4 passed

- [ ] **Step 5: ruff + 全回归**

Run: `$env:PYTHONPATH="src"; ruff check src/septmuse/orchestration/memory.py tests/unit/test_update.py`
Expected: `All checks passed!`

Run: `$env:PYTHONPATH="src"; python -m pytest tests/ -q --ignore=tests/unit/test_rbac_rest_openai.py -k "not test_mount_routes" 2>&1 | Select-Object -Last 3`
Expected: 511 passed（507 + 4 新增）, 9 skipped, 1 deselected

---

### Task 3: TypedMemoryStore — Block CRUD

**Files:**
- Modify: `src/septmuse/storage/typed_store.py`（import Block + 加 5 个 Block 方法）
- Test: `tests/unit/test_update.py`（追加 TestBlockStore）

**Interfaces:**
- Produces:
  - `TypedMemoryStore.get_blocks(agent_id: str) -> list[Block]`
  - `TypedMemoryStore.save_block(block: Block) -> Block`
  - `TypedMemoryStore.update_block_value(agent_id: str, label: str, value: str) -> Block | None`
  - `TypedMemoryStore.delete_block(agent_id: str, label: str) -> bool`
  - `TypedMemoryStore.ensure_default_blocks(agent_id: str) -> list[Block]`

- [ ] **Step 1: 写 Block CRUD 的失败测试**

在 `tests/unit/test_update.py` 末尾追加：

```python
from septmuse.schemas.block import Block, default_blocks
from septmuse.storage.typed_store import TypedMemoryStore


class TestBlockStore:
    def test_save_and_get_blocks(self, tmp_path):
        store = TypedMemoryStore(db_path=str(tmp_path / "test.db"))
        block = Block(agent_id="agent-1", label="human", value="Name: Alice")
        store.save_block(block)
        loaded = store.get_blocks("agent-1")
        assert len(loaded) == 1
        assert loaded[0].label == "human"
        assert loaded[0].value == "Name: Alice"

    def test_save_block_update_existing(self, tmp_path):
        store = TypedMemoryStore(db_path=str(tmp_path / "test.db"))
        block = Block(agent_id="agent-1", label="human", value="v1")
        store.save_block(block)
        block.value = "v2"
        store.save_block(block)
        loaded = store.get_blocks("agent-1")
        assert len(loaded) == 1
        assert loaded[0].value == "v2"

    def test_update_block_value(self, tmp_path):
        store = TypedMemoryStore(db_path=str(tmp_path / "test.db"))
        store.save_block(Block(agent_id="agent-1", label="human", value="old"))
        result = store.update_block_value("agent-1", "human", "new")
        assert result is not None
        assert result.value == "new"

    def test_update_block_value_not_found(self, tmp_path):
        store = TypedMemoryStore(db_path=str(tmp_path / "test.db"))
        result = store.update_block_value("agent-1", "nonexistent", "x")
        assert result is None

    def test_delete_block(self, tmp_path):
        store = TypedMemoryStore(db_path=str(tmp_path / "test.db"))
        store.save_block(Block(agent_id="agent-1", label="human", value="x"))
        ok = store.delete_block("agent-1", "human")
        assert ok is True
        assert store.get_blocks("agent-1") == []

    def test_delete_block_not_found(self, tmp_path):
        store = TypedMemoryStore(db_path=str(tmp_path / "test.db"))
        ok = store.delete_block("agent-1", "nonexistent")
        assert ok is False

    def test_ensure_default_blocks(self, tmp_path):
        store = TypedMemoryStore(db_path=str(tmp_path / "test.db"))
        blocks = store.ensure_default_blocks("agent-1")
        assert len(blocks) == 2
        labels = [b.label for b in blocks]
        assert "human" in labels
        assert "persona" in labels

    def test_ensure_default_blocks_idempotent(self, tmp_path):
        store = TypedMemoryStore(db_path=str(tmp_path / "test.db"))
        store.ensure_default_blocks("agent-1")
        blocks = store.ensure_default_blocks("agent-1")
        assert len(blocks) == 2  # 不重复创建
```

- [ ] **Step 2: 运行测试验证失败**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_update.py::TestBlockStore -v 2>&1 | Select-Object -Last 10`
Expected: FAIL（`TypedMemoryStore` 无 `save_block`/`get_blocks` 方法 → `AttributeError`）

- [ ] **Step 3: 实现 Block CRUD**

在 `src/septmuse/storage/typed_store.py`：

1. 顶部 import 加 Block（在 `from septmuse.schemas.strength import MemoryStrength` 之后）：

```python
from septmuse.schemas.block import Block, default_blocks
```

2. 在 `TypedMemoryStore` 类末尾（`__del__` 之前或最后一个方法之后）加：

```python
    # ------------------------------------------------------------------
    # 工作记忆 Block CRUD (架构文档 §3.1.1, 对齐 Letta Block)
    # ------------------------------------------------------------------

    def get_blocks(self, agent_id: str) -> list[Block]:
        """加载 agent 的全部 block (持久化 → 内存)。"""
        with Session(self.engine) as session:
            stmt = select(Block).where(Block.agent_id == agent_id)
            return list(session.exec(stmt).all())

    def save_block(self, block: Block) -> Block:
        """保存 block (INSERT or UPDATE, 按 id upsert)。"""
        with Session(self.engine) as session:
            existing = session.get(Block, block.id)
            if existing:
                existing.label = block.label
                existing.value = block.value
                existing.limit = block.limit
                existing.read_only = block.read_only
                existing.tags = block.tags
                existing.touch()
                session.add(existing)
            else:
                session.add(block)
            session.commit()
            session.refresh(existing or block)
            return existing or block

    def update_block_value(self, agent_id: str, label: str, value: str) -> Block | None:
        """更新 block value (对齐 WorkingMemory.update_block_value)。"""
        with Session(self.engine) as session:
            stmt = select(Block).where(Block.agent_id == agent_id, Block.label == label)
            block = session.exec(stmt).first()
            if not block:
                return None
            block.value = value
            block.touch()
            session.add(block)
            session.commit()
            session.refresh(block)
            return block

    def delete_block(self, agent_id: str, label: str) -> bool:
        """删除 block。"""
        with Session(self.engine) as session:
            stmt = select(Block).where(Block.agent_id == agent_id, Block.label == label)
            block = session.exec(stmt).first()
            if not block:
                return False
            session.delete(block)
            session.commit()
            return True

    def ensure_default_blocks(self, agent_id: str) -> list[Block]:
        """确保 agent 有默认 block (human + persona), 无则创建。"""
        blocks = self.get_blocks(agent_id)
        if blocks:
            return blocks
        for block in default_blocks(agent_id):
            self.save_block(block)
        return self.get_blocks(agent_id)
```

- [ ] **Step 4: 运行测试验证通过**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_update.py::TestBlockStore -v 2>&1 | Select-Object -Last 10`
Expected: 8 passed

- [ ] **Step 5: ruff + 全回归**

Run: `$env:PYTHONPATH="src"; ruff check src/septmuse/storage/typed_store.py tests/unit/test_update.py`
Expected: `All checks passed!`

Run: `$env:PYTHONPATH="src"; python -m pytest tests/ -q --ignore=tests/unit/test_rbac_rest_openai.py -k "not test_mount_routes" 2>&1 | Select-Object -Last 3`
Expected: 519 passed（511 + 8 新增）, 9 skipped, 1 deselected

---

### Task 4: WorkingMemory — store 参数 + 自动持久化

**Files:**
- Modify: `src/septmuse/content_types/working/block.py`（WorkingMemory 加 store 参数）
- Test: `tests/unit/test_update.py`（追加 TestWorkingMemoryPersist）

**Interfaces:**
- Consumes: `TypedMemoryStore.save_block`（Task 3 产出）
- Produces: `WorkingMemory.__init__(agent_id, blocks=None, store=None)` — store 非空时 set_block/update_block_value/core_memory_append/core_memory_replace 自动调 store.save_block

- [ ] **Step 1: 写 WorkingMemory 持久化的失败测试**

在 `tests/unit/test_update.py` 末尾追加：

```python
from septmuse.content_types.working.block import WorkingMemory


class TestWorkingMemoryPersist:
    def test_store_none_backward_compat(self):
        """store=None 纯内存模式 (向后兼容现有 test_block.py)。"""
        wm = WorkingMemory(agent_id="agent-1")
        wm.update_block_value("human", "value")
        assert wm.get_block("human").value == "value"

    def test_update_block_value_persists(self, tmp_path):
        store = TypedMemoryStore(db_path=str(tmp_path / "test.db"))
        store.ensure_default_blocks("agent-1")
        blocks = store.get_blocks("agent-1")
        wm = WorkingMemory("agent-1", blocks=blocks, store=store)
        wm.update_block_value("human", "persisted value")
        # 重新 load 验证持久化
        reloaded = store.get_blocks("agent-1")
        human = [b for b in reloaded if b.label == "human"][0]
        assert human.value == "persisted value"

    def test_core_memory_append_persists(self, tmp_path):
        store = TypedMemoryStore(db_path=str(tmp_path / "test.db"))
        store.ensure_default_blocks("agent-1")
        blocks = store.get_blocks("agent-1")
        wm = WorkingMemory("agent-1", blocks=blocks, store=store)
        wm.core_memory_append("human", "Name: Alice")
        reloaded = store.get_blocks("agent-1")
        human = [b for b in reloaded if b.label == "human"][0]
        assert "Name: Alice" in human.value

    def test_core_memory_replace_persists(self, tmp_path):
        store = TypedMemoryStore(db_path=str(tmp_path / "test.db"))
        store.ensure_default_blocks("agent-1")
        blocks = store.get_blocks("agent-1")
        wm = WorkingMemory("agent-1", blocks=blocks, store=store)
        wm.core_memory_append("human", "Likes: Python, hiking")
        wm.core_memory_replace("human", "hiking", "skiing")
        reloaded = store.get_blocks("agent-1")
        human = [b for b in reloaded if b.label == "human"][0]
        assert "skiing" in human.value
        assert "hiking" not in human.value

    def test_set_block_persists(self, tmp_path):
        store = TypedMemoryStore(db_path=str(tmp_path / "test.db"))
        wm = WorkingMemory("agent-1", store=store)
        wm.set_block(Block(agent_id="agent-1", label="task", value="do x"))
        reloaded = store.get_blocks("agent-1")
        labels = [b.label for b in reloaded]
        assert "task" in labels
```

- [ ] **Step 2: 运行测试验证失败**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_update.py::TestWorkingMemoryPersist -v 2>&1 | Select-Object -Last 10`
Expected: FAIL（`WorkingMemory.__init__` 不接受 `store` 参数 → `TypeError`）

- [ ] **Step 3: 修改 WorkingMemory 加 store 参数 + 自动持久化**

修改 `src/septmuse/content_types/working/block.py`：

1. 顶部加 `from typing import Any`（已有则跳过）：

```python
from typing import Any
```

2. `__init__` 加 `store` 参数：

```python
    def __init__(
        self,
        agent_id: str,
        blocks: list[Block] | None = None,
        store: Any | None = None,
    ) -> None:
        """初始化工作记忆。

        Args:
            agent_id: 归属 agent ID (跨 agent 共享键)
            blocks: 初始块列表; None 时用 default_blocks(agent_id)
            store: TypedMemoryStore | None; 非空时操作后自动持久化
        """
        self.agent_id = agent_id
        self.store = store
        self.blocks: list[Block] = blocks if blocks is not None else default_blocks(agent_id)
        logger.debug("working_memory_init", agent_id=agent_id, block_count=len(self.blocks))
```

3. `set_block` 加持久化（在 `self.blocks[i] = block` 和 `self.blocks.append(block)` 之后）：

```python
    def set_block(self, block: Block) -> None:
        """设置块: 同 label 替换, 否则 append (对齐 letta set_block)。"""
        for i, b in enumerate(self.blocks):
            if b.label == block.label:
                self.blocks[i] = block
                if self.store is not None:
                    self.store.save_block(block)
                return
        self.blocks.append(block)
        if self.store is not None:
            self.store.save_block(block)
```

4. `update_block_value` 加持久化（在 `block.value = value; block.touch()` 之后）：

```python
    def update_block_value(self, label: str, value: str) -> None:
        """更新块 value (对齐 letta update_block_value)。

        Raises:
            ValueError: value 非 str 或 label 不存在
        """
        if not isinstance(value, str):
            raise ValueError("Provided value must be a string")
        for block in self.blocks:
            if block.label == label:
                block.value = value
                block.touch()
                if self.store is not None:
                    self.store.save_block(block)
                return
        raise ValueError(f"Block with label {label} does not exist")
```

5. `core_memory_append` 加持久化（在 `self.update_block_value(...)` 之后。注意 `update_block_value` 已持久化，所以 `core_memory_append` 不需要额外持久化）—— **无需改动**，因为 `core_memory_append` 调 `update_block_value`，后者已持久化。

6. `core_memory_replace` 同理—— **无需改动**，因为 `core_memory_replace` 也调 `update_block_value`。

- [ ] **Step 4: 运行测试验证通过**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_update.py::TestWorkingMemoryPersist -v 2>&1 | Select-Object -Last 10`
Expected: 5 passed

- [ ] **Step 5: 验证现有 test_block.py 不退化**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_block.py -v 2>&1 | Select-Object -Last 10`
Expected: 全部 passed（store=None 向后兼容）

- [ ] **Step 6: ruff + 全回归**

Run: `$env:PYTHONPATH="src"; ruff check src/septmuse/content_types/working/block.py tests/unit/test_update.py`
Expected: `All checks passed!`

Run: `$env:PYTHONPATH="src"; python -m pytest tests/ -q --ignore=tests/unit/test_rbac_rest_openai.py -k "not test_mount_routes" 2>&1 | Select-Object -Last 3`
Expected: 524 passed（519 + 5 新增）, 9 skipped, 1 deselected

---

### Task 5: facade 层 — get_working_memory + block 方法

**Files:**
- Modify: `src/septmuse/orchestration/memory.py`（Memory 加 block 方法 + import WorkingMemory）
- Test: `tests/unit/test_update.py`（追加 TestFacadeBlock）

**Interfaces:**
- Consumes: `TypedMemoryStore.ensure_default_blocks`（Task 3）+ `WorkingMemory(store=)`（Task 4）
- Produces:
  - `Memory.get_working_memory(agent_id: str) -> WorkingMemory`
  - `Memory.get_blocks(agent_id: str) -> list[dict[str, Any]]`
  - `Memory.update_block(agent_id: str, label: str, value: str) -> dict[str, Any]`
  - `Memory.core_memory_append(agent_id: str, label: str, content: str) -> dict[str, Any]`
  - `Memory.core_memory_replace(agent_id: str, label: str, old_content: str, new_content: str) -> dict[str, Any]`

- [ ] **Step 1: 写 facade block 的失败测试**

在 `tests/unit/test_update.py` 末尾追加：

```python
class TestFacadeBlock:
    def test_get_blocks_creates_defaults(self, tmp_path):
        m = _make_memory(tmp_path)
        blocks = m.get_blocks("agent-1")
        labels = [b["label"] for b in blocks]
        assert "human" in labels
        assert "persona" in labels

    def test_update_block(self, tmp_path):
        m = _make_memory(tmp_path)
        result = m.update_block("agent-1", "human", "Name: Alice")
        assert result["event"] == "UPDATE"
        assert result["value"] == "Name: Alice"
        # 验证持久化 — 新 Memory 实例读同一 db
        m2 = _make_memory(tmp_path)
        blocks = m2.get_blocks("agent-1")
        human = [b for b in blocks if b["label"] == "human"][0]
        assert human["value"] == "Name: Alice"

    def test_core_memory_append(self, tmp_path):
        m = _make_memory(tmp_path)
        m.update_block("agent-1", "human", "Name: Alice")
        result = m.core_memory_append("agent-1", "human", "Likes: Python")
        assert result["event"] == "APPEND"
        assert "Name: Alice" in result["value"]
        assert "Likes: Python" in result["value"]

    def test_core_memory_replace(self, tmp_path):
        m = _make_memory(tmp_path)
        m.update_block("agent-1", "human", "Likes: Python, hiking")
        result = m.core_memory_replace("agent-1", "human", "hiking", "skiing")
        assert result["event"] == "REPLACE"
        assert "skiing" in result["value"]
        assert "hiking" not in result["value"]
```

- [ ] **Step 2: 运行测试验证失败**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_update.py::TestFacadeBlock -v 2>&1 | Select-Object -Last 10`
Expected: FAIL（`Memory` 无 `get_blocks`/`update_block` 方法 → `AttributeError`）

- [ ] **Step 3: 实现 facade block 方法**

在 `src/septmuse/orchestration/memory.py`：

1. 顶部 import 加 WorkingMemory（在 `from septmuse.content_types.procedural import ProceduralMemory` 之后）：

```python
from septmuse.content_types.working.block import WorkingMemory
```

2. 在 `Memory` 类的 `update` 方法之后（Task 2 加的）加 block 方法：

```python
    # ------------------------------------------------------------------
    # 工作记忆 Block (架构文档 §3.1.1, 对齐 Letta Block)
    # ------------------------------------------------------------------

    def get_working_memory(self, agent_id: str) -> WorkingMemory:
        """获取 agent 的工作记忆 (从 TypedMemoryStore 加载, 自动持久化)。"""
        blocks = self.typed_store.ensure_default_blocks(agent_id)
        return WorkingMemory(agent_id, blocks=blocks, store=self.typed_store)

    def get_blocks(self, agent_id: str) -> list[dict[str, Any]]:
        """列出 agent 的全部 block。"""
        wm = self.get_working_memory(agent_id)
        return [b.model_dump() for b in wm.blocks]

    def update_block(self, agent_id: str, label: str, value: str) -> dict[str, Any]:
        """更新 block value (对齐 Letta update_block_value)。"""
        wm = self.get_working_memory(agent_id)
        wm.update_block_value(label, value)
        block = wm.get_block(label)
        return {"id": block.id, "label": block.label, "value": block.value, "event": "UPDATE"}

    def core_memory_append(self, agent_id: str, label: str, content: str) -> dict[str, Any]:
        """追加 block 内容 (对齐 Letta core_memory_append)。"""
        wm = self.get_working_memory(agent_id)
        wm.core_memory_append(label, content)
        block = wm.get_block(label)
        return {"id": block.id, "label": block.label, "value": block.value, "event": "APPEND"}

    def core_memory_replace(self, agent_id: str, label: str, old_content: str, new_content: str) -> dict[str, Any]:
        """替换 block 内容片段 (对齐 Letta core_memory_replace)。"""
        wm = self.get_working_memory(agent_id)
        wm.core_memory_replace(label, old_content, new_content)
        block = wm.get_block(label)
        return {"id": block.id, "label": block.label, "value": block.value, "event": "REPLACE"}
```

- [ ] **Step 4: 运行测试验证通过**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_update.py::TestFacadeBlock -v 2>&1 | Select-Object -Last 10`
Expected: 4 passed

- [ ] **Step 5: ruff + 全回归**

Run: `$env:PYTHONPATH="src"; ruff check src/septmuse/orchestration/memory.py tests/unit/test_update.py`
Expected: `All checks passed!`

Run: `$env:PYTHONPATH="src"; python -m pytest tests/ -q --ignore=tests/unit/test_rbac_rest_openai.py -k "not test_mount_routes" 2>&1 | Select-Object -Last 3`
Expected: 528 passed（524 + 4 新增）, 9 skipped, 1 deselected

---

### Task 6: REST — PUT /memories/{id} + Block REST 端点

**Files:**
- Modify: `src/septmuse/api/rest/__init__.py`（加 PUT 端点 + Block REST 端点）
- Test: `tests/unit/test_update.py`（追加 TestRestUpdate）

**Interfaces:**
- Consumes: `Memory.update`（Task 2）+ `Memory.get_blocks/update_block/core_memory_append/core_memory_replace`（Task 5）

- [ ] **Step 1: 写 REST 的失败测试**

在 `tests/unit/test_update.py` 末尾追加：

```python
from fastapi.testclient import TestClient

from septmuse.api.rest import create_app


def _make_app(tmp_path):
    m = _make_memory(tmp_path)
    return create_app(m), m


class TestRestUpdate:
    def test_put_memory(self, tmp_path):
        app, m = _make_app(tmp_path)
        result = m.add("旧内容", user_id="alice")
        mid = result["results"][0]["id"]
        client = TestClient(app)
        resp = client.put(f"/memories/{mid}", json={"text": "新内容"})
        assert resp.status_code == 200
        assert resp.json()["event"] == "UPDATE"

    def test_put_memory_not_found(self, tmp_path):
        app, _ = _make_app(tmp_path)
        client = TestClient(app)
        resp = client.put("/memories/nonexistent", json={"text": "x"})
        assert resp.status_code == 404

    def test_get_blocks(self, tmp_path):
        app, _ = _make_app(tmp_path)
        client = TestClient(app)
        resp = client.get("/memories/working/blocks/agent-1")
        assert resp.status_code == 200
        labels = [b["label"] for b in resp.json()]
        assert "human" in labels

    def test_put_block(self, tmp_path):
        app, _ = _make_app(tmp_path)
        client = TestClient(app)
        resp = client.put("/memories/working/blocks/agent-1/human", json={"value": "Name: Alice"})
        assert resp.status_code == 200
        assert resp.json()["value"] == "Name: Alice"

    def test_post_append(self, tmp_path):
        app, m = _make_app(tmp_path)
        m.update_block("agent-1", "human", "Name: Alice")
        client = TestClient(app)
        resp = client.post("/memories/working/blocks/agent-1/human/append", json={"content": "Likes: Python"})
        assert resp.status_code == 200
        assert "Likes: Python" in resp.json()["value"]

    def test_post_replace(self, tmp_path):
        app, m = _make_app(tmp_path)
        m.update_block("agent-1", "human", "Likes: Python, hiking")
        client = TestClient(app)
        resp = client.post(
            "/memories/working/blocks/agent-1/human/replace",
            json={"old_content": "hiking", "new_content": "skiing"},
        )
        assert resp.status_code == 200
        assert "skiing" in resp.json()["value"]
```

- [ ] **Step 2: 运行测试验证失败**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_update.py::TestRestUpdate -v 2>&1 | Select-Object -Last 10`
Expected: FAIL（无 PUT 端点 → 405 Method Not Allowed）

- [ ] **Step 3: 实现 REST 端点**

在 `src/septmuse/api/rest/__init__.py` 的 `register_routes` 函数内：

1. 在 Pydantic 请求模型区域（`CaptureRequest` 之后）加：

```python
class UpdateMemoryRequest(BaseModel):
    text: str | None = Field(default=None, description="新内容")
    metadata: dict[str, Any] | None = Field(default=None, description="新 metadata")


class BlockUpdateRequest(BaseModel):
    value: str = Field(description="新 block 内容")


class BlockAppendRequest(BaseModel):
    content: str = Field(description="追加内容")


class BlockReplaceRequest(BaseModel):
    old_content: str = Field(description="被替换的旧内容")
    new_content: str = Field(description="新内容")
```

2. 在 `register_routes` 函数内（`health` 端点之前）加路由：

```python
    @app.put("/memories/{memory_id}")
    async def update_memory(memory_id: str, req: UpdateMemoryRequest) -> dict[str, Any]:
        """更新记忆 (对齐 mem0 PUT /memories/{id})。"""
        result = app.state.memory.update(memory_id, req.text, metadata=req.metadata)
        if result.get("event") == "NOT_FOUND":
            raise HTTPException(status_code=404, detail=f"memory {memory_id} not found")
        return result

    @app.get("/memories/working/blocks/{agent_id}")
    async def list_blocks(agent_id: str) -> list[dict[str, Any]]:
        """列出 agent 的全部 block。"""
        return app.state.memory.get_blocks(agent_id)

    @app.put("/memories/working/blocks/{agent_id}/{label}")
    async def update_block(agent_id: str, label: str, req: BlockUpdateRequest) -> dict[str, Any]:
        """更新 block value (架构 §11.2)。"""
        return app.state.memory.update_block(agent_id, label, req.value)

    @app.post("/memories/working/blocks/{agent_id}/{label}/append")
    async def append_block(agent_id: str, label: str, req: BlockAppendRequest) -> dict[str, Any]:
        """追加 block 内容 (对齐 Letta core_memory_append)。"""
        return app.state.memory.core_memory_append(agent_id, label, req.content)

    @app.post("/memories/working/blocks/{agent_id}/{label}/replace")
    async def replace_block(agent_id: str, label: str, req: BlockReplaceRequest) -> dict[str, Any]:
        """替换 block 内容片段 (对齐 Letta core_memory_replace)。"""
        return app.state.memory.core_memory_replace(agent_id, label, req.old_content, req.new_content)
```

注意：`PUT /memories/{memory_id}` 路由必须放在 `GET /memories/{memory_id}` 之后（FastAPI 路由匹配顺序），但 `GET /memories/working/blocks/{agent_id}` 必须放在 `GET /memories/{memory_id}` 之前，否则 `working` 会被当作 `memory_id`。调整路由顺序：把 working blocks 路由放在所有 `/memories/{memory_id}` 路由之前。

- [ ] **Step 4: 运行测试验证通过**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_update.py::TestRestUpdate -v 2>&1 | Select-Object -Last 10`
Expected: 6 passed

- [ ] **Step 5: ruff + 全回归**

Run: `$env:PYTHONPATH="src"; ruff check src/septmuse/api/rest/__init__.py tests/unit/test_update.py`
Expected: `All checks passed!`

Run: `$env:PYTHONPATH="src"; python -m pytest tests/ -q --ignore=tests/unit/test_rbac_rest_openai.py -k "not test_mount_routes" 2>&1 | Select-Object -Last 3`
Expected: 534 passed（528 + 6 新增）, 9 skipped, 1 deselected

---

### Task 7: CLI — update + block 子命令

**Files:**
- Modify: `src/septmuse/cli/main.py`（加 update + block 子命令）
- Test: `tests/unit/test_cli.py`（追加 TestCliUpdate + TestCliBlock）

**Interfaces:**
- Consumes: `Memory.update`（Task 2）+ `Memory.get_blocks/update_block`（Task 5）

- [ ] **Step 1: 写 CLI 的失败测试**

在 `tests/unit/test_cli.py` 末尾追加：

```python
class TestCliUpdate:
    def test_update(self, tmp_path, monkeypatch, capsys):
        db = tmp_path / "test.db"
        _run_cli(["init", "--user", "alice", "--db-path", str(db)], monkeypatch, capsys)
        add_rc, add_out, _ = _run_cli(["add", "旧内容", "--user", "alice", "--db-path", str(db)], monkeypatch, capsys)
        mid = json.loads(add_out).get("memory_id")
        rc, out, _ = _run_cli(
            ["update", mid, "新内容", "--db-path", str(db)],
            monkeypatch,
            capsys,
        )
        assert rc == 0
        data = json.loads(out)
        assert data["event"] == "UPDATE"

    def test_update_not_found(self, tmp_path, monkeypatch, capsys):
        db = tmp_path / "test.db"
        _run_cli(["init", "--user", "alice", "--db-path", str(db)], monkeypatch, capsys)
        rc, out, _ = _run_cli(
            ["update", "nonexistent", "x", "--db-path", str(db)],
            monkeypatch,
            capsys,
        )
        assert rc == 0  # 不算失败, 返回 NOT_FOUND
        data = json.loads(out)
        assert data["event"] == "NOT_FOUND"


class TestCliBlock:
    def test_block_set_and_list(self, tmp_path, monkeypatch, capsys):
        db = tmp_path / "test.db"
        _run_cli(["init", "--user", "alice", "--db-path", str(db)], monkeypatch, capsys)
        rc, _, _ = _run_cli(
            ["block", "set", "agent-1", "human", "Name: Alice", "--db-path", str(db)],
            monkeypatch,
            capsys,
        )
        assert rc == 0
        rc, out, _ = _run_cli(
            ["block", "list", "agent-1", "--db-path", str(db)],
            monkeypatch,
            capsys,
        )
        assert rc == 0
        data = json.loads(out)
        labels = [b["label"] for b in data]
        assert "human" in labels
```

- [ ] **Step 2: 运行测试验证失败**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_cli.py::TestCliUpdate tests/unit/test_cli.py::TestCliBlock -v 2>&1 | Select-Object -Last 10`
Expected: FAIL（无 `update`/`block` 子命令 → argparse 报错）

- [ ] **Step 3: 实现 CLI update + block 子命令**

在 `src/septmuse/cli/main.py` 的 `_build_parser` 函数，在 `# serve` 之前加：

```python
    # update
    p_update = sub.add_parser("update", help="更新记忆")
    p_update.add_argument("memory_id", help="记忆 ID")
    p_update.add_argument("content", help="新内容")
    p_update.add_argument("--user", default=os.getenv("SEPTMUSE_USER_ID", "default"), help="用户 ID")
    p_update.add_argument("--db-path", default=None, help="SQLite 路径")
    p_update.set_defaults(func=_cmd_update)

    # block
    p_block = sub.add_parser("block", help="工作记忆 Block 操作")
    block_sub = p_block.add_subparsers(dest="block_cmd", required=True)
    p_block_set = block_sub.add_parser("set", help="设置 block value")
    p_block_set.add_argument("agent_id", help="agent ID")
    p_block_set.add_argument("label", help="block 标签")
    p_block_set.add_argument("value", help="新内容")
    p_block_set.add_argument("--db-path", default=None, help="SQLite 路径")
    p_block_set.set_defaults(func=_cmd_block_set)
    p_block_list = block_sub.add_parser("list", help="列出 block")
    p_block_list.add_argument("agent_id", help="agent ID")
    p_block_list.add_argument("--db-path", default=None, help="SQLite 路径")
    p_block_list.set_defaults(func=_cmd_block_list)
```

然后在文件中加实现函数（在 `_cmd_dump` 之后）：

```python
def _cmd_update(args: argparse.Namespace) -> int:
    """更新记忆。"""
    m = _make_memory(args.db_path)
    result = m.update(args.memory_id, args.content)
    print(json.dumps(result, ensure_ascii=False))
    return 0


def _cmd_block_set(args: argparse.Namespace) -> int:
    """设置 block value。"""
    m = _make_memory(args.db_path)
    result = m.update_block(args.agent_id, args.label, args.value)
    print(json.dumps(result, ensure_ascii=False))
    return 0


def _cmd_block_list(args: argparse.Namespace) -> int:
    """列出 block。"""
    m = _make_memory(args.db_path)
    blocks = m.get_blocks(args.agent_id)
    print(json.dumps(blocks, ensure_ascii=False, indent=2))
    return 0
```

- [ ] **Step 4: 运行测试验证通过**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_cli.py::TestCliUpdate tests/unit/test_cli.py::TestCliBlock -v 2>&1 | Select-Object -Last 10`
Expected: 3 passed

- [ ] **Step 5: ruff + 全回归**

Run: `$env:PYTHONPATH="src"; ruff check src/septmuse/cli/main.py tests/unit/test_cli.py`
Expected: `All checks passed!`

Run: `$env:PYTHONPATH="src"; python -m pytest tests/ -q --ignore=tests/unit/test_rbac_rest_openai.py -k "not test_mount_routes" 2>&1 | Select-Object -Last 3`
Expected: 537 passed（534 + 3 新增）, 9 skipped, 1 deselected

---

### Task 8: 全回归 + 冒烟验证

**Files:**
- 无新文件，仅验证

- [ ] **Step 1: ruff 全量验证**

Run: `$env:PYTHONPATH="src"; ruff check src/ tests/`
Expected: `All checks passed!`

- [ ] **Step 2: 全回归验证**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/ -q --ignore=tests/unit/test_rbac_rest_openai.py -k "not test_mount_routes" 2>&1 | Select-Object -Last 5`
Expected: 537 passed, 9 skipped, 1 deselected

- [ ] **Step 3: 冒烟测试 — CLI**

Run: `$env:PYTHONPATH="src"; python -m septmuse.cli.main --help`
Expected: 列出 init/add/search/dump/update/block/serve/mcp/version 9 命令

Run: `$env:PYTHONPATH="src"; python -c "from septmuse import Memory, MemoryConfig; from septmuse.providers.embedders.hash import HashEmbedder; import tempfile, os; d=tempfile.mkdtemp(); m=Memory(config=MemoryConfig(db_path=os.path.join(d,'t.db')), embedder=HashEmbedder()); r=m.add('hello', user_id='a'); mid=r['results'][0]['id']; print(m.update(mid, 'world')); print(m.update_block('ag1','human','Alice')); print(m.get_blocks('ag1'))"`
Expected: 输出 update + block 操作的 JSON 结果

---

## Self-Review

**1. Spec coverage:**
- §4.1.1 MemoryStore ABC update → Task 1 ✓
- §4.1.2 SQLiteMemoryStore.update → Task 1 ✓
- §4.1.3 Memory facade update → Task 2 ✓
- §4.1.4 REST PUT /memories/{id} → Task 6 ✓
- §4.1.5 CLI update 命令 → Task 7 ✓
- §4.2.1 TypedMemoryStore Block CRUD → Task 3 ✓
- §4.2.2 WorkingMemory store 参数 → Task 4 ✓
- §4.2.3 Memory facade block 方法 → Task 5 ✓
- §4.2.4 REST Block 端点 → Task 6 ✓
- §4.2.5 CLI block 命令 → Task 7 ✓

**2. Placeholder scan:** 无 TBD/TODO，所有代码步骤有完整代码。

**3. Type consistency:**
- `MemoryStore.update(memory_id, content, embedding, *, metadata) -> bool` — Task 1 定义，Task 2 Memory.update 调用 ✓
- `TypedMemoryStore.save_block(block) -> Block` — Task 3 定义，Task 4 WorkingMemory 调用 ✓
- `TypedMemoryStore.ensure_default_blocks(agent_id) -> list[Block]` — Task 3 定义，Task 5 get_working_memory 调用 ✓
- `WorkingMemory(agent_id, blocks=None, store=None)` — Task 4 定义，Task 5 get_working_memory 调用 ✓
- `Memory.update(memory_id, data, *, user_id, metadata) -> dict` — Task 2 定义，Task 6 REST 调用、Task 7 CLI 调用 ✓
- `Memory.update_block(agent_id, label, value) -> dict` — Task 5 定义，Task 6/7 调用 ✓

无问题。

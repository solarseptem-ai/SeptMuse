# SeptMuse 记忆更新能力设计规格

> 日期: 2026-07-20
> 对应架构文档: `agent-memory-architecture.md` §3.1.1 Block + §11.2 API 草案
> 借鉴源: mem0 `update(memory_id, data)` + Letta `core_memory_replace` / `set_block`
> 状态: 待实现

---

## 1. 背景

当前 SeptMuse 缺少记忆更新能力：

- **长时记忆**：Memory facade 有 `add`/`search`/`get`/`delete` 但**无 `update`**。SQLiteMemoryStore 有 `history` 表（old_memory/new_memory/event 字段已就绪）但无 update 方法。mem0 有 `update(memory_id, data)` + REST `PUT /memories/{memory_id}`，SeptMuse 未对齐。
- **工作记忆 Block**：`WorkingMemory` 类有 `set_block`/`update_block_value`/`core_memory_append`/`core_memory_replace`，但**只在内存**，不持久化。`Block` schema 已是 `SQLModel, table=True`（表 `septmuse_blocks`），`TypedMemoryStore` 已有 SQLModel engine 会建 blocks 表，但**无 Block CRUD 方法**。架构 §11.2 定义了 `PUT /memories/working/blocks/{label}`，未实现。Memory facade 也**未暴露** WorkingMemory 的操作。

## 2. 目标

实现两个层面的记忆更新：

| 层面 | 对齐 | 范围 |
|------|------|------|
| A. 长时记忆 update | mem0 `update(memory_id, data)` | verbatim content 更新 + 重新嵌入 + metadata + history 记录 |
| B. 工作记忆 Block replace | Letta `core_memory_replace` + 架构 §11.2 | Block 持久化 + set/update/append/replace + REST PUT |

## 3. 非目标

- semantic/episodic/procedural 的 update（它们有各自的 typed_store，后续单独做）
- PGVectorStore 的 update（可选后端，后续做）
- e2e 端到端测试（#3 缺口，后续单独做）
- Block 驱逐/压缩逻辑（架构 §3.1.1 "治理增量"，阶段3）

## 4. 设计

### 4.1 A. 长时记忆 update

#### 4.1.1 MemoryStore ABC

在 `storage/base.py` 的 `MemoryStore` ABC 加 `update` 抽象方法：

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

#### 4.1.2 SQLiteMemoryStore.update

在 `storage/sqlite/store.py` 的 `SQLiteMemoryStore` 加 `update` 实现：

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
        # 先读旧值 (不存在或已删除则返回 False)
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

#### 4.1.3 Memory facade

在 `orchestration/memory.py` 的 `Memory` 类加 `update` 方法（对齐 mem0 `update(memory_id, data)`）：

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
        user_id: 用户 ID (验证归属权, 不匹配则报错)
        metadata: 新 metadata; None=不改

    Returns:
        {"id": memory_id, "memory": 新content, "event": "UPDATE"}
        不存在/已删除: {"id": memory_id, "event": "NOT_FOUND"}
    """
    existing = self.store.get(memory_id)
    if existing is None:
        return {"id": memory_id, "event": "NOT_FOUND"}

    # 注: 不做 user_id 归属权验证 (对齐 mem0 update — 依赖调用方传对 memory_id)。
    # user_id 参数保留用于日志, 不参与查询条件。

    new_content = data if data is not None else existing["memory"]
    new_embedding = self.embedder.embed(new_content)

    # metadata 合并: 新 metadata 覆盖旧 metadata 的同名键
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

#### 4.1.4 REST PUT 端点

在 `api/rest/__init__.py` 的 `register_routes` 加：

```python
class UpdateMemoryRequest(BaseModel):
    text: str | None = Field(default=None, description="新内容")
    metadata: dict[str, Any] | None = Field(default=None, description="新 metadata")

@app.put("/memories/{memory_id}")
async def update_memory(memory_id: str, req: UpdateMemoryRequest) -> dict[str, Any]:
    """更新记忆 (对齐 mem0 PUT /memories/{id})。"""
    result = app.state.memory.update(memory_id, req.text, metadata=req.metadata)
    if result.get("event") == "NOT_FOUND":
        raise HTTPException(status_code=404, detail=f"memory {memory_id} not found")
    return result
```

#### 4.1.5 CLI update 命令

在 `cli/main.py` 加 `update` 子命令：

```
septmuse update <memory_id> "新内容" [--user USER_ID] [--db-path PATH]
```

- 位置参数: memory_id
- 位置参数: 新内容文本
- `--user`: 用户 ID（验证归属权，可选）
- `--db-path`: SQLite 路径
- 行为: 调 `Memory.update(memory_id, data, metadata=None)`
- 输出: JSON `{"id", "memory", "event": "UPDATE"}` 或 `{"id", "event": "NOT_FOUND"}`

### 4.2 B. 工作记忆 Block replace

#### 4.2.1 TypedMemoryStore Block CRUD

在 `storage/typed_store.py` 的 `TypedMemoryStore` 加 Block CRUD 方法。需先在文件顶部 import Block：

```python
from septmuse.schemas.block import Block, default_blocks
```

方法：

```python
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
            # UPDATE: 同 id 替换字段
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

#### 4.2.2 WorkingMemory 持久化

在 `content_types/working/block.py` 的 `WorkingMemory.__init__` 加 `store` 参数：

```python
def __init__(
    self,
    agent_id: str,
    blocks: list[Block] | None = None,
    store: Any | None = None,  # TypedMemoryStore | None, 避免循环 import
) -> None:
    self.agent_id = agent_id
    self.store = store
    self.blocks = blocks if blocks is not None else default_blocks(agent_id)
    logger.debug(...)
```

在 `set_block` / `update_block_value` / `core_memory_append` / `core_memory_replace` 操作后，若 `self.store` 非空则自动调 `self.store.save_block(block)` 持久化。

示例（`update_block_value`）：

```python
def update_block_value(self, label: str, value: str) -> None:
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

`store=None` 时回退纯内存模式（向后兼容现有测试）。

#### 4.2.3 Memory facade Block 方法

在 `orchestration/memory.py` 的 `Memory` 类加：

```python
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
    wm.update_block_value(label, value)  # 自动持久化
    block = wm.get_block(label)
    return {"id": block.id, "label": block.label, "value": block.value, "event": "UPDATE"}

def core_memory_append(self, agent_id: str, label: str, content: str) -> dict[str, Any]:
    """追加 block 内容 (对齐 Letta core_memory_append)。"""
    wm = self.get_working_memory(agent_id)
    wm.core_memory_append(label, content)  # 自动持久化
    block = wm.get_block(label)
    return {"id": block.id, "label": block.label, "value": block.value, "event": "APPEND"}

def core_memory_replace(self, agent_id: str, label: str, old_content: str, new_content: str) -> dict[str, Any]:
    """替换 block 内容片段 (对齐 Letta core_memory_replace)。"""
    wm = self.get_working_memory(agent_id)
    wm.core_memory_replace(label, old_content, new_content)  # 自动持久化
    block = wm.get_block(label)
    return {"id": block.id, "label": block.label, "value": block.value, "event": "REPLACE"}
```

需在文件顶部 import WorkingMemory：

```python
from septmuse.content_types.working.block import WorkingMemory
```

#### 4.2.4 REST 端点（架构 §11.2）

在 `api/rest/__init__.py` 的 `register_routes` 加：

```python
class BlockUpdateRequest(BaseModel):
    value: str = Field(description="新 block 内容")

class BlockAppendRequest(BaseModel):
    content: str = Field(description="追加内容")

class BlockReplaceRequest(BaseModel):
    old_content: str = Field(description="被替换的旧内容")
    new_content: str = Field(description="新内容")

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

#### 4.2.5 CLI block 命令（可选）

```
septmuse block set <agent_id> <label> <value> [--db-path PATH]
septmuse block list <agent_id> [--db-path PATH]
```

## 5. 数据流

### 5.1 长时记忆 update 数据流

```
septmuse update mem-xxx "新内容" --user alice
  → Memory.update("mem-xxx", "新内容", metadata=None)
  → store.get("mem-xxx") 验证存在
  → embedder.embed("新内容") 生成新嵌入
  → store.update("mem-xxx", "新内容", embedding, metadata=旧meta)
  → UPDATE memories SET content=?, embedding=?, updated_at=? WHERE id=?
  → INSERT INTO history (event='UPDATE', old_memory=旧, new_memory=新)
  → stdout: {"id":"mem-xxx","memory":"新内容","event":"UPDATE"}
```

### 5.2 Block replace 数据流

```
PUT /memories/working/blocks/agent-1/human  {"value":"Name: Alice"}
  → Memory.update_block("agent-1", "human", "Name: Alice")
  → typed_store.ensure_default_blocks("agent-1") 加载/创建 blocks
  → WorkingMemory(agent-1, blocks, store=typed_store)
  → wm.update_block_value("human", "Name: Alice")
  → block.value = "Name: Alice", block.touch()
  → store.save_block(block) → SQLModel Session UPDATE
  → {"id":"block-xxx","label":"human","value":"Name: Alice","event":"UPDATE"}
```

## 6. 错误处理

| 错误场景 | 处理 |
|---------|------|
| update 不存在的 memory_id | store.update 返回 False, facade 返回 `{"event":"NOT_FOUND"}`, REST 404 |
| Block label 不存在 | WorkingMemory.update_block_value raise ValueError, REST 400 |
| core_memory_replace old_content 未找到 | WorkingMemory raise ValueError, REST 400 |
| Block read_only=True 时 update | WorkingMemory 拒绝 (后续加 read_only 检查, 当前不拦) |

## 7. 测试策略

### 7.1 长时记忆 update 测试

`tests/unit/test_update.py`（新建）或追加到 `test_memory.py`：

- `test_update_content` — add → update → search 验证新内容可检索
- `test_update_not_found` — update 不存在的 id 返回 NOT_FOUND
- `test_update_metadata_only` — data=None 只改 metadata
- `test_update_history_recorded` — update 后 history 表有 event='UPDATE' 记录
- `test_update_re_embedding` — update 后旧 query 不匹配/新 query 匹配

### 7.2 Block CRUD 测试

`tests/unit/test_block_update.py`（新建）或追加到 `test_block.py`：

- `test_block_persist` — set_block → 重新 load → value 保留
- `test_update_block_value` — update_block_value → load → 新值
- `test_core_memory_append_persist` — append → load → 追加内容保留
- `test_core_memory_replace_persist` — replace → load → 替换内容保留
- `test_ensure_default_blocks` — 新 agent_id → 自动创建 human + persona
- `test_delete_block` — delete → load → 不存在

### 7.3 REST 端点测试

追加到 `test_rbac_rest_openai.py` 或新建：

- `test_put_memory` — PUT /memories/{id} → 200 + 更新后内容
- `test_put_memory_not_found` — PUT 不存在的 id → 404
- `test_put_block` — PUT /memories/working/blocks/{agent}/{label} → 200
- `test_get_blocks` — GET /memories/working/blocks/{agent} → 列表
- `test_post_append` — POST /append → 追加后内容
- `test_post_replace` — POST /replace → 替换后内容

### 7.4 CLI 测试

追加到 `test_cli.py`：

- `test_cli_update` — add → update → search 验证新内容
- `test_cli_update_not_found` — update 不存在的 id → 非零退出码
- `test_cli_block_set` — block set → block list 验证值

### 7.5 回归

- 现有 `test_block.py` 的 WorkingMemory 测试（store=None 纯内存模式）全绿
- 现有 `test_memory.py` 的 add/search/get/delete 测试全绿
- 全回归 502 passed 不降

## 8. 文件清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `src/septmuse/storage/base.py` | 修改 | MemoryStore ABC 加 `update` 抽象方法 |
| `src/septmuse/storage/sqlite/store.py` | 修改 | SQLiteMemoryStore 加 `update` 实现 |
| `src/septmuse/storage/typed_store.py` | 修改 | 加 Block CRUD（import Block + 5 方法） |
| `src/septmuse/content_types/working/block.py` | 修改 | WorkingMemory 加 `store` 参数 + 自动持久化 |
| `src/septmuse/orchestration/memory.py` | 修改 | Memory 加 update + get_working_memory + block 方法 |
| `src/septmuse/api/rest/__init__.py` | 修改 | 加 PUT /memories/{id} + Block REST 端点 |
| `src/septmuse/cli/main.py` | 修改 | 加 update + block 子命令 |
| `tests/unit/test_update.py` | 新建 | 长时记忆 update 测试 |
| `tests/unit/test_block_update.py` | 新建 | Block 持久化 + CRUD 测试 |
| `tests/unit/test_cli.py` | 修改 | 追加 update + block CLI 测试 |

## 9. 验收标准

- [ ] `Memory.update("mem-xxx", "新内容")` 返回 `{"event":"UPDATE"}`, 不存在返回 `{"event":"NOT_FOUND"}`
- [ ] update 后 history 表有 `event='UPDATE'` 记录（old_memory + new_memory）
- [ ] update 后旧嵌入被替换（search 用新 query 能匹配，旧 query 可能不匹配）
- [ ] `Memory.update_block("agent-1", "human", "value")` 持久化，重新 load 保留
- [ ] `Memory.core_memory_append("agent-1", "human", "content")` 追加后持久化
- [ ] `Memory.core_memory_replace("agent-1", "human", "old", "new")` 替换后持久化
- [ ] `PUT /memories/{memory_id}` 返回 200, 不存在返回 404
- [ ] `PUT /memories/working/blocks/{agent_id}/{label}` 持久化 Block
- [ ] `GET /memories/working/blocks/{agent_id}` 返回 Block 列表
- [ ] `septmuse update <id> "新内容"` CLI 可用
- [ ] `septmuse block set <agent> <label> <value>` CLI 可用
- [ ] `ruff check src/ tests/` 全绿
- [ ] `pytest tests/unit/test_update.py tests/unit/test_block_update.py` 全绿
- [ ] `pytest tests/ -q` 全回归不退化（502 passed 不降）

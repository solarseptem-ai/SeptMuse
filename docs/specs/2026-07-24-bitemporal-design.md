# 双时态建模设计文档（P2-Task 1）

> 日期：2026-07-24
> 前置文档：`docs/plans/development-roadmap.md`（P2 Phase）
> 范围：P2-Task 1（双时态建模 — valid_at/invalid_at/expired_at + 手动失效 + 时态查询）
> 不包含：P2-Task 2（时态区间查询，依赖本 Task）、P2-Task 3（消息压缩，独立）、LLM 自动矛盾检测（P3-Task 3 解锁后）
> 借鉴来源：graphiti `EntityEdge` 双时态字段 + `resolve_edge_contradictions` 失效逻辑 + SeptMuse `_migrate_add_state_columns` 迁移模式

---

## 1. 架构总览

```
┌─────────────────────────────────────────────────────────────┐
│                   Bitemporal Memory                          │
│                                                              │
│  memories 表新增 3 列:                                       │
│    valid_at    TEXT  — 事实开始为真的时间 (用户指定)          │
│    invalid_at  TEXT  — 停止为真的时间 (手动失效时设置)       │
│    expired_at  TEXT  — 系统标记失效的 wall-clock             │
│                                                              │
│  迁移: _migrate_add_temporal_columns()                       │
│    幂等: PRAGMA table_info 检查列存在性                      │
│    复用 _migrate_add_state_columns 模式                     │
│                                                              │
│  Memory facade:                                              │
│    add(valid_at="2024-01-01")  → 写入时设置事实有效期        │
│    invalidate(memory_id)       → 手动标记事实不再为真        │
│    search_at(time, query, user_id) → 时态查询:               │
│      WHERE valid_at <= time                                  │
│        AND (invalid_at IS NULL OR invalid_at > time)         │
│        AND state = 'active'                                  │
│        AND is_deleted = 0                                    │
│                                                              │
│  不包含: LLM 自动矛盾检测 (P3-Task 1 解锁后插入 add() 流程)  │
└─────────────────────────────────────────────────────────────┘
```

### 关键决策

- **手动失效**：`m.invalidate(memory_id)` 设置 `invalid_at` + `expired_at`，不删除记忆。LLM 自动矛盾检测留给 P3（在 `add()` 中插入矛盾检测步骤，不改存储层和检索层）。
- **3 个新列**：`valid_at`（事实开始为真）、`invalid_at`（停止为真，NULL=仍为真）、`expired_at`（系统标记失效的 wall-clock）。
- **valid_at 默认 None**：不设 valid_at 的记忆视为"无时间约束"，search_at 也能查到（向后兼容）。
- **ISO 8601 字符串**：与现有 `created_at` 一致，SQLite TEXT 存储。
- **前向兼容**：P3 加 LLM 时只需在 `add()` 的写入后插入矛盾检测步骤，不改存储层和检索层。`invalidate()` 手动接口保留。

### 时间语义

| 字段 | 含义 | 谁设置 | 何时设置 |
|------|------|--------|----------|
| `valid_at` | 事实开始为真的时间 | 用户（add 时指定） | add() 调用时 |
| `invalid_at` | 事实停止为真的时间 | 用户手动 / LLM 自动 (P3) | invalidate() / add() 矛盾检测 |
| `expired_at` | 系统标记失效的 wall-clock | 系统自动 | invalidate() / add() 矛盾检测 |
| `created_at` | 记忆写入时间（已有） | 系统自动 | add() 调用时 |

`invalid_at` ≠ `expired_at`：`invalid_at` 是事实层面的"何时停止为真"（可能与写入时间不同），`expired_at` 是系统层面的"何时标记失效"（wall-clock）。

---

## 2. Schema 迁移

### 2.1 `_migrate_add_temporal_columns()`

新增迁移方法到 `SQLiteMemoryStore`，复用 `_migrate_add_state_columns` 模式：

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

### 2.2 调用时机

在 `SQLiteMemoryStore.__init__` 中，在 `_migrate_add_state_columns()` 之后调用：

```python
self._migrate_add_state_columns()
self._create_access_logs_table()
self._migrate_add_temporal_columns()  # 新增
```

---

## 3. Memory Facade 变更

### 3.1 `Memory.add()` 扩展

新增 `valid_at` 参数：

```python
def add(
    self,
    messages: Any,
    *,
    user_id: str,
    agent_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    infer: bool | None = None,
    auto_extract_entities: bool = True,
    valid_at: str | None = None,           # 新增: ISO 8601
) -> dict[str, Any]:
```

- `valid_at=None`（默认）：不设时间约束，向后兼容
- `valid_at="2024-01-01"`：写入时设置到 memories 表的 `valid_at` 列
- 存储层 `SQLiteMemoryStore.add()` 新增 `valid_at` 可选参数，写入 SQL INSERT

### 3.2 `Memory.invalidate()` 新增

```python
def invalidate(
    self,
    memory_id: str,
    *,
    invalid_at: str | None = None,
) -> dict[str, Any]:
    """手动标记事实不再为真 (借鉴 graphiti resolve_edge_contradictions)。

    设置 invalid_at + expired_at, 不删除记忆。
    invalid_at=None 时用 utc_now()。

    Returns:
        {"id": memory_id, "invalid_at": "...", "expired_at": "...", "event": "INVALIDATE"}
        {"id": memory_id, "event": "NOT_FOUND"} 如果记忆不存在
    """
```

逻辑：
1. `existing = self.store.get(memory_id)` — 检查记忆存在
2. `invalid_at = invalid_at or _utcnow_iso()` — 默认当前时间
3. `expired_at = _utcnow_iso()` — 系统 wall-clock
4. `UPDATE memories SET invalid_at=?, expired_at=? WHERE id=?`
5. 返回结果 dict

### 3.3 `Memory.search_at()` 新增

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
    叠加 hybrid 检索 (向量+BM25+entity boost) + reranker。

    valid_at IS NULL 的记忆视为"无时间约束", 始终返回 (向后兼容)。
    """
```

逻辑：
1. 调用 `store.get_temporal_valid(reference_time, user_id)` 用 SQL WHERE 过滤得到时态有效候选集
2. 在候选集上做 hybrid 检索（向量 + BM25 + entity boost + 可选 reranker）
3. 返回结果

实现方式：`search_at` 先获取候选集，再用 hybrid 检索排序。对中小规模记忆集（当前场景）性能足够。大规模优化（候选集索引）留给后续。

### 3.4 `SQLiteMemoryStore` 新增方法

```python
def get_temporal_valid(self, reference_time: str, *, user_id: str) -> list[dict[str, Any]]:
    """时态过滤: 返回某时刻为真的全部记忆 (供 search_at 调用)。

    WHERE user_id = ? AND is_deleted = 0 AND state = 'active'
      AND (valid_at IS NULL OR valid_at <= ?)
      AND (invalid_at IS NULL OR invalid_at > ?)
    """
```

---

## 4. CLI / REST / MCP 集成

### CLI

- `add` 命令新增 `--valid-at` 可选参数
- 新增 `invalidate` 命令：`python -m septmuse.cli.main invalidate <memory_id>`
- `search` 命令新增 `--at` 可选参数（触发 `search_at` 时态查询）

### REST

- `POST /memories` 新增 `valid_at` body 字段
- 新增 `POST /memories/{id}/invalidate` 端点
- `POST /memories/search` 新增 `reference_time` body 字段（触发时态查询）

### MCP

- `add_memory` 工具新增 `valid_at` 参数
- 新增 `invalidate_memory` 工具
- `search_memory` 工具新增 `reference_time` 参数

---

## 5. 测试策略

### 5.1 测试文件布局

| 文件 | 内容 | 预计测试数 |
|------|------|-----------|
| `tests/unit/test_temporal.py` | 迁移 + invalidate + search_at | ~15 |
| `tests/e2e/test_temporal_e2e.py` | 跨会话时态持久化 | 3 |

### 5.2 测试要点

**迁移测试**：
- 新 DB 直接有 3 列
- 旧 DB（无列）迁移后增加 3 列
- 幂等：重复迁移不报错

**invalidate 测试**：
- invalidate 后 `invalid_at` + `expired_at` 被设置
- 不存在的 memory_id 返回 NOT_FOUND
- 默认 `invalid_at` = utc_now

**search_at 测试**：
- `valid_at=None` 的记忆始终返回（向后兼容）
- `valid_at <= reference_time` 的记忆返回
- `valid_at > reference_time` 的记忆不返回（事实还未开始）
- `invalid_at <= reference_time` 的记忆不返回（事实已失效）
- `invalid_at > reference_time` 的记忆返回（事实在该时刻仍为真）
- 时态过滤后叠加 hybrid 检索

**e2e 测试**：
- 跨会话：写入 valid_at → 新实例 search_at → 正确过滤
- invalidate 后 search_at 返回新事实不返回旧事实
- valid_at=None 的记忆在 search_at 中始终返回

### 5.3 验收标准

- 现有 800 passed 测试零回归
- 新增 ~18 测试 → 总计 ~818 passed
- `ruff check` + `ruff format --check` clean

---

## 6. 文件变更清单

### 新增文件

| 文件 | 内容 |
|------|------|
| `tests/unit/test_temporal.py` | ~15 单元测试 |
| `tests/e2e/test_temporal_e2e.py` | 3 e2e 测试 |

### 修改文件

| 文件 | 变更 |
|------|------|
| `src/septmuse/storage/sqlite/store.py` | `_migrate_add_temporal_columns()` + `add(valid_at=)` + `get_temporal_valid()` |
| `src/septmuse/orchestration/memory.py` | `add(valid_at=)` + `invalidate()` + `search_at()` |
| `src/septmuse/cli/main.py` | `add --valid-at` + `invalidate` 命令 + `search --at` |
| `src/septmuse/api/rest/__init__.py` | `POST /memories` +valid_at + `POST /memories/{id}/invalidate` + `search` +reference_time |
| `src/septmuse/api/mcp/tools.py` | `add_memory` +valid_at + `invalidate_memory` + `search_memory` +reference_time |
| `CHANGELOG.md` | Added — 双时态建模 |
| `AGENTS.md` | +双时态章节 |

---

## 7. 借鉴来源映射

| 设计要素 | 借鉴来源 | 具体文件/类 |
|----------|----------|------------|
| valid_at/invalid_at/expired_at 三字段 | graphiti `EntityEdge` | `opensource/graphiti/graphiti_core/edges.py:263` |
| 矛盾失效逻辑 (invalid_at = new.valid_at) | graphiti `resolve_edge_contradictions` | `opensource/graphiti/graphiti_core/utils/maintenance/edge_operations.py:538` |
| ALTER TABLE 迁移模式 | SeptMuse `_migrate_add_state_columns` | `src/septmuse/storage/sqlite/store.py:129` |
| 时态过滤 SQL | graphiti temporal filters + cognee `temporal_retriever` | `opensource/cognee/cognee/modules/retrieval/temporal_retriever.py` |
| 手动失效 (非 LLM) | SeptMuse 创新设计（graphiti 用 LLM，SeptMuse 降级为手动） | — |

---

## 8. 不包含（Out of Scope）

- **LLM 自动矛盾检测**：P3-Task 1（LLM Provider）+ P3-Task 3（冲突解决）范围。P3 到了在 `add()` 中插入矛盾检测步骤：候选检索 → LLM 判定 → 自动失效。不改存储层和检索层。
- **P2-Task 2 时态区间查询**：依赖本 Task 的 valid_at/invalid_at 列。本 Task 只做 `search_at(reference_time)`，区间查询 `search_interval(start, end)` 留给 P2-Task 2。
- **LLM 时间戳提取**：graphiti 的 `_extract_edge_timestamps` 用 LLM 从自然语言抽 valid_at。本 Task 用户手动指定 valid_at，LLM 提取留给 P3。

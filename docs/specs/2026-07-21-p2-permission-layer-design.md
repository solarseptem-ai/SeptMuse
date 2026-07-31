# P2 权限层设计 — SeptMuse 记忆基础设施补齐

- **状态**: Draft — 待用户 review
- **日期**: 2026-07-21
- **范围**: P2（4 阶段中的第 2 阶段）
- **依赖**: P1 存储抽象层已完成（655 passed, 22 skipped）
- **依赖决策**: 紧急 4 项 / 仅加 app_id 列 / 仅 API 层权限检查

## 1. 背景

P1 存储抽象层已补齐可插拔后端（VectorStoreBase + KeywordIndexBase + GraphStore 扩展 + 混合检索 RRF），655 passed 零退化。P2 聚焦权限层——生产部署前置条件。

SeptMuse 现状（源码实证）：
- `api/auth.py`（110 行）：仅 API key 认证（Bearer/X-API-Key），无 ACL
- `concerns/sharing/rbac.py`（153 行）：RBAC 骨架（Role/Permission/AccessGrant + RBACManager），但**内存态、未集成 API**
- `concerns/sharing/user_id.py`（105 行）：user_id 共享查询（仅查询，无权限检查）
- memories 表：仅 `is_deleted`（布尔），无 state/app_id/archived_at/deleted_at
- 无 MemoryAccessLog、无 AccessControl、无审计日志

mem0 openmemory 对比（codegraph 实证）：
- `models.py:37-109`：User/App/Memory 三级 ORM + state 枚举（active/paused/archived/deleted）
- `models.py:132-145`：AccessControl 表（subject/object/effect）
- `models.py:148-158`：ArchivePolicy 表（criteria/days_to_archive）
- `models.py:161-173`：MemoryStatusHistory 表（state 变更追踪）
- `models.py:176-188`：MemoryAccessLog 表（memory_id/app_id/accessed_at/access_type/metadata）
- `permissions.py:8-53`：check_memory_access_permissions 4 层检查
- `mcp_server.py`：search/list/delete 都先 check + 记日志

## 2. 目标与非目标

### 目标

1. memories 表加 state 状态机（active/paused/archived/deleted）+ app_id + archived_at + deleted_at
2. MemoryAccessLog 审计日志表 + 记录函数
3. check_memory_access_permissions 4 层权限检查
4. REST/MCP API 层集成权限检查（403）+ 访问日志

### 非目标

- 不建 User/App 表（仅加 app_id 列，继续用 user_id 字符串）
- 不建 AccessControl 表（P2.2 未来扩展）
- 不建 ArchivePolicy 表（P2.3 未来扩展）
- 不建 MemoryStatusHistory 表（P2.4 未来扩展）
- 不持久化 RBACManager（P2.9 未来扩展）
- 不在 Memory facade / TypedMemoryStore 加权限检查（仅 API 层）
- 不实现自动归档定时任务

## 3. 架构

### 3.1 权限层分层

```
┌─────────────────────────────────────────────────────────┐
│ Client (curl/MCP/CLI)                                   │
├─────────────────────────────────────────────────────────┤
│ API Layer                                               │
│  auth.py (已有)   → ApiKeyMiddleware (401 认证)         │ 不变
│  REST __init__.py → search/list/delete (新增权限检查)   │ P2.7
│  MCP tools.py     → search_memory 等 (新增权限检查)     │ P2.7
├─────────────────────────────────────────────────────────┤
│ Governance Layer (P2 新增)                              │
│  governance/permissions.py                             │
│    MemoryState enum (active/paused/archived/deleted)    │ P2.6
│    check_memory_access_permissions() 4 层检查          │ P2.6
│  governance/access_log.py                              │
│    record_access() 记 MemoryAccessLog                  │ P2.5
├─────────────────────────────────────────────────────────┤
│ Storage Layer                                           │
│  SQLiteCompositeStore                                   │
│    memories 表 +state/app_id/archived_at/deleted_at    │ P2.1
│    memory_access_logs 表 (新建)                        │ P2.5
├─────────────────────────────────────────────────────────┤
│ Memory facade (orchestration/memory.py)                 │ 不变 (纯逻辑)
│ TypedMemoryStore (concerns 层)                          │ 不变
└─────────────────────────────────────────────────────────┘
```

### 3.2 关键设计决策

1. **仅 API 层加权限检查**：Memory facade + TypedMemoryStore + concerns 层全不变，655 测试零退化基线。

2. **memories 表 ALTER TABLE 加列**（向后兼容）：
   - `app_id TEXT`（默认 NULL，旧数据无 app 归属）
   - `state TEXT DEFAULT 'active'`（旧数据自动 active）
   - `archived_at TEXT`（默认 NULL）
   - `deleted_at TEXT`（默认 NULL，与现有 `is_deleted` 并存）
   - 新索引：`idx_memory_user_state` + `idx_memory_app_state` + `idx_memory_user_app`

3. **401 vs 403 语义**：
   - 401：认证失败（API key 错误/缺失，已有 `auth.py` 处理）
   - 403：授权失败（memory state 非 active / app 无权限，P2 新增）

4. **MemoryAccessLog 异步记录**：不阻塞主请求，用 `try/except` 吞错（日志失败不应影响业务）。

5. **is_deleted 并存**：delete 同时设 `is_deleted=1` + `state='deleted'`（向后兼容，旧查询 `is_deleted=0` 仍有效）。

6. **state 默认 active**：旧数据无 state 列 → ALTER TABLE 后默认 'active'，无需迁移脚本。

## 4. 接口定义

### 4.1 MemoryState enum（`governance/permissions.py`）

```python
class MemoryState(str, Enum):
    """记忆状态 (对齐 mem0 MemoryState enum)。"""
    ACTIVE = "active"
    PAUSED = "paused"
    ARCHIVED = "archived"
    DELETED = "deleted"
```

### 4.2 check_memory_access_permissions（`governance/permissions.py`）

```python
def check_memory_access_permissions(
    store: MemoryStore,
    memory_id: str,
    app_id: str | None = None,
) -> tuple[bool, str]:
    """4 层权限检查 (借鉴 mem0 permissions.py:8-53)。

    Args:
        store: 记忆存储后端
        memory_id: 目标记忆 ID
        app_id: 访问方应用 ID; None 表示用户自己访问

    Returns:
        (allowed, reason): True=放行, False=拒绝 + 原因
    """
    # 层1: memory 存在 + state=active
    mem = store.get(memory_id)
    if not mem:
        return False, "memory not found"
    state = mem.get("state", MemoryState.ACTIVE)
    if state != MemoryState.ACTIVE:
        return False, f"memory state is {state} (not active)"

    # 层2: 无 app_id → 用户自己访问, 放行
    if not app_id:
        return True, "self access"

    # 层3: app_id 非空即 active (SeptMuse 无 App 表, 简化)
    if not app_id.strip():
        return False, "app_id is empty"

    # 层4: app 白名单 (SeptMuse 无 AccessControl 表, 默认全部可访问)
    # P2.2 未来加 AccessControl 表时在此扩展
    return True, f"app {app_id} access granted"
```

**简化说明**（对齐用户决策"仅加 app_id 列，不建 App 表"）：
- 层3 简化为"app_id 非空即 active"（无 App 表可查 is_active）
- 层4 默认全部可访问（无 AccessControl 表，P2.2 未来扩展）

### 4.3 record_access（`governance/access_log.py`）

```python
def record_access(
    store: MemoryStore,
    memory_id: str,
    app_id: str | None,
    access_type: str,
    metadata: dict[str, Any] | None = None,
) -> str | None:
    """记录记忆访问日志 (借鉴 mem0 MemoryAccessLog)。

    Args:
        store: 记忆存储后端 (必须支持 _record_access_log 方法)
        memory_id: 被访问的记忆 ID
        app_id: 访问方应用 ID
        access_type: "search" / "get" / "delete" / "list"
        metadata: 额外信息 {"query":.., "score":..}

    Returns:
        log_id 或 None (记录失败时返回 None, 不抛异常)
    """
    try:
        if hasattr(store, "_record_access_log"):
            return store._record_access_log(memory_id, app_id, access_type, metadata)
        logger.warning("store_does_not_support_access_log", store=type(store).__name__)
        return None
    except Exception as e:
        logger.warning("access_log_failed", error=str(e), memory_id=memory_id)
        return None
```

### 4.4 SQLiteCompositeStore 扩展（`storage/sqlite/store.py`）

**ALTER TABLE 加 4 列**（向后兼容）：
```python
def _migrate_add_state_columns(self) -> None:
    """P2 迁移: 加 state/app_id/archived_at/deleted_at 列。"""
    with self._lock:
        cols = {row[1] for row in self.conn.execute("PRAGMA table_info(memories)")}
        if "state" not in cols:
            self.conn.execute("ALTER TABLE memories ADD COLUMN state TEXT DEFAULT 'active'")
        if "app_id" not in cols:
            self.conn.execute("ALTER TABLE memories ADD COLUMN app_id TEXT")
        if "archived_at" not in cols:
            self.conn.execute("ALTER TABLE memories ADD COLUMN archived_at TEXT")
        if "deleted_at" not in cols:
            self.conn.execute("ALTER TABLE memories ADD COLUMN deleted_at TEXT")
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memory_user_state ON memories(user_id, state)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memory_app_state ON memories(app_id, state)"
        )
        self.conn.commit()
```

**memory_access_logs 新表**：
```python
def _create_access_logs_table(self) -> None:
    """创建 memory_access_logs 表 (借鉴 mem0 models.py:176-188)。"""
    with self._lock:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_access_logs (
                id          TEXT PRIMARY KEY,
                memory_id   TEXT NOT NULL,
                app_id      TEXT,
                accessed_at TEXT NOT NULL,
                access_type TEXT NOT NULL,
                metadata    TEXT DEFAULT '{}'
            )
            """
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_access_memory_time ON memory_access_logs(memory_id, accessed_at)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_access_app_time ON memory_access_logs(app_id, accessed_at)"
        )
        self.conn.commit()
```

**_record_access_log + get_access_logs**：
```python
def _record_access_log(
    self, memory_id: str, app_id: str | None, access_type: str,
    metadata: dict[str, Any] | None,
) -> str:
    """记录访问日志到 memory_access_logs 表。"""
    log_id = f"log-{uuid.uuid4()}"
    with self._lock:
        self.conn.execute(
            "INSERT INTO memory_access_logs (id, memory_id, app_id, accessed_at, access_type, metadata) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (log_id, memory_id, app_id, _utcnow_iso(), access_type,
             json.dumps(metadata or {}, ensure_ascii=False)),
        )
        self.conn.commit()
    return log_id

def get_access_logs(self, memory_id: str, limit: int = 100) -> list[dict[str, Any]]:
    """查询记忆的访问日志 (审计用)。"""
    with self._lock:
        rows = self.conn.execute(
            "SELECT id, memory_id, app_id, accessed_at, access_type, metadata "
            "FROM memory_access_logs WHERE memory_id = ? ORDER BY accessed_at DESC LIMIT ?",
            (memory_id, limit),
        ).fetchall()
    return [
        {"id": r[0], "memory_id": r[1], "app_id": r[2], "accessed_at": r[3],
         "access_type": r[4], "metadata": json.loads(r[5]) if r[5] else {}}
        for r in rows
    ]
```

**delete 改为 state=deleted**（保留 is_deleted 向后兼容）：
```python
def delete(self, memory_id: str) -> None:
    """软删除: state='deleted' + deleted_at + is_deleted=1 (双写兼容)。"""
    # 现有逻辑保留 (is_deleted=1 + history)
    # 新增: state='deleted' + deleted_at
```

**search/get_all 加 state 过滤**：
```python
# search: filters 加 state=active (或 state IS NULL 向后兼容旧数据)
# get_all: WHERE (state = 'active' OR state IS NULL)
```

### 4.5 MemoryStore ABC 扩展（`storage/base.py`）

```python
class MemoryStore(ABC):
    # ... 现有方法不变 ...

    def get_access_logs(self, memory_id: str, limit: int = 100) -> list[dict[str, Any]]:
        """查询记忆访问日志 (审计用)。默认返回空 (子类有日志表时覆盖)。"""
        return []
```

### 4.6 REST/MCP 集成

**REST `api/rest/__init__.py`**：
```python
@app.get("/memories/{memory_id}")
async def get_memory(memory_id: str, app_id: str | None = None):
    mem = app.state.memory
    allowed, reason = check_memory_access_permissions(mem.store, memory_id, app_id)
    if not allowed:
        raise HTTPException(status_code=403, detail=reason)
    record_access(mem.store, memory_id, app_id, "get")
    return mem.store.get(memory_id)

@app.delete("/memories/{memory_id}")
async def delete_memory(memory_id: str, app_id: str | None = None):
    mem = app.state.memory
    allowed, reason = check_memory_access_permissions(mem.store, memory_id, app_id)
    if not allowed:
        raise HTTPException(status_code=403, detail=reason)
    record_access(mem.store, memory_id, app_id, "delete")
    mem.store.delete(memory_id)
    return {"deleted": memory_id}
```

**MCP `api/mcp/tools.py`**（search_memory 加日志）：
```python
@mcp.tool(description="搜索记忆。")
async def search_memory(query: str, user_id: str = "", top_k: int = 5, app_id: str = ""):
    # ... 现有搜索逻辑 ...
    for r in results:
        record_access(mem.store, r["id"], app_id or None, "search",
                      metadata={"query": query, "score": r.get("score")})
    return json.dumps({"results": results}, ensure_ascii=False, indent=2)
```

## 5. 数据流

### 5.1 搜索记忆流程

```
Client GET /memories/{id}?app_id=myapp
  ↓
auth.py ApiKeyMiddleware (401 认证)
  ↓
REST get_memory(memory_id, app_id)
  ↓
check_memory_access_permissions(store, memory_id, app_id)
  ├→ 层1: store.get(memory_id) → mem 存在? state=active?
  │   └→ 不存在/state 非 active → return (False, reason) → 403
  ├→ 层2: app_id 为空? → return (True, "self access")
  ├→ 层3: app_id 非空? → return (True, "app {id} granted")
  └→ 层4: [预留 AccessControl 表扩展点]
  ↓
allowed=True → record_access(store, memory_id, app_id, "get")
  ├→ store._record_access_log(...)
  │   └→ INSERT memory_access_logs + commit
  └→ 失败 → logger.warning + return None (不阻塞)
  ↓
return store.get(memory_id)
```

### 5.2 state 状态转换

```
                ┌─────────┐
       add()    │ active  │  delete()
   ──────────→  │         │  ──────────→ ┌──────────┐
                └────┬────┘              │ deleted  │
                     │                   └──────────┘
              pause()│
                     ▼
                ┌─────────┐
                │ paused  │
                └─────────┘
   (archived 状态 P2 暂不实现自动归档, 预留枚举值)
```

## 6. 实现清单

| # | 文件 | 类型 | 行数估 | 借鉴源 |
|---|------|------|--------|--------|
| 1 | `concerns/governance/permissions.py` | 新增 | ~70 | mem0 `permissions.py:8-53` + `MemoryState` |
| 2 | `concerns/governance/access_log.py` | 新增 | ~50 | mem0 `MemoryAccessLog` |
| 3 | `storage/sqlite/store.py` | 扩展 | +~80 | mem0 `models.py:85-188` |
| 4 | `storage/base.py` | 扩展 | +~15 | store ABC 加 get_access_logs 默认实现 |
| 5 | `api/rest/__init__.py` | 扩展 | +~40 | mem0 mcp_server.py 权限检查模式 |
| 6 | `api/mcp/tools.py` | 扩展 | +~20 | mem0 mcp_server.py 日志记录 |
| 7 | `storage/vector/pgvector.py` | 扩展 | +~40 | 同 SQLite（P2.1 迁移 + 日志） |

**测试**：
| # | 文件 | 测试数 | 覆盖 |
|---|------|--------|------|
| 8 | `tests/unit/test_permissions.py` | ~10 | 4 层检查 + 边界 |
| 9 | `tests/unit/test_access_log.py` | ~6 | 记录 + 查询 + 吞错 |
| 10 | `tests/unit/test_memory_state.py` | ~6 | state 状态机 + 双写兼容 |
| 11 | `tests/unit/test_api_permission_integration.py` | ~8 | REST 403 + 日志 |
| 12 | 现有 655 测试 | 零退化 | 基线 |

## 7. 错误处理

| 场景 | 策略 |
|------|------|
| memory 不存在 | check 返回 (False, "memory not found") → 403 |
| memory state 非 active | check 返回 (False, "memory state is X") → 403 |
| app_id 为空字符串 | check 返回 (False, "app_id is empty") → 403 |
| store 不支持 _record_access_log | record_access 返回 None + warning log |
| _record_access_log 抛异常 | record_access 吞掉 + warning log |
| ALTER TABLE 列已存在 | 跳过（PRAGMA table_info 检查） |
| 旧数据无 state 列 | 迁移后默认 'active' |
| 旧数据 is_deleted=1 | search 用 `is_deleted=0 OR state='active'` 双重过滤 |
| PG store 扩展 | 同 SQLite 模式（ALTER TABLE + memory_access_logs 表） |

## 8. 执行顺序（TDD + 增量）

```
Step 1: MemoryState enum + check_memory_access_permissions  (test_permissions.py)
Step 2: record_access + MemoryAccessLog 操作               (test_access_log.py)
Step 3: SQLiteCompositeStore 扩展 (ALTER TABLE + 日志表 + state 过滤)  (test_memory_state.py)
Step 4: MemoryStore ABC 加 get_access_logs 默认实现
Step 5: REST search/get/delete 集成权限检查 + 日志          (test_api_permission_integration.py)
Step 6: MCP tools 集成日志记录
Step 7: PGVectorStore 扩展 (同 SQLite 模式)
Step 8: 全量回归 ruff + pytest 655+30 新测试全绿
```

## 9. 验证标准

- [ ] `ruff check src/ tests/` All checks passed
- [ ] `pytest tests/unit -q` 655 + 30 = 685 passed, 22 skipped（零退化）
- [ ] `pytest tests/e2e -q` 23 passed（e2e 不受影响）
- [ ] `SEPTMUSE_API_KEY=test python -m septmuse.api.rest` 启动成功
- [ ] `curl -H "Authorization: Bearer test" http://localhost:8000/memories/{deleted_id}` → 403
- [ ] `curl -H "Authorization: Bearer test" http://localhost:8000/memories/{active_id}` → 200 + 日志记录
- [ ] 旧 DB 迁移成功（无 state 列 → 加列 + 默认 active）
- [ ] Memory facade + TypedMemoryStore 零改动（655 基线不变）

## 10. 风险与缓解

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| ALTER TABLE 破坏 655 测试 | 低 | 高 | PRAGMA 检查列存在 + state 默认 active + is_deleted 并存 |
| state 过滤漏掉旧数据 | 中 | 高 | `WHERE (state = 'active' OR state IS NULL)` 双重过滤 |
| API 层 403 破坏 e2e | 中 | 中 | e2e 用 active 记忆，不触发 403 |
| record_access 阻塞请求 | 低 | 中 | try/except 吞错 + 不阻塞 |
| PG store ALTER TABLE 失败 | 低 | 低 | PG 测试 skip，不影响回归 |
| app_id 参数不向后兼容 | 低 | 中 | app_id 默认 None，旧请求不传 app_id |

## 11. 后续阶段预告

- **P3 时态层**：schemas 加 valid_at/invalid_at/expired_at/reference_time（借鉴 graphiti EntityEdge）
- **P4 编排+扩展**：Pipeline DAG（借鉴 cognee）+ vision（借鉴 mem0）+ auto_dream（借鉴 ReMe）

每个阶段独立 spec → plan → implementation 循环。

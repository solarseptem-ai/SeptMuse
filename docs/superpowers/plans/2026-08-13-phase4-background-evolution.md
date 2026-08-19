# Phase 4: 后台演化调度器 + LLM 恢复补跑

> spec: `docs/superpowers/specs/2026-08-13-smart-memory-butler-design.md` §Phase 4

## 目标
agent 不用时 SeptMuse 自己整理记忆。LLM 恢复后自动补跑降级跳过的任务。

## Task 1: LLMCatchUpQueue

**文件**: `src/septmuse/evolution/catchup.py` + `tests/unit/test_catchup_queue.py`

SQLite 表 `septmuse_catchup_queue`，存 LLM 不可用时降级跳过的任务。

```python
class CatchUpTask(SQLModel, table=True):
    __tablename__ = "septmuse_catchup_queue"
    id: str = PrimaryKey(str, default=uuid4)
    task_type: str          # importance_assessment / fact_extraction / conflict_judgment / summarization / classification
    memory_id: str
    user_id: str
    payload: str            # JSON
    status: str = "pending" # pending / completed / failed
    created_at: str
    completed_at: str | None = None

class LLMCatchUpQueue:
    def __init__(self, engine): ...
    def enqueue(self, task_type, memory_id, *, user_id, payload=None) -> str
    def dequeue(self, limit=10) -> list[dict]
    def mark_completed(self, task_id) -> None
    def mark_failed(self, task_id, error) -> None
    def get_pending_count(self) -> int
```

**测试**:
- enqueue → dequeue → mark_completed
- 多条 FIFO 顺序
- pending_count 正确
- mark_failed 设 status

## Task 2: EvolutionScheduler

**文件**: `src/septmuse/evolution/scheduler.py` + `tests/unit/test_evolution_scheduler.py`

```python
class EvolutionScheduler:
    def __init__(self, memory, *, interval=300, enabled=True):
        # task_intervals (seconds): forgetting=3600, conflict=300, distillation=1800, graph=600
        # last_run: dict[str, float] = {name: 0.0}

    def start(self) -> None       # 启动 daemon Timer
    def stop(self, timeout=10) -> None  # Event.set + join
    def _run_cycle(self) -> None  # 检查 due tasks → 跑 → 重新 schedule
    def _run_task(self, name, user_id) -> None  # 单任务执行，max 30s
    def _process_catchup(self) -> None  # LLM ping → dequeue → 补跑
    def _get_user_ids(self) -> list[str]  # 从 store 查所有 user_id
```

**设计要点**:
- `threading.Timer` daemon 线程，不阻塞主线程
- `threading.Event` stop 信号
- `threading.Lock` 防并发 _run_cycle
- 每个任务 30s 超时（用 `threading.Timer` 在子线程跑，主调度线程不卡）
- LLM ping：调 `llm.complete("ping", "ping")` 成功 = 可用
- 4 个演化任务：
  - forgetting → `ForgettingRetriever.archive_stale(user_id=)`
  - conflict → `ConflictResolver.resolve_conflicts(user_id=)`
  - distillation → `SessionReflector.reflect(user_id=, limit=20)`
  - graph → `DreamIntegrator.dream(user_id=)`

**测试**:
- start/stop 不崩
- _run_cycle 跑 due 任务、跳过未 due 的
- _process_catchup：LLM 不可用不跑、可用时 dequeue 补跑
- _get_user_ids 返回正确列表
- stop 等 _run_cycle 完成（max 10s）

## Task 3: 集成到 lifespan + remember() 降级入队

**修改文件**:
- `src/septmuse/api/rest/__init__.py` — lifespan start/stop scheduler
- `src/septmuse/api/mcp/server.py` — run_stdio start/stop scheduler
- `src/septmuse/memory/main.py` — remember() LLM None 时 enqueue catch-up
- `src/septmuse/evolution/__init__.py` — export 新类

**lifespan 集成**:
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # start
    scheduler = EvolutionScheduler(sync_memory, ...)
    scheduler.start()
    app.state.scheduler = scheduler
    yield
    # stop
    scheduler.stop(timeout=10)
    # existing engine.dispose()
```

**remember() 降级入队**:
- 当 `self.llm is None` 且 memory 已存储：
  - `catchup_queue.enqueue("fact_extraction", memory_id, user_id=user_id, payload={"text": stored_text})`
- catch-up queue 延迟初始化（`self._catchup` property，用 store.engine）

**环境变量**:
- `SEPTMUSE_EVOLUTION_INTERVAL` (default 300)
- `SEPTMUSE_EVOLUTION_ENABLED` (default true)

## Task 4: 全量回归

- `ruff check src/ tests/`
- `pytest tests/unit/ -q`
- `pytest tests/e2e/ -q`

## 依赖

```
Task 1 (CatchUpQueue) → Task 2 (Scheduler 用 queue) → Task 3 (集成) → Task 4 (回归)
```

# 测试策略

English/中文

## 测试命令

```bash
# 必须设 PYTHONPATH=src，否则 import septmuse 失败
$env:PYTHONPATH = "src"
$env:SEPTMUSE_EMBEDDER = "hash"        # 测试用 hash embedder（零模型加载）
$env:SEPTMUSE_TOKENIZER = "space"     # 测试用 space 分词（conftest 强制）
$env:SEPTMUSE_RATE_LIMIT = "0"        # 测试禁限流
$env:SEPTMUSE_VECTOR_BACKEND = "sqlite"  # 测试用 sqlite 后端（避免 Qdrant 锁冲突）

PYTHONPATH=src pytest tests/unit/ -q              # 单元 + 集成（1833 collected, 62 failed pre-existing, 15 skipped）
PYTHONPATH=src pytest tests/e2e/ -q               # 端到端（32 tests, 9 files）
PYTHONPATH=src pytest tests/unit/test_memory.py::test_add_returns_id -q   # 单测试
```

CI（`.github/workflows/ci.yml`）跑 `ruff check` + `ruff format --check` + `pytest tests/unit/ tests/e2e/`（Python 3.10/3.11/3.12 矩阵）+ `python -m build`。

## 测试怪癖

- **15 skipped 是正常的**：`pytest.mark.integration` 标记的测试（chroma/qdrant/neo4j/rank-bm25）需装对应 extras；pgvector + AGE 测试需 `SEPTMUSE_TEST_PG_DSN`；Neo4j 需 `SEPTMUSE_TEST_NEO4J_URI`；parametric 需 torch+peft；activation 需 torch；onnx/auto embedder 需 `pip install septmuse[onnx]`。
- **62 failed 是 pre-existing**：`test_llm_providers.py`（13）、`test_attributed_to.py`（8）、`test_user_profile.py`（7）、`test_qdrant_vector_store.py`（4，Qdrant 锁冲突）、`test_procedural_auto.py`（3）、`test_permissions.py`（3）、`test_rest_profile.py`（3）、`test_facade_integration.py`（3，CapturePipeline not callable）、`test_mem_os.py`（2）、`test_recall_profile_inject.py`（2）、`test_update_relink.py`（2）等 — 需要真实 API key / Qdrant 后端 / 未实现功能（attributed_to, get_user_profile），与重构无关。
- **e2e 测试**：`tests/e2e/` 9 文件 32 测试，全部用 `tmp_path` 文件 DB，测跨会话持久化。`pytest tests/e2e/` 必须通过（1 pre-existing failure: `test_capture_persists`）。
- **`pytest_asyncio_mode = "auto"`**：async 测试无需 `@pytest.mark.asyncio`。
- **`--strict-markers --strict-config`**：新增 marker 必须在 `pyproject.toml [tool.pytest.ini_options] markers` 注册。

## 测试保护规则

- 现有全部单元测试、接口测试案例固定不动，严禁改测试代码、调断言、删用例来规避业务缺陷。
- 测试不通过仅通过优化业务逻辑、补齐容错、完善参数校验、修复分支漏洞解决。
- 仅可新增测试用例覆盖新功能，禁止任何绕过测试的取巧写法。

## SQLite 测试陷阱

- 默认 `~/.septmuse/septmuse.db`；`db_path=None` 触发此默认；`":memory:"` 可用但 **FastAPI TestClient 跨线程会连到新空库** — REST/e2e 测试一律用 `tmp_path / "test.db"` 文件路径（见 `tests/e2e/*.py` 的 `db = str(tmp_path / "e2e.db")` 模式）。
- 单元测试用 `:memory:` 时，DatabaseService 自动用 `StaticPool` 共享单连接（跨线程并发检索看到同一内存库）。
- `record_access` 吞错（日志失败不阻塞业务），用 `hasattr` 检查 store 是否有 `get_access_logs` 方法（向后兼容旧 store）。

## 向量存储测试

- 测试：`tests/unit/test_vector_stores/`（7 文件，19 passed + 1 skipped）。

# 可观测性指标系统设计

> 日期：2026-08-06
> 前置：`docs/plans/optimization-roadmap-v2.md` Phase 0 基础设施优化（已完成）
> 技术选型：Prometheus（`prometheus_client` 纯 Python，无外部依赖）
> 范围：REST API + CLI + MCP 全路径的 RED / 资源 / 业务三维指标

---

## 1. 目标

为 SeptMuse 添加 Prometheus 可观测性指标系统，覆盖性能（RED 方法）、资源、业务三个维度。`SEPTMUSE_METRICS=true` 环境变量 opt-in，未启用时所有埋点 no-op（零开销）。`/metrics` 端点返回 Prometheus 格式，Grafana 可视化。

**非目标**：不引入 OpenTelemetry Collector / 分布式追踪（单机部署不需要），不实现告警规则（由 Grafana 配置），不做多进程指标聚合（单 worker 场景；多 worker 留作未来扩展）。

## 2. 设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 指标暴露策略 | 环境变量 opt-in（`SEPTMUSE_METRICS=true`） | 安全默认，生产可控 |
| 耗时指标类型 | 全部 Histogram | 能算 p50/p99，Gauge 只适合当前值 |
| 埋点注入策略 | B+C 混合（中间件 + facade 装饰器 + 底层 ABC 模板方法） | 全覆盖，零遗漏 |
| 底层 ABC 埋点方式 | 模板方法模式（`embed()` → 子类 `_embed()`） | 所有 provider 自动继承 |
| 业务指标刷新 | Pull-on-scrape（CustomCollector） | 零侵入写路径，Prometheus 惯例 |
| `/metrics` 认证 | 免 API key（加入 EXEMPT_PATHS） | 与 /health 一致，网络层隔离 |
| `prometheus_client` 依赖 | 核心依赖（非 optional extra） | 纯 Python ~50KB，opt-in 是运行时开关 |

## 3. 模块结构

```
src/septmuse/observability/
├── __init__.py          # 暴露 init_metrics(app, store, config) 入口
├── collector.py          # MetricsCollector 单例 — 封装所有 Counter/Histogram/Gauge
├── middleware.py         # PrometheusMiddleware — 自动记录 RED 指标
├── business.py           # BusinessMetricsCollector — pull-on-scrape 业务指标
└── hooks.py              # track_operation 装饰器 + 埋点辅助函数
```

## 4. MetricsCollector 单例

`collector.py` — 所有 Prometheus 指标集中定义，未启用时全部 no-op。

### 4.1 接口

```python
class MetricsCollector:
    """单例。未启用时所有方法 no-op（一个 if 分支，零开销）。"""

    _instance: MetricsCollector | None = None

    @classmethod
    def get(cls) -> MetricsCollector:
        """全局获取单例。首次调用时初始化为 disabled。"""
        ...

    def configure(self, enabled: bool) -> None:
        """初始化所有 Prometheus 指标。enabled=False 时保持 no-op。"""
        ...

    def inc(self, name: str, labels: dict[str, str] | None = None, amount: float = 1) -> None:
        """Counter/Gauge 增量。"""
        ...

    def dec(self, name: str, labels: dict[str, str] | None = None, amount: float = 1) -> None:
        """Gauge 减量（in_flight 等）。"""
        ...

    def observe(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        """Histogram 观测。"""
        ...

    def set_gauge(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        """Gauge 设值（业务指标绝对值）。"""
        ...

    @classmethod
    def reset(cls) -> None:
        """测试用 — 清单例 + 从 REGISTRY 注销所有自定义指标。"""
        ...
```

### 4.2 行为

- `configure(enabled=False)`（默认）：`inc()/dec()/observe()/set_gauge()` 全部直接 return，不触碰 `prometheus_client`
- `configure(enabled=True)`：用 `prometheus_client.Counter/Histogram/Gauge` 创建所有指标，存入 `self._metrics` dict
- `reset()`：`_instance = None` + 遍历 `REGISTRY._names_to_collectors` 删除所有 `septmuse_*` 指标（测试隔离用）

### 4.3 指标定义（configure 内）

| 指标名 | 类型 | 标签 |
|--------|------|------|
| `septmuse_api_requests_total` | Counter | endpoint, method, status |
| `septmuse_api_request_duration_seconds` | Histogram | endpoint, method |
| `septmuse_api_requests_in_flight` | Gauge | endpoint |
| `septmuse_memory_operation_duration_seconds` | Histogram | operation |
| `septmuse_embed_duration_seconds` | Histogram | backend |
| `septmuse_embed_batch_duration_seconds` | Histogram | backend |
| `septmuse_embed_cache_hits_total` | Counter | — |
| `septmuse_embed_cache_misses_total` | Counter | — |
| `septmuse_llm_call_duration_seconds` | Histogram | provider |
| `septmuse_llm_calls_total` | Counter | provider, status |
| `septmuse_hybrid_search_components_seconds` | Histogram | component |
| `septmuse_uptime_seconds` | Gauge | — |

Histogram buckets 使用 `prometheus_client` 默认值（`ExponentialBuckets(0.005, 2, 20)`，覆盖 5ms~5242s），适合 API 延迟和 embed/LLM 耗时。

## 5. 埋点注入

### 5.1 REST 中间件（全自动）

`middleware.py` — `PrometheusMiddleware(BaseHTTPMiddleware)` 拦截所有 HTTP 请求：

```
dispatch(request, call_next):
    endpoint = request.url.path
    method = request.method

    # in_flight +1
    collector.inc("api_requests_in_flight", {"endpoint": endpoint})

    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start

    # in_flight -1
    collector.dec("api_requests_in_flight", {"endpoint": endpoint})

    # Counter: 请求总数
    collector.inc("api_requests_total", {"endpoint": endpoint, "method": method, "status": str(response.status_code)})

    # Histogram: 请求延迟
    collector.observe("api_request_duration_seconds", elapsed, {"endpoint": endpoint, "method": method})
```

### 5.2 Facade 装饰器（手动加 6 方法）

`hooks.py` — `@track_operation(operation)` 装饰器，包裹 `ExperimentalMemory` 的核心方法：

```python
def track_operation(operation: str):
    """装饰器 — 记录 memory_operation_duration_seconds。"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            collector = MetricsCollector.get()
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                collector.observe("memory_operation_duration_seconds",
                                  time.perf_counter() - start,
                                  labels={"operation": operation})
        return wrapper
    return decorator
```

应用到的 facade 方法：

| 方法 | operation 标签 |
|------|----------------|
| `ExperimentalMemory.add()` | `add` |
| `ExperimentalMemory.search()` | `search` |
| `ExperimentalMemory.update()` | `update` |
| `ExperimentalMemory.delete()` | `delete` |
| `ExperimentalMemory.get()` | `get` |
| `ExperimentalMemory.invalidate()` | `invalidate` |

`HybridRetriever` 内部三路检索也用装饰器记录 `hybrid_search_components_seconds`：

| 方法 | component 标签 |
|------|----------------|
| `VectorStore.query()` | `vector` |
| `KeywordIndexBase.search()` | `keyword` |
| `GraphSearcher.bfs()` | `graph` |

### 5.3 底层 ABC 模板方法（全覆盖）

**EmbedderBase**（`src/septmuse/embedders/base.py`）：

```python
class EmbedderBase(ABC):
    @abstractmethod
    def _embed(self, text: str, memory_action: str = "none") -> list[float]:
        """子类实现实际嵌入逻辑。"""
        ...

    def embed(self, text: str, memory_action: str = "none") -> list[float]:
        """嵌入文本为向量（带可观测性埋点）。"""
        collector = MetricsCollector.get()
        start = time.perf_counter()
        try:
            return self._embed(text, memory_action)
        finally:
            collector.observe("embed_duration_seconds",
                              time.perf_counter() - start,
                              labels={"backend": self._backend_name()})

    def _backend_name(self) -> str:
        """返回后端名称（用于指标标签）。

        子类应设置 self.backend_name 属性（如 "hash"、"bge-zh"、"openai"）。
        未设置时 fallback 用类名小写。
        """
        return getattr(self, "backend_name", type(self).__name__.lower())
```

所有 embedder 子类：`def embed(` → `def _embed(`（机械重命名，14 个文件）。

每个子类需添加 `backend_name` 类属性或实例属性，用于指标标签（如 `backend_name = "hash"`、`backend_name = "bge-zh"`）。

**embed_batch 模板方法**：

```python
class EmbedderBase(ABC):
    def _embed_batch(self, texts: list[str], memory_action: str = "none") -> list[list[float]]:
        """子类可 override 做真批量推理（如 OnnxEmbedder）。默认循环 _embed()。"""
        return [self._embed(t, memory_action) for t in texts]

    def embed_batch(self, texts: list[str], memory_action: str = "none") -> list[list[float]]:
        """批量嵌入（带可观测性埋点）。"""
        collector = MetricsCollector.get()
        start = time.perf_counter()
        try:
            return self._embed_batch(texts, memory_action)
        finally:
            collector.observe("embed_batch_duration_seconds",
                              time.perf_counter() - start,
                              labels={"backend": self._backend_name()})
```

OnnxEmbedder 的 `embed_batch` → `_embed_batch` 重命名（1 个文件，已有真批量推理）。

**CachedEmbedder**（`src/septmuse/embedders/cached.py`）：

```python
class CachedEmbedder(EmbedderBase):
    def _embed(self, text: str, memory_action: str = "none") -> list[float]:
        key = (text, memory_action)
        if key in self._cache:
            MetricsCollector.get().inc("embed_cache_hits_total")
            return self._cache[key].copy()
        MetricsCollector.get().inc("embed_cache_misses_total")
        result = self._inner._embed(text, memory_action)
        self._cache[key] = result
        return result.copy()
```

**LLMBase**（`src/septmuse/providers/llms/base.py`）：

```python
class LLMBase(ABC):
    @abstractmethod
    def _complete(self, system_prompt: str, user_prompt: str) -> str:
        """子类实现实际 LLM 调用。"""
        ...

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """调用 LLM（带可观测性埋点）。"""
        collector = MetricsCollector.get()
        start = time.perf_counter()
        try:
            result = self._complete(system_prompt, user_prompt)
            collector.inc("llm_calls_total", {"provider": self._provider_name(), "status": "success"})
            return result
        except Exception:
            collector.inc("llm_calls_total", {"provider": self._provider_name(), "status": "error"})
            raise
        finally:
            collector.observe("llm_call_duration_seconds",
                              time.perf_counter() - start,
                              labels={"provider": self._provider_name()})
```

所有 LLM 子类：`def complete(` → `def _complete(`（机械重命名，4 个文件）。

每个子类需添加 `provider_name` 类属性或 `_provider_name()` 方法（如 `provider_name = "openai"`、`provider_name = "ollama"`）。基类提供默认 `_provider_name()`：`return getattr(self, "provider_name", type(self).__name__.lower())`。

## 6. BusinessMetricsCollector

`business.py` — pull-on-scrape，实现 `prometheus_client.Collector` 接口。

```python
class BusinessMetricsCollector(Collector):
    """prometheus Collector — 每次 scrape 触发 collect()，查 DB 算值。"""

    def __init__(self, store, db_path: str | None = None):
        self.store = store
        self.db_path = db_path
        self._start_time = time.time()

    def _query(self, sql: str) -> list[tuple]:
        """通过 store.engine 执行原始 SQL，返回行列表。"""
        from sqlalchemy import text
        with self.store.engine.connect() as conn:
            return conn.execute(text(sql)).fetchall()

    def collect(self) -> Iterator[MetricBase]:
        # memories_total (按 state)
        for state, count in self._query("SELECT state, COUNT(*) FROM memories GROUP BY state"):
            ...  # yield GaugeMetricFamily with labels

        # db_size_bytes
        if self.db_path:
            yield GaugeMetricFamily("septmuse_db_size_bytes", "DB 文件大小", value=os.path.getsize(self.db_path))

        # ... 其余同理
```

注册方式：`prometheus_client.REGISTRY.register(collector)`，Prometheus 拉取时自动调用 `collect()`。

查询 SQL（全部简单 COUNT，SQLite <1ms）：
- `SELECT state, COUNT(*) FROM memories GROUP BY state`
- `SELECT COUNT(*) FROM memory_blocks`
- `SELECT COUNT(*) FROM septmuse_entities`
- `SELECT SUM(LENGTH(value)), SUM(LENGTH(metadata)) FROM memories`（memory_size_bytes 按 type=value/metadata）
- `SELECT COUNT(*) FROM memory_vectors`（vector_index_size）

## 7. init_metrics 入口

`__init__.py` — 在 `create_app()` 中调用。

```python
def init_metrics(app: FastAPI, store, config) -> None:
    """初始化可观测性指标系统。opt-in：SEPTMUSE_METRICS=true 时启用。"""
    import os
    from septmuse.core.logging import get_logger
    logger = get_logger(__name__)

    if os.getenv("SEPTMUSE_METRICS", "").lower() not in ("1", "true", "yes"):
        return  # opt-in，未设不挂载

    MetricsCollector.get().configure(enabled=True)
    logger.info("metrics_enabled", endpoint="/metrics")

    # 挂 REST 中间件
    app.add_middleware(PrometheusMiddleware)

    # 业务指标 collector
    from prometheus_client import REGISTRY
    REGISTRY.register(BusinessMetricsCollector(store, getattr(config, "db_path", None)))

    # /metrics 加入豁免路径（免 API key）
    from septmuse.api.auth import EXEMPT_PATHS
    EXEMPT_PATHS.add("/metrics")

    # /metrics 端点
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    metrics_path = os.getenv("SEPTMUSE_METRICS_PATH", "/metrics")

    @app.get(metrics_path, tags=["observability"])
    async def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
```

调用位置：`create_app()` 中 `setup_auth(app)` 之后，`register_routes(app, ...)` 之前。

## 8. 进程指标

`prometheus_client` 自带 `ProcessCollector`（RSS、线程数、GC、CPU、fd），默认启用，不需要额外代码。`GC` 指标也自动收集。

## 9. 测试策略

### 9.1 全局状态管理

`prometheus_client` 的 `REGISTRY` 是全局单例，同名指标不能重复注册。测试用 `MetricsCollector.reset()` 隔离。

```python
# tests/unit/test_observability/conftest.py
import pytest
from septmuse.observability.collector import MetricsCollector

@pytest.fixture(autouse=True)
def _reset_metrics():
    """每个测试前重置 metrics 单例 + 注销所有自定义指标。"""
    MetricsCollector.reset()
    yield
    MetricsCollector.reset()
```

### 9.2 测试文件

```
tests/unit/test_observability/
├── conftest.py              # _reset_metrics autouse fixture
├── test_collector.py         # 单例、no-op 行为、inc/observe/set_gauge
├── test_middleware.py        # RED 指标：请求计数、状态码、延迟、in_flight
├── test_business_collector.py # pull-on-scrape：mock store，验证 SQL 查询
├── test_hooks.py             # @track_operation 装饰器
└── test_embed_abc.py         # 模板方法：_embed 调用后 embed_duration_seconds 记录
```

### 9.3 测试要点

- **no-op 验证**：`configure(enabled=False)` 后 `inc()/observe()` 不报错、不产生指标
- **中间件测试**：用 FastAPI `TestClient` 发请求，`REGISTRY.collect()` 断言 `api_requests_total` 增量、`api_request_duration_seconds` 有 bucket
- **业务 collector 测试**：mock store 的 `engine` 属性返回内存 SQLite engine，插入测试数据，验证 `collect()` yield 的 MetricFamily 值正确
- **模板方法测试**：创建 dummy embedder 子类实现 `_embed()`，调用 `embed()`，验证 `embed_duration_seconds` 被记录
- **CachedEmbedder 测试**：验证 cache hit/miss 计数正确

## 10. 环境变量

| 变量 | 默认 | 作用 |
|------|------|------|
| `SEPTMUSE_METRICS` | 未设 | `true`/`1`/`yes` 启用 /metrics 端点 + 全部埋点 |
| `SEPTMUSE_METRICS_PATH` | `/metrics` | 端点路径（可自定义） |

## 11. pyproject.toml 改动

`prometheus_client` 加入 `[project.dependencies]`（核心依赖，非 optional extra）。

## 12. AGENTS.md 更新

新增「可观测性」段落，记录：
- `SEPTMUSE_METRICS` 环境变量
- `/metrics` 端点
- `MetricsCollector` 单例 + opt-in 行为
- 模板方法重命名（`embed` → `_embed`，`complete` → `_complete`）
- 测试 fixture（`_reset_metrics` autouse）

## 13. 修改范围

| 文件 | 改动 |
|------|------|
| `src/septmuse/observability/__init__.py` | 新建 — `init_metrics()` 入口 |
| `src/septmuse/observability/collector.py` | 新建 — `MetricsCollector` 单例 |
| `src/septmuse/observability/middleware.py` | 新建 — `PrometheusMiddleware` |
| `src/septmuse/observability/business.py` | 新建 — `BusinessMetricsCollector` |
| `src/septmuse/observability/hooks.py` | 新建 — `@track_operation` 装饰器 |
| `src/septmuse/embedders/base.py` | 模板方法：`embed()` → `_embed()` + `embed_batch()` → `_embed_batch()` + 埋点 + `_backend_name()` |
| `src/septmuse/embedders/cached.py` | `_embed()` + cache hit/miss 计数 |
| `src/septmuse/embedders/hash.py` | `embed` → `_embed` 重命名 + `backend_name = "hash"` |
| `src/septmuse/embedders/onnx.py` | `embed` → `_embed` + `embed_batch` → `_embed_batch` 重命名 + `backend_name` |
| `src/septmuse/embedders/auto_onnx.py` | `embed` → `_embed` 重命名 + `backend_name` |
| `src/septmuse/embedders/openai.py` | `embed` → `_embed` 重命名 + `backend_name` |
| `src/septmuse/embedders/mock.py` | `embed` → `_embed` 重命名 + `backend_name` |
| `src/septmuse/embedders/ollama.py` | `embed` → `_embed` 重命名 + `backend_name` |
| `src/septmuse/embedders/together.py` | `embed` → `_embed` 重命名 + `backend_name` |
| `src/septmuse/embedders/lmstudio.py` | `embed` → `_embed` 重命名 + `backend_name` |
| `src/septmuse/embedders/azure_openai.py` | `embed` → `_embed` 重命名 + `backend_name` |
| `src/septmuse/embedders/gemini.py` | `embed` → `_embed` 重命名 + `backend_name` |
| `src/septmuse/embedders/vertexai.py` | `embed` → `_embed` 重命名 + `backend_name` |
| `src/septmuse/embedders/huggingface.py` | `embed` → `_embed` 重命名 + `backend_name` |
| `src/septmuse/embedders/aws_bedrock.py` | `embed` → `_embed` 重命名 + `backend_name` |
| `src/septmuse/embedders/fastembed.py` | `embed` → `_embed` 重命名 + `backend_name` |
| `src/septmuse/embedders/langchain.py` | `embed` → `_embed` 重命名 + `backend_name` |
| `src/septmuse/providers/llms/base.py` | 模板方法：`complete()` → `_complete()` + 埋点 + `_provider_name()` |
| `src/septmuse/providers/llms/openai.py` | `complete` → `_complete` 重命名 + `provider_name` |
| `src/septmuse/providers/llms/ollama.py` | `complete` → `_complete` 重命名 + `provider_name` |
| `src/septmuse/providers/llms/anthropic.py` | `complete` → `_complete` 重命名 + `provider_name` |
| `src/septmuse/providers/llms/dashscope.py` | `complete` → `_complete` 重命名 + `provider_name` |
| `src/septmuse/api/rest/__init__.py` | `create_app()` 调用 `init_metrics()` |
| `src/septmuse/api/auth.py` | `EXEMPT_PATHS` 从 `frozenset` 改为 `set`（允许运行时 add `/metrics`） |
| `src/septmuse/memory/main.py` | facade 6 方法加 `@track_operation` |
| `src/septmuse/retrieval/hybrid.py` | 3 路检索加 component 装饰器 |
| `pyproject.toml` | `prometheus_client` 加入核心依赖 |
| `AGENTS.md` | 新增可观测性段落 |
| `tests/unit/test_observability/` | 6 测试文件 |
| **总计** | ~30 文件 |

## 14. 未来扩展

- **多进程**：`PROMETHEUS_MULTIPROC_DIR` 环境变量 + `MultiProcessCollector`（uvicorn `--workers N` 场景）
- **分布式追踪**：OpenTelemetry（跨服务调用链追踪）
- **Grafana Dashboard**：JSON 模板 + `docs/observability/grafana-dashboard.json`
- **告警规则**：Prometheus AlertManager rule 文件

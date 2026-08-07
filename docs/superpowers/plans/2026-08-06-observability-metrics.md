# 可观测性指标系统 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 SeptMuse 添加 Prometheus 可观测性指标系统，覆盖 RED / 资源 / 业务三维指标，`SEPTMUSE_METRICS=true` opt-in。

**Architecture:** `MetricsCollector` 单例封装所有 Prometheus 指标，未启用时 no-op。三层埋点：REST 中间件（全自动 RED）+ facade 装饰器（业务操作粒度）+ 底层 ABC 模板方法（embed/LLM 全覆盖）。`BusinessMetricsCollector` pull-on-scrape 查 DB。

**Tech Stack:** prometheus_client (纯 Python ~50KB), FastAPI/Starlette middleware, SQLAlchemy text() for raw SQL

**Spec:** `docs/superpowers/specs/2026-08-06-observability-metrics-design.md`

## Global Constraints

- **包名** `septmuse`，src/ layout，`PYTHONPATH=src` 运行测试
- **ruff line-length 120**，`ruff check` 必须通过。**禁用 `ruff format`**（Windows 清空文件 bug）
- **代码注释中文**，不暴露开源库参考来源
- **测试**：用户指示"开发完统一测试"——全部代码开发完后统一跑 ruff + pytest
- **基类名称**：`Embedder`（非 `EmbedderBase`），`LLM`（非 `LLMBase`）
- **LLM 路径**：`src/septmuse/llms/`（非 `src/septmuse/providers/llms/`）
- **embed 签名**：`def embed(self, text: str, memory_action: str | None = None) -> list[float]`
- **conda 环境**：`SeptMuse`，Python 3.12
- **测试保护**：现有测试固定不动，禁止改测试绕过缺陷

---

## File Structure

| 文件 | 职责 |
|------|------|
| `src/septmuse/observability/__init__.py` | `init_metrics(app, store, config)` 入口 |
| `src/septmuse/observability/collector.py` | `MetricsCollector` 单例 — 所有指标定义 + inc/dec/observe/set_gauge |
| `src/septmuse/observability/middleware.py` | `PrometheusMiddleware` — REST RED 指标自动记录 |
| `src/septmuse/observability/business.py` | `BusinessMetricsCollector` — pull-on-scrape 业务指标 |
| `src/septmuse/observability/hooks.py` | `track_operation` 装饰器 + `time_block` 上下文管理器 |

---

### Task 1: MetricsCollector 单例 + pyproject.toml 依赖

**Files:**
- Create: `src/septmuse/observability/__init__.py` (空占位，Task 8 填充)
- Create: `src/septmuse/observability/collector.py`
- Modify: `pyproject.toml` (加 prometheus_client 到 `[project.dependencies]`)

**Interfaces:**
- Produces: `MetricsCollector.get() -> MetricsCollector`, `MetricsCollector.reset()`, `MetricsCollector.configure(enabled: bool)`, `MetricsCollector.inc(name, labels, amount)`, `MetricsCollector.dec(name, labels, amount)`, `MetricsCollector.observe(name, value, labels)`, `MetricsCollector.set_gauge(name, value, labels)`

- [ ] **Step 1: 添加 prometheus_client 到 pyproject.toml**

在 `[project.dependencies]` 列表中添加 `"prometheus-client>=0.16.0"`。

```toml
# 在 dependencies 列表中添加
"prometheus-client>=0.16.0",
```

- [ ] **Step 2: 安装依赖**

Run: `pip install prometheus-client`
Expected: Successfully installed

- [ ] **Step 3: 创建 observability 包**

创建 `src/septmuse/observability/__init__.py`，内容为空（Task 8 填充 `init_metrics`）。

```python
"""SeptMuse 可观测性指标系统。"""
```

- [ ] **Step 4: 创建 MetricsCollector 单例**

创建 `src/septmuse/observability/collector.py`：

```python
"""MetricsCollector 单例 — 封装所有 Prometheus 指标。

未启用时 (SEPTMUSE_METRICS 未设) 所有方法 no-op，零开销。
启用时在 configure() 中创建所有 Counter/Histogram/Gauge，存入 _metrics dict。
"""

from __future__ import annotations

from typing import Any

# prometheus_client 延迟 import — configure(enabled=True) 时才加载


class MetricsCollector:
    """单例。未启用时所有方法 no-op。"""

    _instance: MetricsCollector | None = None

    @classmethod
    def get(cls) -> MetricsCollector:
        """全局获取单例。首次调用时初始化为 disabled。"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        self._enabled = False
        self._metrics: dict[str, Any] = {}

    @property
    def enabled(self) -> bool:
        return self._enabled

    def configure(self, enabled: bool) -> None:
        """初始化所有 Prometheus 指标。enabled=False 时保持 no-op。"""
        self._enabled = enabled
        if not enabled:
            return

        from prometheus_client import Counter, Gauge, Histogram

        self._metrics["api_requests_total"] = Counter(
            "septmuse_api_requests_total",
            "API 请求总数",
            labelnames=["endpoint", "method", "status"],
        )
        self._metrics["api_request_duration_seconds"] = Histogram(
            "septmuse_api_request_duration_seconds",
            "API 请求延迟",
            labelnames=["endpoint", "method"],
        )
        self._metrics["api_requests_in_flight"] = Gauge(
            "septmuse_api_requests_in_flight",
            "当前在途请求数",
            labelnames=["endpoint"],
        )
        self._metrics["memory_operation_duration_seconds"] = Histogram(
            "septmuse_memory_operation_duration_seconds",
            "记忆操作延迟",
            labelnames=["operation"],
        )
        self._metrics["embed_duration_seconds"] = Histogram(
            "septmuse_embed_duration_seconds",
            "嵌入延迟",
            labelnames=["backend"],
        )
        self._metrics["embed_batch_duration_seconds"] = Histogram(
            "septmuse_embed_batch_duration_seconds",
            "批量嵌入延迟",
            labelnames=["backend"],
        )
        self._metrics["embed_cache_hits_total"] = Counter(
            "septmuse_embed_cache_hits_total",
            "嵌入缓存命中次数",
        )
        self._metrics["embed_cache_misses_total"] = Counter(
            "septmuse_embed_cache_misses_total",
            "嵌入缓存未命中次数",
        )
        self._metrics["llm_call_duration_seconds"] = Histogram(
            "septmuse_llm_call_duration_seconds",
            "LLM 调用延迟",
            labelnames=["provider"],
        )
        self._metrics["llm_calls_total"] = Counter(
            "septmuse_llm_calls_total",
            "LLM 调用总数",
            labelnames=["provider", "status"],
        )
        self._metrics["hybrid_search_components_seconds"] = Histogram(
            "septmuse_hybrid_search_components_seconds",
            "混合检索各组件延迟",
            labelnames=["component"],
        )
        self._metrics["uptime_seconds"] = Gauge(
            "septmuse_uptime_seconds",
            "进程运行时间",
        )

    def inc(self, name: str, labels: dict[str, str] | None = None, amount: float = 1) -> None:
        """Counter/Gauge 增量。"""
        if not self._enabled:
            return
        metric = self._metrics.get(name)
        if metric is None:
            return
        if labels:
            metric.labels(**labels).inc(amount)
        else:
            metric.inc(amount)

    def dec(self, name: str, labels: dict[str, str] | None = None, amount: float = 1) -> None:
        """Gauge 减量（in_flight 等）。"""
        if not self._enabled:
            return
        metric = self._metrics.get(name)
        if metric is None:
            return
        if labels:
            metric.labels(**labels).dec(amount)
        else:
            metric.dec(amount)

    def observe(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        """Histogram 观测。"""
        if not self._enabled:
            return
        metric = self._metrics.get(name)
        if metric is None:
            return
        if labels:
            metric.labels(**labels).observe(value)
        else:
            metric.observe(value)

    def set_gauge(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        """Gauge 设值（业务指标绝对值）。"""
        if not self._enabled:
            return
        metric = self._metrics.get(name)
        if metric is None:
            return
        if labels:
            metric.labels(**labels).set(value)
        else:
            metric.set(value)

    @classmethod
    def reset(cls) -> None:
        """测试用 — 清单例 + 从 REGISTRY 注销所有 septmuse_* 指标。"""
        if cls._instance is not None and cls._instance._enabled:
            try:
                from prometheus_client import REGISTRY

                to_remove = [
                    name for name in REGISTRY._names_to_collectors if name.startswith("septmuse_")
                ]
                for name in to_remove:
                    collector = REGISTRY._names_to_collectors.pop(name, None)
                    if collector is not None:
                        REGISTRY.unregister(collector)
            except Exception:
                pass
        cls._instance = None
```

- [ ] **Step 5: 验证 import 无报错**

Run: `$env:PYTHONPATH = "src"; python -c "from septmuse.observability.collector import MetricsCollector; c = MetricsCollector.get(); c.inc('test'); print('OK')"`
Expected: OK

---

### Task 2: hooks.py — track_operation 装饰器 + time_block 上下文管理器

**Files:**
- Create: `src/septmuse/observability/hooks.py`

**Interfaces:**
- Consumes: `MetricsCollector.get()` from Task 1
- Produces: `track_operation(operation: str)` 装饰器（同步+异步兼容）, `time_block(metric_name, labels)` 上下文管理器

- [ ] **Step 1: 创建 hooks.py**

```python
"""埋点辅助 — track_operation 装饰器 + time_block 上下文管理器。

track_operation: 装饰 Memory facade 方法，自动记录 memory_operation_duration_seconds。
time_block: 用于 HybridRetriever 内部函数的 component 级计时。
"""

from __future__ import annotations

import asyncio
import functools
import time
from collections.abc import Iterator
from contextlib import contextmanager

from septmuse.observability.collector import MetricsCollector


def track_operation(operation: str):
    """装饰器 — 记录 memory_operation_duration_seconds（同步+异步兼容）。

    Args:
        operation: 操作名 (add/search/update/delete/get/invalidate)
    """

    def decorator(func):
        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                collector = MetricsCollector.get()
                start = time.perf_counter()
                try:
                    return await func(*args, **kwargs)
                finally:
                    collector.observe(
                        "memory_operation_duration_seconds",
                        time.perf_counter() - start,
                        labels={"operation": operation},
                    )

            return async_wrapper

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            collector = MetricsCollector.get()
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                collector.observe(
                    "memory_operation_duration_seconds",
                    time.perf_counter() - start,
                    labels={"operation": operation},
                )

        return sync_wrapper

    return decorator


@contextmanager
def time_block(metric_name: str, labels: dict[str, str] | None = None) -> Iterator[None]:
    """上下文管理器 — 记录指定 Histogram 指标的耗时。

    用于 HybridRetriever 内部函数的 component 级计时：

        with time_block("hybrid_search_components_seconds", {"component": "vector"}):
            results = self.store.search(...)
    """
    collector = MetricsCollector.get()
    start = time.perf_counter()
    try:
        yield
    finally:
        collector.observe(metric_name, time.perf_counter() - start, labels=labels)
```

- [ ] **Step 2: 验证 import**

Run: `$env:PYTHONPATH = "src"; python -c "from septmuse.observability.hooks import track_operation, time_block; print('OK')"`
Expected: OK

---

### Task 3: Embedder 模板方法 + 全部 embedder 子类重命名

**Files:**
- Modify: `src/septmuse/embedders/base.py` — 模板方法：`embed` → `_embed` + `embed_batch` → `_embed_batch`
- Modify: `src/septmuse/embedders/_openai_compatible.py` — `embed` → `_embed`, `embed_batch` → `_embed_batch`
- Modify: `src/septmuse/embedders/hash.py` — `embed` → `_embed` + `backend_name`
- Modify: `src/septmuse/embedders/onnx.py` — `embed` → `_embed`, `embed_batch` → `_embed_batch` + `backend_name`
- Modify: `src/septmuse/embedders/auto.py` — `embed` → `_embed` + `backend_name`
- Modify: `src/septmuse/embedders/mock.py` — `embed` → `_embed` + `backend_name`
- Modify: `src/septmuse/embedders/sentence_transformers.py` — `embed` → `_embed` + `backend_name`
- Modify: `src/septmuse/embedders/langchain.py` — `embed` → `_embed` + `backend_name`
- Modify: `src/septmuse/embedders/aws_bedrock.py` — `embed` → `_embed` + `backend_name`
- Modify: `src/septmuse/embedders/huggingface.py` — `embed` → `_embed` + `backend_name`
- Modify: `src/septmuse/embedders/fastembed.py` — `embed` → `_embed`, `embed_batch` → `_embed_batch` + `backend_name`
- Modify: `src/septmuse/embedders/langdetect.py` — 检查是否有 `embed` 方法（可能只是工具函数）
- Note: `openai.py`, `azure_openai.py`, `together.py`, `lmstudio.py`, `gemini.py`, `vertexai.py` 继承 `_OpenAICompatibleEmbedder`，不 override `embed`，不需改

**Interfaces:**
- Consumes: `MetricsCollector.get()` from Task 1
- Produces: `Embedder.embed()` 非抽象方法（带埋点），`Embedder._embed()` 抽象方法（子类实现），`Embedder.embed_batch()` 非抽象方法（带埋点），`Embedder._embed_batch()` 非抽象方法（子类可 override），`Embedder._backend_name()`

- [ ] **Step 1: 修改 base.py — 模板方法**

将 `src/septmuse/embedders/base.py` 的 `embed` 改为 `_embed`（abstract），添加 `embed()` wrapper，`embed_batch` 改为 `_embed_batch`，添加 `embed_batch()` wrapper，添加 `_backend_name()`：

```python
"""嵌入模型抽象基类。

所有 embedder 实现此接口, 用于把文本转为向量供相似检索。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import time

from septmuse.observability.collector import MetricsCollector


class Embedder(ABC):
    """嵌入模型抽象。"""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """返回嵌入维度。"""
        ...

    @abstractmethod
    def _embed(self, text: str, memory_action: str | None = None) -> list[float]:
        """嵌入单条文本, 返回归一化向量。子类实现。"""
        ...

    def embed(self, text: str, memory_action: str | None = None) -> list[float]:
        """嵌入单条文本（带可观测性埋点）。"""
        collector = MetricsCollector.get()
        start = time.perf_counter()
        try:
            return self._embed(text, memory_action)
        finally:
            collector.observe(
                "embed_duration_seconds",
                time.perf_counter() - start,
                labels={"backend": self._backend_name()},
            )

    def _embed_batch(self, texts: list[str], memory_action: str | None = None) -> list[list[float]]:
        """批量嵌入 — 默认逐条调用 _embed(), 子类可 override 实现真批量推理。"""
        return [self._embed(t, memory_action) for t in texts]

    def embed_batch(self, texts: list[str], memory_action: str | None = None) -> list[list[float]]:
        """批量嵌入（带可观测性埋点）。"""
        collector = MetricsCollector.get()
        start = time.perf_counter()
        try:
            return self._embed_batch(texts, memory_action)
        finally:
            collector.observe(
                "embed_batch_duration_seconds",
                time.perf_counter() - start,
                labels={"backend": self._backend_name()},
            )

    def _backend_name(self) -> str:
        """返回后端名称（用于指标标签）。子类应设置 self.backend_name。"""
        return getattr(self, "backend_name", type(self).__name__.lower())
```

- [ ] **Step 2: 修改 _openai_compatible.py — embed → _embed, embed_batch → _embed_batch**

在 `src/septmuse/embedders/_openai_compatible.py` 中：
- `def embed(` → `def _embed(`
- `def embed_batch(` → `def _embed_batch(`
- 在 `__init__` 中添加 `self.backend_name = "openai_compatible"`（子类会在自己的 __init__ 中覆盖）

- [ ] **Step 3: 对每个有 `def embed(` 的 embedder 子类执行重命名**

对以下文件，将 `def embed(` 改为 `def _embed(`，并在 `__init__` 或类体中添加 `backend_name` 属性：

| 文件 | backend_name |
|------|-------------|
| `hash.py` | `"hash"` |
| `onnx.py` | `"onnx"` |
| `auto.py` | `"auto_onnx"` |
| `mock.py` | `"mock"` |
| `sentence_transformers.py` | `"st"` |
| `langchain.py` | `"langchain"` |
| `aws_bedrock.py` | `"aws_bedrock"` |
| `huggingface.py` | `"huggingface"` |
| `fastembed.py` | `"fastembed"` |

对 `onnx.py` 和 `fastembed.py`：同时将 `def embed_batch(` 改为 `def _embed_batch(`。

对 `langdetect.py`：检查是否有 `def embed(` — 如果有则重命名，如果没有则跳过。

对 `_openai_compatible.py` 的子类（`openai.py`, `azure_openai.py`, `together.py`, `lmstudio.py`, `gemini.py`, `vertexai.py`）：在 `__init__` 中设置 `self.backend_name` 为对应名称（如 `"openai"`, `"azure_openai"` 等），不需要重命名 `embed`（继承自 `_OpenAICompatibleEmbedder`）。

- [ ] **Step 4: 验证所有 embedder import 无报错**

Run: `$env:PYTHONPATH = "src"; python -c "from septmuse.embedders.hash import HashEmbedder; e = HashEmbedder(); print(e.embed('test')); print(e._backend_name())"`
Expected: 嵌入向量 + `hash`

---

### Task 4: CachedEmbedder cache 计数 + _embed 委托

**Files:**
- Modify: `src/septmuse/embedders/cached.py`

**Interfaces:**
- Consumes: `MetricsCollector` from Task 1, `Embedder._embed` / `Embedder._embed_batch` from Task 3

- [ ] **Step 1: 修改 CachedEmbedder**

在 `src/septmuse/embedders/cached.py` 中：
- `def embed(` → `def _embed(`
- `def embed_batch(` → `def _embed_batch(`
- 在 `_embed` 中：cache hit 时 `MetricsCollector.get().inc("embed_cache_hits_total")`，cache miss 时 `MetricsCollector.get().inc("embed_cache_misses_total")`
- 在 `_embed` 中：调用 `self._inner._embed()` 而非 `self._inner.embed()`（避免双重计时）
- 在 `_embed_batch` 中：同样改用 `self._inner._embed_batch()` + cache hit/miss 计数
- 添加 `backend_name = "cached"` 类属性

修改后的 `_embed` 方法：

```python
def _embed(self, text: str, memory_action: str | None = None) -> list[float]:
    from septmuse.observability.collector import MetricsCollector

    cache_key = (text, memory_action)
    with self._lock:
        cached = self._cache.get(cache_key)
        if cached is not None:
            self._cache.move_to_end(cache_key)
            self._hits += 1
            MetricsCollector.get().inc("embed_cache_hits_total")
            return list(cached)
        self._misses += 1
        MetricsCollector.get().inc("embed_cache_misses_total")

    vec = self._inner._embed(text, memory_action=memory_action)

    with self._lock:
        self._cache[cache_key] = vec
        self._cache.move_to_end(cache_key)
        if len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)
    return list(vec)
```

同样修改 `_embed_batch` 中的 `self._inner.embed_batch(...)` → `self._inner._embed_batch(...)`，并在 cache hit/miss 时添加计数。

- [ ] **Step 2: 验证 CachedEmbedder**

Run: `$env:PYTHONPATH = "src"; python -c "from septmuse.embedders.cached import CachedEmbedder; from septmuse.embedders.hash import HashEmbedder; c = CachedEmbedder(HashEmbedder()); print(c.embed('test')); print(c.embed('test')); print(c.stats)"`
Expected: 嵌入向量 + 嵌入向量 + `{'hits': 1, 'misses': 1, ...}`

---

### Task 5: LLM 模板方法 + 全部 LLM 子类重命名

**Files:**
- Modify: `src/septmuse/llms/base.py` — 模板方法：`complete` → `_complete`
- Modify: `src/septmuse/llms/openai.py` — `complete` → `_complete` + `provider_name`
- Modify: `src/septmuse/llms/ollama.py` — `complete` → `_complete` + `provider_name`
- Modify: `src/septmuse/llms/anthropic.py` — `complete` → `_complete` + `provider_name`
- Modify: `src/septmuse/llms/dashscope.py` — `complete` → `_complete` + `provider_name`
- Modify: `src/septmuse/llms/deepseek.py` — `complete` → `_complete` + `provider_name`
- Modify: `src/septmuse/llms/gemini.py` — `complete` → `_complete` + `provider_name`
- Modify: `src/septmuse/llms/groq.py` — `complete` → `_complete` + `provider_name`
- Modify: `src/septmuse/llms/litellm.py` — `complete` → `_complete` + `provider_name`
- Modify: `src/septmuse/llms/mock.py` — `complete` → `_complete` + `provider_name`

**Interfaces:**
- Consumes: `MetricsCollector.get()` from Task 1
- Produces: `LLM.complete()` 非抽象方法（带埋点），`LLM._complete()` 抽象方法（子类实现），`LLM._provider_name()`

- [ ] **Step 1: 修改 base.py — 模板方法**

将 `src/septmuse/llms/base.py` 的 `complete` 改为 `_complete`（abstract），添加 `complete()` wrapper + `_provider_name()`：

```python
"""LLM 抽象基类。

所有 LLM 实现此接口, 用于记忆抽取 (infer=True)。
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod

from septmuse.observability.collector import MetricsCollector


class LLM(ABC):
    """LLM 抽象。"""

    @abstractmethod
    def _complete(self, system_prompt: str, user_prompt: str) -> str:
        """同步补全, 返回 LLM 输出文本。子类实现。"""
        ...

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """调用 LLM（带可观测性埋点）。"""
        collector = MetricsCollector.get()
        start = time.perf_counter()
        try:
            result = self._complete(system_prompt, user_prompt)
            collector.inc(
                "llm_calls_total",
                labels={"provider": self._provider_name(), "status": "success"},
            )
            return result
        except Exception:
            collector.inc(
                "llm_calls_total",
                labels={"provider": self._provider_name(), "status": "error"},
            )
            raise
        finally:
            collector.observe(
                "llm_call_duration_seconds",
                time.perf_counter() - start,
                labels={"provider": self._provider_name()},
            )

    def _provider_name(self) -> str:
        """返回 provider 名称（用于指标标签）。"""
        return getattr(self, "provider_name", type(self).__name__.lower())
```

- [ ] **Step 2: 对每个 LLM 子类执行重命名**

对以下文件，将 `def complete(` 改为 `def _complete(`，并在 `__init__` 或类体中添加 `provider_name` 属性：

| 文件 | provider_name |
|------|--------------|
| `openai.py` | `"openai"` |
| `ollama.py` | `"ollama"` |
| `anthropic.py` | `"anthropic"` |
| `dashscope.py` | `"dashscope"` |
| `deepseek.py` | `"deepseek"` |
| `gemini.py` | `"gemini"` |
| `groq.py` | `"groq"` |
| `litellm.py` | `"litellm"` |
| `mock.py` | `"mock"` |

- [ ] **Step 3: 验证 MockLLM**

Run: `$env:PYTHONPATH = "src"; python -c "from septmuse.llms.mock import MockLLM; llm = MockLLM(); print(llm.complete('sys', 'hello')); print(llm._provider_name())"`
Expected: mock 输出 + `mock`

---

### Task 6: PrometheusMiddleware

**Files:**
- Create: `src/septmuse/observability/middleware.py`

**Interfaces:**
- Consumes: `MetricsCollector` from Task 1

- [ ] **Step 1: 创建 middleware.py**

```python
"""PrometheusMiddleware — 自动记录 REST API RED 指标。

RED = Rate / Errors / Duration
- Rate: api_requests_total (Counter, endpoint+method+status)
- Duration: api_request_duration_seconds (Histogram, endpoint+method)
- In-flight: api_requests_in_flight (Gauge, endpoint)
"""

from __future__ import annotations

import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from septmuse.observability.collector import MetricsCollector


class PrometheusMiddleware(BaseHTTPMiddleware):
    """自动记录所有 HTTP 请求的 RED 指标。"""

    async def dispatch(self, request: Request, call_next) -> Response:
        collector = MetricsCollector.get()
        endpoint = request.url.path
        method = request.method

        collector.inc("api_requests_in_flight", labels={"endpoint": endpoint})

        start = time.perf_counter()
        try:
            response = await call_next(request)
            return response
        finally:
            elapsed = time.perf_counter() - start
            collector.dec("api_requests_in_flight", labels={"endpoint": endpoint})
            status = str(getattr(response, "status_code", 500))
            collector.inc(
                "api_requests_total",
                labels={"endpoint": endpoint, "method": method, "status": status},
            )
            collector.observe(
                "api_request_duration_seconds",
                elapsed,
                labels={"endpoint": endpoint, "method": method},
            )
```

- [ ] **Step 2: 验证 import**

Run: `$env:PYTHONPATH = "src"; python -c "from septmuse.observability.middleware import PrometheusMiddleware; print('OK')"`
Expected: OK

---

### Task 7: BusinessMetricsCollector

**Files:**
- Create: `src/septmuse/observability/business.py`

**Interfaces:**
- Consumes: `store.engine` (SQLAlchemy Engine) from ORMMemoryStore, `os.path.getsize` for db_size
- Produces: `BusinessMetricsCollector(store, db_path)` — 实现 `prometheus_client.Collector`

- [ ] **Step 1: 创建 business.py**

```python
"""BusinessMetricsCollector — pull-on-scrape 业务指标。

实现 prometheus_client.Collector 接口，Prometheus 拉取 /metrics 时触发 collect()，
查询 DB 算值，返回 GaugeMetricFamily。
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from typing import Any

from prometheus_client import Collector, GaugeMetricFamily


class BusinessMetricsCollector(Collector):
    """prometheus Collector — 每次 scrape 触发 collect()，查 DB 算值。"""

    def __init__(self, store: Any, db_path: str | None = None) -> None:
        self.store = store
        self.db_path = db_path
        self._start_time = time.time()

    def _query(self, sql: str) -> list[tuple]:
        """通过 store.engine 执行原始 SQL，返回行列表。"""
        try:
            from sqlalchemy import text

            with self.store.engine.connect() as conn:
                return list(conn.execute(text(sql)).fetchall())
        except Exception:
            return []

    def collect(self) -> Iterator[GaugeMetricFamily]:
        # memories_total (按 state)
        states = self._query("SELECT state, COUNT(*) FROM memories GROUP BY state")
        if states:
            fam = GaugeMetricFamily(
                "septmuse_memories_total",
                "记忆总数（按 state）",
                labels=["state"],
            )
            for state, count in states:
                fam.add_metric([str(state)], float(count))
            yield fam

        # blocks_total
        blocks = self._query("SELECT COUNT(*) FROM memory_blocks")
        if blocks:
            yield GaugeMetricFamily(
                "septmuse_blocks_total",
                "block 总数",
                value=float(blocks[0][0]),
            )

        # entities_total
        entities = self._query("SELECT COUNT(*) FROM septmuse_entities")
        if entities:
            yield GaugeMetricFamily(
                "septmuse_entities_total",
                "实体总数",
                value=float(entities[0][0]),
            )

        # memory_size_bytes (按 type)
        sizes = self._query(
            "SELECT 'value' AS type, SUM(LENGTH(content)) AS size FROM memories "
            "UNION ALL "
            "SELECT 'metadata' AS type, SUM(LENGTH(metadata_json)) AS size FROM memories"
        )
        if sizes:
            fam = GaugeMetricFamily(
                "septmuse_memory_size_bytes",
                "记忆数据大小",
                labels=["type"],
            )
            for type_name, size in sizes:
                fam.add_metric([str(type_name)], float(size or 0))
            yield fam

        # db_size_bytes
        if self.db_path and os.path.exists(self.db_path):
            yield GaugeMetricFamily(
                "septmuse_db_size_bytes",
                "DB 文件大小",
                value=float(os.path.getsize(self.db_path)),
            )

        # vector_index_size
        vectors = self._query("SELECT COUNT(*) FROM memory_vectors")
        if vectors:
            yield GaugeMetricFamily(
                "septmuse_vector_index_size",
                "向量索引条数",
                value=float(vectors[0][0]),
            )

        # uptime_seconds
        yield GaugeMetricFamily(
            "septmuse_uptime_seconds",
            "进程运行时间",
            value=time.time() - self._start_time,
        )
```

- [ ] **Step 2: 验证 import**

Run: `$env:PYTHONPATH = "src"; python -c "from septmuse.observability.business import BusinessMetricsCollector; print('OK')"`
Expected: OK

---

### Task 8: init_metrics 入口 + create_app 集成 + auth 改动

**Files:**
- Modify: `src/septmuse/observability/__init__.py` — 填充 `init_metrics()`
- Modify: `src/septmuse/api/rest/__init__.py:440` — `create_app()` 调用 `init_metrics()`
- Modify: `src/septmuse/api/auth.py:40` — `EXEMPT_PATHS` 从 `frozenset` 改为 `set`

**Interfaces:**
- Consumes: `MetricsCollector` (Task 1), `PrometheusMiddleware` (Task 6), `BusinessMetricsCollector` (Task 7)
- Produces: `init_metrics(app, store, config)` — 在 create_app 中调用

- [ ] **Step 1: 修改 auth.py — EXEMPT_PATHS frozenset → set**

在 `src/septmuse/api/auth.py` 第 40 行，将 `frozenset` 改为 `set`：

```python
EXEMPT_PATHS = set(
    {
        "/docs",
        "/redoc",
        "/openapi.json",
        "/health",
        "/favicon.ico",
    }
)
```

- [ ] **Step 2: 填充 observability/__init__.py**

```python
"""SeptMuse 可观测性指标系统。"""

from __future__ import annotations

import os


def init_metrics(app, store, config) -> None:
    """初始化可观测性指标系统。opt-in：SEPTMUSE_METRICS=true 时启用。

    Args:
        app: FastAPI 实例
        store: MemoryStore 实例（用于业务指标查询）
        config: MemoryConfig（用于 db_path）
    """
    from septmuse.core.logging import get_logger
    from septmuse.observability.collector import MetricsCollector

    logger = get_logger(__name__)

    if os.getenv("SEPTMUSE_METRICS", "").lower() not in ("1", "true", "yes"):
        return

    MetricsCollector.get().configure(enabled=True)
    logger.info("metrics_enabled", endpoint="/metrics")

    # 挂 REST 中间件
    from septmuse.observability.middleware import PrometheusMiddleware

    app.add_middleware(PrometheusMiddleware)

    # 业务指标 collector
    from prometheus_client import REGISTRY

    from septmuse.observability.business import BusinessMetricsCollector

    db_path = getattr(config, "db_path", None)
    REGISTRY.register(BusinessMetricsCollector(store, db_path))

    # /metrics 加入豁免路径（免 API key）
    from septmuse.api.auth import EXEMPT_PATHS

    metrics_path = os.getenv("SEPTMUSE_METRICS_PATH", "/metrics")
    EXEMPT_PATHS.add(metrics_path)

    # /metrics 端点
    from fastapi import Response
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

    @app.get(metrics_path, tags=["observability"])
    async def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
```

- [ ] **Step 3: 修改 create_app — 调用 init_metrics**

在 `src/septmuse/api/rest/__init__.py` 的 `create_app()` 函数中，在 `setup_auth(app)` 之后（约第 562 行）、`register_routes(app, sync_memory, async_memory)` 之前（约第 577 行），添加：

```python
    # 可观测性指标 (opt-in: SEPTMUSE_METRICS=true)
    from septmuse.observability import init_metrics

    init_metrics(app, sync_memory.store, sync_memory.config)
```

- [ ] **Step 4: 验证 create_app 不报错**

Run: `$env:PYTHONPATH = "src"; python -c "from septmuse.api.rest import create_app; app = create_app(); print('OK')"`
Expected: OK（无 SEPTMUSE_METRICS 不会挂载 /metrics）

---

### Task 9: Facade 装饰器 — ExperimentalMemory 6 方法

**Files:**
- Modify: `src/septmuse/memory/main.py` — 在 `Memory` 类的 6 个方法上加 `@track_operation`

**Interfaces:**
- Consumes: `track_operation` from Task 2
- 注意：`ExperimentalMemory` 继承 `Memory`，装饰加在 `Memory` 的方法上即可

- [ ] **Step 1: 添加 import**

在 `src/septmuse/memory/main.py` 顶部添加：

```python
from septmuse.observability.hooks import track_operation
```

- [ ] **Step 2: 装饰 6 个方法**

在 `Memory` 类的以下方法上添加 `@track_operation("xxx")` 装饰器（在 `def` 上方）：

| 方法 | 装饰器 |
|------|--------|
| `add` (约 204 行) | `@track_operation("add")` |
| `search` | `@track_operation("search")` |
| `update` | `@track_operation("update")` |
| `delete` | `@track_operation("delete")` |
| `get` | `@track_operation("get")` |
| `invalidate` | `@track_operation("invalidate")` |

注意：`add` 方法很长，装饰器加在最上方 `def add(` 之前。

- [ ] **Step 3: 验证不报错**

Run: `$env:PYTHONPATH = "src"; python -c "from septmuse.memory.main import Memory; print('OK')"`
Expected: OK

---

### Task 10: HybridRetriever 组件计时

**Files:**
- Modify: `src/septmuse/retrieval/hybrid.py` — 在 `_vector_path` / `_keyword_path` / `_entity_path` 内部函数中加 `time_block`

**Interfaces:**
- Consumes: `time_block` from Task 2

- [ ] **Step 1: 添加 import**

在 `src/septmuse/retrieval/hybrid.py` 顶部添加：

```python
from septmuse.observability.hooks import time_block
```

- [ ] **Step 2: 在三个内部函数中加 time_block**

在 `HybridRetriever.search()` 方法的 `_vector_path` / `_keyword_path` / `_entity_path` 内部函数中，用 `with time_block(...)` 包裹核心逻辑：

```python
def _vector_path() -> list[dict[str, Any]]:
    with time_block("hybrid_search_components_seconds", {"component": "vector"}):
        emb = self.embedder.embed(query)
        return self.store.search(
            emb,
            user_id=user_id,
            session_id=session_id,
            top_k=internal_limit,
            threshold=threshold,
            filters=filters,
        )

def _keyword_path() -> list[dict[str, Any]] | None:
    with time_block("hybrid_search_components_seconds", {"component": "keyword"}):
        try:
            return self.store.keyword_search(
                query, user_id=user_id, session_id=session_id, top_k=internal_limit
            )
        except Exception:
            return None

def _entity_path() -> dict[str, float]:
    with time_block("hybrid_search_components_seconds", {"component": "entity"}):
        if self.entity_extractor is None or self.entity_store is None:
            return {}
        try:
            entities = self.entity_extractor.extract(query)
            boosts: dict[str, float] = {}
            for entity in entities:
                matches = self.entity_store.search(entity.text, user_id=user_id, top_k=10)
                for match in matches:
                    linked_ids = match.get("linked_memory_ids", [])
                    n = len(linked_ids)
                    boost = 0.5 / (RRF_K + 1) if n > 0 else 0.0
                    for eid in linked_ids:
                        boosts[eid] = boosts.get(eid, 0.0) + boost
            return boosts
        except Exception as e:
            logger.warning("entity_boost_failed", error=str(e))
            return {}
```

- [ ] **Step 3: 验证 import**

Run: `$env:PYTHONPATH = "src"; python -c "from septmuse.retrieval.hybrid import HybridRetriever; print('OK')"`
Expected: OK

---

### Task 11: 统一测试

**Files:**
- Create: `tests/unit/test_observability/conftest.py`
- Create: `tests/unit/test_observability/test_collector.py`
- Create: `tests/unit/test_observability/test_middleware.py`
- Create: `tests/unit/test_observability/test_business_collector.py`
- Create: `tests/unit/test_observability/test_hooks.py`
- Create: `tests/unit/test_observability/test_embed_abc.py`

**Interfaces:**
- Consumes: All previous tasks

- [ ] **Step 1: 创建 conftest.py**

```python
"""可观测性测试 conftest — 每个测试前重置 MetricsCollector 单例。"""

import pytest

from septmuse.observability.collector import MetricsCollector


@pytest.fixture(autouse=True)
def _reset_metrics():
    """每个测试前重置 metrics 单例 + 注销所有自定义指标。"""
    MetricsCollector.reset()
    yield
    MetricsCollector.reset()
```

- [ ] **Step 2: 创建 test_collector.py — MetricsCollector 单例 + no-op 行为**

```python
"""MetricsCollector 单例测试 — 单例、no-op、inc/dec/observe/set_gauge。"""

from septmuse.observability.collector import MetricsCollector


def test_singleton():
    """get() 返回同一实例。"""
    a = MetricsCollector.get()
    b = MetricsCollector.get()
    assert a is b


def test_noop_when_disabled():
    """未 configure 时所有方法不报错。"""
    c = MetricsCollector.get()
    assert not c.enabled
    c.inc("api_requests_total", {"endpoint": "/", "method": "GET", "status": "200"})
    c.dec("api_requests_in_flight", {"endpoint": "/"})
    c.observe("embed_duration_seconds", 0.001, {"backend": "hash"})
    c.set_gauge("septmuse_uptime_seconds", 42.0)


def test_inc_after_configure():
    """configure(enabled=True) 后 inc 记录到 prometheus_client。"""
    c = MetricsCollector.get()
    c.configure(enabled=True)
    assert c.enabled
    c.inc("api_requests_total", {"endpoint": "/", "method": "GET", "status": "200"})
    c.inc("api_requests_total", {"endpoint": "/", "method": "GET", "status": "200"})
    from prometheus_client import REGISTRY

    samples = list(REGISTRY.collect())
    api_metric = [s for s in samples if s.name == "septmuse_api_requests_total"]
    assert len(api_metric) > 0
    total = sum(s.value for s in api_metric[0].samples)
    assert total >= 2.0


def test_reset_clears_singleton():
    """reset() 后 get() 返回新实例。"""
    a = MetricsCollector.get()
    a.configure(enabled=True)
    MetricsCollector.reset()
    b = MetricsCollector.get()
    assert a is not b
    assert not b.enabled
```

- [ ] **Step 3: 创建 test_middleware.py — RED 指标**

```python
"""PrometheusMiddleware 测试 — 请求计数、状态码、延迟、in_flight。"""

import os

from fastapi import FastAPI
from fastapi.testclient import TestClient

from septmuse.observability.collector import MetricsCollector
from septmuse.observability.middleware import PrometheusMiddleware


def _make_app() -> FastAPI:
    app = FastAPI()

    @app.get("/test")
    async def test_endpoint():
        return {"ok": True}

    MetricsCollector.get().configure(enabled=True)
    app.add_middleware(PrometheusMiddleware)
    return app


def test_request_counted():
    """GET /test 后 api_requests_total 增加。"""
    app = _make_app()
    client = TestClient(app)
    client.get("/test")

    from prometheus_client import REGISTRY

    samples = list(REGISTRY.collect())
    req_metric = [s for s in samples if s.name == "septmuse_api_requests_total"]
    assert len(req_metric) > 0
    found = any(
        s.labels.get("endpoint") == "/test" and s.labels.get("status") == "200"
        for s in req_metric[0].samples
    )
    assert found


def test_duration_recorded():
    """GET /test 后 api_request_duration_seconds 有 bucket。"""
    app = _make_app()
    client = TestClient(app)
    client.get("/test")

    from prometheus_client import REGISTRY

    samples = list(REGISTRY.collect())
    dur_metric = [s for s in samples if s.name == "septmuse_api_request_duration_seconds"]
    assert len(dur_metric) > 0
    found = any(s.labels.get("endpoint") == "/test" for s in dur_metric[0].samples)
    assert found
```

- [ ] **Step 4: 创建 test_business_collector.py — pull-on-scrape**

```python
"""BusinessMetricsCollector 测试 — mock store，验证 collect() 指标值。"""

from septmuse.observability.business import BusinessMetricsCollector


class MockStore:
    """模拟 store，提供 engine 属性。"""

    def __init__(self, rows_map: dict[str, list[tuple]]):
        self._rows_map = rows_map

    @property
    def engine(self):
        return MockEngine(self._rows_map)


class MockEngine:
    def __init__(self, rows_map):
        self._rows_map = rows_map

    def connect(self):
        return MockConnection(self._rows_map)


class MockConnection:
    def __init__(self, rows_map):
        self._rows_map = rows_map

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def execute(self, stmt):
        sql = str(stmt)
        for key, rows in self._rows_map.items():
            if key in sql:
                return MockResult(rows)
        return MockResult([])


class MockResult:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows


def test_collect_memories_total():
    """collect() 返回 memories_total 指标（按 state）。"""
    store = MockStore({
        "GROUP BY state": [("active", 10), ("deleted", 2)],
        "COUNT(*) FROM memory_blocks": [(5,)],
        "COUNT(*) FROM septmuse_entities": [(3,)],
        "LENGTH(content)": [("value", 1000), ("metadata", 500)],
        "COUNT(*) FROM memory_vectors": [(10,)],
    })
    collector = BusinessMetricsCollector(store, db_path=None)
    metrics = list(collector.collect())
    names = [m.name for m in metrics]
    assert "septmuse_memories_total" in names
    assert "septmuse_uptime_seconds" in names


def test_collect_no_db_no_crash():
    """db_path=None 时不报错。"""
    store = MockStore({})
    collector = BusinessMetricsCollector(store, db_path=None)
    metrics = list(collector.collect())
    names = [m.name for m in metrics]
    assert "septmuse_uptime_seconds" in names
```

- [ ] **Step 5: 创建 test_hooks.py — track_operation 装饰器**

```python
"""track_operation 装饰器测试。"""

from septmuse.observability.collector import MetricsCollector
from septmuse.observability.hooks import track_operation, time_block


def test_track_operation_sync():
    """同步函数装饰器记录延迟。"""
    @track_operation("add")
    def add(a, b):
        return a + b

    MetricsCollector.get().configure(enabled=True)
    result = add(1, 2)
    assert result == 3

    from prometheus_client import REGISTRY

    samples = list(REGISTRY.collect())
    op_metric = [s for s in samples if s.name == "septmuse_memory_operation_duration_seconds"]
    assert len(op_metric) > 0
    found = any(s.labels.get("operation") == "add" for s in op_metric[0].samples)
    assert found


def test_track_operation_async():
    """异步函数装饰器记录延迟。"""
    import asyncio

    @track_operation("search")
    async def search(q):
        return f"result:{q}"

    MetricsCollector.get().configure(enabled=True)
    result = asyncio.get_event_loop().run_until_complete(search("hello"))
    assert result == "result:hello"

    from prometheus_client import REGISTRY

    samples = list(REGISTRY.collect())
    op_metric = [s for s in samples if s.name == "septmuse_memory_operation_duration_seconds"]
    assert len(op_metric) > 0
    found = any(s.labels.get("operation") == "search" for s in op_metric[0].samples)
    assert found


def test_time_block():
    """time_block 上下文管理器记录耗时。"""
    MetricsCollector.get().configure(enabled=True)
    with time_block("hybrid_search_components_seconds", {"component": "vector"}):
        pass

    from prometheus_client import REGISTRY

    samples = list(REGISTRY.collect())
    hybrid_metric = [s for s in samples if s.name == "septmuse_hybrid_search_components_seconds"]
    assert len(hybrid_metric) > 0
    found = any(s.labels.get("component") == "vector" for s in hybrid_metric[0].samples)
    assert found


def test_noop_when_disabled():
    """未启用时装饰器不报错。"""
    @track_operation("add")
    def add(a, b):
        return a + b

    result = add(1, 2)
    assert result == 3
```

- [ ] **Step 6: 创建 test_embed_abc.py — 模板方法**

```python
"""Embedder 模板方法测试 — embed() 调用后 embed_duration_seconds 记录。"""

from septmuse.embedders.base import Embedder
from septmuse.embedders.hash import HashEmbedder
from septmuse.embedders.cached import CachedEmbedder
from septmuse.observability.collector import MetricsCollector


def test_embed_records_duration():
    """embed() 调用后 embed_duration_seconds 被记录。"""
    MetricsCollector.get().configure(enabled=True)
    embedder = HashEmbedder()
    embedder.backend_name = "hash"
    result = embedder.embed("hello")
    assert len(result) > 0

    from prometheus_client import REGISTRY

    samples = list(REGISTRY.collect())
    emb_metric = [s for s in samples if s.name == "septmuse_embed_duration_seconds"]
    assert len(emb_metric) > 0
    found = any(s.labels.get("backend") == "hash" for s in emb_metric[0].samples)
    assert found


def test_embed_batch_records_duration():
    """embed_batch() 调用后 embed_batch_duration_seconds 被记录。"""
    MetricsCollector.get().configure(enabled=True)
    embedder = HashEmbedder()
    embedder.backend_name = "hash"
    results = embedder.embed_batch(["hello", "world"])
    assert len(results) == 2

    from prometheus_client import REGISTRY

    samples = list(REGISTRY.collect())
    batch_metric = [s for s in samples if s.name == "septmuse_embed_batch_duration_seconds"]
    assert len(batch_metric) > 0
    found = any(s.labels.get("backend") == "hash" for s in batch_metric[0].samples)
    assert found


def test_cached_embedder_cache_counters():
    """CachedEmbedder cache hit/miss 计数。"""
    MetricsCollector.get().configure(enabled=True)
    inner = HashEmbedder()
    inner.backend_name = "hash"
    cached = CachedEmbedder(inner)
    cached.embed("hello")
    cached.embed("hello")

    from prometheus_client import REGISTRY

    samples = list(REGISTRY.collect())
    hits = [s for s in samples if s.name == "septmuse_embed_cache_hits_total"]
    misses = [s for s in samples if s.name == "septmuse_embed_cache_misses_total"]
    assert len(hits) > 0
    assert len(misses) > 0
    hit_total = sum(s.value for s in hits[0].samples)
    miss_total = sum(s.value for s in misses[0].samples)
    assert hit_total >= 1.0
    assert miss_total >= 1.0


def test_noop_when_not_configured():
    """未 configure 时 embed() 仍然正常工作。"""
    embedder = HashEmbedder()
    result = embedder.embed("hello")
    assert len(result) > 0
```

- [ ] **Step 7: 跑全部 observability 测试**

Run: `$env:PYTHONPATH = "src"; python -m pytest tests/unit/test_observability/ -v --tb=short`
Expected: ALL PASSED

- [ ] **Step 8: 跑 ruff check**

Run: `ruff check src/septmuse/observability/ tests/unit/test_observability/`
Expected: All checks passed

---

### Task 12: 全量回归测试 + AGENTS.md 更新

**Files:**
- Modify: `AGENTS.md` — 新增可观测性段落

- [ ] **Step 1: 跑全量测试**

Run: `$env:PYTHONPATH = "src"; python -m pytest tests/unit/ tests/e2e/ -q --tb=no`
Expected: 1319+ passed / 16 failed (pre-existing) / 23+ skipped（与基线一致，新增测试全绿）

- [ ] **Step 2: 跑 ruff check 全量**

Run: `ruff check src/ tests/`
Expected: All checks passed

- [ ] **Step 3: 更新 AGENTS.md**

在 `AGENTS.md` 中新增「可观测性」段落（在合适位置，如 LLM Provider 段落之后）：

```markdown
### Observability (可观测性)

- `SEPTMUSE_METRICS`（默认未设）— `true`/`1`/`yes` 启用 Prometheus 指标 + `/metrics` 端点。
- `SEPTMUSE_METRICS_PATH`（默认 `/metrics`）— 端点路径（可自定义）。
- `prometheus_client` 是核心依赖（纯 Python ~50KB）。
- 未启用时所有埋点 no-op（`MetricsCollector` 单例检查 `enabled` 标志）。
- **三层埋点**：REST 中间件（RED）+ facade `@track_operation`（业务操作）+ 底层 ABC 模板方法（embed/LLM 全覆盖）。
- **模板方法重命名**：`Embedder.embed()` → 子类实现 `_embed()`；`LLM.complete()` → 子类实现 `_complete()`。
- **业务指标** pull-on-scrape：`BusinessMetricsCollector` 实现 `prometheus_client.Collector`，每次 scrape 查 DB 算值。
- `/metrics` 免 API key（加入 `EXEMPT_PATHS`），生产环境通过网络层隔离。
- 测试：`tests/unit/test_observability/`（6 文件），`conftest.py` 有 `_reset_metrics` autouse fixture。
```

同时在 AGENTS.md 的环境变量表中添加：

```markdown
| `SEPTMUSE_METRICS` | 未设 | `true` 启用 /metrics 端点 + 全部埋点 |
| `SEPTMUSE_METRICS_PATH` | `/metrics` | 端点路径 |
```

- [ ] **Step 4: 最终验证**

Run: `$env:PYTHONPATH = "src"; $env:SEPTMUSE_METRICS = "true"; python -c "from septmuse.api.rest import create_app; app = create_app(); from fastapi.testclient import TestClient; c = TestClient(app); r = c.get('/metrics'); print(r.status_code, 'septmuse_' in r.text)"`
Expected: 200 True

- [ ] **Step 5: Commit（用户确认后）**

```bash
git add src/septmuse/observability/ src/septmuse/embedders/ src/septmuse/llms/ src/septmuse/api/ src/septmuse/memory/main.py src/septmuse/retrieval/hybrid.py pyproject.toml AGENTS.md tests/unit/test_observability/
git commit -m "feat: add Prometheus observability metrics system

- MetricsCollector singleton with opt-in via SEPTMUSE_METRICS=true
- Three-layer instrumentation: REST middleware + facade decorators + ABC template method
- EmbedderBase template: embed() -> _embed(), embed_batch() -> _embed_batch()
- LLM template: complete() -> _complete()
- BusinessMetricsCollector: pull-on-scrape via prometheus Collector
- CachedEmbedder: cache hit/miss counters
- 6 test files in tests/unit/test_observability/
"
```

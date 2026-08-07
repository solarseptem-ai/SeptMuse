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
        self._business_collector: Any = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    def configure(self, enabled: bool) -> None:
        """初始化所有 Prometheus 指标。enabled=False 时保持 no-op。"""
        if self._enabled and self._metrics and enabled:
            return
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

                # 收集唯一 collector 对象（一个 Counter 在 _names_to_collectors 下注册多个名字），
                # 交给 unregister 内部统一清理 _names_to_collectors + _collector_to_names。
                collectors = {
                    c for name, c in REGISTRY._names_to_collectors.items()
                    if name.startswith("septmuse_")
                }
                for collector in collectors:
                    REGISTRY.unregister(collector)
            except Exception:
                pass
        cls._instance = None

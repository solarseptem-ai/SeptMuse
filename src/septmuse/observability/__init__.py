"""SeptMuse 可观测性指标系统。"""

from __future__ import annotations

import contextlib
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

    mc = MetricsCollector.get()
    mc.configure(enabled=True)
    logger.info("metrics_enabled", endpoint="/metrics")

    # 挂 REST 中间件
    from septmuse.observability.middleware import PrometheusMiddleware

    app.add_middleware(PrometheusMiddleware)

    # 业务指标 collector（幂等：重复调用 create_app 时先注销旧实例再注册新的）
    from prometheus_client import REGISTRY

    from septmuse.observability.business import BusinessMetricsCollector

    db_path = getattr(config, "db_path", None)
    bc = BusinessMetricsCollector(store, db_path)
    if mc._business_collector is not None:
        with contextlib.suppress(Exception):
            REGISTRY.unregister(mc._business_collector)
    mc._business_collector = bc
    REGISTRY.register(bc)

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

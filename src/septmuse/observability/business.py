"""BusinessMetricsCollector — pull-on-scrape 业务指标。

实现 prometheus_client.Collector 接口，Prometheus 拉取 /metrics 时触发 collect()，
查询 DB 算值，返回 GaugeMetricFamily。
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from typing import Any

from prometheus_client.core import GaugeMetricFamily
from prometheus_client.registry import Collector


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

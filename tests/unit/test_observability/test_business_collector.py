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

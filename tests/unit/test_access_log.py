#  Copyright 2026 The sonhhxg0529 Authors. All Rights Reserved.
"""record_access + MemoryAccessLog 测试。"""

from __future__ import annotations

from septmuse.governance.audit import record_access


class _MockStore:
    """模拟 store, 支持 _record_access_log。"""

    def __init__(self, should_fail: bool = False):
        self.logs: list[dict] = []
        self._should_fail = should_fail

    def _record_access_log(self, memory_id, app_id, access_type, metadata):
        if self._should_fail:
            raise RuntimeError("mock failure")
        log = {"memory_id": memory_id, "app_id": app_id, "access_type": access_type, "metadata": metadata or {}}
        self.logs.append(log)
        return f"log-{len(self.logs)}"


class _StoreWithoutLog:
    """模拟 store, 不支持 _record_access_log。"""

    pass


def test_record_access_creates_log_entry():
    store = _MockStore()
    log_id = record_access(store, "m1", "app1", "get", {"k": "v"})
    assert log_id is not None
    assert log_id.startswith("log-")
    assert len(store.logs) == 1
    assert store.logs[0]["memory_id"] == "m1"
    assert store.logs[0]["access_type"] == "get"


def test_record_access_with_metadata():
    store = _MockStore()
    record_access(store, "m1", "app1", "search", {"query": "foo", "score": 0.9})
    assert store.logs[0]["metadata"] == {"query": "foo", "score": 0.9}


def test_record_access_none_metadata():
    store = _MockStore()
    record_access(store, "m1", None, "list", None)
    assert store.logs[0]["metadata"] == {}


def test_record_access_failure_returns_none():
    store = _MockStore(should_fail=True)
    log_id = record_access(store, "m1", "app1", "get", None)
    assert log_id is None


def test_record_access_unsupported_store_returns_none():
    store = _StoreWithoutLog()
    log_id = record_access(store, "m1", "app1", "get", None)
    assert log_id is None


def test_record_access_swallows_exceptions():
    store = _MockStore(should_fail=True)
    # 不应抛异常
    result = record_access(store, "m1", "app1", "delete", None)
    assert result is None

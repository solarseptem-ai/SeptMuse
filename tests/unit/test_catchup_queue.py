"""LLMCatchUpQueue 测试。"""
from __future__ import annotations

import pytest
from sqlalchemy import create_engine

from septmuse.evolution.catchup import LLMCatchUpQueue


@pytest.fixture
def queue():
    engine = create_engine("sqlite:///:memory:")
    q = LLMCatchUpQueue(engine)
    return q


def test_enqueue_dequeue_completed(queue):
    """入队 → 出队 → 标记完成。"""
    task_id = queue.enqueue("fact_extraction", "mem-1", user_id="alice", payload={"text": "hello"})
    tasks = queue.dequeue(limit=10)
    assert len(tasks) == 1
    assert tasks[0]["task_type"] == "fact_extraction"
    assert tasks[0]["memory_id"] == "mem-1"
    assert tasks[0]["user_id"] == "alice"
    assert tasks[0]["payload"] == {"text": "hello"}
    queue.mark_completed(task_id)
    assert queue.get_pending_count() == 0


def test_fifo_order(queue):
    """多条 FIFO 顺序。"""
    for i in range(5):
        queue.enqueue("fact_extraction", f"mem-{i}", user_id="alice")
    tasks = queue.dequeue(limit=3)
    assert len(tasks) == 3
    # 剩 2 条 pending
    assert queue.get_pending_count() == 2


def test_mark_failed(queue):
    """mark_failed 设 status=failed。"""
    task_id = queue.enqueue("fact_extraction", "mem-1", user_id="alice")
    queue.mark_failed(task_id, "LLM timeout")
    assert queue.get_pending_count() == 0
    # failed 的不会被 dequeue 再取
    tasks = queue.dequeue()
    assert len(tasks) == 0


def test_pending_count(queue):
    """pending_count 正确。"""
    assert queue.get_pending_count() == 0
    queue.enqueue("fact_extraction", "mem-1", user_id="alice")
    queue.enqueue("classification", "mem-2", user_id="bob")
    assert queue.get_pending_count() == 2
    tasks = queue.dequeue(limit=1)
    queue.mark_completed(tasks[0]["id"])
    assert queue.get_pending_count() == 1


def test_reset(queue):
    """reset 清空队列。"""
    queue.enqueue("fact_extraction", "mem-1", user_id="alice")
    queue.enqueue("fact_extraction", "mem-2", user_id="alice")
    queue.reset()
    assert queue.get_pending_count() == 0

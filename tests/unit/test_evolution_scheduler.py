"""EvolutionScheduler 测试。"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

from septmuse.evolution.scheduler import EvolutionScheduler


def test_start_stop():
    """start/stop 不崩。"""
    mem = MagicMock()
    mem.llm = None
    mem.store.engine = MagicMock()
    scheduler = EvolutionScheduler(mem, interval=0.1, enabled=True)
    scheduler.start()
    time.sleep(0.3)
    scheduler.stop(timeout=2)


def test_disabled_does_not_start():
    """enabled=False 不启动。"""
    mem = MagicMock()
    scheduler = EvolutionScheduler(mem, enabled=False)
    scheduler.start()
    assert scheduler._timer is None


def test_run_cycle_runs_due_tasks():
    """_run_cycle 跑 due 任务、跳过未 due 的。"""
    mem = MagicMock()
    mem.llm = None
    mem.store.engine = MagicMock()
    scheduler = EvolutionScheduler(mem, interval=0.1, enabled=True)
    # 首次 last_run 全 0, 所有任务 due
    with (
        patch.object(scheduler, "_run_task") as mock_run,
        patch.object(scheduler, "_process_catchup"),
        patch.object(scheduler, "_get_user_ids", return_value=["alice"]),
    ):
        scheduler._run_cycle()
    # 4 个任务都跑过
    assert mock_run.call_count == 4


def test_run_cycle_skips_not_due():
    """设置 last_run 为最近, 跳过未 due 的。"""
    mem = MagicMock()
    mem.llm = None
    mem.store.engine = MagicMock()
    scheduler = EvolutionScheduler(mem, interval=0.1, enabled=True)
    now = time.time()
    scheduler._last_run = {name: now for name in scheduler.TASK_INTERVALS}
    with (
        patch.object(scheduler, "_run_task") as mock_run,
        patch.object(scheduler, "_process_catchup"),
        patch.object(scheduler, "_get_user_ids", return_value=["alice"]),
    ):
        scheduler._run_cycle()
    # 所有任务刚跑过, 不再 due
    assert mock_run.call_count == 0


def test_process_catchup_llm_unavailable():
    """LLM 不可用不跑补跑。"""
    mem = MagicMock()
    mem.llm = None
    mem.store.engine = MagicMock()
    scheduler = EvolutionScheduler(mem, enabled=True)
    # _llm_available False -> 不调 dequeue
    with patch.object(scheduler, "_catchup_queue") as mock_queue:
        scheduler._process_catchup()
    mock_queue.dequeue.assert_not_called()


def test_process_catchup_llm_available():
    """LLM 可用时 dequeue 补跑。"""
    mem = MagicMock()
    mem.llm = MagicMock()
    mem.store.engine = MagicMock()
    scheduler = EvolutionScheduler(mem, enabled=True)
    # 模拟 LLM ping 成功
    mem.llm.complete.return_value = "ok"
    # 模拟 catchup_queue
    mock_queue = MagicMock()
    mock_queue.dequeue.return_value = []  # 空队列
    scheduler._cu_queue = mock_queue
    scheduler._process_catchup()
    mock_queue.dequeue.assert_called_once()


def test_get_user_ids():
    """_get_user_ids 从 store 查 distinct user_id。"""
    mem = MagicMock()
    mem.store = MagicMock()
    mock_engine = MagicMock()
    mock_conn = MagicMock()
    mock_result = MagicMock()
    mock_result.fetchall.return_value = [("alice",), ("bob",)]
    mock_conn.execute.return_value = mock_result
    mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
    mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)
    mem.store.engine = mock_engine

    scheduler = EvolutionScheduler(mem, enabled=True)
    uids = scheduler._get_user_ids()
    assert "alice" in uids
    assert "bob" in uids


def test_stop_waits_for_cycle():
    """stop 等 _run_cycle 完成。"""
    mem = MagicMock()
    mem.llm = None
    mem.store.engine = MagicMock()
    scheduler = EvolutionScheduler(mem, interval=0.1, enabled=True)

    # 模拟 _run_cycle 慢
    def slow_cycle():
        with scheduler._lock:
            time.sleep(0.2)

    with (
        patch.object(scheduler, "_run_cycle", side_effect=slow_cycle),
        patch.object(scheduler, "_schedule_next"),
    ):
        t = threading.Thread(target=scheduler._run_cycle)
        t.start()
        time.sleep(0.05)
        scheduler.stop(timeout=2)
    # 不超时不崩


def test_run_task_catches_exception():
    """_run_task 吞异常不崩。"""
    mem = MagicMock()
    scheduler = EvolutionScheduler(mem, enabled=True)
    # 调一个不存在的 user_id, 各种 mock 不匹配 -> 异常被吞
    scheduler._run_task("forgetting", "ghost")
    # 不崩就行

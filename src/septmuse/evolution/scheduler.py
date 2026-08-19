#  Copyright 2026 The sonhhxg0529 Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""后台演化调度器。"""

from __future__ import annotations

import os
import threading
import time
from typing import Any, ClassVar

from septmuse.core.logging import get_logger

logger = get_logger(__name__)


class EvolutionScheduler:
    """后台演化调度器。"""

    TASK_INTERVALS: ClassVar[dict[str, int]] = {
        "forgetting": 3600,  # 1h
        "conflict": 300,  # 5min
        "distillation": 1800,  # 30min
        "graph": 600,  # 10min
    }

    def __init__(self, memory: Any, *, interval: int | None = None, enabled: bool | None = None) -> None:
        """初始化调度器。"""
        self.memory = memory
        self._interval = interval or int(os.getenv("SEPTMUSE_EVOLUTION_INTERVAL", "300"))
        self._enabled = (
            enabled
            if enabled is not None
            else os.getenv("SEPTMUSE_EVOLUTION_ENABLED", "true").lower() in ("1", "true", "yes")
        )
        self._stop_event = threading.Event()
        self._timer: threading.Timer | None = None
        self._lock = threading.Lock()
        self._last_run: dict[str, float] = {name: 0.0 for name in self.TASK_INTERVALS}
        self._self_loop = None  # 延迟初始化 SelfEvolutionLoop

    def start(self) -> None:
        """启动 daemon Timer 线程。"""
        if not self._enabled:
            return
        self._stop_event.clear()
        self._schedule_next()

    def stop(self, timeout: float = 10.0) -> None:
        """停止调度器, 等当前任务完成最多 timeout 秒。"""
        self._stop_event.set()
        if self._timer is not None:
            self._timer.cancel()
        acquired = self._lock.acquire(timeout=timeout)
        if acquired:
            self._lock.release()

    def _schedule_next(self) -> None:
        """安排下一个 cycle 的 Timer。"""
        if self._stop_event.is_set():
            return
        self._timer = threading.Timer(self._interval, self._run_cycle)
        self._timer.daemon = True
        self._timer.start()

    def _run_cycle(self) -> None:
        """一轮调度: 检查 due 任务 -> 跑 -> 重新 schedule 下一次。"""
        with self._lock:
            now = time.time()
            try:
                user_ids = self._get_user_ids()
            except Exception as e:
                logger.warning("get_user_ids_failed", error=str(e))
                user_ids = []

            for name, interval in self.TASK_INTERVALS.items():
                if now - self._last_run.get(name, 0.0) >= interval:
                    for uid in user_ids:
                        self._run_task(name, uid)
                    self._last_run[name] = now

            try:
                self._process_catchup()
            except Exception as e:
                logger.warning("process_catchup_failed", error=str(e))

        self._schedule_next()

    def _run_task(self, name: str, user_id: str) -> None:
        """执行单个演化任务。吞异常, 只 log warning。"""
        try:
            if name == "forgetting":
                from septmuse.retrieval.forgetting import ForgettingRetriever

                ForgettingRetriever(self.memory.typed_store).archive_stale(user_id=user_id)
            elif name == "conflict":
                self.memory.evolution.resolve_conflicts(user_id=user_id)
            elif name == "distillation":
                self.memory.evolution.reflect(user_id=user_id, limit=20)
            elif name == "graph":
                self.memory.evolution.dream(user_id=user_id)
        except Exception as e:
            logger.warning("evolution_task_failed", task=name, user_id=user_id, error=str(e))

    def _get_self_loop(self) -> Any:
        """延迟初始化 SelfEvolutionLoop。"""
        if self._self_loop is None:
            try:
                from septmuse.evolution.self_loop import SelfEvolutionLoop

                self._self_loop = SelfEvolutionLoop(
                    self.memory.store,
                    self.memory.graph_store,
                    self.memory.embedder,
                    self.memory.typed_store,
                    llm=self.memory.llm,
                )
            except Exception as e:
                logger.warning("self_loop_init_failed", error=str(e))
                self._self_loop = None
        return self._self_loop

    # ------------------------------------------------------------------
    # 事件驱动触发
    # ------------------------------------------------------------------

    def trigger_immediate(self, *, user_id: str | None = None) -> None:
        """立即触发一次自进化循环。"""
        if not self._enabled:
            return
        loop = self._get_self_loop()
        if loop is None:
            return
        try:
            if user_id is not None:
                loop.improve(user_id=user_id)
                logger.info("trigger_immediate_done", user_id=user_id)
            else:
                user_ids = self._get_user_ids()
                for uid in user_ids:
                    loop.improve(user_id=uid)
                logger.info("trigger_immediate_all_done", user_count=len(user_ids))
        except Exception as e:
            logger.warning("trigger_immediate_failed", error=str(e))

    def trigger_immediate_for_user(self, user_id: str) -> None:
        """对单用户立即触发自进化循环。"""
        self.trigger_immediate(user_id=user_id)

    def trigger_refine(self, *, user_id: str) -> list[Any]:
        """立即触发规则精炼。"""
        loop = self._get_self_loop()
        if loop is None:
            return []
        return loop.refiner.refine_all(user_id=user_id)

    def _process_catchup(self) -> None:
        """检查 LLM 可用 -> dequeue 补跑。LLM 不可用就跳过。"""
        if not self._llm_available():
            return
        queue = self._catchup_queue
        if queue is None:
            from septmuse.evolution.catchup import LLMCatchUpQueue

            queue = LLMCatchUpQueue(self.memory.store.engine)
            self._catchup_queue = queue

        tasks = queue.dequeue(limit=5)
        for task in tasks:
            self._reprocess_task(task, queue)

    def _reprocess_task(self, task: dict[str, Any], queue: Any) -> None:
        """补跑单个 catch-up 任务。"""
        task_id = task["id"]
        task_type = task.get("task_type", "")
        user_id = task.get("user_id", "default")
        try:
            if task_type == "fact_extraction":
                extractor = getattr(self.memory, "extractor", None)
                if extractor is not None:
                    payload = task.get("payload", {})
                    extractor.extract_and_store(payload.get("text", ""), user_id=user_id)
            queue.mark_completed(task_id)
        except Exception as e:
            logger.warning("catchup_task_failed", task_id=task_id, task_type=task_type, error=str(e))
            queue.mark_failed(task_id, str(e))

    def _get_user_ids(self) -> list[str]:
        """从 store 查所有 distinct user_id。"""
        store = self.memory.store
        if hasattr(store, "get_user_ids"):
            return store.get_user_ids()
        # fallback: raw SQL (旧 store 兼容)
        from sqlalchemy import text

        with store.engine.connect() as conn:
            result = conn.execute(text("SELECT DISTINCT user_id FROM memories WHERE is_deleted = 0"))
            return [row[0] for row in result.fetchall() if row[0]]

    def _llm_available(self) -> bool:
        """检查 LLM 是否可用。"""
        if self.memory.llm is None:
            return False
        try:
            self.memory.llm.complete("ping", "ping")
            return True
        except Exception:
            return False

    @property
    def _catchup_queue(self) -> Any:
        """延迟初始化的 LLMCatchUpQueue。"""
        return getattr(self, "_cu_queue", None)

    @_catchup_queue.setter
    def _catchup_queue(self, value: Any) -> None:
        self._cu_queue = value

    @_catchup_queue.deleter
    def _catchup_queue(self) -> None:
        self._cu_queue = None

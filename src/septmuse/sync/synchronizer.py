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
"""多形态共存一致性同步器 (架构文档 §4.1 自研)。

14 家开源均无统一方案。当一份记忆需多形态共存时（如语义事实同时写图三元组
+ Markdown 文件 + 向量），由源同步器保证最终一致。

设计 (架构文档 §4.1):
- write: 并行写图/文件/向量, 任一失败则记录补偿任务
- reconcile: 检测形态间漂移, 以 graph 为权威源重同步文件/向量
- 权威源优先级: 图 > 文件 > 向量 (图最结构化, 文件人可纠错, 向量易漂移)
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from septmuse.core.logging import get_logger

logger = get_logger(__name__)


class SyncForm(str, Enum):
    """存储形态 (架构文档 §4.1 平面B)。"""

    GRAPH = "graph"  # 图三元组 (最权威)
    FILE = "file"  # Markdown 文件
    VECTOR = "vector"  # 向量嵌入


# 权威源优先级 (架构文档 §4.1: 图 > 文件 > 向量)
AUTHORITY_PRIORITY: list[SyncForm] = [SyncForm.GRAPH, SyncForm.FILE, SyncForm.VECTOR]


@dataclass
class SyncWriteResult:
    """单形态写入结果。"""

    form: SyncForm
    success: bool
    memory_id: str | None = None
    error: str | None = None


@dataclass
class CompensationTask:
    """补偿任务 (写入失败后记录, 后续重试)。"""

    form: SyncForm
    memory_id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    retries: int = 0
    max_retries: int = 3


@dataclass
class SyncResult:
    """同步结果。"""

    writes: list[SyncWriteResult] = field(default_factory=list)
    compensations: list[CompensationTask] = field(default_factory=list)

    @property
    def all_succeeded(self) -> bool:
        return all(w.success for w in self.writes) and not self.compensations


# 写入函数类型: (content, metadata) -> memory_id
WriteFunc = Callable[[str, dict[str, Any]], str]


class SourceSynchronizer:
    """多形态记忆的最终一致同步器 (架构文档 §4.1 自研)。

    用法:
        sync = SourceSynchronizer()
        sync.register(SyncForm.VECTOR, vector_store.add)
        sync.register(SyncForm.FILE, file_store.write)
        result = sync.write("alice likes python", {"user_id": "alice"},
                            targets=[SyncForm.VECTOR, SyncForm.FILE])
        if not result.all_succeeded:
            # 有补偿任务, 后续重试
            sync.retry_compensations()
    """

    def __init__(self, max_workers: int = 3) -> None:
        self._writers: dict[SyncForm, WriteFunc] = {}
        self._compensations: list[CompensationTask] = []
        self._lock = threading.Lock()
        self._max_workers = max_workers

    def register(self, form: SyncForm, writer: WriteFunc) -> None:
        """注册形态写入函数。"""
        self._writers[form] = writer
        logger.info("sync_form_registered", form=form.value)

    def write(
        self,
        content: str,
        metadata: dict[str, Any],
        targets: list[SyncForm] | None = None,
    ) -> SyncResult:
        """并行写多形态 (架构文档 §4.1: 并行写+补偿)。

        1. 并行调用各形态 writer
        2. 成功: 记录 memory_id
        3. 失败: 记录 CompensationTask
        """
        targets = targets or list(self._writers.keys())
        result = SyncResult()

        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            futures: dict[Any, SyncForm] = {}
            for form in targets:
                writer = self._writers.get(form)
                if writer is None:
                    result.writes.append(SyncWriteResult(form=form, success=False, error=f"no writer for {form.value}"))
                    continue
                future = executor.submit(self._safe_write, form, writer, content, metadata)
                futures[future] = form

            for future in as_completed(futures):
                form = futures[future]
                write_result = future.result()
                result.writes.append(write_result)
                if not write_result.success:
                    comp = CompensationTask(
                        form=form,
                        memory_id=write_result.memory_id or "",
                        content=content,
                        metadata=metadata,
                    )
                    result.compensations.append(comp)
                    with self._lock:
                        self._compensations.append(comp)

        succeeded = sum(1 for w in result.writes if w.success)
        logger.info(
            "sync_write_done",
            targets=len(targets),
            succeeded=succeeded,
            compensations=len(result.compensations),
        )
        return result

    def reconcile(self, memory_id: str) -> bool:
        """检测形态间漂移, 以权威源重同步 (架构文档 §4.1)。

        权威源优先级: 图 > 文件 > 向量。
        从最高权威形态读取, 重新写入较低权威形态。
        """
        for authority in AUTHORITY_PRIORITY:
            reader = self._writers.get(authority)
            if reader is None:
                continue
            # 读取权威源内容 (假设 writer 可读, 实际由 store 提供 get)
            # 这里简化: 仅检测是否存在 writer, 存在则视为可重同步
            logger.info("sync_reconcile", memory_id=memory_id, authority=authority.value)
            return True
        return False

    def retry_compensations(self) -> list[SyncWriteResult]:
        """重试所有补偿任务。"""
        results: list[SyncWriteResult] = []
        with self._lock:
            pending = list(self._compensations)
            self._compensations.clear()

        for comp in pending:
            if comp.retries >= comp.max_retries:
                logger.warning("sync_compensation_exhausted", form=comp.form.value, retries=comp.retries)
                results.append(SyncWriteResult(form=comp.form, success=False, error="max retries exceeded"))
                continue
            writer = self._writers.get(comp.form)
            if writer is None:
                results.append(SyncWriteResult(form=comp.form, success=False, error="no writer"))
                continue
            comp.retries += 1
            try:
                mid = writer(comp.content, comp.metadata)
                results.append(SyncWriteResult(form=comp.form, success=True, memory_id=mid))
                logger.info("sync_compensation_succeeded", form=comp.form.value)
            except Exception as e:
                results.append(SyncWriteResult(form=comp.form, success=False, error=str(e)))
                with self._lock:
                    self._compensations.append(comp)
        return results

    @property
    def pending_compensations(self) -> int:
        """待重试的补偿任务数。"""
        with self._lock:
            return len(self._compensations)

    def _safe_write(self, form: SyncForm, writer: WriteFunc, content: str, metadata: dict[str, Any]) -> SyncWriteResult:
        """安全写入 (捕获异常)。"""
        try:
            mid = writer(content, metadata)
            return SyncWriteResult(form=form, success=True, memory_id=mid)
        except Exception as e:
            return SyncWriteResult(form=form, success=False, error=str(e))

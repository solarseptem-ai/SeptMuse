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
"""LLM 降级补跑队列。"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import text
from sqlmodel import Field, Session, SQLModel, select

from septmuse.core.logging import get_logger

logger = get_logger(__name__)

_CREATE_TABLE_SQL = """\
CREATE TABLE IF NOT EXISTS septmuse_catchup_queue (
    id TEXT PRIMARY KEY,
    task_type TEXT NOT NULL,
    memory_id TEXT DEFAULT '',
    user_id TEXT DEFAULT 'default',
    payload TEXT DEFAULT '{}',
    status TEXT DEFAULT 'pending',
    error TEXT DEFAULT '',
    created_at TEXT NOT NULL,
    completed_at TEXT
)
"""


class CatchUpTask(SQLModel, table=True):
    """LLM 补跑任务记录。"""

    __tablename__ = "septmuse_catchup_queue"

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    task_type: str = Field(index=True)
    memory_id: str = Field(default="")
    user_id: str = Field(default="default", index=True)
    payload: str = Field(default="{}")
    status: str = Field(default="pending", index=True)  # pending / processing / completed / failed
    error: str = Field(default="")
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: str | None = Field(default=None)


class LLMCatchUpQueue:
    """LLM 降级补跑队列。"""

    def __init__(self, engine: Any) -> None:
        self._engine = engine
        self._ensure_table()

    def _ensure_table(self) -> None:
        # Raw SQL CREATE TABLE IF NOT EXISTS — 不依赖 SQLModel.metadata 注册时序,
        # 只建本表, 避免在全新 engine 上连带建出其它已注册表。
        with Session(self._engine) as session:
            session.exec(text(_CREATE_TABLE_SQL))
            session.exec(
                text("CREATE INDEX IF NOT EXISTS ix_catchup_status ON septmuse_catchup_queue (status)")
            )
            session.exec(
                text("CREATE INDEX IF NOT EXISTS ix_catchup_user ON septmuse_catchup_queue (user_id)")
            )
            session.exec(
                text("CREATE INDEX IF NOT EXISTS ix_catchup_task_type ON septmuse_catchup_queue (task_type)")
            )
            session.commit()

    def enqueue(
        self,
        task_type: str,
        memory_id: str,
        *,
        user_id: str = "default",
        payload: dict[str, Any] | None = None,
    ) -> str:
        """入队一个补跑任务。"""
        task = CatchUpTask(
            task_type=task_type,
            memory_id=memory_id,
            user_id=user_id,
            payload=json.dumps(payload or {}, ensure_ascii=False),
        )
        with Session(self._engine) as session:
            session.add(task)
            session.commit()
            logger.info(
                "catchup_enqueued", task_type=task_type, memory_id=memory_id, user_id=user_id
            )
            return task.id

    def dequeue(self, limit: int = 10) -> list[dict[str, Any]]:
        """取出 pending 任务并标记为 processing。"""
        with Session(self._engine) as session:
            stmt = (
                select(CatchUpTask)
                .where(CatchUpTask.status == "pending")
                .order_by(CatchUpTask.created_at)
                .limit(limit)
            )
            rows = session.exec(stmt).all()
            result = [
                {
                    "id": r.id,
                    "task_type": r.task_type,
                    "memory_id": r.memory_id,
                    "user_id": r.user_id,
                    "payload": json.loads(r.payload) if r.payload else {},
                    "created_at": r.created_at,
                }
                for r in rows
            ]
            for r in rows:
                r.status = "processing"
                session.add(r)
            session.commit()
            return result

    def mark_completed(self, task_id: str) -> None:
        """标记任务完成。"""
        with Session(self._engine) as session:
            row = session.get(CatchUpTask, task_id)
            if row:
                row.status = "completed"
                row.completed_at = datetime.now(timezone.utc).isoformat()
                session.add(row)
                session.commit()

    def mark_failed(self, task_id: str, error: str) -> None:
        """标记任务失败 (可重试)。"""
        with Session(self._engine) as session:
            row = session.get(CatchUpTask, task_id)
            if row:
                row.status = "failed"
                row.error = error[:500]
                session.add(row)
                session.commit()

    def get_pending_count(self) -> int:
        """获取 pending 任务数。"""
        with Session(self._engine) as session:
            stmt = select(CatchUpTask).where(CatchUpTask.status == "pending")
            return len(session.exec(stmt).all())

    def reset(self) -> None:
        """清空队列表 (测试用)。"""
        with Session(self._engine) as session:
            session.exec(text("DELETE FROM septmuse_catchup_queue"))
            session.commit()

"""ResetPipeline — 记忆重置流水线。

提取自 Memory.reset()，负责存储层清理 (SQL 表清空 + vector/keyword/entity 重置)。
Facade 层缓存/注册表清理仍由 Memory 自行处理。
"""

from __future__ import annotations

import contextlib
from typing import Any

from septmuse.core.logging import get_logger

logger = get_logger(__name__)


class ResetPipeline:
    """记忆重置流水线 — 存储层清理。

    职责:
    - store.reset() (memories / history / access_logs / vector / keyword)
    - typed_store 关联表清空 (facts / episodic / procedural)
    - entity_store.reset()

    注意: facade 层缓存 (hybrid_retriever, dedup_window, registry) 由 Memory 自行清理。
    """

    def __init__(
        self,
        *,
        store: Any,
        entity_store: Any | None = None,
    ) -> None:
        self.store = store
        self.entity_store = entity_store

    def reset(self) -> dict[str, Any]:
        """重置存储层数据。"""
        self.store.reset()

        engine = getattr(self.store, "engine", None)
        if engine is not None:
            from sqlalchemy import text
            from sqlmodel import Session

            with Session(engine) as session:
                for table_name in [
                    "septmuse_facts",
                    "septmuse_episodic",
                    "septmuse_procedural",
                ]:
                    with contextlib.suppress(Exception):
                        session.exec(text(f"DELETE FROM {table_name}"))
                session.commit()

        if self.entity_store is not None:
            with contextlib.suppress(Exception):
                self.entity_store.reset()

        logger.info("memory_reset_pipeline_done")
        return {"status": "reset", "message": "storage layers cleared"}

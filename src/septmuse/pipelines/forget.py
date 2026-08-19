"""ForgetPipeline — 记忆遗忘流水线。

提取自 Memory.forget()，负责所有权检查 → invalidate → delete 编排。
"""

from __future__ import annotations

from typing import Any

from septmuse.core.logging import get_logger
from septmuse.storage.base import MemoryStore, TemporalStore

logger = get_logger(__name__)


class ForgetPipeline:
    """记忆遗忘流水线 (原 Memory.forget 内联逻辑)。

    职责:
    - 所有权校验
    - 先 invalidate (双时态失效)
    - 再 delete (软删除 + 实体引用清理)
    """

    def __init__(
        self,
        *,
        store: MemoryStore,
        entity_store: Any | None = None,
    ) -> None:
        self.store = store
        self.entity_store = entity_store

    def forget(self, memory_id: str, *, user_id: str) -> dict[str, Any]:
        """V2 forget: 先 invalidate 再 delete (决策 5)。"""
        existing = self.store.get(memory_id)
        if existing is not None:
            owner = existing.get("user_id")
            if owner is not None and owner != user_id:
                logger.warning("forget_ownership_mismatch", memory_id=memory_id, user_id=user_id, owner=owner)
                return {
                    "event": "FORGET",
                    "error": "ownership_mismatch",
                    "invalidated_at": None,
                    "deleted_at": None,
                }

        invalidated_at = None
        try:
            if isinstance(self.store, TemporalStore):
                inv = self.store.invalidate(memory_id)
                if isinstance(inv, dict):
                    invalidated_at = inv.get("invalid_at") or inv.get("invalidated_at")
        except Exception as e:
            logger.warning("forget_invalidate_failed", error=str(e))

        deleted_at = None
        try:
            if existing is not None:
                if self.entity_store is not None:
                    self.entity_store.remove_memory_from_entities(memory_id)
                self.store.delete(memory_id)
                from datetime import datetime, timezone

                deleted_at = datetime.now(timezone.utc).isoformat()
        except Exception as e:
            logger.warning("forget_delete_failed", error=str(e))

        logger.info("forget_done", memory_id=memory_id, user_id=user_id)
        return {"event": "FORGET", "invalidated_at": invalidated_at, "deleted_at": deleted_at}

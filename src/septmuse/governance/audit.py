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
"""记忆访问审计日志。"""

from __future__ import annotations

from typing import Any

from septmuse.core.logging import get_logger
from septmuse.storage.async_base import AsyncMemoryStore
from septmuse.storage.base import MemoryStore

logger = get_logger(__name__)


# ── sync ──────────────────────────────────────────────────────────────────


def record_access(
    store: MemoryStore,
    memory_id: str,
    app_id: str | None,
    access_type: str,
    metadata: dict[str, Any] | None = None,
) -> str | None:
    """记录记忆访问日志 (sync 版)。"""
    try:
        if hasattr(store, "_record_access_log"):
            return store._record_access_log(memory_id, app_id, access_type, metadata)
        logger.warning("store_does_not_support_access_log", store=type(store).__name__)
        return None
    except Exception as e:
        logger.warning("access_log_failed", error=str(e), memory_id=memory_id)
        return None


# ── async ─────────────────────────────────────────────────────────────────


async def async_record_access(
    store: AsyncMemoryStore,
    memory_id: str,
    app_id: str | None,
    access_type: str,
    metadata: dict[str, Any] | None = None,
) -> str | None:
    """异步记录记忆访问日志 (async 版)。"""
    try:
        if hasattr(store, "_record_access_log"):
            return await store._record_access_log(memory_id, app_id, access_type, metadata)
        logger.warning("async_store_does_not_support_access_log", store=type(store).__name__)
        return None
    except Exception as e:
        logger.warning("async_access_log_failed", error=str(e), memory_id=memory_id)
        return None

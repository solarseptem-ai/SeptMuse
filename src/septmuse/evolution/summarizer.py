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
"""消息压缩 Summarizer。"""

from __future__ import annotations

from typing import Any

from septmuse.core.logging import get_logger
from septmuse.llms.base import LLM
from septmuse.storage.base import MemoryStore
from septmuse.storage.relational_stores.typed_store import TypedMemoryStore

logger = get_logger(__name__)

SUMMARY_PROMPT = "Summarize the following memories concisely, preserving key facts and context:\n\n"


class Summarizer:
    """消息压缩 Summarizer。"""

    def __init__(self, store: MemoryStore, typed_store: TypedMemoryStore, llm: LLM | None = None) -> None:
        self.store = store
        self.typed_store = typed_store
        self.llm = llm

    def compress(
        self,
        *,
        user_id: str,
        mode: str = "static",
        buffer_size: int = 20,
    ) -> dict[str, Any]:
        """压缩消息。"""
        all_mems = self.store.get_all(user_id=user_id)
        if not isinstance(all_mems, list):
            all_mems = all_mems.get("results", []) if isinstance(all_mems, dict) else []

        if len(all_mems) <= buffer_size:
            logger.info("compress_skipped", user_id=user_id, count=len(all_mems), buffer_size=buffer_size)
            return {"compressed": False, "evicted": 0, "kept": len(all_mems), "summary_id": None}

        if mode == "partial":
            evict_count = max(1, int(len(all_mems) * 0.3))
            evicted = all_mems[:evict_count]
            kept = all_mems[evict_count:]
        else:
            evicted = all_mems[:-buffer_size] if buffer_size > 0 else all_mems
            kept = all_mems[len(evicted) :]

        summary_text = self._summarize(evicted)

        summary_episode = self.typed_store.add_episode(
            summary_text,
            user_id=user_id,
            event_type="summary",
        )

        for mem in evicted:
            mem_id = mem.get("id") if isinstance(mem, dict) else mem
            if not mem_id:
                continue
            self.store.delete(mem_id)

        logger.info(
            "compress_done",
            user_id=user_id,
            mode=mode,
            evicted=len(evicted),
            kept=len(kept),
            summary_id=summary_episode.id,
        )

        return {
            "compressed": True,
            "evicted": len(evicted),
            "kept": len(kept),
            "summary_id": summary_episode.id,
        }

    def _summarize(self, memories: list[dict[str, Any]]) -> str:
        """LLM 递归摘要 (无 LLM 时用拼接降级)。"""
        texts: list[str] = []
        for mem in memories:
            if isinstance(mem, dict):
                texts.append(mem.get("memory", mem.get("content", "")))
            else:
                texts.append(str(mem))

        joined = "\n".join(t for t in texts if t)

        if self.llm is None:
            truncated = joined[:500] + "..." if len(joined) > 500 else joined
            return f"[Summary] {len(memories)} messages compressed:\n{truncated}"

        try:
            return self.llm.complete(SUMMARY_PROMPT, joined)
        except Exception as e:
            logger.warning("summarize_llm_failed", error=str(e))
            truncated = joined[:500] + "..." if len(joined) > 500 else joined
            return f"[Summary] {len(memories)} messages compressed:\n{truncated}"

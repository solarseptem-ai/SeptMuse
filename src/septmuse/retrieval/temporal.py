"""时态区间查询 + LLM 自然语言时间抽取 (借鉴 graphiti search_filters + cognee temporal_retriever)。

功能:
- search_interval(start, end, query, user_id): 查询时间区间内为真的事实
- extract_time_range(query): LLM 从自然语言抽取时间区间 ("上周" → {"start", "end"})
- search_natural(query, user_id): 自然语言时态查询 (先抽时间, 有则时态过滤, 无则回退普通检索)
"""

from __future__ import annotations

import json
import re
from typing import Any

from septmuse.core.logging import get_logger
from septmuse.embedders.base import Embedder
from septmuse.llms.base import LLM
from septmuse.storage.base import MemoryStore

logger = get_logger(__name__)

TIME_EXTRACTION_PROMPT = """Extract time range from the user query. Output ONLY a JSON object.

Rules:
- If the query mentions a specific time range (e.g., "last week", "2024年6月", "yesterday"), extract start and end as ISO 8601 dates.
- "last week" → {"start": "<monday of last week>", "end": "<sunday of last week + 1 day>"}
- "2024年6月" → {"start": "2024-06-01", "end": "2024-07-01"}
- "yesterday" → {"start": "<yesterday>", "end": "<today>"}
- If no time range is mentioned, output {"start": null, "end": null}.
- Use today's date as reference: {today}

Examples:
Input: "Alice上周在做什么" → {{"start": "2026-07-14", "end": "2026-07-21"}}
Input: "Alice的工作经历" → {{"start": null, "end": null}}
"""


class TemporalRetriever:
    """时态区间查询 + LLM 自然语言时间抽取 (借鉴 cognee temporal_retriever)。

    用法:
        retriever = TemporalRetriever(store, embedder, llm)
        results = retriever.search_interval("2024-06-01", "2024-07-01", "Alice", user_id="u1")
        results = retriever.search_natural("Alice上周在做什么", user_id="u1")
    """

    def __init__(
        self,
        store: MemoryStore,
        embedder: Embedder,
        llm: LLM | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.llm = llm

    def search_interval(
        self,
        start: str,
        end: str,
        query: str,
        *,
        user_id: str,
        top_k: int = 5,
        threshold: float = 0.1,
    ) -> list[dict[str, Any]]:
        """时间区间查询: 返回 [start, end) 内为真的相关记忆。

        条件: valid_at <= end AND (invalid_at IS NULL OR invalid_at > start)
        valid_at IS NULL 的记忆视为"无时间约束", 始终返回 (向后兼容)。
        """
        valid_memories = self.store.get_temporal_interval(start, end, user_id=user_id)
        if not valid_memories:
            return []

        valid_ids = {m["id"] for m in valid_memories}

        emb = self.embedder.embed(query)
        search_results = self.store.search(emb, user_id=user_id, top_k=top_k * 2, threshold=threshold)
        filtered = [r for r in search_results if r["id"] in valid_ids]

        valid_map = {m["id"]: m for m in valid_memories}
        for r in filtered:
            vm = valid_map.get(r["id"])
            if vm:
                r["valid_at"] = vm.get("valid_at")
                r["invalid_at"] = vm.get("invalid_at")

        logger.info(
            "search_interval_done",
            user_id=user_id,
            start=start,
            end=end,
            candidates=len(valid_memories),
            returned=len(filtered[:top_k]),
        )
        return filtered[:top_k]

    def extract_time_range(self, query: str) -> dict[str, str | None] | None:
        """LLM 从自然语言抽取时间区间。

        Returns: {"start": "2024-07-01", "end": "2024-07-08"} 或 None (无时间信息)。
        无 LLM 时返回 None (回退普通检索)。
        """
        if self.llm is None:
            return None

        from datetime import datetime, timezone

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        prompt = TIME_EXTRACTION_PROMPT.replace("{today}", today)

        raw = self.llm.complete(prompt, f"Input:\n{query}")
        return self._parse_time_response(raw)

    @staticmethod
    def _parse_time_response(raw: str) -> dict[str, str | None] | None:
        """解析 LLM 时间抽取响应。"""
        cleaned = re.sub(r"^```[a-zA-Z0-9]*\n|\n```$", "", raw.strip())
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("time_parse_failed", raw=raw[:100])
            return None

        if not isinstance(data, dict):
            return None

        start = data.get("start")
        end = data.get("end")

        if start is None and end is None:
            return None

        start_str = str(start) if start else None
        end_str = str(end) if end else None

        if not start_str and not end_str:
            return None

        return {"start": start_str, "end": end_str}

    def search_natural(
        self,
        query: str,
        *,
        user_id: str,
        top_k: int = 5,
        threshold: float = 0.1,
    ) -> list[dict[str, Any]]:
        """自然语言时态查询 (验收: LLM 从"上周Alice在做什么"抽取时间区间)。

        1. LLM 抽取时间区间
        2. 有时间区间 → search_interval (时态过滤)
        3. 无时间信息 → 回退普通 search (验收: 无时间信息时回退普通检索)
        """
        time_range = self.extract_time_range(query)

        if time_range:
            start = time_range.get("start")
            end = time_range.get("end")
            if start and end:
                logger.info("natural_query_with_time", query=query[:50], time_range=time_range)
                return self.search_interval(
                    start,
                    end,
                    query,
                    user_id=user_id,
                    top_k=top_k,
                    threshold=threshold,
                )

        logger.info("natural_query_no_time_fallback", query=query[:50])
        emb = self.embedder.embed(query)
        return self.store.search(emb, user_id=user_id, top_k=top_k, threshold=threshold)

"""异步记忆 facade — 9 个 async 方法，提供 async/sync 双版本 API。

REST API 用 AsyncMemory，CLI/MCP 用 Memory（sync）。
store 层真 async（aiosqlite），embedder/LLM sync 用 asyncio.to_thread 包装。
"""
from __future__ import annotations

import asyncio
from typing import Any

from septmuse.configs.base import MemoryConfig
from septmuse.configs.defaults import default_config
from septmuse.core.logging import get_logger
from septmuse.embedders.base import Embedder
from septmuse.llms.base import LLM
from septmuse.storage.async_base import AsyncMemoryStore
from septmuse.storage.async_sqlite.store import AsyncSQLiteMemoryStore

logger = get_logger(__name__)


def _normalize_messages(messages: Any) -> list[str]:
    """消息标准化（复用 sync 版逻辑）。"""
    if isinstance(messages, str):
        return [messages]
    if isinstance(messages, list):
        texts: list[str] = []
        for msg in messages:
            if isinstance(msg, dict):
                texts.append(msg.get("content", ""))
            elif isinstance(msg, str):
                texts.append(msg)
        return [t for t in texts if t.strip()]
    return []


class AsyncMemory:
    """异步记忆 facade。

    用法:
        mem = AsyncMemory()
        result = await mem.add("hello", user_id="alice")
        results = await mem.search("hello", user_id="alice")
    """

    def __init__(
        self,
        config: MemoryConfig | None = None,
        *,
        embedder: Embedder | None = None,
        store: AsyncMemoryStore | None = None,
        llm: LLM | None = None,
    ) -> None:
        self.config = config or default_config()
        self.embedder = embedder or self._resolve_embedder()
        self.store = store or AsyncSQLiteMemoryStore(db_path=self.config.db_path)
        self.llm = llm
        if self.llm is None and self.config.llm is not None:
            self.llm = self._resolve_llm()
        logger.info("async_memory_init", db_path=str(self.config.db_path))

    def _resolve_embedder(self) -> Embedder:
        from septmuse.services.providers import embedder_provider
        return embedder_provider.resolve(self.config.embedder.backend, config=self.config.embedder)

    def _resolve_llm(self) -> LLM | None:
        from septmuse.services.providers import llm_provider
        return llm_provider.resolve(self.config.llm.backend, config=self.config.llm)

    async def add(
        self,
        messages: Any,
        *,
        user_id: str,
        agent_id: str | None = None,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        infer: bool | None = None,
        valid_at: str | None = None,
        auto_extract_entities: bool = True,
    ) -> dict[str, Any]:
        """异步添加记忆。"""
        texts = _normalize_messages(messages)
        if not texts:
            return {"results": [], "relations": []}

        embeddings = await asyncio.to_thread(self.embedder.embed_batch, texts)

        results: list[dict[str, Any]] = []
        for text, emb in zip(texts, embeddings, strict=True):
            mid = await self.store.add(
                text, emb, user_id=user_id, agent_id=agent_id,
                session_id=session_id, metadata=metadata, valid_at=valid_at,
            )
            results.append({"id": mid, "memory": text, "event": "ADD"})

        logger.info("async_add_done", user_id=user_id, count=len(results))
        return {"results": results, "relations": []}

    async def search(
        self, query: str, *, user_id: str, session_id: str | None = None,
        top_k: int = 5, threshold: float = 0.1, filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """异步检索记忆。"""
        emb = await asyncio.to_thread(self.embedder.embed, query)
        return await self.store.search(
            emb, user_id=user_id, session_id=session_id, top_k=top_k, threshold=threshold, filters=filters
        )

    async def update(
        self, memory_id: str, content: str, *, metadata: dict[str, Any] | None = None
    ) -> bool:
        """异步更新记忆。"""
        emb = await asyncio.to_thread(self.embedder.embed, content)
        return await self.store.update(memory_id, content, emb, metadata=metadata)

    async def delete(self, memory_id: str) -> None:
        """异步软删除。"""
        await self.store.delete(memory_id)

    async def delete_all(self, *, user_id: str) -> int:
        """异步批量删除该用户所有记忆。"""
        memories = await self.store.get_all(user_id=user_id)
        for m in memories:
            await self.store.delete(m["id"])
        return len(memories)

    async def get(self, memory_id: str) -> dict[str, Any] | None:
        """异步取单条。"""
        return await self.store.get(memory_id)

    async def get_all(
        self, *, user_id: str, session_id: str | None = None, filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """异步列出全部。"""
        return await self.store.get_all(user_id=user_id, session_id=session_id, filters=filters)

    async def get_history(self, memory_id: str) -> list[dict[str, Any]]:
        """异步获取变更历史。"""
        return await self.store.get_history(memory_id)

    async def invalidate(self, memory_id: str, *, invalid_at: str | None = None) -> dict[str, Any]:
        """异步标记事实不再为真。"""
        return await self.store.invalidate(memory_id, invalid_at=invalid_at)

    async def close(self) -> None:
        """异步释放资源。"""
        await self.store.close()

"""异步记忆 facade — 基础方法真 async (aiosqlite)，高级方法 to_thread 桥接 sync。

REST API 用 AsyncMemory，CLI/MCP 用 Memory（sync）。
store 层真 async（aiosqlite），embedder/LLM sync 用 asyncio.to_thread 包装。
高级方法 (cognify/reflect/compress/search_graph 等) 通过内部 sync ExperimentalMemory 实例
用 asyncio.to_thread 委托，共享同一 DB 文件。
"""
from __future__ import annotations

import asyncio
from typing import Any

from septmuse.configs.base import MemoryConfig
from septmuse.configs.defaults import default_config
from septmuse.core.logging import get_logger
from septmuse.embedders.base import Embedder
from septmuse.embedders.cached import CachedEmbedder
from septmuse.embedders.resolver import resolve_embedder
from septmuse.llms.base import LLM
from septmuse.storage.async_base import AsyncMemoryStore

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

    高级方法 (cognify/reflect/compress/search_graph/search_interval 等) 通过
    asyncio.to_thread 委托内部 sync ExperimentalMemory，共享同一 DB 文件。
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
        if not isinstance(self.embedder, CachedEmbedder):
            self.embedder = CachedEmbedder(self.embedder)
        self.store = store or self._resolve_store()
        self.llm = llm
        if self.llm is None and self.config.llm is not None:
            self.llm = self._resolve_llm()

        # 内部 sync 实例（共享同一 DB 文件 + vector_store，高级方法用 to_thread 桥接）
        from septmuse.experimental import ExperimentalMemory

        sync_store = None
        vs = getattr(self.store, "_vector_store", None)
        ki = getattr(self.store, "_keyword_index", None)
        if vs is not None:
            from sqlalchemy import create_engine as _ce

            sync_url = (
                str(self.config.db_url or f"sqlite:///{self.config.db_path or ':memory:'}")
                .replace("+aiosqlite", "")
                .replace("+aiomysql", "+pymysql")
                .replace("+asyncpg", "+psycopg2")
            )
            sync_engine = _ce(sync_url)
            from septmuse.storage.relational_stores.orm_store import ORMMemoryStore

            sync_store = ORMMemoryStore(sync_engine, vs, ki)

        self._sync = ExperimentalMemory(
            config=self.config,
            embedder=self.embedder,
            store=sync_store,
            llm=self.llm,
        )

        logger.info("async_memory_init", db_path=str(self.config.db_path))

    def _resolve_embedder(self) -> Embedder:
        return resolve_embedder(self.config)

    def _resolve_llm(self) -> LLM | None:
        from septmuse.services.providers import llm_provider
        return llm_provider.resolve(self.config.llm.backend, config=self.config.llm)

    def _resolve_store(self) -> AsyncMemoryStore:
        """解析 store: 统一走 AsyncORMMemoryStore (DatabaseService 自动回退 SQLite 零配置)。"""
        from septmuse.storage.relational_stores.factory import RelationalStoreFactory

        return RelationalStoreFactory.create_async(self.config)

    # ------------------------------------------------------------------
    # 基础 API（真 async: aiosqlite + to_thread embedder）
    # ------------------------------------------------------------------

    async def add(
        self,
        messages: Any,
        *,
        user_id: str,
        agent_id: str | None = None,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        infer: bool | None = None,
        memory_type: str | None = None,
        valid_at: str | None = None,
        auto_extract_entities: bool = True,
        expiration_date: str | None = None,
        attributed_to: str | None = None,
        namespace: str = "default",
        subject: str | None = None,
        predicate: str | None = None,
        object: str | None = None,
        event_type: str = "fact",
        rule: str | None = None,
    ) -> dict[str, Any]:
        """异步添加记忆。

        memory_type != None 或 infer 决策为 True → 委托 sync (to_thread)。
        infer=False verbatim → 真 async 路径 (P1 参数: expiration_date/attributed_to/actor_id)。
        """
        # 类型化记忆 + 决策模式 → 委托 sync
        should_delegate = (
            memory_type is not None
            or infer is True
            or (infer is None and self.llm is not None)
        )
        if should_delegate:
            return await asyncio.to_thread(
                self._sync.add,
                messages,
                user_id=user_id,
                agent_id=agent_id,
                session_id=session_id,
                metadata=metadata,
                infer=infer,
                memory_type=memory_type,
                valid_at=valid_at,
                auto_extract_entities=auto_extract_entities,
                expiration_date=expiration_date,
                attributed_to=attributed_to,
                namespace=namespace,
                subject=subject,
                predicate=predicate,
                object=object,
                event_type=event_type,
                rule=rule,
            )

        # verbatim 真 async 路径
        from septmuse.core.validation import validate_entity_id

        user_id = validate_entity_id(user_id, "user_id") or "default"
        if agent_id is not None:
            agent_id = validate_entity_id(agent_id, "agent_id")
        if session_id is not None:
            session_id = validate_entity_id(session_id, "session_id")

        # 过期日期归一化
        from septmuse.memory.main import _normalize_expiration_date

        normalized_exp = _normalize_expiration_date(expiration_date)
        if normalized_exp is not None:
            if metadata is None:
                metadata = {}
            metadata = {**metadata, "expiration_date": normalized_exp}

        # 多说话人归因
        if attributed_to is not None:
            if metadata is None:
                metadata = {}
            metadata = {**metadata, "attributed_to": attributed_to}

        # actor_id 从 message["name"] 提取
        if isinstance(messages, list) and messages:
            first_msg = messages[0] if isinstance(messages[0], dict) else {}
            actor_id = first_msg.get("name")
            if actor_id is not None:
                if metadata is None:
                    metadata = {}
                metadata = {**metadata, "actor_id": actor_id}

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
        self,
        query: str,
        *,
        user_id: str,
        session_id: str | None = None,
        top_k: int = 5,
        threshold: float = 0.1,
        filters: dict[str, Any] | None = None,
        hybrid: bool = False,
        reranker: str | None = None,
        recipe: str | None = None,
        explain: bool = False,
        search_filter: dict[str, Any] | None = None,
        forgetting: bool = False,
        token_budget: int | None = None,
        inject_prompt: bool = False,
        hyde: bool = False,
        query_rewrite: bool = False,
        show_expired: bool = False,
    ) -> list[dict[str, Any]] | dict[str, Any]:
        """异步检索记忆。

        有高级参数时 → 委托 sync (hybrid/reranker/recipe 等)。
        基础检索 (纯向量) → 真 async 路径 + 过期过滤。
        """
        # 高级参数检测 → 委托 sync
        has_advanced = (
            hybrid or reranker is not None or recipe is not None or explain
            or search_filter is not None or forgetting or token_budget is not None
            or inject_prompt or hyde or query_rewrite
        )
        if has_advanced:
            return await asyncio.to_thread(
                self._sync.search,
                query,
                user_id=user_id,
                session_id=session_id,
                top_k=top_k,
                threshold=threshold,
                filters=filters,
                hybrid=hybrid,
                reranker=reranker,
                recipe=recipe,
                explain=explain,
                search_filter=search_filter,
                forgetting=forgetting,
                token_budget=token_budget,
                inject_prompt=inject_prompt,
                hyde=hyde,
                query_rewrite=query_rewrite,
                show_expired=show_expired,
            )

        # 基础纯向量检索 (真 async)
        from septmuse.core.validation import validate_entity_id, validate_search_query

        user_id = validate_entity_id(user_id, "user_id") or "default"
        query = validate_search_query(query)

        emb = await asyncio.to_thread(self.embedder.embed, query)
        results = await self.store.search(
            emb, user_id=user_id, session_id=session_id, top_k=top_k, threshold=threshold, filters=filters
        )

        # 过期过滤
        if not show_expired:
            from septmuse.memory.main import _is_expired

            results = [r for r in results if not _is_expired(r.get("metadata"))]

        return results

    async def update(
        self,
        memory_id: str,
        content: str,
        *,
        metadata: dict[str, Any] | None = None,
        user_id: str | None = None,
    ) -> dict[str, Any] | bool:
        """异步更新记忆 (含实体重链接).

        user_id 提供时 → 委托 sync (含实体清理+重链接).
        无 user_id → 真 async (仅更新 content+embedding).
        """
        if user_id is not None and self._sync is not None:
            return await asyncio.to_thread(
                self._sync.update,
                memory_id,
                content,
                metadata=metadata,
                user_id=user_id,
            )
        # 降级: 真 async (无实体重链接)
        emb = await asyncio.to_thread(self.embedder.embed, content)
        return await self.store.update(memory_id, content, emb, metadata=metadata)

    async def delete(self, memory_id: str, *, user_id: str | None = None) -> dict[str, Any] | None:
        """异步删除记忆 (user_id 提供时含 invalidate + 图清理)."""
        if user_id is not None and self._sync is not None:
            return await asyncio.to_thread(
                self._sync.forget,
                memory_id,
                user_id=user_id,
            )
        # 降级: 仅软删除
        await self.store.delete(memory_id)
        return None

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

    # ------------------------------------------------------------------
    # 高级 API（to_thread 桥接 sync ExperimentalMemory，共享同一 DB 文件）
    # ------------------------------------------------------------------

    async def cognify(self, text: str, *, user_id: str, agent_id: str | None = None) -> dict[str, Any]:
        """异步构建知识图谱（存记忆→抽三元组→存实体/关系→建链接）。"""
        return await asyncio.to_thread(self._sync.cognify, text, user_id=user_id, agent_id=agent_id)

    async def search_graph_fused(
        self, query: str, *, user_id: str, seed_memory_id: str,
        max_depth: int = 2, relation: str | None = None, top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """异步 BFS + 向量结果 RRF 融合检索。"""
        return await asyncio.to_thread(
            self._sync.search_graph_fused, query,
            user_id=user_id, seed_memory_id=seed_memory_id,
            max_depth=max_depth, relation=relation, top_k=top_k,
        )

    async def get_entity_relations(self, entity_name: str, *, user_id: str) -> list[dict[str, Any]]:
        """异步实体间关系遍历（双向）。"""
        return await asyncio.to_thread(
            self._sync.get_entity_relations, entity_name, user_id=user_id
        )

    async def reflect(self, *, user_id: str, limit: int = 20) -> dict[str, Any]:
        """异步会话蒸馏（提取教训→procedural rules）。"""
        return await asyncio.to_thread(self._sync.reflect, user_id=user_id, limit=limit)

    async def resolve_conflicts(self, *, user_id: str) -> dict[str, Any]:
        """异步解决矛盾事实（软删除旧 fact + invalidate verbatim）。"""
        return await asyncio.to_thread(self._sync.resolve_conflicts, user_id=user_id)

    async def deduplicate_entities(self, *, user_id: str) -> dict[str, Any]:
        """异步实体去重三段式（精确归一 + 模糊相似 + LLM 兜底）。"""
        return await asyncio.to_thread(self._sync.deduplicate_entities, user_id=user_id)

    async def compress(self, *, user_id: str, mode: str = "static", buffer_size: int = 20) -> dict[str, Any]:
        """异步消息压缩（static/partial 两种模式）。"""
        return await asyncio.to_thread(
            self._sync.compress, user_id=user_id, mode=mode, buffer_size=buffer_size
        )

    async def search_at(
        self, reference_time: str, query: str, *,
        user_id: str, session_id: str | None = None, top_k: int = 5, threshold: float = 0.1,
    ) -> list[dict[str, Any]]:
        """异步时态查询（查询某时刻为真的相关记忆）。"""
        return await asyncio.to_thread(
            self._sync.search_at, reference_time, query,
            user_id=user_id, session_id=session_id, top_k=top_k, threshold=threshold,
        )

    async def search_interval(
        self, start: str, end: str, query: str, *,
        user_id: str, session_id: str | None = None, top_k: int = 5, threshold: float = 0.1,
    ) -> list[dict[str, Any]]:
        """异步时间区间查询（[start, end) 内为真的相关记忆）。"""
        return await asyncio.to_thread(
            self._sync.search_interval, start, end, query,
            user_id=user_id, session_id=session_id, top_k=top_k, threshold=threshold,
        )

    async def search_natural(
        self, query: str, *, user_id: str, top_k: int = 5, threshold: float = 0.1,
    ) -> list[dict[str, Any]]:
        """异步自然语言时态查询（LLM 抽时间区间→时态过滤→无则回退普通检索）。"""
        return await asyncio.to_thread(
            self._sync.search_natural, query,
            user_id=user_id, top_k=top_k, threshold=threshold,
        )

    async def get_access_logs(self, memory_id: str, limit: int = 100) -> list[dict[str, Any]]:
        """异步查询记忆访问日志。"""
        return await self.store.get_access_logs(memory_id, limit=limit)

    async def get_active_rules(self, *, user_id: str, namespace: str = "default") -> list[dict[str, Any]]:
        """异步获取应注入的规则（废弃规则不返回）。"""
        return await asyncio.to_thread(self._sync.get_active_rules, user_id=user_id, namespace=namespace)

    async def rules_to_prompt(self, *, user_id: str, namespace: str = "default") -> str:
        """异步编译规则为 prompt 注入文本。"""
        return await asyncio.to_thread(self._sync.rules_to_prompt, user_id=user_id, namespace=namespace)

    # ------------------------------------------------------------------
    # V2 编排 API (to_thread 桥接 sync, 共享同一 DB 文件)
    # ------------------------------------------------------------------

    async def remember(
        self,
        messages: Any,
        *,
        user_id: str,
        agent_id: str | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """异步捕获 → add(决策) → episodic raw_log → working_memory."""
        if self._sync is None:
            # 降级: 简单 add
            return await self.add(messages, user_id=user_id, agent_id=agent_id, session_id=session_id)
        return await asyncio.to_thread(
            self._sync.remember,
            messages,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
        )

    async def recall(
        self,
        query: str,
        *,
        user_id: str,
        top_k: int = 5,
        recipe: str | None = None,
        inject_profile: bool = False,
        auto_compress: bool = False,
    ) -> dict[str, Any]:
        """异步检索 → 遗忘加权 → token预算 → block+规则注入 → 画像注入."""
        if self._sync is None:
            # 降级: 简单 search
            results = await self.search(query, user_id=user_id, top_k=top_k)
            return {"memories": results, "injected_prompt": "", "route": None, "strategy": None, "used_tokens": 0}
        return await asyncio.to_thread(
            self._sync.recall,
            query,
            user_id=user_id,
            top_k=top_k,
            recipe=recipe,
            inject_profile=inject_profile,
            auto_compress=auto_compress,
        )

    async def forget(self, memory_id: str, *, user_id: str) -> dict[str, Any]:
        """异步 invalidate → delete → 图清理."""
        if self._sync is None:
            await self.store.delete(memory_id)
            return {"memory_id": memory_id, "event": "FORGET"}
        return await asyncio.to_thread(
            self._sync.forget,
            memory_id,
            user_id=user_id,
        )

    async def improve(self, *, user_id: str, limit: int = 50) -> dict[str, Any]:
        """异步 dream + reflect + conflict + coverage."""
        if self._sync is None:
            return {"dream": {"links_created": 0, "processed": 0}, "rules": 0, "conflicts": {}, "coverage": {}}
        return await asyncio.to_thread(
            self._sync.improve,
            user_id=user_id,
            limit=limit,
        )

    async def close(self) -> None:
        """异步释放资源。"""
        await self.store.close()

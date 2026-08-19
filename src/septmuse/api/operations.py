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
"""MemoryOperations — 统一业务操作适配层。"""

from __future__ import annotations

import functools
import json
import os
from typing import Any

from septmuse.core.logging import get_logger

logger = get_logger(__name__)


def _get_memory_class() -> Any:
    from septmuse.memory.main import Memory

    return Memory


# ------------------------------------------------------------------
# 通用装饰器: 自动捕获异常并包装为结构化错误
# ------------------------------------------------------------------


def _safe_op(func):
    """装饰器: 捕获异常并包装为 {error, code} 结构。"""

    @functools.wraps(func)
    def wrapper(self: MemoryOperations, *args: Any, **kwargs: Any) -> Any:
        try:
            return func(self, *args, **kwargs)
        except Exception as e:
            op_name = func.__name__
            logger.exception(f"op_{op_name}_error")
            return {"error": str(e), "code": f"{op_name.upper()}_FAILED"}

    return wrapper


# ------------------------------------------------------------------
# MemoryOperations
# ------------------------------------------------------------------


class MemoryOperations:
    """统一业务操作层 — 所有 API 入口共享。

    Args:
        memory: Memory facade 实例; None 时内部懒加载创建 (CLI 便利)。
    """

    def __init__(self, memory: Any | None = None) -> None:
        self._memory = memory
        self._db_path: str | None = None

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    @property
    def memory(self) -> Any:
        """底层 Memory facade 实例。"""
        return self._resolve_memory()

    def _resolve_memory(self) -> Any:
        """获取 Memory 实例 (懒加载)。"""
        if self._memory is None:
            self._memory = self._create_memory()
        return self._memory

    def _create_memory(self) -> Any:
        """创建默认 Memory 实例 (SQLite + HashEmbedder)。"""
        from septmuse.configs.defaults import MemoryConfig
        from septmuse.embedders.hash import HashEmbedder

        db_path = self._db_path or os.getenv("SEPTMUSE_DB_PATH")
        config = MemoryConfig(db_path=db_path)
        return _get_memory_class()(config=config, embedder=HashEmbedder())

    @staticmethod
    def resolve_user_id(explicit: str | None = None) -> str:
        """解析 user_id: 显式参数 -> SEPTMUSE_USER_ID 环境变量 -> 'default'。"""
        return explicit or os.getenv("SEPTMUSE_USER_ID", "default")

    @staticmethod
    def to_json(data: Any, *, indent: int | None = 2) -> str:
        """统一 JSON 序列化 (ensure_ascii=False, default=str)。"""
        return json.dumps(data, ensure_ascii=False, default=str, indent=indent)

    # ------------------------------------------------------------------
    # 核心 CRUD 操作
    # ------------------------------------------------------------------

    @_safe_op
    def add_typed(self, content: str, memory_type: str, *, user_id: str) -> dict[str, Any]:
        """添加类型化记忆 (semantic/episodic/procedural)。"""
        return self._resolve_memory().add_typed(content, memory_type, user_id=user_id)

    @_safe_op
    def add(
        self,
        content: str,
        *,
        user_id: str,
        agent_id: str | None = None,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        infer: bool = False,
        memory_type: str | None = None,
        valid_at: str | None = None,
    ) -> dict[str, Any]:
        """添加记忆。"""
        return self._resolve_memory().add(
            content,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            metadata=metadata,
            infer=infer,
            memory_type=memory_type,
            valid_at=valid_at,
        )

    @_safe_op
    def search(
        self,
        query: str,
        *,
        user_id: str,
        session_id: str | None = None,
        top_k: int = 5,
        threshold: float = 0.1,
        filters: dict[str, Any] | None = None,
        reranker: str | None = None,
        explain: bool = False,
    ) -> list[dict[str, Any]]:
        """搜索记忆。"""
        return self._resolve_memory().search(
            query,
            user_id=user_id,
            session_id=session_id,
            top_k=top_k,
            threshold=threshold,
            filters=filters,
            reranker=reranker,
            explain=explain,
        )

    @_safe_op
    def get_all(
        self,
        *,
        user_id: str,
        session_id: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """列出全部记忆。"""
        return self._resolve_memory().get_all(user_id=user_id, session_id=session_id, filters=filters)

    @_safe_op
    def get(self, memory_id: str) -> dict[str, Any] | None:
        """取单条记忆。"""
        return self._resolve_memory().get(memory_id)

    @_safe_op
    def update(
        self,
        memory_id: str,
        content: str | None = None,
        *,
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """更新记忆。"""
        return self._resolve_memory().update(memory_id, content, user_id=user_id, metadata=metadata)

    @_safe_op
    def delete(self, memory_id: str) -> dict[str, Any]:
        """删除单条记忆。"""
        return self._resolve_memory().delete(memory_id)

    @_safe_op
    def delete_all(self, *, user_id: str, session_id: str | None = None) -> dict[str, Any]:
        """批量删除记忆。"""
        return self._resolve_memory().delete_all(user_id=user_id, session_id=session_id)

    @_safe_op
    def get_history(self, memory_id: str) -> list[dict[str, Any]]:
        """获取记忆变更历史。"""
        return self._resolve_memory().get_history(memory_id)

    @_safe_op
    def invalidate(self, memory_id: str, *, invalid_at: str | None = None) -> dict[str, Any]:
        """标记记忆失效。"""
        return self._resolve_memory().invalidate(memory_id, invalid_at=invalid_at)

    # ------------------------------------------------------------------
    # 实体操作
    # ------------------------------------------------------------------

    @_safe_op
    def search_entities(self, query: str, *, user_id: str, top_k: int = 5) -> dict[str, Any]:
        """搜索实体。"""
        results = self._resolve_memory().search_entities(query, user_id=user_id, top_k=top_k)
        return {"results": results}

    @_safe_op
    def list_entities(self, *, user_id: str, entity_type: str | None = None, limit: int = 100) -> dict[str, Any]:
        """列出实体。"""
        results = self._resolve_memory().list_entities(user_id=user_id, entity_type=entity_type, limit=limit)
        return {"results": results}

    # ------------------------------------------------------------------
    # Working memory block
    # ------------------------------------------------------------------

    @_safe_op
    def get_blocks(self, agent_id: str) -> list[dict[str, Any]]:
        """列出 agent 的 block。"""
        return self._resolve_memory().get_blocks(agent_id)

    @_safe_op
    def update_block(self, agent_id: str, label: str, value: str) -> dict[str, Any]:
        """更新 block。"""
        return self._resolve_memory().update_block(agent_id, label, value)

    @_safe_op
    def core_memory_append(self, agent_id: str, label: str, content: str) -> dict[str, Any]:
        """追加 block 内容。"""
        return self._resolve_memory().core_memory_append(agent_id, label, content)

    @_safe_op
    def core_memory_replace(self, agent_id: str, label: str, old_content: str, new_content: str) -> dict[str, Any]:
        """替换 block 内容片段。"""
        return self._resolve_memory().core_memory_replace(agent_id, label, old_content, new_content)

    # ------------------------------------------------------------------
    # V2 编排 (remember / recall / improve / forget)
    # ------------------------------------------------------------------

    @_safe_op
    def remember(
        self,
        text: str,
        *,
        user_id: str,
        session_id: str | None = None,
        agent_id: str | None = None,
    ) -> dict[str, Any]:
        """V2 remember 入口。"""
        return self._resolve_memory().remember(text, user_id=user_id, session_id=session_id, agent_id=agent_id)

    @_safe_op
    def recall(
        self,
        query: str,
        *,
        user_id: str,
        session_id: str | None = None,
        top_k: int = 5,
        token_budget: int = 2000,
    ) -> dict[str, Any]:
        """V2 recall 入口。"""
        return self._resolve_memory().recall(
            query,
            user_id=user_id,
            session_id=session_id,
            top_k=top_k,
            inject_prompt=True,
            token_budget=token_budget,
        )

    @_safe_op
    def improve(self, *, user_id: str, limit: int = 20) -> dict[str, Any]:
        """V2 improve 入口。"""
        return self._resolve_memory().improve(user_id=user_id, limit=limit)

    @_safe_op
    def forget(self, memory_id: str, *, user_id: str) -> dict[str, Any]:
        """V2 forget 入口。"""
        return self._resolve_memory().forget(memory_id, user_id=user_id)

    # ------------------------------------------------------------------
    # 扩展操作
    # ------------------------------------------------------------------

    @_safe_op
    def cognify(self, text: str, *, user_id: str, agent_id: str | None = None) -> dict[str, Any]:
        """构建知识图谱。"""
        return self._resolve_memory().cognify(text, user_id=user_id, agent_id=agent_id)

    @_safe_op
    def reflect(self, *, user_id: str, limit: int = 50) -> dict[str, Any]:
        """会话蒸馏。"""
        return self._resolve_memory().reflect(user_id=user_id, limit=limit)

    @_safe_op
    def search_graph(self, memory_id: str, *, max_depth: int = 3) -> dict[str, Any]:
        """BFS 图遍历检索。"""
        results = self._resolve_memory().search_graph(memory_id, max_depth=max_depth)
        return {"results": results}

    @_safe_op
    def search_interval(self, start: str, end: str, query: str, *, user_id: str) -> dict[str, Any]:
        """时间区间查询。"""
        results = self._resolve_memory().search_interval(start, end, query, user_id=user_id)
        return {"results": results}

    @_safe_op
    def get_access_logs(self, memory_id: str, *, limit: int = 100) -> list[dict[str, Any]]:
        """查询记忆访问日志。"""
        store = self._resolve_memory().store
        if not hasattr(store, "get_access_logs"):
            return []
        return store.get_access_logs(memory_id, limit)

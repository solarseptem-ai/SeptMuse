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
"""FastAPI REST 端点 (架构文档 §11.2 草案)。

对齐架构文档 §11.2 API 草案:
- POST   /memories              # 添加记忆
- GET    /memories              # 列出记忆
- GET    /memories/{memory_id}  # 取单条记忆
- PUT    /memories/{memory_id}  # 更新记忆 (对齐 mem0)
- DELETE /memories/{memory_id}  # 删除记忆
- GET    /memories/working/blocks/{agent_id}              # 列出 block
- PUT    /memories/working/blocks/{agent_id}/{label}      # 更新 block
- POST   /memories/working/blocks/{agent_id}/{label}/append  # 追加 block
- POST   /memories/working/blocks/{agent_id}/{label}/replace  # 替换 block 片段
- POST   /memories/search       # 统一检索 (元认知路由)
- POST   /memories/search/causal # 反事实因果查询
- GET    /memories/meta/coverage # 元认知覆盖报告
- POST   /memories/rehearse     # 主动复述
- GET    /agents/{user_id}/memories # 跨 agent 共享读

用法:
    app = create_app()
    # uvicorn septmuse.api.rest:app --reload
    # 或
    # from septmuse.api.rest import create_app; app = create_app()
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, model_validator

from septmuse import MemoryConfig
from septmuse.embedders.hash import HashEmbedder
from septmuse.experimental import ExperimentalMemory
from septmuse.governance.access_log import record_access as record_access
from septmuse.governance.async_access_log import async_record_access
from septmuse.governance.async_permissions import async_check_memory_access_permissions
from septmuse.governance.permissions import check_memory_access_permissions as check_memory_access_permissions
from septmuse.memory.async_main import AsyncMemory

# ======================================================================
# Pydantic 请求模型
# ======================================================================


class AddMemoryRequest(BaseModel):
    content: str = Field(description="记忆内容")
    user_id: str = Field(description="用户 ID")
    agent_id: str | None = Field(default=None, description="agent ID")
    session_id: str | None = Field(default=None, description="会话 ID (对齐 mem0 run_id)")
    memory_type: str = Field(default="verbatim", description="verbatim|semantic|episodic|procedural")
    infer: bool | None = Field(default=None, description="LLM 抽取")
    valid_at: str | None = Field(default=None, description="事实开始为真的时间 (ISO 8601)")

    @model_validator(mode="before")
    @classmethod
    def _accept_messages_alias(cls, data):
        if isinstance(data, dict) and "messages" in data and "content" not in data:
            data["content"] = data["messages"]
        return data


class SearchRequest(BaseModel):
    query: str = Field(description="查询文本")
    user_id: str = Field(description="用户 ID")
    session_id: str | None = Field(default=None, description="会话 ID (对齐 mem0 run_id)")
    filters: dict[str, Any] | None = Field(default=None, description="mem0 风格 filters dict")
    top_k: int = Field(default=5, description="返回数")
    threshold: float = Field(default=0.1, description="相似阈值")
    reranker: str | None = Field(default=None, description="reranker: noop/mmr/cross_encoder/llm")
    explain: bool = Field(default=False, description="返回 score_details")


class CausalRequest(BaseModel):
    cause_event_id: str
    effect_event_id: str
    user_id: str


class RehearseRequest(BaseModel):
    memory_id: str | None = Field(default=None, description="指定记忆 ID (None=批量)")
    user_id: str = Field(description="用户 ID")


class CaptureRequest(BaseModel):
    text: str = Field(description="捕获文本")
    user_id: str = Field(description="用户 ID")
    agent_id: str | None = Field(default=None)
    session_id: str | None = Field(default=None, description="会话 ID (对齐 mem0 run_id)")


class UpdateMemoryRequest(BaseModel):
    text: str | None = Field(default=None, description="新内容")
    metadata: dict[str, Any] | None = Field(default=None, description="新 metadata")


class BlockUpdateRequest(BaseModel):
    value: str = Field(description="新 block 内容")


class BlockAppendRequest(BaseModel):
    content: str = Field(description="追加内容")


class BlockReplaceRequest(BaseModel):
    old_content: str = Field(description="被替换的旧内容")
    new_content: str = Field(description="新内容")


class InvalidateRequest(BaseModel):
    invalid_at: str | None = Field(default=None, description="失效时间 (ISO 8601), 默认当前时间")


# ======================================================================
# App 工厂
# ======================================================================


def register_routes(app: FastAPI, memory: ExperimentalMemory, async_memory: AsyncMemory | None = None) -> None:
    """注册 REST 路由到已有 FastAPI app。

    供 CLI serve --with-rest 使用 (同一 app 同时挂 MCP + REST)。
    """
    app.state.memory = memory
    app.state.async_memory = async_memory or memory

    @app.post("/memories", status_code=201)
    async def add_memory(req: AddMemoryRequest) -> dict[str, Any]:
        """添加记忆 (架构文档 §11.2)。"""
        if req.memory_type == "semantic":
            parts = req.content.split(None, 2)
            subject = parts[0] if len(parts) > 0 else req.content
            predicate = parts[1] if len(parts) > 1 else "is"
            obj = parts[2] if len(parts) > 2 else ""
            return app.state.memory.add_fact(subject, predicate, obj, user_id=req.user_id)
        elif req.memory_type == "episodic":
            return app.state.memory.add_episode(req.content, user_id=req.user_id)
        elif req.memory_type == "procedural":
            return app.state.memory.add_rule(req.content, user_id=req.user_id)
        else:
            result = await app.state.async_memory.add(
                req.content,
                user_id=req.user_id,
                agent_id=req.agent_id,
                session_id=req.session_id,
                infer=req.infer,
                valid_at=req.valid_at,
            )
            # 实体抽取走 sync 路径（async add 不做实体抽取，sync/async 共享 DB 文件）
            for r in result.get("results", []):
                app.state.memory._extract_and_store_entities(
                    r["memory"], r["id"], user_id=req.user_id, agent_id=req.agent_id
                )
            return result

    @app.get("/memories")
    async def list_memories(
        user_id: str = Query(..., description="用户 ID"),
        session_id: str | None = Query(default=None, description="会话 ID"),
        app_id: str | None = None,
    ) -> dict[str, Any]:
        """列出记忆 (对齐 mem0 get_all)。"""
        results = await app.state.async_memory.get_all(user_id=user_id, session_id=session_id)
        store = app.state.async_memory.store
        for r in results:
            await async_record_access(store, r["id"], app_id, "list")
        return {"results": results}

    @app.get("/memories/working/blocks/{agent_id}")
    async def list_blocks(agent_id: str) -> list[dict[str, Any]]:
        """列出 agent 的全部 block。"""
        return app.state.memory.get_blocks(agent_id)

    @app.put("/memories/working/blocks/{agent_id}/{label}")
    async def update_block(agent_id: str, label: str, req: BlockUpdateRequest) -> dict[str, Any]:
        """更新 block value (架构 §11.2)。"""
        return app.state.memory.update_block(agent_id, label, req.value)

    @app.post("/memories/working/blocks/{agent_id}/{label}/append")
    async def append_block(agent_id: str, label: str, req: BlockAppendRequest) -> dict[str, Any]:
        """追加 block 内容 (对齐 Letta core_memory_append)。"""
        return app.state.memory.core_memory_append(agent_id, label, req.content)

    @app.post("/memories/working/blocks/{agent_id}/{label}/replace")
    async def replace_block(agent_id: str, label: str, req: BlockReplaceRequest) -> dict[str, Any]:
        """替换 block 内容片段 (对齐 Letta core_memory_replace)。"""
        return app.state.memory.core_memory_replace(agent_id, label, req.old_content, req.new_content)

    @app.get("/memories/{memory_id}")
    async def get_memory(memory_id: str, app_id: str | None = None) -> dict[str, Any]:
        """取单条记忆。

        权限层: async_check_memory_access_permissions 校验存在性 + state=active。
        403=存在但非 active (deleted/archived/paused); 404=从未存在。
        """
        store = app.state.async_memory.store
        allowed, reason = await async_check_memory_access_permissions(store, memory_id, app_id)
        if not allowed:
            history = await app.state.async_memory.get_history(memory_id)
            if "not found" in reason and not history:
                raise HTTPException(status_code=404, detail=f"memory {memory_id} not found")
            raise HTTPException(status_code=403, detail=reason)
        await async_record_access(store, memory_id, app_id, "get")
        result = await app.state.async_memory.get(memory_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"memory {memory_id} not found")
        return result

    @app.put("/memories/{memory_id}")
    async def update_memory(memory_id: str, req: UpdateMemoryRequest) -> dict[str, Any]:
        """更新记忆 (对齐 mem0 PUT /memories/{id})。"""
        text = req.text or ""
        success = await app.state.async_memory.update(memory_id, text, metadata=req.metadata)
        if not success:
            raise HTTPException(status_code=404, detail=f"memory {memory_id} not found")
        return {"event": "UPDATE", "memory_id": memory_id, "memory": text}

    @app.get("/memories/{memory_id}/history")
    async def get_history(memory_id: str) -> list[dict[str, Any]]:
        """获取记忆变更历史 (对齐 mem0 GET /memories/{id}/history)。"""
        return await app.state.async_memory.get_history(memory_id)

    @app.get("/memories/{memory_id}/access-logs")
    async def get_access_logs(
        memory_id: str,
        limit: int = Query(default=100, ge=1, description="返回日志数上限"),
    ) -> dict[str, Any]:
        """获取记忆访问日志 (审计用, 架构 §11.3)。"""
        store = app.state.async_memory.store
        logs = await store.get_access_logs(memory_id, limit)
        return {"logs": logs}

    @app.delete("/memories/{memory_id}")
    async def delete_memory(memory_id: str, app_id: str | None = None) -> dict[str, str]:
        """删除记忆 (软删除)。

        权限层: 校验存在性 + state=active 后才允许删除。
        403=存在但非 active; 404=从未存在。
        """
        store = app.state.async_memory.store
        allowed, reason = await async_check_memory_access_permissions(store, memory_id, app_id)
        if not allowed:
            history = await app.state.async_memory.get_history(memory_id)
            if "not found" in reason and not history:
                raise HTTPException(status_code=404, detail=f"memory {memory_id} not found")
            raise HTTPException(status_code=403, detail=reason)
        await async_record_access(store, memory_id, app_id, "delete")
        await app.state.async_memory.delete(memory_id)
        return {"event": "DELETE", "memory_id": memory_id}

    @app.post("/memories/{memory_id}/invalidate")
    async def invalidate_memory(memory_id: str, req: InvalidateRequest) -> dict[str, Any]:
        """手动标记事实不再为真 (设置 invalid_at + expired_at, 不删除记忆)。"""
        return await app.state.async_memory.invalidate(memory_id, invalid_at=req.invalid_at)

    @app.post("/memories/search")
    async def search_memories(req: SearchRequest) -> dict[str, Any]:
        """统一检索 (元认知路由)。"""
        if req.reranker or req.explain:
            # 高级检索（reranker/explain），回退到 sync ExperimentalMemory
            results = app.state.memory.search(
                req.query,
                user_id=req.user_id,
                session_id=req.session_id,
                top_k=req.top_k,
                threshold=req.threshold,
                reranker=req.reranker,
                explain=req.explain,
            )
        else:
            # 基础检索，用 async
            results = await app.state.async_memory.search(
                req.query,
                user_id=req.user_id,
                session_id=req.session_id,
                top_k=req.top_k,
                threshold=req.threshold,
                filters=req.filters,
            )
        return {"results": results}

    @app.post("/memories/search/causal")
    async def causal_search(req: CausalRequest) -> dict[str, Any]:
        """反事实因果查询 (架构文档 §6.1)。"""
        return app.state.memory.counterfactual(req.cause_event_id, req.effect_event_id, user_id=req.user_id)

    @app.get("/memories/meta/coverage")
    async def coverage_report(
        user_id: str = Query(..., description="用户 ID"),
    ) -> dict[str, Any]:
        """元认知覆盖报告 (架构文档 §6.3 L1)。"""
        return app.state.memory.coverage_report(user_id=user_id)

    @app.post("/memories/rehearse")
    async def rehearse(req: RehearseRequest) -> dict[str, Any]:
        """主动复述 (架构文档 §6.2)。"""
        if req.memory_id:
            return app.state.memory.rehearse(req.memory_id, user_id=req.user_id)
        candidates = app.state.memory.find_rehearse_candidates(user_id=req.user_id)
        for c in candidates:
            app.state.memory.rehearse(c["memory_id"], user_id=req.user_id)
        return {"rehearsed": len(candidates)}

    @app.post("/memories/capture")
    async def capture(req: CaptureRequest) -> dict[str, Any]:
        """PostToolUse 捕获 (架构文档 §5.1)。"""
        return app.state.memory.capture(req.text, user_id=req.user_id, agent_id=req.agent_id)

    @app.get("/agents/{user_id}/memories")
    async def get_shared_memories(user_id: str) -> dict[str, Any]:
        """跨 agent 共享读 (架构文档 §5.5)。"""
        agents = app.state.memory.list_agents(user_id)
        return {"user_id": user_id, "agents": agents, "is_cross_agent": app.state.memory.is_cross_agent(user_id)}

    @app.get("/entities")
    async def search_entities(
        query: str = Query(..., description="搜索查询"),
        user_id: str = Query(default="default", description="用户 ID"),
        top_k: int = Query(default=5, ge=1, description="返回数"),
    ) -> dict[str, Any]:
        """搜索实体 (Task 8)。"""
        results = app.state.memory.search_entities(query, user_id=user_id, top_k=top_k)
        return {"results": results}

    @app.get("/entities/list")
    async def list_entities(
        user_id: str = Query(default="default", description="用户 ID"),
        entity_type: str | None = Query(default=None, description="实体类型过滤"),
        limit: int = Query(default=100, ge=1, description="返回数"),
    ) -> dict[str, Any]:
        """列出实体 (Task 8)。"""
        results = app.state.memory.list_entities(user_id=user_id, entity_type=entity_type, limit=limit)
        return {"results": results}

    @app.get("/health")
    async def health() -> dict[str, str]:
        """健康检查。"""
        return {"status": "ok"}


def create_app(memory=None) -> FastAPI:
    """创建 FastAPI app (可注入 Memory 实例或 MemoryConfig 便于测试)。

    用法:
        app = create_app()
        # uvicorn septmuse.api.rest:app

    测试:
        app = create_app(MemoryConfig(db_path=str(tmp_path / "rest.db")))
        # TestClient(app).post("/memories", json={...})
    """
    if memory is None:
        config = MemoryConfig()
        sync_memory = ExperimentalMemory(config=config, embedder=HashEmbedder())
        async_memory = AsyncMemory(config=config, embedder=HashEmbedder())
    elif isinstance(memory, MemoryConfig):
        sync_memory = ExperimentalMemory(config=memory, embedder=HashEmbedder())
        async_memory = AsyncMemory(config=memory, embedder=HashEmbedder())
    else:
        # 已有 Memory 实例注入（向后兼容）
        sync_memory = memory
        async_memory = AsyncMemory(config=memory.config, embedder=memory.embedder)

    app = FastAPI(
        title="SeptMuse Memory API",
        description="Agent 记忆系统 REST API (架构文档 §11.2)",
        version="0.1.0",
    )
    from septmuse.api.auth import setup_auth

    setup_auth(app)
    register_routes(app, sync_memory, async_memory)
    return app


# 默认 app (uvicorn septmuse.api.rest:app)
app = create_app()

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
- PUT    /memories/{memory_id}  # 更新记忆
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

import http
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator

from septmuse import MemoryConfig
from septmuse.embedders.hash import HashEmbedder
from septmuse.experimental import ExperimentalMemory
from septmuse.governance.access import async_check_memory_access_permissions
from septmuse.governance.audit import async_record_access
from septmuse.memory.async_main import AsyncMemory

# ======================================================================
# Pydantic 请求模型
# ======================================================================


class AddMemoryRequest(BaseModel):
    content: str = Field(description="记忆内容", examples=["用户偏好用 Python 写后端"])
    user_id: str = Field(description="用户 ID", examples=["user-001"])
    agent_id: str | None = Field(default=None, description="agent ID", examples=["agent-1"])
    session_id: str | None = Field(default=None, description="会话 ID", examples=["session-1"])
    memory_type: str = Field(
        default="verbatim",
        description="verbatim|semantic|episodic|procedural",
        examples=["verbatim"],
    )
    infer: bool | None = Field(default=None, description="LLM 抽取", examples=[False])
    valid_at: str | None = Field(
        default=None, description="事实开始为真的时间 (ISO 8601)", examples=["2024-01-01T00:00:00"]
    )

    @model_validator(mode="before")
    @classmethod
    def _accept_messages_alias(cls, data):
        if isinstance(data, dict) and "messages" in data and "content" not in data:
            data["content"] = data["messages"]
        return data


class SearchRequest(BaseModel):
    query: str = Field(description="查询文本", examples=["Python 后端框架"])
    user_id: str = Field(description="用户 ID", examples=["user-001"])
    session_id: str | None = Field(default=None, description="会话 ID", examples=["session-1"])
    filters: dict[str, Any] | None = Field(
        default=None,
        description="字段过滤字典",
        examples=[{"agent_id": "agent-1", "session_id": "session-1"}],
    )
    top_k: int = Field(default=5, description="返回数", examples=[5])
    threshold: float = Field(default=0.1, description="相似阈值", examples=[0.1])
    reranker: str | None = Field(
        default=None,
        description="reranker: noop/mmr/cross_encoder/llm",
        examples=["mmr"],
    )
    explain: bool = Field(default=False, description="返回 score_details", examples=[False])


class CausalRequest(BaseModel):
    cause_event_id: str = Field(description="原因事件 ID", examples=["evt-001"])
    effect_event_id: str = Field(description="结果事件 ID", examples=["evt-002"])
    user_id: str = Field(description="用户 ID", examples=["user-001"])


class RehearseRequest(BaseModel):
    memory_id: str | None = Field(
        default=None, description="指定记忆 ID (None=批量)", examples=["mem-001"]
    )
    user_id: str = Field(description="用户 ID", examples=["user-001"])


class CaptureRequest(BaseModel):
    text: str = Field(description="捕获文本", examples=["工具调用结果：查询返回 3 条记录"])
    user_id: str = Field(description="用户 ID", examples=["user-001"])
    agent_id: str | None = Field(default=None, description="agent ID", examples=["agent-1"])
    session_id: str | None = Field(default=None, description="会话 ID", examples=["session-1"])


class UpdateMemoryRequest(BaseModel):
    text: str | None = Field(default=None, description="新内容", examples=["更新后的记忆内容"])
    metadata: dict[str, Any] | None = Field(
        default=None, description="新 metadata", examples=[{"source": "user-edit"}]
    )


class BlockUpdateRequest(BaseModel):
    value: str = Field(description="新 block 内容", examples=["用户偏好：深色模式"])


class BlockAppendRequest(BaseModel):
    content: str = Field(description="追加内容", examples=["；夜间模式"])


class BlockReplaceRequest(BaseModel):
    old_content: str = Field(description="被替换的旧内容", examples=["深色模式"])
    new_content: str = Field(description="新内容", examples=["深色+夜间模式"])


class InvalidateRequest(BaseModel):
    invalid_at: str | None = Field(
        default=None, description="失效时间 (ISO 8601), 默认当前时间", examples=["2024-12-31T23:59:59"]
    )


# ======================================================================
# 响应模型 + 错误响应 (RFC 7807 Problem Details)
# ======================================================================


class ErrorResponse(BaseModel):
    """RFC 7807 Problem Details 错误响应。"""

    type: str = Field(default="about:blank", description="错误类型 URI")
    title: str = Field(description="错误标题 (HTTP 状态短语)")
    status: int = Field(description="HTTP 状态码")
    detail: str = Field(description="错误详情")
    instance: str | None = Field(default=None, description="请求路径")


class DeleteResponse(BaseModel):
    event: str = Field(default="DELETE", description="事件类型")
    memory_id: str = Field(description="被删除的记忆 ID")


class UpdateResponse(BaseModel):
    event: str = Field(default="UPDATE", description="事件类型")
    memory_id: str = Field(description="记忆 ID")
    memory: str = Field(description="更新后的内容")


class HealthResponse(BaseModel):
    status: str = Field(description="服务状态")


class PaginatedResponse(BaseModel):
    results: list[dict[str, Any]] = Field(description="记忆列表")
    total: int = Field(description="总数")
    limit: int = Field(description="每页上限")
    offset: int = Field(description="偏移量")


# 通用错误响应 schema (403/404)
ERROR_RESPONSES = {
    403: {"model": ErrorResponse, "description": "授权失败 — 记忆 state 不允许访问"},
    404: {"model": ErrorResponse, "description": "记忆不存在"},
}


# ======================================================================
# App 工厂
# ======================================================================


def register_routes(app: FastAPI, memory: ExperimentalMemory, async_memory: AsyncMemory | None = None) -> None:
    """注册 REST 路由到已有 FastAPI app。

    供 CLI serve --with-rest 使用 (同一 app 同时挂 MCP + REST)。
    """
    app.state.memory = memory
    app.state.async_memory = async_memory or memory

    @app.post("/memories", status_code=201, tags=["memories"])
    async def add_memory(req: AddMemoryRequest) -> dict[str, Any]:
        """添加记忆 (架构文档 §11.2)。

        支持 verbatim（原文）/ semantic（事实三元组）/ episodic（情节）/ procedural（规则）四类。
        verbatim 模式可配合 `infer=true` 走 LLM 抽取事实。
        """
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

    @app.get("/memories", tags=["memories"], response_model=PaginatedResponse)
    async def list_memories(
        user_id: str = Query(..., description="用户 ID"),
        session_id: str | None = Query(default=None, description="会话 ID"),
        limit: int = Query(default=100, ge=1, description="每页上限"),
        offset: int = Query(default=0, ge=0, description="偏移量"),
        app_id: str | None = None,
    ) -> dict[str, Any]:
        """列出记忆 (分页)。

        按 `user_id` 过滤，可选 `session_id` 二次过滤。
        `limit`/`offset` 分页，返回 `total` 为过滤后总数。
        """
        results = await app.state.async_memory.get_all(user_id=user_id, session_id=session_id)
        total = len(results)
        paged = results[offset : offset + limit]
        store = app.state.async_memory.store
        for r in paged:
            await async_record_access(store, r["id"], app_id, "list")
        return {"results": paged, "total": total, "limit": limit, "offset": offset}

    @app.get("/memories/working/blocks/{agent_id}", tags=["working-memory"])
    async def list_blocks(agent_id: str) -> list[dict[str, Any]]:
        """列出 agent 的全部 block。"""
        return app.state.memory.get_blocks(agent_id)

    @app.put("/memories/working/blocks/{agent_id}/{label}", tags=["working-memory"])
    async def update_block(agent_id: str, label: str, req: BlockUpdateRequest) -> dict[str, Any]:
        """更新 block value (架构 §11.2)。"""
        return app.state.memory.update_block(agent_id, label, req.value)

    @app.post("/memories/working/blocks/{agent_id}/{label}/append", tags=["working-memory"])
    async def append_block(agent_id: str, label: str, req: BlockAppendRequest) -> dict[str, Any]:
        """追加 block 内容。"""
        return app.state.memory.core_memory_append(agent_id, label, req.content)

    @app.post("/memories/working/blocks/{agent_id}/{label}/replace", tags=["working-memory"])
    async def replace_block(agent_id: str, label: str, req: BlockReplaceRequest) -> dict[str, Any]:
        """替换 block 内容片段。"""
        return app.state.memory.core_memory_replace(agent_id, label, req.old_content, req.new_content)

    @app.get("/memories/{memory_id}", tags=["memories"], responses={**ERROR_RESPONSES})
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

    @app.put("/memories/{memory_id}", tags=["memories"], response_model=UpdateResponse, responses={**ERROR_RESPONSES})
    async def update_memory(memory_id: str, req: UpdateMemoryRequest) -> dict[str, Any]:
        """更新记忆内容或 metadata。"""
        text = req.text or ""
        success = await app.state.async_memory.update(memory_id, text, metadata=req.metadata)
        if not success:
            raise HTTPException(status_code=404, detail=f"memory {memory_id} not found")
        return {"event": "UPDATE", "memory_id": memory_id, "memory": text}

    @app.get("/memories/{memory_id}/history", tags=["audit"])
    async def get_history(memory_id: str) -> list[dict[str, Any]]:
        """获取记忆变更历史。"""
        return await app.state.async_memory.get_history(memory_id)

    @app.get("/memories/{memory_id}/access-logs", tags=["audit"])
    async def get_access_logs(
        memory_id: str,
        limit: int = Query(default=100, ge=1, description="返回日志数上限"),
    ) -> dict[str, Any]:
        """获取记忆访问日志 (审计用, 架构 §11.3)。"""
        store = app.state.async_memory.store
        logs = await store.get_access_logs(memory_id, limit)
        return {"logs": logs}

    @app.delete("/memories/{memory_id}", tags=["memories"], response_model=DeleteResponse, responses={**ERROR_RESPONSES})
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

    @app.post("/memories/{memory_id}/invalidate", tags=["memories"])
    async def invalidate_memory(memory_id: str, req: InvalidateRequest) -> dict[str, Any]:
        """手动标记事实不再为真 (设置 invalid_at + expired_at, 不删除记忆)。

        用于双时态场景：事实曾经为真，但现已失效。
        """
        return await app.state.async_memory.invalidate(memory_id, invalid_at=req.invalid_at)

    @app.post("/memories/search", tags=["search"])
    async def search_memories(req: SearchRequest) -> dict[str, Any]:
        """统一检索 (元认知路由)。

        向量 + 关键词 + 实体图三路融合，支持:
        - `filters`: 字段过滤字典
        - `reranker`: noop / mmr / cross_encoder / llm
        - `explain`: 返回 score_details 评分详情
        """
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

    @app.post("/memories/search/causal", tags=["search"])
    async def causal_search(req: CausalRequest) -> dict[str, Any]:
        """反事实因果查询 (架构文档 §6.1)。"""
        return app.state.memory.counterfactual(req.cause_event_id, req.effect_event_id, user_id=req.user_id)

    @app.get("/memories/meta/coverage", tags=["metacognition"])
    async def coverage_report(
        user_id: str = Query(..., description="用户 ID"),
    ) -> dict[str, Any]:
        """元认知覆盖报告 (架构文档 §6.3 L1)。"""
        return app.state.memory.coverage_report(user_id=user_id)

    @app.post("/memories/rehearse", tags=["metacognition"])
    async def rehearse(req: RehearseRequest) -> dict[str, Any]:
        """主动复述 (架构文档 §6.2)。"""
        if req.memory_id:
            return app.state.memory.rehearse(req.memory_id, user_id=req.user_id)
        candidates = app.state.memory.find_rehearse_candidates(user_id=req.user_id)
        for c in candidates:
            app.state.memory.rehearse(c["memory_id"], user_id=req.user_id)
        return {"rehearsed": len(candidates)}

    @app.post("/memories/capture", tags=["capture"])
    async def capture(req: CaptureRequest) -> dict[str, Any]:
        """PostToolUse 捕获 (架构文档 §5.1)。"""
        return app.state.memory.capture(req.text, user_id=req.user_id, agent_id=req.agent_id)

    @app.get("/agents/{user_id}/memories", tags=["sharing"])
    async def get_shared_memories(user_id: str) -> dict[str, Any]:
        """跨 agent 共享读 (架构文档 §5.5)。"""
        agents = app.state.memory.list_agents(user_id)
        return {"user_id": user_id, "agents": agents, "is_cross_agent": app.state.memory.is_cross_agent(user_id)}

    @app.get("/entities", tags=["entities"])
    async def search_entities(
        query: str = Query(..., description="搜索查询"),
        user_id: str = Query(default="default", description="用户 ID"),
        top_k: int = Query(default=5, ge=1, description="返回数"),
    ) -> dict[str, Any]:
        """搜索实体 (Task 8)。"""
        results = app.state.memory.search_entities(query, user_id=user_id, top_k=top_k)
        return {"results": results}

    @app.get("/entities/list", tags=["entities"])
    async def list_entities(
        user_id: str = Query(default="default", description="用户 ID"),
        entity_type: str | None = Query(default=None, description="实体类型过滤"),
        limit: int = Query(default=100, ge=1, description="返回数"),
    ) -> dict[str, Any]:
        """列出实体 (Task 8)。"""
        results = app.state.memory.list_entities(user_id=user_id, entity_type=entity_type, limit=limit)
        return {"results": results}

    @app.get("/health", tags=["health"], response_model=HealthResponse)
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

    import os

    enable_docs = os.getenv("SEPTMUSE_ENABLE_DOCS", "true").lower() in ("1", "true", "yes")
    docs_url = "/docs" if enable_docs else None
    redoc_url = "/redoc" if enable_docs else None
    openapi_url = "/openapi.json" if enable_docs else None

    app = FastAPI(
        title="SeptMuse Memory API",
        description="""
## SeptMuse Memory API

Agent 记忆系统 REST API — 三维正交架构（内容类型 × 存储形态 × 横切关注点）。

### Features

- **记忆 CRUD**: 添加 / 列出 / 获取 / 更新 / 删除 / 失效记忆，支持 verbatim / semantic / episodic / procedural 四类
- **工作记忆**: Block 级读写（append / replace），agent 私有核心记忆区
- **统一检索**: 向量 + 关键词 + 实体图三路融合，支持 filters 过滤、reranker 重排、explain 评分详情
- **因果查询**: 反事实因果分析（架构 §6.1）
- **元认知**: 覆盖报告（§6.3 L1）+ 主动复述（§6.2）
- **审计**: 记忆变更历史 + 访问日志（§11.3）
- **共享**: 跨 agent 共享读（§5.5）
- **实体**: 实体搜索 / 列表（Task 8）
- **治理**: 4 层权限 + 软删除 + 双时态（valid_at / invalid_at）

### Architecture

- **Facade**: `Memory` / `AsyncMemory`（零配置入口）
- **存储**: SQLite 组合后端（vector + keyword + graph），可换 pgvector / chroma / qdrant / age / neo4j
- **嵌入**: HashEmbedder（默认离线）/ ONNX / sentence-transformers
- **LLM**: OpenAI / Anthropic / Ollama / DashScope（可选，verbatim 模式不需要）
- **权限**: `SEPTMUSE_API_KEY` 未设=开发模式（无认证）；已设=生产模式（401 未认证）

### Quick Start

    pip install septmuse
    python -m septmuse.cli.main serve
    # 访问 http://localhost:8000/docs

### Notes

- **score 统一为相似度 [0,1]**，越高越相似
- **state 状态机**: active / paused / archived / deleted
- **401 vs 403**: 401=认证（API key 缺失/错误）；403=授权（记忆 state 不允许访问）
        """,
        version="0.1.0",
        docs_url=docs_url,
        redoc_url=redoc_url,
        openapi_url=openapi_url,
        contact={
            "name": "SeptMuse",
            "url": "https://github.com/solarseptem-ai/SeptMuse",
        },
        license_info={
            "name": "Apache 2.0",
            "url": "https://www.apache.org/licenses/LICENSE-2.0",
        },
        openapi_tags=[
            {
                "name": "memories",
                "description": "记忆 CRUD — 添加 / 列出 / 获取 / 更新 / 删除 / 失效，支持 verbatim / semantic / episodic / procedural 四类内容",
            },
            {
                "name": "working-memory",
                "description": "工作记忆 Block 操作 — agent 私有核心记忆区，支持 append / replace 片段级编辑",
            },
            {
                "name": "search",
                "description": "统一检索 — 向量 + 关键词 + 实体图三路融合，支持 filters / reranker / explain / 反事实因果查询",
            },
            {
                "name": "metacognition",
                "description": "元认知 — 覆盖报告 (§6.3 L1) + 主动复述 (§6.2)",
            },
            {
                "name": "capture",
                "description": "PostToolUse 捕获 — 自动从工具调用结果抽取记忆 (§5.1)",
            },
            {
                "name": "audit",
                "description": "审计 — 记忆变更历史 + 访问日志 (§11.3)",
            },
            {
                "name": "sharing",
                "description": "跨 agent 共享读 (§5.5)",
            },
            {
                "name": "entities",
                "description": "实体搜索 / 列表 (Task 8)",
            },
            {
                "name": "health",
                "description": "健康检查 + 系统状态",
            },
        ],
    )
    from septmuse.api.auth import setup_auth

    setup_auth(app)

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        """RFC 7807 Problem Details 错误响应。"""
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                title=http.HTTPStatus(exc.status_code).phrase,
                status=exc.status_code,
                detail=str(exc.detail),
                instance=str(request.url.path),
            ).model_dump(),
        )

    register_routes(app, sync_memory, async_memory)
    return app


# 默认 app (uvicorn septmuse.api.rest:app)
app = create_app()

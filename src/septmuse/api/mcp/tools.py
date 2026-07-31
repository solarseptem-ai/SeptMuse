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
"""MCP 工具集 — @mcp.tool 注册 9 个记忆工具。

源码参考 mem0 mcp_server.py 的 @mcp.tool 模式:
- 基础 5 工具 (对齐 mem0): add_memories / search_memory / list_memories / delete_memories / delete_all_memories
- SeptMuse 扩展 4 工具 (架构文档 §13.4, 阶段1 先 stub, 后续阶段实现):
  remember_episode / causal_query / rehearse / coverage_report

user_id 解析: 优先显式参数, 缺省回退 contextvar (兼容 stdio/http)

注: 不用 `from __future__ import annotations`, 因 FastMCP func_metadata 把返回注解
传给 pydantic create_model, 字符串化注解会导致 PydanticUserError (non-annotated)。
源码参考 mem0 mcp_server.py 同样不用 future annotations。
"""

import json

from septmuse.api.mcp.context import user_id_var
from septmuse.api.mcp.server import get_memory_safe, mcp
from septmuse.core.logging import get_logger

logger = get_logger(__name__)


def _resolve_user_id(explicit: str):
    """解析 user_id: 显式参数优先, 回退 contextvar。"""
    return explicit or user_id_var.get()


# ---------------------------------------------------------------------------
# 基础 5 工具 (对齐 mem0 mcp_server.py)
# ---------------------------------------------------------------------------


@mcp.tool(
    description="添加记忆。用户告知任何偏好/事实时调用。infer=False 存原文不抽取 (默认), infer=True 需 LLM 抽取事实。session_id 可选, 用于会话隔离 (对齐 mem0 run_id)。"
)
async def add_memories(content: str, user_id: str = "", infer: bool = False, session_id: str = ""):
    """添加记忆 (对齐 mem0 add_memories, 参数名 content 与 core_memory_append 一致)。

    session_id: 会话 ID (对齐 mem0 run_id; 空字符串=不限)。
    """
    uid = _resolve_user_id(user_id)
    if not uid:
        return "Error: user_id not provided (传 user_id 参数或设 SEPTMUSE_USER_ID 环境变量)"

    mem = get_memory_safe()
    if not mem:
        return "Error: Memory system is currently unavailable. Please try again later."

    try:
        response = mem.add(
            content,
            user_id=uid,
            session_id=session_id or None,
            metadata={"source": "mcp"},
            infer=infer,
        )
        return json.dumps(response, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.exception("mcp_add_error")
        return f"Error adding to memory: {e}"


@mcp.tool(description="搜索记忆。每次用户提问时调用, 召回相关记忆。session_id 可选, 用于会话隔离 (对齐 mem0 run_id)。")
async def search_memory(query: str, user_id: str = "", top_k: int = 5, session_id: str = ""):
    """搜索记忆 (对齐 mem0 search_memory)。

    session_id: 仅搜该会话的记忆 (空字符串=不限, 对齐 mem0 run_id)。
    """
    uid = _resolve_user_id(user_id)
    if not uid:
        return "Error: user_id not provided"

    mem = get_memory_safe()
    if not mem:
        return "Error: Memory system is currently unavailable."

    try:
        results = mem.search(query, user_id=uid, session_id=session_id or None, top_k=top_k)
        return json.dumps({"results": results}, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.exception("mcp_search_error")
        return f"Error searching memory: {e}"


@mcp.tool(description="列出用户全部记忆。")
async def list_memories(user_id: str = ""):
    """列出全部记忆 (对齐 mem0 list_memories)。"""
    uid = _resolve_user_id(user_id)
    if not uid:
        return "Error: user_id not provided"

    mem = get_memory_safe()
    if not mem:
        return "Error: Memory system is currently unavailable."

    try:
        return json.dumps(mem.get_all(user_id=uid), ensure_ascii=False, indent=2)
    except Exception as e:
        logger.exception("mcp_list_error")
        return f"Error getting memories: {e}"


@mcp.tool(description="按 ID 删除指定记忆 (软删除)。")
async def delete_memories(memory_ids: list[str], user_id: str = ""):
    """按 ID 删除 (对齐 mem0 delete_memories)。"""
    uid = _resolve_user_id(user_id)
    if not uid:
        return "Error: user_id not provided"

    mem = get_memory_safe()
    if not mem:
        return "Error: Memory system is currently unavailable."

    deleted = 0
    for mid in memory_ids:
        try:
            mem.delete(mid)
            deleted += 1
        except Exception as e:
            logger.warning("mcp_delete_failed", memory_id=mid, error=str(e))
    return f"Successfully deleted {deleted}/{len(memory_ids)} memories"


@mcp.tool(description="删除用户全部记忆。")
async def delete_all_memories(user_id: str = ""):
    """删除全部 (对齐 mem0 delete_all_memories)。"""
    uid = _resolve_user_id(user_id)
    if not uid:
        return "Error: user_id not provided"

    mem = get_memory_safe()
    if not mem:
        return "Error: Memory system is currently unavailable."

    try:
        allm = mem.get_all(user_id=uid)
        count = 0
        for item in allm.get("results", []):
            mem.delete(item["id"])
            count += 1
        return f"Successfully deleted all {count} memories"
    except Exception as e:
        logger.exception("mcp_delete_all_error")
        return f"Error deleting all memories: {e}"


# ---------------------------------------------------------------------------
# SeptMuse 扩展 4 工具 (架构文档 §13.4, 阶段1 stub, 后续阶段实现)
# ---------------------------------------------------------------------------


@mcp.tool(description="记录成功交互的推理经验 (观察/思考/行动/结果)。借鉴 LangMem Episode。")
async def remember_episode(observation: str, thoughts: str, action: str, outcome: str, user_id: str = ""):
    """记录推理经验 (架构文档 §3.2.1)。

    注: 参数名用 outcome 而非 result, 因 FastMCP 把返回值包装为 pydantic model 的
    `result` 字段, 参数名 result 会与之冲突 (pydantic PydanticUserError)。
    """
    uid = _resolve_user_id(user_id)
    if not uid:
        return "Error: user_id not provided"
    mem = get_memory_safe()
    if not mem:
        return "Error: Memory system is currently unavailable."
    try:
        result = mem.add_episode(
            f"obs: {observation}; act: {action}; result: {outcome}",
            user_id=uid,
            event_type="reasoning",
            observation=observation,
            thoughts=thoughts,
            action=action,
            result=outcome,
        )
        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception as e:
        return f"Error recording episode: {e}"


@mcp.tool(description="反事实因果查询: 若某事件未发生,结果是否仍成立 (架构文档 §6.1)")
async def causal_query(cause_event_id: str, hypothesized_effect: str, user_id: str = ""):
    """因果查询 (架构文档 §6.1)。"""
    uid = _resolve_user_id(user_id)
    if not uid:
        return "Error: user_id not provided"
    mem = get_memory_safe()
    if not mem:
        return "Error: Memory system is currently unavailable."
    try:
        result = mem.counterfactual(cause_event_id, hypothesized_effect, user_id=uid)
        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception as e:
        return f"Error in causal query: {e}"


@mcp.tool(description="触发主动复述强化低强度高价值记忆 (架构文档 §6.2)")
async def rehearse(user_id: str = "", memory_id: str = ""):
    """主动复述 (架构文档 §6.2)。memory_id 为空时批量复述候选。"""
    uid = _resolve_user_id(user_id)
    if not uid:
        return "Error: user_id not provided"
    mem = get_memory_safe()
    if not mem:
        return "Error: Memory system is currently unavailable."
    try:
        if memory_id:
            result = mem.rehearse(memory_id, user_id=uid)
            return json.dumps(result, ensure_ascii=False, default=str)
        candidates = mem.find_rehearse_candidates(user_id=uid)
        count = 0
        for c in candidates:
            mem.rehearse(c["memory_id"], user_id=uid)
            count += 1
        return json.dumps({"rehearsed": count, "candidates": len(candidates)}, ensure_ascii=False, default=str)
    except Exception as e:
        return f"Error in rehearse: {e}"


@mcp.tool(description="生成元认知覆盖报告: agent 记住了什么/记不住什么 (架构文档 §6.3)")
async def coverage_report(user_id: str = ""):
    """覆盖报告 (架构文档 §6.3)。"""
    uid = _resolve_user_id(user_id)
    if not uid:
        return "Error: user_id not provided"
    mem = get_memory_safe()
    if not mem:
        return "Error: Memory system is currently unavailable."
    try:
        result = mem.coverage_report(user_id=uid)
        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception as e:
        return f"Error generating coverage report: {e}"


# ---------------------------------------------------------------------------
# SeptMuse 扩展 6 工具 (update + block + history, 对齐 mem0 plugin 9 工具)
# ---------------------------------------------------------------------------


@mcp.tool(description="更新已有记忆的内容或 metadata。")
async def update_memory(memory_id: str, content: str = "", user_id: str = "", metadata: str = ""):
    """更新记忆 (对齐 mem0 update_memory)。metadata 传 JSON 字符串或空。"""
    uid = _resolve_user_id(user_id)
    if not uid:
        return "Error: user_id not provided"
    mem = get_memory_safe()
    if not mem:
        return "Error: Memory system is currently unavailable."
    try:
        meta = json.loads(metadata) if metadata else None
        result = mem.update(memory_id, content or None, user_id=uid, metadata=meta)
        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception as e:
        return f"Error updating memory: {e}"


@mcp.tool(description="更新工作记忆 Block 的值 (对齐 Letta update_block_value)。")
async def update_block(agent_id: str, label: str, value: str, user_id: str = ""):
    """更新 block value。"""
    mem = get_memory_safe()
    if not mem:
        return "Error: Memory system is currently unavailable."
    try:
        result = mem.update_block(agent_id, label, value)
        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception as e:
        return f"Error updating block: {e}"


@mcp.tool(description="追加内容到工作记忆 Block (对齐 Letta core_memory_append)。")
async def core_memory_append(agent_id: str, label: str, content: str, user_id: str = ""):
    """追加 block 内容。"""
    mem = get_memory_safe()
    if not mem:
        return "Error: Memory system is currently unavailable."
    try:
        result = mem.core_memory_append(agent_id, label, content)
        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception as e:
        return f"Error appending to block: {e}"


@mcp.tool(description="替换工作记忆 Block 中的内容片段 (对齐 Letta core_memory_replace)。")
async def core_memory_replace(agent_id: str, label: str, old_content: str, new_content: str, user_id: str = ""):
    """替换 block 内容片段。"""
    mem = get_memory_safe()
    if not mem:
        return "Error: Memory system is currently unavailable."
    try:
        result = mem.core_memory_replace(agent_id, label, old_content, new_content)
        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception as e:
        return f"Error replacing block content: {e}"


@mcp.tool(description="列出 agent 的工作记忆 Block 列表。")
async def get_blocks(agent_id: str, user_id: str = ""):
    """列出 block。"""
    mem = get_memory_safe()
    if not mem:
        return "Error: Memory system is currently unavailable."
    try:
        blocks = mem.get_blocks(agent_id)
        return json.dumps(blocks, ensure_ascii=False, default=str)
    except Exception as e:
        return f"Error getting blocks: {e}"


@mcp.tool(description="查看记忆的变更历史 (ADD/UPDATE/DELETE 记录)。")
async def get_memory_history(memory_id: str, user_id: str = ""):
    """查看记忆历史。"""
    mem = get_memory_safe()
    if not mem:
        return "Error: Memory system is currently unavailable."
    try:
        history = mem.get_history(memory_id)
        return json.dumps(history, ensure_ascii=False, default=str)
    except Exception as e:
        return f"Error getting history: {e}"


@mcp.tool(description="标记记忆不再为真 (双时态: 设置 invalid_at + expired_at, 不删除记忆)。")
async def invalidate_memory(memory_id: str, user_id: str = "", invalid_at: str = ""):
    """标记事实失效 (双时态建模, 借鉴 graphiti EntityEdge)。"""
    mem = get_memory_safe()
    if not mem:
        return "Error: Memory system is currently unavailable."
    try:
        result = mem.invalidate(memory_id, invalid_at=invalid_at or None)
        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception as e:
        return f"Error invalidating memory: {e}"


@mcp.tool(description="搜索实体 (精确匹配 + 向量相似度)。")
async def search_entities(query: str, user_id: str = "", top_k: int = 5):
    """搜索实体。"""
    mem = get_memory_safe()
    if not mem:
        return "Error: Memory system is currently unavailable."
    try:
        if not user_id:
            return "Error: user_id not provided."
        results = mem.search_entities(query, user_id=user_id, top_k=top_k)
        return json.dumps(results, ensure_ascii=False, default=str)
    except Exception as e:
        return f"Error searching entities: {e}"


@mcp.tool(description="列出用户全部未删除实体。")
async def list_entities(user_id: str = "", entity_type: str = "", limit: int = 100):
    """列出实体。"""
    mem = get_memory_safe()
    if not mem:
        return "Error: Memory system is currently unavailable."
    try:
        if not user_id:
            return "Error: user_id not provided."
        results = mem.list_entities(user_id=user_id, entity_type=entity_type or None, limit=limit)
        return json.dumps(results, ensure_ascii=False, default=str)
    except Exception as e:
        return f"Error listing entities: {e}"


logger.info("mcp_tools_registered", count=18)

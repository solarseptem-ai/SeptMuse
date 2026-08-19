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
"""MCP 工具集 — @mcp.tool 注册 22 个记忆工具。"""

import asyncio
import json

from septmuse.api.mcp.context import user_id_var
from septmuse.api.mcp.server import get_memory_safe, mcp
from septmuse.core.logging import get_logger

logger = get_logger(__name__)


def _resolve_user_id(explicit: str) -> str:
    """解析 user_id: 显式参数优先, 回退 contextvar。"""
    return explicit or user_id_var.get()


# ---------------------------------------------------------------------------
# MCP 工具执行模板 — 消除每个工具的重复样板
# ---------------------------------------------------------------------------


async def _mcp_run(func_name: str, user_id: str, func):
    """通用 MCP 工具执行模板。"""
    uid = _resolve_user_id(user_id)
    if not uid:
        return "Error: user_id not provided"
    mem = get_memory_safe()
    if not mem:
        return "Error: Memory system is currently unavailable. Please try again later."
    try:
        result = await asyncio.to_thread(func, mem, uid)
        if isinstance(result, str):
            return result
        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception as e:
        logger.exception(f"mcp_{func_name}_error")
        return f"Error in {func_name}: {e}"


async def _mcp_run_simple(func_name: str, func):
    """简版 MCP 工具执行模板 (无 user_id 解析)。"""
    mem = get_memory_safe()
    if not mem:
        return "Error: Memory system is currently unavailable. Please try again later."
    try:
        result = await asyncio.to_thread(func, mem)
        if isinstance(result, str):
            return result
        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception as e:
        logger.exception(f"mcp_{func_name}_error")
        return f"Error in {func_name}: {e}"


# ---------------------------------------------------------------------------
# 基础 5 工具
# ---------------------------------------------------------------------------


@mcp.tool(
    description="添加记忆到长期存储。content=记忆文本, user_id=用户ID, infer=True时用LLM抽取事实(默认False存原文), session_id=会话隔离。用户告知任何偏好、事实或个人信息时调用此工具。"
)
async def add_memories(content: str, user_id: str = "", infer: bool = False, session_id: str = ""):
    """添加记忆 (参数名 content 与 core_memory_append 一致)。

    session_id: 会话 ID (空字符串=不限)。
    """
    return await _mcp_run(
        "add_memories",
        user_id,
        lambda mem, uid: mem.add(
            content,
            user_id=uid,
            session_id=session_id or None,
            metadata={"source": "mcp"},
            infer=infer,
        ),
    )


@mcp.tool(
    description="搜索记忆召回相关内容。query=查询文本, user_id=用户ID, top_k=返回数量(默认5), session_id=会话过滤, filters=字段过滤字典。用户提问时调用以召回长期记忆。"
)
async def search_memory(
    query: str, user_id: str = "", top_k: int = 5, session_id: str = "", filters: dict | None = None
):
    """搜索记忆。

    session_id: 仅搜该会话的记忆 (空字符串=不限)。
    filters: 字段过滤字典 (如 {"session_id":"s1", "agent_id":"a1"}), None=不过滤。
    """
    return await _mcp_run(
        "search_memory",
        user_id,
        lambda mem, uid: {"results": mem.search(query, user_id=uid, session_id=session_id or None, top_k=top_k, filters=filters)},
    )


@mcp.tool(
    description="列出用户全部记忆。user_id=用户ID(缺省读contextvar)。需要查看用户所有已存储记忆时调用,不做语义搜索。"
)
async def list_memories(user_id: str = ""):
    """列出全部记忆。"""
    return await _mcp_run("list_memories", user_id, lambda mem, uid: mem.get_all(user_id=uid))


@mcp.tool(
    description="按ID批量删除记忆(软删除,标记state=deleted)。memory_ids=记忆ID列表, user_id=用户ID。用户要求删除特定记忆时调用,记忆不会被物理删除。"
)
async def delete_memories(memory_ids: list[str], user_id: str = ""):
    """按 ID 删除。"""
    def _do(mem, uid):
        deleted = 0
        for mid in memory_ids:
            try:
                mem.delete(mid)
                deleted += 1
            except Exception as e:
                logger.warning("mcp_delete_failed", memory_id=mid, error=str(e))
        return f"Successfully deleted {deleted}/{len(memory_ids)} memories"

    return await _mcp_run("delete_memories", user_id, _do)


@mcp.tool(
    description="删除用户全部记忆(软删除)。user_id=用户ID。用户要求清空所有记忆时调用,危险操作不可恢复,建议先确认。"
)
async def delete_all_memories(user_id: str = ""):
    """删除全部。"""
    def _do(mem, uid):
        allm = mem.get_all(user_id=uid)
        count = 0
        for item in allm.get("results", []):
            mem.delete(item["id"])
            count += 1
        return f"Successfully deleted all {count} memories"

    return await _mcp_run("delete_all_memories", user_id, _do)


# ---------------------------------------------------------------------------
# SeptMuse 扩展 4 工具
# ---------------------------------------------------------------------------


@mcp.tool(
    description="记录成功交互的推理经验。observation=观察, thoughts=思考, action=行动, outcome=结果, user_id=用户ID。任务成功完成时调用以沉淀经验用于后续复用。"
)
async def remember_episode(observation: str, thoughts: str, action: str, outcome: str, user_id: str = ""):
    """记录推理经验。"""
    return await _mcp_run(
        "remember_episode",
        user_id,
        lambda mem, uid: mem.add_episode(
            f"obs: {observation}; act: {action}; result: {outcome}",
            user_id=uid,
            event_type="reasoning",
            observation=observation,
            thoughts=thoughts,
            action=action,
            result=outcome,
        ),
    )


@mcp.tool(
    description="反事实因果查询:若某事件未发生,结果是否仍成立。cause_event_id=原因事件ID, hypothesized_effect=假设结果, user_id=用户ID。需要分析因果关系或做假设推理时调用。"
)
async def causal_query(cause_event_id: str, hypothesized_effect: str, user_id: str = ""):
    """因果查询。"""
    return await _mcp_run(
        "causal_query",
        user_id,
        lambda mem, uid: mem.counterfactual(cause_event_id, hypothesized_effect, user_id=uid),
    )


@mcp.tool(
    description="触发主动复述强化低强度高价值记忆。user_id=用户ID, memory_id=指定记忆(空则批量复述候选)。记忆强度衰减时调用以回升强度防止遗忘。"
)
async def rehearse(user_id: str = "", memory_id: str = ""):
    """主动复述。"""
    def _do(mem, uid):
        if memory_id:
            return mem.rehearse(memory_id, user_id=uid)
        return mem.rehearse_batch(user_id=uid)

    return await _mcp_run("rehearse", user_id, _do)


@mcp.tool(
    description="生成元认知覆盖报告:agent记住了什么/记不住什么。user_id=用户ID。需要自省记忆覆盖情况、发现知识盲区时调用,返回强弱领域和覆盖分数。"
)
async def coverage_report(user_id: str = ""):
    """覆盖报告。"""
    return await _mcp_run("coverage_report", user_id, lambda mem, uid: mem.coverage_report(user_id=uid))


# ---------------------------------------------------------------------------
# SeptMuse 扩展 6 工具 (update + block + history)
# ---------------------------------------------------------------------------


@mcp.tool(
    description="更新已有记忆的内容或metadata。memory_id=记忆ID, content=新内容(空则不改), user_id=用户ID, metadata=JSON字符串(空则不改)。记忆内容需修正或补充时调用。"
)
async def update_memory(memory_id: str, content: str = "", user_id: str = "", metadata: str = ""):
    """更新记忆。metadata 传 JSON 字符串或空。"""
    def _do(mem, uid):
        meta = json.loads(metadata) if metadata else None
        return mem.update(memory_id, content or None, user_id=uid, metadata=meta)

    return await _mcp_run("update_memory", user_id, _do)


@mcp.tool(
    description="更新工作记忆Block的值(整体替换)。agent_id=agent标识, label=block标签, value=新值, user_id=用户ID。需要修改agent工作记忆整体内容时调用。"
)
async def update_block(agent_id: str, label: str, value: str, user_id: str = ""):
    """更新 block value。"""
    return await _mcp_run_simple("update_block", lambda mem: mem.update_block(agent_id, label, value))


@mcp.tool(
    description="追加内容到工作记忆Block末尾。agent_id=agent标识, label=block标签, content=追加文本, user_id=用户ID。需要向工作记忆增量补充信息时调用。"
)
async def core_memory_append(agent_id: str, label: str, content: str, user_id: str = ""):
    """追加 block 内容。"""
    return await _mcp_run_simple("core_memory_append", lambda mem: mem.core_memory_append(agent_id, label, content))


@mcp.tool(
    description="替换工作记忆Block中的内容片段。agent_id=agent标识, label=block标签, old_content=待替换片段, new_content=新片段, user_id=用户ID。需精确修改工作记忆中某段文字时调用。"
)
async def core_memory_replace(agent_id: str, label: str, old_content: str, new_content: str, user_id: str = ""):
    """替换 block 内容片段。"""
    return await _mcp_run_simple(
        "core_memory_replace",
        lambda mem: mem.core_memory_replace(agent_id, label, old_content, new_content),
    )


@mcp.tool(
    description="列出agent的工作记忆Block列表。agent_id=agent标识, user_id=用户ID。需要查看agent当前工作记忆全部block内容时调用。"
)
async def get_blocks(agent_id: str, user_id: str = ""):
    """列出 block。"""
    return await _mcp_run_simple("get_blocks", lambda mem: mem.get_blocks(agent_id))


@mcp.tool(
    description="查看记忆的变更历史(ADD/UPDATE/DELETE记录)。memory_id=记忆ID, user_id=用户ID。需要审计记忆修改轨迹或排查数据变更时调用。"
)
async def get_memory_history(memory_id: str, user_id: str = ""):
    """查看记忆历史。"""
    return await _mcp_run_simple("get_memory_history", lambda mem: mem.get_history(memory_id))


@mcp.tool(
    description="标记记忆不再为真(双时态:设置invalid_at+expired_at,不删除记忆)。memory_id=记忆ID, user_id=用户ID, invalid_at=失效时间。事实过期或被推翻时调用,保留历史轨迹。"
)
async def invalidate_memory(memory_id: str, user_id: str = "", invalid_at: str = ""):
    """标记事实失效 (双时态建模)。"""
    return await _mcp_run_simple(
        "invalidate_memory",
        lambda mem: mem.invalidate(memory_id, invalid_at=invalid_at or None),
    )


@mcp.tool(
    description="搜索实体(精确匹配+向量相似度)。query=查询文本, user_id=用户ID(必填), top_k=返回数量(默认5)。需要查找人物、地点、概念等实体时调用。"
)
async def search_entities(query: str, user_id: str = "", top_k: int = 5):
    """搜索实体。"""
    def _do(mem, uid):
        if not uid:
            return "Error: user_id not provided."
        return mem.search_entities(query, user_id=uid, top_k=top_k)

    return await _mcp_run("search_entities", user_id, _do)


@mcp.tool(
    description="列出用户全部未删除实体。user_id=用户ID(必填), entity_type=类型过滤(空则全部), limit=返回上限(默认100)。需要浏览用户所有实体清单时调用。"
)
async def list_entities(user_id: str = "", entity_type: str = "", limit: int = 100):
    """列出实体。"""
    def _do(mem, uid):
        if not uid:
            return "Error: user_id not provided."
        return mem.list_entities(user_id=uid, entity_type=entity_type or None, limit=limit)

    return await _mcp_run("list_entities", user_id, _do)


# ---------------------------------------------------------------------------
# V2 编排 API 工具 (记忆管家 — remember/recall/improve/forget)
# ---------------------------------------------------------------------------


@mcp.tool(
    description="记忆管家：自动捕获文本→抽取事实→分类存储→更新工作记忆。agent只需传入对话文本，SeptMuse自主决定存什么、怎么分类。每次用户透露个人信息、偏好、计划或重要事实时调用。"
)
async def remember(text: str, user_id: str = "", agent_id: str = "", session_id: str = "") -> str:
    """记忆管家写入入口。"""
    return await _mcp_run(
        "remember",
        user_id,
        lambda mem, uid: mem.remember(
            text,
            user_id=uid,
            session_id=session_id or None,
            agent_id=agent_id or None,
        ),
    )


@mcp.tool(
    description="记忆管家：智能检索记忆→压缩摘要→token裁剪→注入prompt→策略推荐。返回即用上下文包（摘要+记忆列表+冲突标注+行动建议）。agent需要回忆相关信息时调用，取代简单search。"
)
async def recall(
    query: str,
    user_id: str = "",
    agent_id: str = "",
    session_id: str = "",
    top_k: int = 5,
    token_budget: int = 2000,
) -> str:
    """记忆管家召回入口。"""
    return await _mcp_run(
        "recall",
        user_id,
        lambda mem, uid: mem.recall(
            query,
            user_id=uid,
            session_id=session_id or None,
            top_k=top_k,
            inject_prompt=True,
            token_budget=token_budget,
        ),
    )


@mcp.tool(
    description="记忆管家：整理记忆→蒸馏教训→解决冲突→图谱生长→覆盖分析。在对话结束或空闲时调用，让SeptMuse整理和强化记忆。tasks可选指定子任务(distill/resolve_conflicts/grow_graph/analyze_coverage)，不传跑全部。"
)
async def improve(user_id: str = "", tasks: str = "") -> str:
    """记忆管家演化入口。"""
    return await _mcp_run(
        "improve",
        user_id,
        lambda mem, uid: mem.improve(user_id=uid, limit=20),
    )


@mcp.tool(
    description="记忆管家：遗忘指定记忆（先invalidate标记失效，再软删除）。memory_id=要遗忘的记忆ID。用于清理过时信息或用户要求忘记的内容。"
)
async def forget(user_id: str = "", memory_id: str = "") -> str:
    """记忆管家遗忘入口。"""
    def _do(mem, uid):
        if not memory_id:
            return "Error: memory_id is required"
        return mem.forget(memory_id, user_id=uid)

    return await _mcp_run("forget", user_id, _do)


logger.info("mcp_tools_registered", count=22)

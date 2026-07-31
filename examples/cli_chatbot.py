"""SeptMuse CLI Chatbot 示例 — 验证 facade 可消费性 + 反推路线图优先级。

这是 office-hours 设计文档的落地实现：
    docs/specs/2026-07-27-office-hours-integration-example-agent.md

目标：用一个最小 CLI chatbot 验证 Memory facade 能否支撑真实 agent 对话，
记录顺畅/别扭/缺失三类观察，用于重排 21 Task 路线图优先级。

默认 verbatim 模式（infer=False，零 LLM 依赖）：
    - 每条用户消息原文存入记忆
    - 检索相关记忆注入 context（top-3）
    - 规则模板拼接回复（无 LLM 生成）

--llm 模式（需配置 SEPTMUSE_LLM 环境变量）：
    - infer=True 走 LLM 抽取事实
    - 对话回复用真实 LLM 生成

运行:
    PYTHONPATH=src python examples/cli_chatbot.py
    PYTHONPATH=src python examples/cli_chatbot.py --user alice
    PYTHONPATH=src python examples/cli_chatbot.py --llm
    PYTHONPATH=src python examples/cli_chatbot.py --llm mock

交互命令:
    直接输入文本  — 存记忆 + 检索 + 回复
    /search <关键词> — 只检索，不存
    /get <memory_id> — 查特定记忆
    /update <memory_id> <新内容> — 修正记忆
    /list — 列出全部记忆
    /quit — 退出并输出结构化反馈 JSON
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field

from septmuse import Memory
from septmuse.providers.embedders.hash import HashEmbedder


@dataclass
class ToolUsage:
    """记录每个 facade 方法的调用情况。"""

    method: str
    count: int = 0
    experience: str = "smooth"
    notes: list[str] = field(default_factory=list)


@dataclass
class ChatbotState:
    """chatbot 运行状态 + 反馈收集。"""

    user_id: str
    memory: Memory
    infer: bool
    tool_usage: dict[str, ToolUsage] = field(default_factory=dict)
    missing_capabilities: list[str] = field(default_factory=list)

    def record_call(self, method: str, experience: str = "smooth", note: str = "") -> None:
        if method not in self.tool_usage:
            self.tool_usage[method] = ToolUsage(method=method)
        self.tool_usage[method].count += 1
        if experience == "awkward":
            self.tool_usage[method].experience = "awkward"
        if note:
            self.tool_usage[method].notes.append(note)

    def record_missing(self, capability: str) -> None:
        self.missing_capabilities.append(capability)


def build_memory(llm_mode: str | None) -> Memory:
    """根据 --llm 参数构建 Memory 实例。"""
    embedder = HashEmbedder(dim=128)

    if llm_mode is None:
        return Memory(embedder=embedder)

    if llm_mode == "mock":
        from septmuse.providers.llms.mock import MockLLM

        return Memory(embedder=embedder, llm=MockLLM())

    real_llm_provider = os.environ.get("SEPTMUSE_LLM", "")
    if not real_llm_provider:
        print(
            f"错误: --llm {llm_mode} 需要设置 SEPTMUSE_LLM 环境变量 (openai/ollama/anthropic/dashscope)",
            file=sys.stderr,
        )
        sys.exit(1)

    os.environ["SEPTMUSE_LLM"] = llm_mode
    from septmuse.configs.defaults import default_config

    config = default_config()
    config.infer = True
    return Memory(config=config, embedder=embedder)


def format_reply(user_input: str, memories: list[dict], infer: bool) -> str:
    """规则模板拼接回复（默认模式，零 LLM 依赖）。"""
    if not memories:
        return f"我还没有关于你的记忆。你刚才说：「{user_input}」— 我记住了。"

    memory_texts = [f"  - {m['memory']} (score={m.get('score', 0):.3f})" for m in memories[:3]]
    memory_block = "\n".join(memory_texts)

    mode_label = "LLM 抽取" if infer else "原文存储"
    return f"根据我的记忆（{mode_label}模式）：\n{memory_block}\n\n你刚才说：「{user_input}」— 已记住。"


def llm_reply(user_input: str, memories: list[dict], memory_obj: Memory, user_id: str) -> str:
    """用真实 LLM 生成回复（--llm 模式）。"""
    if memory_obj.llm is None:
        return "[LLM 未配置] 回退到规则模板。\n" + format_reply(user_input, memories, False)

    context = "\n".join(f"- {m['memory']}" for m in memories[:3]) or "（无相关记忆）"
    system_prompt = f"你是用户的记忆助手。以下是与用户相关的记忆：\n{context}\n\n请基于记忆回复用户。简洁友好。"
    try:
        return memory_obj.llm.complete(system_prompt, user_input)
    except Exception as e:
        return f"[LLM 回复失败: {e}] 回退到规则模板。\n" + format_reply(user_input, memories, False)


def handle_search(state: ChatbotState, query: str) -> None:
    """处理 /search 命令。"""
    if not query:
        print("用法: /search <关键词>")
        return

    results = state.memory.search(query, user_id=state.user_id, top_k=5)
    state.record_call("search", note=f"query='{query[:30]}' hits={len(results)}")

    if not results:
        print("未找到相关记忆。")
        return

    print(f"=== 检索结果 ({len(results)} 条) ===")
    for r in results:
        score = r.get("score", 0)
        print(f"  [{r['id']}] {r['memory']} (score={score:.3f})")


def handle_get(state: ChatbotState, memory_id: str) -> None:
    """处理 /get 命令。"""
    if not memory_id:
        print("用法: /get <memory_id>")
        return

    result = state.memory.get(memory_id)
    state.record_call("get", note=f"id={memory_id} found={result is not None}")

    if result is None:
        print(f"记忆 {memory_id} 不存在。")
    else:
        print(f"=== 记忆 {memory_id} ===")
        print(f"  content: {result.get('memory', '?')}")
        print(f"  metadata: {result.get('metadata', {})}")
        print(f"  created_at: {result.get('created_at', '?')}")


def handle_update(state: ChatbotState, args: str) -> None:
    """处理 /update 命令。"""
    parts = args.split(maxsplit=1)
    if len(parts) < 2:
        print("用法: /update <memory_id> <新内容>")
        return

    memory_id, new_text = parts
    result = state.memory.update(memory_id, new_text, user_id=state.user_id)

    if result.get("event") == "NOT_FOUND":
        state.record_call("update", "awkward", f"memory_id={memory_id} not found")
        print(f"记忆 {memory_id} 不存在，无法更新。")
    else:
        state.record_call("update", note=f"memory_id={memory_id} updated")
        print(f"已更新: {result.get('memory', new_text)}")


def handle_list(state: ChatbotState) -> None:
    """处理 /list 命令。"""
    all_memories = state.memory.get_all(user_id=state.user_id)
    results = all_memories.get("results", [])
    state.record_call("get_all", note=f"count={len(results)}")

    if not results:
        print("暂无记忆。")
        return

    print(f"=== 全部记忆 ({len(results)} 条) ===")
    for r in results:
        print(f"  [{r.get('id', '?')}] {r.get('memory', '?')[:80]}")


def handle_normal_input(state: ChatbotState, user_input: str) -> None:
    """处理普通用户输入：存 + 检索 + 回复。"""
    results = state.memory.add(user_input, user_id=state.user_id, infer=state.infer)
    add_results = results.get("results", [])
    if add_results:
        memory_id = add_results[0].get("id", "?")
        state.record_call("add", note=f"memory_id={memory_id} infer={state.infer}")
    else:
        state.record_call("add", "awkward", "add returned empty results")

    memories = state.memory.search(user_input, user_id=state.user_id, top_k=3)
    state.record_call("search", note=f"hits={len(memories)}")

    if state.infer and state.memory.llm is not None:
        reply = llm_reply(user_input, memories, state.memory, state.user_id)
    else:
        reply = format_reply(user_input, memories, state.infer)

    print(reply)


def print_feedback(state: ChatbotState) -> None:
    """退出时输出结构化反馈 JSON。"""
    tools_used = []
    for method, usage in state.tool_usage.items():
        tools_used.append(
            {
                "method": method,
                "count": usage.count,
                "experience": usage.experience,
                "notes": usage.notes[:3],
            }
        )

    awkward_methods = [u.method for u in state.tool_usage.values() if u.experience == "awkward"]
    recommendation_parts = []
    if awkward_methods:
        recommendation_parts.append(f"facade 方法 {awkward_methods} 体验别扭，需改进 API 设计")
    if state.missing_capabilities:
        recommendation_parts.append(f"缺失能力: {state.missing_capabilities}")
    if not recommendation_parts:
        recommendation_parts.append("facade API 顺畅，可继续 21 Task 路线图按原优先级推进")

    feedback = {
        "tools_used": tools_used,
        "missing_capabilities": state.missing_capabilities,
        "recommendation": "；".join(recommendation_parts),
    }

    print("\n=== 结构化反馈 ===")
    print(json.dumps(feedback, ensure_ascii=False, indent=2))


def interactive_loop(state: ChatbotState) -> None:
    """主交互循环。"""
    print("=== SeptMuse CLI Chatbot ===")
    print(f"用户: {state.user_id} | 模式: {'LLM 抽取' if state.infer else 'verbatim'}")
    print("命令: /search /get /update /list /quit")
    print("直接输入文本即可对话。\n")

    while True:
        try:
            user_input = input("你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue

        if user_input == "/quit":
            break
        elif user_input.startswith("/search "):
            handle_search(state, user_input[8:].strip())
        elif user_input.startswith("/get "):
            handle_get(state, user_input[5:].strip())
        elif user_input.startswith("/update "):
            handle_update(state, user_input[8:].strip())
        elif user_input == "/list":
            handle_list(state)
        elif user_input.startswith("/"):
            print(f"未知命令: {user_input}。可用: /search /get /update /list /quit")
        else:
            handle_normal_input(state, user_input)

    print_feedback(state)


def main() -> None:
    parser = argparse.ArgumentParser(description="SeptMuse CLI Chatbot 示例")
    parser.add_argument("--user", default="default", help="用户 ID (默认: default)")
    parser.add_argument(
        "--llm",
        choices=["mock", "openai", "ollama", "anthropic", "dashscope"],
        default=None,
        help="LLM 模式 (默认: verbatim 零 LLM; mock=MockLLM 仅抽取; "
        "openai/ollama/anthropic/dashscope=真实 LLM 抽取+生成)",
    )
    args = parser.parse_args()

    memory = build_memory(args.llm)
    infer = args.llm is not None

    state = ChatbotState(user_id=args.user, memory=memory, infer=infer)
    interactive_loop(state)


if __name__ == "__main__":
    main()

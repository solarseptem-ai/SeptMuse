"""SeptMuse 快速上手示例。

零配置: pip install septmuse 即可运行。
本示例注入 HashEmbedder (零模型加载, 离线可用), 与 CLI 默认行为一致。
生产场景可省略 embedder 注入, Memory() 会用 sentence-transformers 本地嵌入。

运行:
    pip install -e .
    python examples/quickstart.py
"""

from septmuse import Memory
from septmuse.providers.embedders.hash import HashEmbedder


def main() -> None:
    # 1. 创建 Memory 实例（注入 HashEmbedder 离线零模型加载, 与 CLI 默认一致）
    #    生产场景: memory = Memory()  # 自动用 sentence-transformers
    memory = Memory(embedder=HashEmbedder(dim=128))

    # 2. 添加记忆（user_id 隔离）
    memory.add("我喜欢 Python 和 vim 键位", user_id="alice")
    memory.add("alice 在北京工作", user_id="alice")
    memory.add("Bob 偏好 TypeScript 和 emacs", user_id="bob")

    # 3. 检索（返回 list[{"id","memory","score","metadata","created_at"}]）
    results = memory.search("alice 喜欢什么编辑器", user_id="alice", top_k=3)
    print("=== alice 的记忆检索 ===")
    for r in results:
        print(f"  - {r['memory']} (score={r['score']:.3f})")

    # 4. 用户隔离验证
    bob_results = memory.search("喜欢什么编辑器", user_id="bob", top_k=3)
    print("\n=== bob 的记忆检索 ===")
    for r in bob_results:
        print(f"  - {r['memory']} (score={r['score']:.3f})")

    # 5. 更新记忆
    target_id = results[0]["id"]
    memory.update(target_id, "alice 喜欢 Python 和 vim 键位（已更新）", user_id="alice")
    print(f"\n=== 更新记忆 {target_id} ===")

    # 6. 查询历史
    history = memory.get_history(target_id)
    print(f"=== {target_id} 历史 ({len(history)} 条) ===")
    for h in history:
        print(f"  - [{h.get('event', '?')}] {h.get('content', h.get('memory', ''))}")

    # 7. 类型化记忆（fact / episode / rule）
    memory.add_fact("alice", "是", "后端工程师", user_id="alice")
    ep = memory.add_episode("alice 周末部署了 v1.2", user_id="alice", event_type="fact")
    memory.add_rule("部署前必须跑全回归", user_id="alice")
    print(f"\n=== 类型化记忆已添加, episode_id={ep['id']} ===")

    # 8. 工作记忆 Block（letta 风格，注意参数是 agent_id）
    wm = memory.get_working_memory(agent_id="alice-assistant")
    wm.update_block_value("persona", "你是 alice 的助手，简洁回答")
    wm.update_block_value("human", "alice 是后端工程师，喜欢 vim")
    print("\n=== alice 工作记忆 Block ===")
    print(wm.compile_to_xml())

    # 9. 因果链（§6 创新，需要 event_id）
    cause_ep = memory.add_episode("部署 v1.2", user_id="alice", event_type="fact")
    effect_ep = memory.add_episode("回滚到 v1.1", user_id="alice", event_type="fact")
    memory.add_causal_edge(cause_ep["id"], effect_ep["id"], user_id="alice", relation="caused")
    causes = memory.find_causes(effect_ep["id"], user_id="alice")
    print(f"\n=== '{effect_ep['id']}' 的原因路径 ===")
    for c in causes:
        print(f"  - path={c['path']} confidence={c['confidence']:.2f}")

    # 10. 元认知（§6 创新）
    coverage = memory.coverage_report(user_id="alice")
    print("\n=== 覆盖报告 ===")
    print(f"  overall_score={coverage.get('overall_score', 0):.2f}")
    print(f"  summary={coverage.get('summary', '')}")


if __name__ == "__main__":
    main()

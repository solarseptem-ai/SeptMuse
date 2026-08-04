"""SeptMuse 记忆系统使用示例 — 添加记忆 + 召回当前会话记忆。

运行方式:
    $env:PYTHONPATH="src"; python examples/memory_usage_demo.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

# 确保能 import septmuse (src/ layout)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def main() -> None:
    # 用临时目录避免污染 ~/.septmuse/
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["SEPTMUSE_DB_PATH"] = str(Path(tmp) / "demo.db")

        from septmuse import Memory

        m = Memory()
        # 确保结束时关闭 store (释放 SQLite 连接, 否则 Windows 无法删除临时文件)
        import atexit

        atexit.register(lambda: m.store.close())

        print("=" * 60)
        print("SeptMuse 记忆系统使用示例")
        print("=" * 60)

        # ------------------------------------------------------------------
        # 1. 添加记忆
        # ------------------------------------------------------------------
        print("\n--- 1. 添加记忆 ---")

        m.add("我喜欢 Python 编程语言", user_id="alice", session_id="s1")
        m.add("我在 Google 担任后端工程师", user_id="alice", session_id="s1")
        m.add(
            [{"role": "user", "content": "什么是 RAG?"},
             {"role": "assistant", "content": "RAG 是检索增强生成技术"}],
            user_id="alice",
            session_id="s1",
        )

        # 不同会话的记忆
        m.add("今天天气很好", user_id="alice", session_id="s2")

        print(f"已添加 {len(m.get_all(user_id='alice', session_id='s1')['results'])} 条记忆到会话 s1")
        print(f"已添加 {len(m.get_all(user_id='alice', session_id='s2')['results'])} 条记忆到会话 s2")

        # ------------------------------------------------------------------
        # 2. 召回当前会话记忆
        # ------------------------------------------------------------------
        print("\n--- 2. 召回当前会话记忆 ---")

        results = m.search("Python", user_id="alice", session_id="s1", top_k=3)
        print(f"搜索 'Python' (会话 s1): 找到 {len(results)} 条")
        for r in results:
            print(f"  [{r['score']:.4f}] {r['memory']}")

        # ------------------------------------------------------------------
        # 3. 混合检索 (BM25 + 向量 RRF 融合)
        # ------------------------------------------------------------------
        print("\n--- 3. 混合检索 ---")

        results = m.search("工作经历", user_id="alice", session_id="s1", top_k=3)
        print(f"搜索 '工作经历' (混合检索): 找到 {len(results)} 条")
        for r in results:
            print(f"  [{r['score']:.4f}] {r['memory']}")

        # ------------------------------------------------------------------
        # 4. 会话隔离验证
        # ------------------------------------------------------------------
        print("\n--- 4. 会话隔离验证 ---")

        s1_results = m.search("天气", user_id="alice", session_id="s1", top_k=5)
        s2_results = m.search("天气", user_id="alice", session_id="s2", top_k=5)
        print(f"搜索 '天气' 在 s1: {len(s1_results)} 条 (应为 0)")
        print(f"搜索 '天气' 在 s2: {len(s2_results)} 条 (应为 1)")

        # ------------------------------------------------------------------
        # 5. 不限会话的全局搜索
        # ------------------------------------------------------------------
        print("\n--- 5. 全局搜索 (不限会话) ---")

        results = m.search("Python", user_id="alice", top_k=5)
        print(f"搜索 'Python' (全部会话): 找到 {len(results)} 条")

        # ------------------------------------------------------------------
        # 6. 获取单条记忆 + 更新 + 删除
        # ------------------------------------------------------------------
        print("\n--- 6. CRUD 完整流程 ---")

        add_result = m.add("测试记忆", user_id="bob", session_id="s3")
        mid = add_result["results"][0]["id"]
        print(f"添加: id={mid}")

        mem = m.get(mid)
        print(f"获取: {mem['memory']}")

        m.update(mid, data="更新后的测试记忆", user_id="bob")
        mem = m.get(mid)
        print(f"更新: {mem['memory']}")

        m.delete(mid)
        mem = m.get(mid)
        print(f"删除: {mem}")

        # ------------------------------------------------------------------
        # 7. 历史记录
        # ------------------------------------------------------------------
        print("\n--- 7. 历史记录 ---")

        history = m.get_history(mid)
        print(f"记忆 {mid} 的变更历史:")
        for h in history:
            print(f"  {h['event']}: {h['old_memory']} → {h['new_memory']}")

        print("\n" + "=" * 60)
        print("示例完成!")
        print("=" * 60)

        m.store.close()


if __name__ == "__main__":
    main()

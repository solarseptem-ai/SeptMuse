"""SeptMuse 真实端点烟测 — 验证 LLM + Embedder + Memory 全链路。

用法:
    $env:PYTHONPATH = "src"
    python examples/smoke_test_ollama.py

前提: Ollama 兼容端点在 http://localhost:7521/v1 运行,
      模型 qwen3.5:latest + bge-m3:latest 已拉取。
"""

from __future__ import annotations

import os
import sys
import tempfile
import traceback
import gc

from septmuse import Memory, MemoryConfig

# ── 环境变量配置 ──
os.environ["SEPTMUSE_LLM"] = "openai"
os.environ["SEPTMUSE_LLM_MODEL"] = "qwen3.5:latest"
os.environ["SEPTMUSE_LLM_BASE_URL"] = "http://localhost:7521/v1"
os.environ["SEPTMUSE_INFER"] = "true"

os.environ["SEPTMUSE_EMBEDDER"] = "openai"
os.environ["SEPTMUSE_EMBEDDER_MODEL"] = "bge-m3:latest"
os.environ["SEPTMUSE_EMBEDDER_BASE_URL"] = "http://localhost:7521/v1"
os.environ["SEPTMUSE_EMBEDDER_DIMS"] = "1024"

PASS = "\033[92m✓ PASS\033[0m"
FAIL = "\033[91m✗ FAIL\033[0m"

results: list[tuple[str, bool, str]] = []

_TMP = tempfile.gettempdir()


def _db_path(name: str) -> str:
    """生成唯一 db 路径, 清理残留文件 (避免 Windows 文件锁)。"""
    base = os.path.join(_TMP, f"septmuse_smoke_{name}")
    for suffix in ["", ".bm25.db", ".vec.db"]:
        p = base + suffix
        if os.path.exists(p):
            try:
                os.unlink(p)
            except OSError:
                pass
    return base


def _cleanup_db(path: str) -> None:
    """关闭后清理 db 文件。"""
    gc.collect()
    for suffix in ["", ".bm25.db", ".vec.db"]:
        p = path + suffix
        if os.path.exists(p):
            try:
                os.unlink(p)
            except OSError:
                pass


def run_test(name: str, fn):
    try:
        fn()
        results.append((name, True, ""))
        print(f"{PASS}  {name}")
    except Exception as e:
        tb = traceback.format_exc()
        results.append((name, False, str(e)))
        print(f"{FAIL}  {name}: {e}")
        print(f"       {tb.splitlines()[-1] if tb else ''}")


# ── 1. LLM 连通性 ──
def test_llm_complete():
    from septmuse.providers.llms import _resolve_llm
    from septmuse.configs.defaults import MemoryConfig

    config = MemoryConfig(
        llm_provider="openai",
        llm_model="qwen3.5:latest",
        llm_base_url="http://localhost:7521/v1",
    )
    llm = _resolve_llm(config)
    assert llm is not None, "LLM 未创建"

    response = llm.complete("You are a helpful assistant.", "Say 'hello' in one word.")
    assert response and len(response) > 0, f"LLM 返回空: {response}"
    print(f"       LLM 响应: {response[:80]}")


# ── 2. Embedder 连通性 ──
def test_embedder_embed():
    from septmuse.orchestration.memory import _resolve_embedder
    from septmuse.configs.defaults import MemoryConfig

    config = MemoryConfig(
        embedder_backend="openai",
        embedder_model="bge-m3:latest",
        embedder_base_url="http://localhost:7521/v1",
        embedder_dims=1024,
    )
    emb = _resolve_embedder(config)

    vec = emb.embed("我喜欢 Python")
    assert len(vec) == 1024, f"维度不对: {len(vec)} (期望 1024)"
    assert any(v != 0 for v in vec), "向量全零"
    print(f"       向量维度: {len(vec)}, 前5值: {vec[:5]}")


def test_embedder_batch():
    from septmuse.orchestration.memory import _resolve_embedder
    from septmuse.configs.defaults import MemoryConfig

    config = MemoryConfig(
        embedder_backend="openai",
        embedder_model="bge-m3:latest",
        embedder_base_url="http://localhost:7521/v1",
        embedder_dims=1024,
    )
    emb = _resolve_embedder(config)

    vecs = emb.embed_batch(["hello", "world", "你好"])
    assert len(vecs) == 3, f"批量数量不对: {len(vecs)}"
    assert all(len(v) == 1024 for v in vecs), "维度不一致"
    print(f"       批量嵌入: 3 条, 全部 1024 维")


# ── 3. Memory 全链路 ──
def test_memory_add_search():
    db = _db_path("basic")
    m = Memory(config=MemoryConfig(db_path=db))
    mid = m.add("我喜欢 Python 和 Go", user_id="alice")
    assert mid, "add 返回空 id"

    results = m.search("Alice 喜欢什么", user_id="alice")
    assert results, "search 返回空"
    assert "Python" in results[0]["memory"], f"结果不含 Python: {results}"
    print(f"       写入: {str(mid)[:8]}..., 检索 score: {results[0]['score']:.3f}")
    m.close()
    _cleanup_db(db)


def test_memory_infer_llm():
    db = _db_path("infer")
    m = Memory(config=MemoryConfig(db_path=db))

    m.add("我爱 Python 并在 Google 当工程师", user_id="alice", infer=True)

    facts = m.search_facts("Python", user_id="alice", top_k=5)
    if facts:
        print(f"       LLM 抽取事实: {len(facts)} 条")
        for f in facts[:3]:
            print(f"         → {f.get('triple', f.get('memory', ''))[:60]}")
    else:
        print("       (LLM 抽取无结果 — 检查 prompt)")
    m.close()
    _cleanup_db(db)


def test_memory_cognify():
    db = _db_path("cognify")
    m = Memory(config=MemoryConfig(db_path=db))

    result = m.cognify("Alice 在 Google 工作, Bob 也在 Google", user_id="alice")
    entities = result.get("entities", [])
    relations = result.get("relations", [])
    print(f"       实体: {len(entities)}, 关系: {len(relations)}")
    if entities:
        names = [e if isinstance(e, str) else e.get("name", str(e)) for e in entities[:5]]
        print(f"         实体列表: {[n[:20] for n in names]}")
    m.close()
    _cleanup_db(db)


def test_memory_reflect():
    db = _db_path("reflect")
    m = Memory(config=MemoryConfig(db_path=db))

    m.add("用户喜欢简洁的代码", user_id="alice")
    m.add("用户不喜欢冗长的注释", user_id="alice")
    m.add("用户偏好函数式编程", user_id="alice")

    result = m.reflect(user_id="alice", limit=10)
    proposed = result.get("proposed", 0)
    accepted = result.get("accepted", 0)
    print(f"       提出课程: {proposed}, 接受: {accepted}")
    if result.get("rule_ids"):
        print(f"       规则 IDs: {result['rule_ids'][:3]}")
    m.close()
    _cleanup_db(db)


def test_memory_resolve_conflicts():
    db = _db_path("conflict")
    m = Memory(config=MemoryConfig(db_path=db))

    m.add_fact(subject="Alice", predicate="works_at", object="Google", user_id="alice")
    m.add_fact(subject="Alice", predicate="works_at", object="Apple", user_id="alice")

    result = m.resolve_conflicts(user_id="alice")
    print(f"       冲突检测: {result}")
    m.close()
    _cleanup_db(db)


# ── 4. 密集检索性能 ──
def test_search_latency():
    import time

    db = _db_path("latency")
    m = Memory(config=MemoryConfig(db_path=db))

    for i in range(20):
        m.add(f"记忆条目 {i}: 关于 Python 的第 {i} 条笔记", user_id="alice")

    start = time.time()
    results = m.search("Python", user_id="alice", top_k=5)
    elapsed = time.time() - start
    print(f"       20 条记忆检索 {len(results)} 结果, 耗时 {elapsed:.3f}s")
    assert elapsed < 5.0, f"检索太慢: {elapsed:.3f}s"
    m.close()
    _cleanup_db(db)


if __name__ == "__main__":
    print("=" * 60)
    print("SeptMuse 真实端点烟测")
    print(f"LLM:     {os.environ['SEPTMUSE_LLM_MODEL']} @ {os.environ['SEPTMUSE_LLM_BASE_URL']}")
    print(f"Embedder: {os.environ['SEPTMUSE_EMBEDDER_MODEL']} @ {os.environ['SEPTMUSE_EMBEDDER_BASE_URL']}")
    print("=" * 60)
    print()

    print("── 1. LLM 连通性 ──")
    run_test("LLM complete()", test_llm_complete)
    print()

    print("── 2. Embedder 连通性 ──")
    run_test("Embedder embed() 单条", test_embedder_embed)
    run_test("Embedder embed_batch() 批量", test_embedder_batch)
    print()

    print("── 3. Memory 全链路 ──")
    run_test("add + search 基础", test_memory_add_search)
    run_test("add(infer=True) LLM 抽取", test_memory_infer_llm)
    run_test("cognify 知识图谱", test_memory_cognify)
    run_test("reflect 反思蒸馏", test_memory_reflect)
    run_test("resolve_conflicts 冲突解决", test_memory_resolve_conflicts)
    print()

    print("── 4. 性能 ──")
    run_test("检索延迟 (<5s)", test_search_latency)

    print()
    print("=" * 60)
    passed = sum(1 for _, ok, _ in results if ok)
    failed = sum(1 for _, ok, _ in results if not ok)
    total = len(results)
    print(f"结果: {passed}/{total} 通过, {failed} 失败")
    if failed:
        print("\n失败详情:")
        for name, ok, msg in results:
            if not ok:
                print(f"  ✗ {name}: {msg}")
    print("=" * 60)
    sys.exit(0 if failed == 0 else 1)

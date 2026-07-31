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
"""Memory facade 闭环单元测试 (用 HashEmbedder 注入, 不依赖网络)。

固化阶段1 MVP 行为:
- add → search 往返
- get_all / get / delete
- 多 user 隔离
- 返回结构对齐 mem0 ({"results": [...]})
"""

from __future__ import annotations

import pytest

from septmuse import Memory, MemoryConfig
from septmuse.embedders.hash import HashEmbedder
from septmuse.experimental import ExperimentalMemory


@pytest.fixture()
def mem() -> Memory:
    return Memory(
        config=MemoryConfig(db_path=":memory:"),
        embedder=HashEmbedder(),
    )


class TestAdd:
    def test_add_str_returns_results(self, mem: Memory) -> None:
        r = mem.add("hello world", user_id="u1")
        assert "results" in r
        assert len(r["results"]) == 1
        assert r["results"][0]["event"] == "ADD"
        assert r["results"][0]["memory"] == "hello world"
        assert r["results"][0]["id"].startswith("mem-")

    def test_add_message_list(self, mem: Memory) -> None:
        msgs = [
            {"role": "user", "content": "msg one"},
            {"role": "assistant", "content": "msg two"},
        ]
        r = mem.add(msgs, user_id="u1")
        assert len(r["results"]) == 2

    def test_add_empty_returns_empty(self, mem: Memory) -> None:
        r = mem.add([], user_id="u1")
        assert r["results"] == []


class TestSearch:
    def test_search_finds_matching(self, mem: Memory) -> None:
        mem.add("python programming language", user_id="u1")
        mem.add("hiking mountains outdoor", user_id="u1")
        hits = mem.search("python code", user_id="u1", top_k=2, threshold=0.0)
        assert len(hits) >= 1
        assert "python" in hits[0]["memory"]

    def test_search_user_isolation(self, mem: Memory) -> None:
        mem.add("alice secret", user_id="alice")
        mem.add("bob secret", user_id="bob")
        hits = mem.search("secret", user_id="alice", top_k=5, threshold=0.0)
        assert all(h["memory"] == "alice secret" for h in hits)
        assert len(hits) == 1

    def test_search_empty_store(self, mem: Memory) -> None:
        hits = mem.search("anything", user_id="u1")
        assert hits == []


class TestGetAllGetDelete:
    def test_get_all(self, mem: Memory) -> None:
        mem.add("first", user_id="u1")
        mem.add("second", user_id="u1")
        allm = mem.get_all(user_id="u1")
        assert len(allm["results"]) == 2

    def test_get_by_id(self, mem: Memory) -> None:
        r = mem.add("find me", user_id="u1")
        mid = r["results"][0]["id"]
        got = mem.get(mid)
        assert got is not None
        assert got["memory"] == "find me"

    def test_get_nonexistent(self, mem: Memory) -> None:
        assert mem.get("mem-nonexistent") is None

    def test_delete_soft(self, mem: Memory) -> None:
        r = mem.add("to delete", user_id="u1")
        mid = r["results"][0]["id"]
        mem.delete(mid)
        # 软删除: get_all 不再含, get 返回 None
        assert len(mem.get_all(user_id="u1")["results"]) == 0
        assert mem.get(mid) is None

    def test_delete_returns_status(self, mem: Memory) -> None:
        r = mem.add("x", user_id="u1")
        result = mem.delete(r["results"][0]["id"])
        assert result["status"] == "deleted"


class TestMultiUser:
    def test_users_isolated(self, mem: Memory) -> None:
        mem.add("alice data", user_id="alice")
        mem.add("bob data", user_id="bob")
        assert len(mem.get_all(user_id="alice")["results"]) == 1
        assert len(mem.get_all(user_id="bob")["results"]) == 1


class TestSearchHybridDefault:
    """P0: search 默认走 hybrid（BM25+向量 RRF 融合）。"""

    def test_search_default_returns_bm25_score(self, mem: Memory) -> None:
        """hybrid 模式返回结果含 bm25_score 字段。"""
        mem.add("python programming language", user_id="u1")
        hits = mem.search("python", user_id="u1", top_k=5, threshold=0.0)
        assert len(hits) >= 1
        # hybrid 模式额外返回 vector_score + bm25_score
        assert "bm25_score" in hits[0]
        assert "vector_score" in hits[0]

    def test_search_hybrid_false_opt_out(self, mem: Memory) -> None:
        """hybrid=False 回退纯向量，不含 bm25_score。"""
        mem.add("python programming language", user_id="u1")
        hits = mem.search("python", user_id="u1", top_k=5, threshold=0.0, hybrid=False)
        assert len(hits) >= 1
        # 纯向量模式不含 bm25_score
        assert "bm25_score" not in hits[0]

    def test_search_hybrid_catches_keyword_match(self, mem: Memory) -> None:
        """HashEmbedder 向量质量差，但 BM25 能按关键词兜底。"""
        mem.add("我喜欢 python 编程", user_id="u1")
        mem.add("今天天气不错", user_id="u1")
        hits = mem.search("python", user_id="u1", top_k=5, threshold=0.0)
        assert len(hits) >= 1
        assert "python" in hits[0]["memory"]


class TestResolveEmbedder:
    """P1: _resolve_embedder 支持 onnx / onnx-zh / auto 选项。"""

    def test_resolve_hash_default(self, monkeypatch) -> None:
        monkeypatch.delenv("SEPTMUSE_EMBEDDER", raising=False)
        from septmuse.memory.main import _resolve_embedder

        emb = _resolve_embedder(MemoryConfig())
        assert isinstance(emb, HashEmbedder)

    def test_resolve_onnx_english(self, monkeypatch) -> None:
        monkeypatch.setenv("SEPTMUSE_EMBEDDER", "onnx")
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            pytest.skip("onnxruntime 未安装")

        from septmuse.memory.main import _resolve_embedder

        emb = _resolve_embedder(MemoryConfig())
        from septmuse.embedders.onnx import OnnxEmbedder

        assert isinstance(emb, OnnxEmbedder)
        assert emb.dimension == 384

    def test_resolve_onnx_zh(self, monkeypatch) -> None:
        monkeypatch.setenv("SEPTMUSE_EMBEDDER", "onnx-zh")
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            pytest.skip("onnxruntime 未安装")

        from septmuse.memory.main import _resolve_embedder

        emb = _resolve_embedder(MemoryConfig())
        from septmuse.embedders.onnx import OnnxEmbedder

        assert isinstance(emb, OnnxEmbedder)

    def test_resolve_auto(self, monkeypatch) -> None:
        monkeypatch.setenv("SEPTMUSE_EMBEDDER", "auto")
        monkeypatch.setenv("SEPTMUSE_LANG", "zh")
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            pytest.skip("onnxruntime 未安装")

        from septmuse.memory.main import _resolve_embedder

        emb = _resolve_embedder(MemoryConfig())
        from septmuse.embedders.auto import AutoOnnxEmbedder

        assert isinstance(emb, AutoOnnxEmbedder)


class TestResolveEmbedderOpenAI:
    """openai embedder 后端: 支持 OpenAI 兼容端点 (Ollama /v1, vLLM 等)。"""

    def test_resolve_openai_embedder_from_config(self, monkeypatch) -> None:
        monkeypatch.delenv("SEPTMUSE_EMBEDDER", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        from septmuse.memory.main import _resolve_embedder

        config = MemoryConfig(
            embedder_backend="openai",
            embedder_model="bge-m3:latest",
            embedder_base_url="http://localhost:7521/v1",
            embedder_dims=1024,
        )
        emb = _resolve_embedder(config)
        from septmuse.embedders.openai import OpenAIEmbedder

        assert isinstance(emb, OpenAIEmbedder)
        assert emb.model == "bge-m3:latest"
        assert emb.dimension == 1024

    def test_resolve_openai_embedder_no_key_no_crash(self, monkeypatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        from septmuse.memory.main import _resolve_embedder

        config = MemoryConfig(
            embedder_backend="openai",
            embedder_model="bge-m3:latest",
            embedder_base_url="http://localhost:7521/v1",
            embedder_dims=1024,
        )
        emb = _resolve_embedder(config)
        assert emb.dimension == 1024

    def test_resolve_openai_embedder_default_dims(self, monkeypatch) -> None:
        monkeypatch.delenv("SEPTMUSE_EMBEDDER", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        from septmuse.memory.main import _resolve_embedder

        config = MemoryConfig(
            embedder_backend="openai",
            embedder_model="text-embedding-3-small",
            embedder_base_url="http://localhost:7521/v1",
        )
        emb = _resolve_embedder(config)
        assert emb.dimension == 1536


class TestResolveLLMBaseUrl:
    """LLM base_url 配置: 支持 OpenAI 兼容端点。"""

    def test_openai_llm_with_base_url(self, monkeypatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        from septmuse.llms import _resolve_llm

        config = MemoryConfig(
            llm_provider="openai",
            llm_model="qwen3.5:latest",
            llm_base_url="http://localhost:7521/v1",
        )
        llm = _resolve_llm(config)
        from septmuse.llms.openai import OpenAILLM

        assert isinstance(llm, OpenAILLM)
        assert llm.model == "qwen3.5:latest"

    def test_openai_llm_no_key_no_crash(self, monkeypatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        from septmuse.llms import _resolve_llm

        config = MemoryConfig(
            llm_provider="openai",
            llm_model="qwen3.5:latest",
            llm_base_url="http://localhost:7521/v1",
        )
        llm = _resolve_llm(config)
        assert llm is not None

    def test_ollama_llm_with_base_url_as_host(self, monkeypatch) -> None:
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
        from septmuse.llms import _resolve_llm

        config = MemoryConfig(
            llm_provider="ollama",
            llm_model="qwen3.5:latest",
            llm_base_url="http://localhost:7521",
        )
        try:
            import ollama  # noqa: F401
        except ImportError:
            pytest.skip("ollama 未安装")
        llm = _resolve_llm(config)
        from septmuse.llms.ollama import OllamaLLM

        assert isinstance(llm, OllamaLLM)
        assert llm.model == "qwen3.5:latest"


class TestEntityIntegration:
    """Memory facade 实体集成测试。"""

    def test_add_auto_extracts_entities(self, tmp_path):
        config = MemoryConfig(db_path=str(tmp_path / "test.db"))
        m = ExperimentalMemory(config=config)
        m.add("Alice works at Google", user_id="u1")

        entities = m.list_entities(user_id="u1")
        texts = {e["entity_text"] for e in entities}
        assert "Alice" in texts or "Google" in texts

        search_results = m.search_entities("Google", user_id="u1")
        assert any(r["entity_text"] == "Google" for r in search_results)
        m.close()

    def test_add_auto_extract_disabled(self, tmp_path):
        config = MemoryConfig(db_path=str(tmp_path / "test.db"))
        m = ExperimentalMemory(config=config)
        m.add("Alice works at Google", user_id="u1", auto_extract_entities=False)

        entities = m.list_entities(user_id="u1")
        assert len(entities) == 0
        m.close()

    def test_delete_cleans_entity_refs(self, tmp_path):
        config = MemoryConfig(db_path=str(tmp_path / "test.db"))
        m = ExperimentalMemory(config=config)
        result = m.add("Alice works at Google", user_id="u1")
        memory_id = result["results"][0]["id"]

        entities_before = m.list_entities(user_id="u1")
        assert len(entities_before) > 0

        m.delete(memory_id)

        entities_after = m.list_entities(user_id="u1")
        google_entities = [e for e in entities_after if e["entity_text"] == "Google"]
        assert len(google_entities) == 0
        m.close()

    def test_extract_entities_no_store(self, tmp_path):
        config = MemoryConfig(db_path=str(tmp_path / "test.db"))
        m = ExperimentalMemory(config=config)
        entities = m.extract_entities("Alice works at Google")
        texts_types = {(e["text"], e["type"]) for e in entities}
        assert any(e[0] == "Alice" for e in texts_types)
        assert any(e[0] == "Google" for e in texts_types)
        m.close()

    def test_list_entities_by_type(self, tmp_path):
        config = MemoryConfig(db_path=str(tmp_path / "test.db"))
        m = ExperimentalMemory(config=config)
        m.add("Alice works at Google using Python", user_id="u1")
        proper_entities = m.list_entities(user_id="u1", entity_type="PROPER")
        assert all(e["entity_type"] == "PROPER" for e in proper_entities)
        m.close()


class TestEntityMethods:
    """Memory facade 5 个实体方法测试。"""

    def test_extract_entities(self, tmp_path):
        from septmuse.configs.defaults import MemoryConfig
        from septmuse.experimental import ExperimentalMemory

        m = ExperimentalMemory(config=MemoryConfig(db_path=str(tmp_path / "test.db")))
        entities = m.extract_entities("Alice works at Google")
        texts_types = {(e["text"], e["type"]) for e in entities}
        assert any(e[0] == "Alice" for e in texts_types)
        assert any(e[0] == "Google" for e in texts_types)
        m.close()

    def test_add_entity_manual(self, tmp_path):
        from septmuse.configs.defaults import MemoryConfig
        from septmuse.experimental import ExperimentalMemory

        m = ExperimentalMemory(config=MemoryConfig(db_path=str(tmp_path / "test.db")))
        result = m.add_entity("CustomEntity", "PROPER", "mem-001", user_id="u1")
        assert result["event"] == "ADD"
        assert result["entity"] == "CustomEntity"
        entities = m.list_entities(user_id="u1")
        assert any(e["entity_text"] == "CustomEntity" for e in entities)
        m.close()

    def test_search_entities(self, tmp_path):
        from septmuse.configs.defaults import MemoryConfig
        from septmuse.experimental import ExperimentalMemory

        m = ExperimentalMemory(config=MemoryConfig(db_path=str(tmp_path / "test.db")))
        m.add("Alice works at Google", user_id="u1")
        results = m.search_entities("Google", user_id="u1")
        assert len(results) > 0
        assert any(r["entity_text"] == "Google" for r in results)
        m.close()

    def test_get_entity_neighbors(self, tmp_path):
        from septmuse.configs.defaults import MemoryConfig
        from septmuse.experimental import ExperimentalMemory

        m = ExperimentalMemory(config=MemoryConfig(db_path=str(tmp_path / "test.db")))
        result = m.add("Alice works at Google", user_id="u1")
        memory_id = result["results"][0]["id"]
        entities = m.list_entities(user_id="u1")
        google_entity = next(e for e in entities if e["entity_text"] == "Google")
        neighbors = m.get_entity_neighbors(google_entity["id"])
        assert memory_id in neighbors
        m.close()

    def test_list_entities_by_type(self, tmp_path):
        from septmuse.configs.defaults import MemoryConfig
        from septmuse.experimental import ExperimentalMemory

        m = ExperimentalMemory(config=MemoryConfig(db_path=str(tmp_path / "test.db")))
        m.add("Alice works at Google using Python", user_id="u1")
        proper_entities = m.list_entities(user_id="u1", entity_type="PROPER")
        assert all(e["entity_type"] == "PROPER" for e in proper_entities)
        m.close()

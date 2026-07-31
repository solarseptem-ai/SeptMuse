#  Copyright 2026 The sonhhxg0529 Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  Unless required by applicable law.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""阶段5 RBAC 权限 + OpenAI provider + REST API 单元测试。"""

from __future__ import annotations

import pytest

from septmuse import MemoryConfig
from septmuse.embedders.hash import HashEmbedder
from septmuse.experimental import ExperimentalMemory
from septmuse.governance.rbac import Permission, RBACManager, Role


@pytest.fixture()
def mem() -> ExperimentalMemory:
    return ExperimentalMemory(
        config=MemoryConfig(db_path=":memory:"),
        embedder=HashEmbedder(),
    )


# ======================================================================
# RBACManager
# ======================================================================


class TestRBACManager:
    def test_grant_and_check_read(self) -> None:
        rbac = RBACManager()
        rbac.grant("alice", "bot1", Role.AGENT, namespaces=["semantic"])
        result = rbac.check("alice", "bot1", Permission.READ, namespace="semantic")
        assert result.allowed
        assert result.role == Role.AGENT

    def test_check_write_agent(self) -> None:
        rbac = RBACManager()
        rbac.grant("alice", "bot1", Role.AGENT)
        result = rbac.check("alice", "bot1", Permission.WRITE)
        assert result.allowed

    def test_check_observer_no_write(self) -> None:
        rbac = RBACManager()
        rbac.grant("alice", "bot2", Role.OBSERVER)
        result = rbac.check("alice", "bot2", Permission.WRITE)
        assert not result.allowed

    def test_self_access(self) -> None:
        rbac = RBACManager()
        result = rbac.check("alice", "alice", Permission.ADMIN)
        assert result.allowed
        assert result.role == Role.OWNER

    def test_no_grant(self) -> None:
        rbac = RBACManager()
        result = rbac.check("alice", "bot1", Permission.READ)
        assert not result.allowed

    def test_namespace_restriction(self) -> None:
        rbac = RBACManager()
        rbac.grant("alice", "bot1", Role.AGENT, namespaces=["semantic"])
        assert rbac.check("alice", "bot1", Permission.READ, namespace="semantic").allowed
        assert not rbac.check("alice", "bot1", Permission.READ, namespace="episodic").allowed

    def test_wildcard_namespace(self) -> None:
        rbac = RBACManager()
        rbac.grant("alice", "bot1", Role.AGENT, namespaces=["*"])
        assert rbac.check("alice", "bot1", Permission.READ, namespace="anything").allowed

    def test_revoke(self) -> None:
        rbac = RBACManager()
        rbac.grant("alice", "bot1", Role.AGENT)
        assert rbac.revoke("alice", "bot1")
        assert not rbac.check("alice", "bot1", Permission.READ).allowed

    def test_revoke_not_found(self) -> None:
        rbac = RBACManager()
        assert not rbac.revoke("alice", "nonexistent")

    def test_list_grants(self) -> None:
        rbac = RBACManager()
        rbac.grant("alice", "bot1", Role.AGENT)
        rbac.grant("alice", "bot2", Role.OBSERVER)
        grants = rbac.list_grants("alice")
        assert len(grants) == 2

    def test_list_agents(self) -> None:
        rbac = RBACManager()
        rbac.grant("alice", "bot1", Role.AGENT)
        rbac.grant("alice", "bot2", Role.OBSERVER)
        agents = rbac.list_agents_for_user("alice")
        assert set(agents) == {"bot1", "bot2"}

    def test_has_any_access(self) -> None:
        rbac = RBACManager()
        rbac.grant("alice", "bot1", Role.AGENT)
        assert rbac.has_any_access("alice", "bot1")
        assert not rbac.has_any_access("alice", "bot2")


# ======================================================================
# REST API
# ======================================================================


class TestRestAPI:
    @pytest.fixture()
    def client(self, tmp_path):
        from fastapi.testclient import TestClient

        from septmuse.api.rest import create_app

        mem = ExperimentalMemory(
            config=MemoryConfig(db_path=str(tmp_path / "test.db")),
            embedder=HashEmbedder(),
        )
        app = create_app(mem)
        return TestClient(app)

    def test_health(self, client) -> None:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_add_memory(self, client) -> None:
        r = client.post("/memories", json={"content": "hello world", "user_id": "alice"})
        assert r.status_code == 201
        data = r.json()
        assert "results" in data

    def test_list_memories(self, client) -> None:
        client.post("/memories", json={"content": "hello", "user_id": "alice"})
        r = client.get("/memories", params={"user_id": "alice"})
        assert r.status_code == 200
        data = r.json()
        assert len(data["results"]) >= 1

    def test_get_memory_not_found(self, client) -> None:
        r = client.get("/memories/nonexistent-id")
        assert r.status_code == 404

    def test_delete_memory(self, client) -> None:
        r = client.post("/memories", json={"content": "temp", "user_id": "alice"})
        mid = r.json()["results"][0]["id"]
        r = client.delete(f"/memories/{mid}")
        assert r.status_code == 200

    def test_search(self, client) -> None:
        client.post("/memories", json={"content": "alice likes python", "user_id": "alice"})
        r = client.post(
            "/memories/search",
            json={"query": "alice", "user_id": "alice", "top_k": 5, "threshold": 0.0},
        )
        assert r.status_code == 200
        results = r.json()
        assert len(results) >= 1

    def test_capture(self, client) -> None:
        r = client.post("/memories/capture", json={"text": "tool output", "user_id": "alice"})
        assert r.status_code == 200
        assert r.json()["captured"]

    def test_coverage_report(self, client) -> None:
        r = client.get("/memories/meta/coverage", params={"user_id": "alice"})
        assert r.status_code == 200
        assert "overall_score" in r.json()

    def test_rehearse(self, client) -> None:
        r = client.post("/memories/rehearse", json={"user_id": "alice"})
        assert r.status_code == 200

    def test_shared_memories(self, client) -> None:
        client.post("/memories", json={"content": "hello", "user_id": "alice", "agent_id": "bot1"})
        r = client.get("/agents/alice/memories")
        assert r.status_code == 200
        data = r.json()
        assert "bot1" in data["agents"]

    def test_search_entities(self, client) -> None:
        client.post("/memories", json={"content": "Alice works at Google", "user_id": "alice"})
        r = client.get("/entities", params={"query": "Google", "user_id": "alice"})
        assert r.status_code == 200
        results = r.json()["results"]
        assert any(e["entity_text"] == "Google" for e in results)

    def test_list_entities(self, client) -> None:
        client.post("/memories", json={"content": "Alice works at Google", "user_id": "alice"})
        r = client.get("/entities/list", params={"user_id": "alice"})
        assert r.status_code == 200
        results = r.json()["results"]
        assert len(results) >= 1
        assert all("entity_text" in e for e in results)

    def test_add_semantic(self, client) -> None:
        r = client.post(
            "/memories",
            json={"content": "alice likes python", "user_id": "alice", "memory_type": "semantic"},
        )
        assert r.status_code == 201
        assert "triple" in r.json()

    def test_add_episodic(self, client) -> None:
        r = client.post(
            "/memories",
            json={"content": "deployed code", "user_id": "alice", "memory_type": "episodic"},
        )
        assert r.status_code == 201
        assert "id" in r.json()


# ======================================================================
# OpenAI Provider (仅测导入+初始化, 不测真实 API 调用)
# ======================================================================


class TestOpenAILLM:
    def test_import(self) -> None:
        from septmuse.llms.openai import OpenAILLM

        assert OpenAILLM is not None

    def test_init_no_key_uses_dummy(self) -> None:
        # 无 api_key 且无环境变量 → 使用 "not-required" dummy key (本地兼容端点)
        import os

        from septmuse.llms.openai import OpenAILLM

        old_key = os.environ.pop("OPENAI_API_KEY", None)
        try:
            llm = OpenAILLM(api_key=None)
            assert llm is not None
            assert llm.model == "gpt-4o-mini"
        finally:
            if old_key:
                os.environ["OPENAI_API_KEY"] = old_key

    def test_init_with_api_key(self) -> None:
        from septmuse.llms.openai import OpenAILLM

        llm = OpenAILLM(api_key="sk-test-fake-key", model="gpt-4o-mini")
        assert llm.model == "gpt-4o-mini"

    def test_inherits_llm_abc(self) -> None:
        from septmuse.llms.base import LLM
        from septmuse.llms.openai import OpenAILLM

        llm = OpenAILLM(api_key="sk-test-fake-key")
        assert isinstance(llm, LLM)

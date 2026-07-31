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
"""MCP server 工具注册与调用单元测试。

固化 (架构文档 §13):
- 9 工具全部注册 (5 基础对齐 mem0 + 4 SeptMuse 扩展)
- 基础工具 add/search/list/delete 闭环 (注入 HashEmbedder Memory)
- user_id 缺失时报错
- 扩展工具接真实实现 (remember_episode/causal_query/rehearse/coverage_report)
"""

from __future__ import annotations

import asyncio
import json

import pytest

from septmuse import MemoryConfig
from septmuse.api.mcp import tools
from septmuse.api.mcp.server import mcp
from septmuse.embedders.hash import HashEmbedder
from septmuse.experimental import ExperimentalMemory


@pytest.fixture(autouse=True)
def _inject_memory(monkeypatch):
    """每个测试注入内存库 Memory (HashEmbedder, 避免 sentence-transformers 加载)。"""
    mem = ExperimentalMemory(config=MemoryConfig(db_path=":memory:"), embedder=HashEmbedder())
    monkeypatch.setattr(tools, "get_memory_safe", lambda: mem)
    # 同时让 server 模块的 get_memory_safe 返回同一实例
    import septmuse.api.mcp.server as srv

    monkeypatch.setattr(srv, "get_memory_safe", lambda: mem)


def _run(coro):
    return asyncio.run(coro)


class TestToolRegistration:
    def test_tools_registered_count(self):
        tools_map = mcp._tool_manager._tools
        assert len(tools_map) == 18

    def test_required_tool_names(self):
        names = set(mcp._tool_manager._tools.keys())
        expected = {
            "add_memories",
            "search_memory",
            "list_memories",
            "delete_memories",
            "delete_all_memories",
            "remember_episode",
            "causal_query",
            "rehearse",
            "coverage_report",
            "update_memory",
            "update_block",
            "core_memory_append",
            "core_memory_replace",
            "get_blocks",
            "get_memory_history",
            "search_entities",
            "list_entities",
            "invalidate_memory",
        }
        assert expected == names


class TestBasicTools:
    def test_add_memories(self):
        r = _run(tools.add_memories(content="hello", user_id="u1"))
        data = json.loads(r)
        assert len(data["results"]) == 1
        assert data["results"][0]["event"] == "ADD"

    def test_search_memory(self):
        _run(tools.add_memories(content="python programming", user_id="u1"))
        r = _run(tools.search_memory(query="python", user_id="u1"))
        data = json.loads(r)
        assert len(data["results"]) >= 1

    def test_list_memories(self):
        _run(tools.add_memories(content="a", user_id="u1"))
        _run(tools.add_memories(content="b", user_id="u1"))
        r = _run(tools.list_memories(user_id="u1"))
        data = json.loads(r)
        assert len(data["results"]) == 2

    def test_delete_memories(self):
        r = _run(tools.add_memories(content="to delete", user_id="u1"))
        mid = json.loads(r)["results"][0]["id"]
        r2 = _run(tools.delete_memories(memory_ids=[mid], user_id="u1"))
        assert "Successfully deleted 1/1" in r2

    def test_delete_all_memories(self):
        _run(tools.add_memories(content="x", user_id="u1"))
        _run(tools.add_memories(content="y", user_id="u1"))
        r = _run(tools.delete_all_memories(user_id="u1"))
        assert "Successfully deleted all 2" in r


class TestUserIsolation:
    def test_users_isolated(self):
        _run(tools.add_memories(content="alice data", user_id="alice"))
        _run(tools.add_memories(content="bob data", user_id="bob"))
        alice = json.loads(_run(tools.list_memories(user_id="alice")))
        bob = json.loads(_run(tools.list_memories(user_id="bob")))
        assert len(alice["results"]) == 1
        assert len(bob["results"]) == 1


class TestMissingUserId:
    def test_add_without_user_id(self):
        r = _run(tools.add_memories(content="x"))
        assert "user_id not provided" in r

    def test_search_without_user_id(self):
        r = _run(tools.search_memory(query="x"))
        assert "user_id not provided" in r


class TestExtensionToolsReal:
    def test_remember_episode(self):
        r = _run(tools.remember_episode(observation="obs", thoughts="th", action="act", outcome="res", user_id="u1"))
        assert "Episode recorded" not in r
        assert "阶段" not in r

    def test_causal_query(self):
        r = _run(tools.causal_query(cause_event_id="x", hypothesized_effect="y", user_id="u1"))
        assert "阶段4" not in r

    def test_rehearse(self):
        r = _run(tools.rehearse(user_id="u1"))
        assert "阶段4" not in r

    def test_coverage_report(self):
        r = _run(tools.coverage_report(user_id="u1"))
        assert "阶段4" not in r


class TestSetupMcpServer:
    def test_mount_routes(self):
        from fastapi import FastAPI

        from septmuse.api.mcp.server import setup_mcp_server

        app = FastAPI()
        setup_mcp_server(app)
        paths = [getattr(r, "path", "") for r in app.routes]
        assert any("/mcp/http/" in p for p in paths)
        assert any("/mcp/sse/" in p for p in paths)


class TestNewTools:
    def test_update_memory(self):
        # 先 add 再 update
        add_r = _run(tools.add_memories(content="old content", user_id="u1", infer=False))
        mid = json.loads(add_r)["results"][0]["id"]
        r = _run(tools.update_memory(memory_id=mid, content="new content", user_id="u1"))
        assert "UPDATE" in r

    def test_update_memory_not_found(self):
        r = _run(tools.update_memory(memory_id="nonexistent", content="x", user_id="u1"))
        assert "NOT_FOUND" in r

    def test_update_block(self):
        r = _run(tools.update_block(agent_id="ag1", label="human", value="Alice", user_id="u1"))
        assert "UPDATE" in r

    def test_core_memory_append_tool(self):
        _run(tools.update_block(agent_id="ag1", label="human", value="base", user_id="u1"))
        r = _run(tools.core_memory_append(agent_id="ag1", label="human", content="appended", user_id="u1"))
        assert "APPEND" in r

    def test_core_memory_replace_tool(self):
        _run(tools.update_block(agent_id="ag1", label="human", value="old text", user_id="u1"))
        r = _run(
            tools.core_memory_replace(agent_id="ag1", label="human", old_content="old", new_content="new", user_id="u1")
        )
        assert "REPLACE" in r

    def test_get_blocks_tool(self):
        r = _run(tools.get_blocks(agent_id="ag1", user_id="u1"))
        assert "human" in r

    def test_get_memory_history(self):
        add_r = _run(tools.add_memories(content="content", user_id="u1", infer=False))
        mid = json.loads(add_r)["results"][0]["id"]
        r = _run(tools.get_memory_history(memory_id=mid, user_id="u1"))
        assert "ADD" in r


class TestEntityTools:
    def test_search_entities(self):
        _run(tools.add_memories(content="Alice works at Google", user_id="u1"))
        r = _run(tools.search_entities(query="Google", user_id="u1"))
        data = json.loads(r)
        assert any(e["entity_text"] == "Google" for e in data)

    def test_list_entities(self):
        _run(tools.add_memories(content="Alice works at Google", user_id="u1"))
        r = _run(tools.list_entities(user_id="u1"))
        data = json.loads(r)
        assert len(data) >= 1
        assert all("entity_text" in e for e in data)

    def test_search_entities_no_user(self):
        r = _run(tools.search_entities(query="x"))
        assert "user_id not provided" in r

    def test_list_entities_no_user(self):
        r = _run(tools.list_entities())
        assert "user_id not provided" in r

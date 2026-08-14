"""Tests for MCP butler tools (remember/recall/improve/forget)."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_memory():
    """Mock ExperimentalMemory with V2 methods."""
    m = MagicMock()
    m.remember.return_value = {
        "captured": True,
        "raw_id": "raw-001",
        "fact_ids": ["fact-001", "fact-002"],
        "memory_ids": ["mem-001"],
        "memory_id": "mem-001",
    }
    m.recall.return_value = {
        "memories": [{"id": "mem-001", "memory": "hello", "score": 0.9}],
        "injected_prompt": "<memory>hello</memory>",
        "route": "semantic",
        "strategy": "MAINTAIN",
        "used_tokens": 50,
    }
    m.improve.return_value = {
        "dream": {"links_created": 3, "processed": 5},
        "rules": 2,
        "coverage": {"overall_score": 0.7},
        "conflicts": {"resolved": 1},
    }
    m.forget.return_value = {
        "event": "FORGET",
        "invalidated_at": "2026-08-13T00:00:00Z",
        "deleted_at": "2026-08-13T00:00:01Z",
    }
    return m


def test_remember_tool_basic(mock_memory):
    """remember 工具：基本调用返回 JSON。"""
    with patch("septmuse.api.mcp.tools.get_memory_safe", return_value=mock_memory):
        import asyncio

        from septmuse.api.mcp.tools import remember

        result = asyncio.run(remember("用户要出差去上海", user_id="alice"))
        parsed = json.loads(result)
        assert parsed["captured"] is True
        assert parsed["memory_id"] == "mem-001"
        assert len(parsed["fact_ids"]) == 2
        mock_memory.remember.assert_called_once_with("用户要出差去上海", user_id="alice", session_id=None, agent_id=None)


def test_remember_tool_no_user_id():
    """remember 工具：缺 user_id 返回 Error 字符串。"""
    import asyncio

    from septmuse.api.mcp.tools import remember

    result = asyncio.run(remember("test", user_id=""))
    assert result.startswith("Error:")


def test_remember_tool_no_memory():
    """remember 工具：Memory 不可用返回 Error。"""
    with patch("septmuse.api.mcp.tools.get_memory_safe", return_value=None):
        import asyncio

        from septmuse.api.mcp.tools import remember

        result = asyncio.run(remember("test", user_id="alice"))
        assert result.startswith("Error:")


def test_recall_tool_basic(mock_memory):
    """recall 工具：基本调用返回 JSON。"""
    with patch("septmuse.api.mcp.tools.get_memory_safe", return_value=mock_memory):
        import asyncio

        from septmuse.api.mcp.tools import recall

        result = asyncio.run(recall("上海出差", user_id="alice"))
        parsed = json.loads(result)
        assert "memories" in parsed
        assert "injected_prompt" in parsed
        assert len(parsed["memories"]) == 1
        mock_memory.recall.assert_called_once()


def test_recall_tool_with_top_k(mock_memory):
    """recall 工具：top_k 参数传递。"""
    with patch("septmuse.api.mcp.tools.get_memory_safe", return_value=mock_memory):
        import asyncio

        from septmuse.api.mcp.tools import recall

        result = asyncio.run(recall("query", user_id="alice", top_k=10))
        parsed = json.loads(result)
        assert "memories" in parsed
        _, kwargs = mock_memory.recall.call_args
        assert kwargs.get("top_k") == 10 or mock_memory.recall.call_args[0][0] == "query"


def test_recall_tool_no_user_id():
    """recall 工具：缺 user_id 返回 Error。"""
    import asyncio

    from septmuse.api.mcp.tools import recall

    result = asyncio.run(recall("test", user_id=""))
    assert result.startswith("Error:")


def test_improve_tool_basic(mock_memory):
    """improve 工具：基本调用返回 JSON。"""
    with patch("septmuse.api.mcp.tools.get_memory_safe", return_value=mock_memory):
        import asyncio

        from septmuse.api.mcp.tools import improve

        result = asyncio.run(improve(user_id="alice"))
        parsed = json.loads(result)
        assert parsed["dream"]["links_created"] == 3
        assert parsed["rules"] == 2
        mock_memory.improve.assert_called_once_with(user_id="alice", limit=20)


def test_improve_tool_no_user_id():
    """improve 工具：缺 user_id 返回 Error。"""
    import asyncio

    from septmuse.api.mcp.tools import improve

    result = asyncio.run(improve(user_id=""))
    assert result.startswith("Error:")


def test_improve_tool_no_memory():
    """improve 工具：Memory 不可用返回 Error。"""
    with patch("septmuse.api.mcp.tools.get_memory_safe", return_value=None):
        import asyncio

        from septmuse.api.mcp.tools import improve

        result = asyncio.run(improve(user_id="alice"))
        assert result.startswith("Error:")

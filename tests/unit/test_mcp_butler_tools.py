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

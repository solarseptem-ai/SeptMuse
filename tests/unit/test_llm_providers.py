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
"""LLM provider 单元测试 (借鉴 mem0 test_llms + SeptMuse test_reranker _MockLLM 模式)。

SDK 不可用时用 sys.modules mock 注入, 不调真实 API。
"""

from __future__ import annotations

import sys
import types

import pytest


def _make_mock_ollama():
    """创建 mock ollama 模块。"""
    module = types.ModuleType("ollama")

    class MockClient:
        def __init__(self, host=None):
            self.host = host

        def chat(self, **kwargs):
            return {"message": {"content": "mock response"}}

    module.Client = MockClient
    return module


def _make_mock_anthropic():
    """创建 mock anthropic 模块。"""
    module = types.ModuleType("anthropic")

    class MockBlock:
        def __init__(self, text):
            self.text = text

    class MockResponse:
        def __init__(self):
            self.content = [MockBlock("claude response")]

    class MockMessages:
        def create(self, **kwargs):
            return MockResponse()

    class MockClient:
        def __init__(self, api_key=None):
            self.api_key = api_key

        @property
        def messages(self):
            return MockMessages()

    module.Anthropic = MockClient
    return module


def _make_mock_dashscope():
    """创建 mock dashscope 模块。"""
    module = types.ModuleType("dashscope")

    class MockMessage:
        def __init__(self, content):
            self.content = content

    class MockChoice:
        def __init__(self, content):
            self.message = MockMessage(content)

    class MockOutput:
        def __init__(self, content):
            self.choices = [MockChoice(content)]

    class MockResponse:
        def __init__(self, content="qwen response"):
            self.output = MockOutput(content)

    class MockGeneration:
        @staticmethod
        def call(**kwargs):
            return MockResponse()

    module.Generation = MockGeneration
    return module


class TestOllamaLLM:
    def test_complete_returns_text(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "ollama", _make_mock_ollama())
        from septmuse.llms.ollama import OllamaLLM

        llm = OllamaLLM(model="qwen2.5:7b")
        result = llm.complete("system prompt", "user prompt")
        assert result == "mock response"

    def test_default_host(self, monkeypatch):
        mock_mod = _make_mock_ollama()
        captured = {}

        class TrackingClient:
            def __init__(self, host=None):
                captured["host"] = host

            def chat(self, **kwargs):
                return {"message": {"content": "ok"}}

        mock_mod.Client = TrackingClient
        monkeypatch.setitem(sys.modules, "ollama", mock_mod)
        from septmuse.llms.ollama import OllamaLLM

        _ = OllamaLLM()
        assert captured["host"] is not None

    def test_custom_host(self, monkeypatch):
        mock_mod = _make_mock_ollama()
        captured = {}

        class TrackingClient:
            def __init__(self, host=None):
                captured["host"] = host

            def chat(self, **kwargs):
                return {"message": {"content": "ok"}}

        mock_mod.Client = TrackingClient
        monkeypatch.setitem(sys.modules, "ollama", mock_mod)
        from septmuse.llms.ollama import OllamaLLM

        _ = OllamaLLM(host="http://gpu-server:11434")
        assert "gpu-server" in captured["host"]

    def test_no_api_key_required(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "ollama", _make_mock_ollama())
        from septmuse.llms.ollama import OllamaLLM

        llm = OllamaLLM()
        assert llm is not None

    def test_messages_format(self, monkeypatch):
        mock_mod = _make_mock_ollama()
        captured = {}

        class TrackingClient:
            def __init__(self, host=None):
                self.host = host

            def chat(self, **kwargs):
                captured["messages"] = kwargs.get("messages", [])
                return {"message": {"content": "ok"}}

        mock_mod.Client = TrackingClient
        monkeypatch.setitem(sys.modules, "ollama", mock_mod)
        from septmuse.llms.ollama import OllamaLLM

        llm = OllamaLLM()
        llm.complete("sys", "usr")
        assert len(captured["messages"]) == 2
        assert captured["messages"][0]["role"] == "system"
        assert captured["messages"][0]["content"] == "sys"
        assert captured["messages"][1]["role"] == "user"
        assert captured["messages"][1]["content"] == "usr"


class TestAnthropicLLM:
    def test_complete_returns_text(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "anthropic", _make_mock_anthropic())
        from septmuse.llms.anthropic import AnthropicLLM

        llm = AnthropicLLM(api_key="sk-ant-test", model="claude-3-5-haiku-latest")
        result = llm.complete("system prompt", "user prompt")
        assert result == "claude response"

    def test_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-env")
        monkeypatch.setitem(sys.modules, "anthropic", _make_mock_anthropic())
        from septmuse.llms.anthropic import AnthropicLLM

        llm = AnthropicLLM()
        assert llm is not None

    def test_system_passed_separately(self, monkeypatch):
        mock_mod = _make_mock_anthropic()
        captured = {}

        class MockMessages:
            def create(self, **kwargs):
                captured["system"] = kwargs.get("system")
                captured["messages"] = kwargs.get("messages", [])

                class MockBlock:
                    text = "ok"

                class MockResponse:
                    def __init__(self):
                        self.content = [MockBlock()]

                return MockResponse()

        class MockClient:
            def __init__(self, api_key=None):
                pass

            @property
            def messages(self):
                return MockMessages()

        mock_mod.Anthropic = MockClient
        monkeypatch.setitem(sys.modules, "anthropic", mock_mod)
        from septmuse.llms.anthropic import AnthropicLLM

        llm = AnthropicLLM(api_key="test")
        llm.complete("sys prompt", "usr prompt")
        assert captured["system"] == "sys prompt"
        assert len(captured["messages"]) == 1
        assert captured["messages"][0]["role"] == "user"

    def test_no_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setitem(sys.modules, "anthropic", _make_mock_anthropic())
        from septmuse.llms.anthropic import AnthropicLLM

        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            AnthropicLLM()


class TestDashScopeLLM:
    def test_complete_returns_text(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "dashscope", _make_mock_dashscope())
        from septmuse.llms.dashscope import DashScopeLLM

        llm = DashScopeLLM(api_key="sk-ds-test", model="qwen-plus")
        result = llm.complete("system prompt", "user prompt")
        assert result == "qwen response"

    def test_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-ds-env")
        monkeypatch.setitem(sys.modules, "dashscope", _make_mock_dashscope())
        from septmuse.llms.dashscope import DashScopeLLM

        llm = DashScopeLLM()
        assert llm is not None

    def test_no_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
        monkeypatch.setitem(sys.modules, "dashscope", _make_mock_dashscope())
        from septmuse.llms.dashscope import DashScopeLLM

        with pytest.raises(ValueError, match="DASHSCOPE_API_KEY"):
            DashScopeLLM()

    def test_default_model(self, monkeypatch):
        mock_mod = _make_mock_dashscope()
        captured = {}

        class MockGeneration:
            @staticmethod
            def call(**kwargs):
                captured["model"] = kwargs.get("model")

                class MockMsg:
                    content = "ok"

                class MockChoice:
                    message = MockMsg()

                class MockOutput:
                    def __init__(self):
                        self.choices = [MockChoice()]

                class MockResponse:
                    output = MockOutput()

                return MockResponse()

        mock_mod.Generation = MockGeneration
        monkeypatch.setitem(sys.modules, "dashscope", mock_mod)
        from septmuse.llms.dashscope import DashScopeLLM

        llm = DashScopeLLM(api_key="test")
        llm.complete("sys", "usr")
        assert captured["model"] == "qwen-plus"


class TestResolveLLM:
    def test_none_provider_returns_none(self):
        from septmuse.configs.defaults import MemoryConfig
        from septmuse.llms import _resolve_llm

        config = MemoryConfig(llm_provider=None)
        assert _resolve_llm(config) is None

    def test_openai_provider(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        mock_module = types.ModuleType("openai")

        class MockOpenAI:
            def __init__(self, **kwargs):
                pass

        mock_module.OpenAI = MockOpenAI
        monkeypatch.setitem(sys.modules, "openai", mock_module)

        from septmuse.configs.defaults import MemoryConfig
        from septmuse.llms import _resolve_llm
        from septmuse.llms.openai import OpenAILLM

        config = MemoryConfig(llm_provider="openai")
        llm = _resolve_llm(config)
        assert isinstance(llm, OpenAILLM)

    def test_ollama_provider(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "ollama", _make_mock_ollama())
        from septmuse.configs.defaults import MemoryConfig
        from septmuse.llms import _resolve_llm
        from septmuse.llms.ollama import OllamaLLM

        config = MemoryConfig(llm_provider="ollama")
        llm = _resolve_llm(config)
        assert isinstance(llm, OllamaLLM)

    def test_anthropic_provider(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        monkeypatch.setitem(sys.modules, "anthropic", _make_mock_anthropic())
        from septmuse.configs.defaults import MemoryConfig
        from septmuse.llms import _resolve_llm
        from septmuse.llms.anthropic import AnthropicLLM

        config = MemoryConfig(llm_provider="anthropic")
        llm = _resolve_llm(config)
        assert isinstance(llm, AnthropicLLM)

    def test_dashscope_provider(self, monkeypatch):
        monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-ds-test")
        monkeypatch.setitem(sys.modules, "dashscope", _make_mock_dashscope())
        from septmuse.configs.defaults import MemoryConfig
        from septmuse.llms import _resolve_llm
        from septmuse.llms.dashscope import DashScopeLLM

        config = MemoryConfig(llm_provider="dashscope")
        llm = _resolve_llm(config)
        assert isinstance(llm, DashScopeLLM)

    def test_unknown_provider_raises(self):
        from septmuse.configs.defaults import MemoryConfig
        from septmuse.llms import _resolve_llm

        config = MemoryConfig(llm_provider="nonexistent")
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            _resolve_llm(config)

    def test_llm_model_override(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        mock_module = types.ModuleType("openai")

        class MockOpenAI:
            def __init__(self, **kwargs):
                pass

        mock_module.OpenAI = MockOpenAI
        monkeypatch.setitem(sys.modules, "openai", mock_module)

        from septmuse.configs.defaults import MemoryConfig
        from septmuse.llms import _resolve_llm
        from septmuse.llms.openai import OpenAILLM

        config = MemoryConfig(llm_provider="openai", llm_model="gpt-4o")
        llm = _resolve_llm(config)
        assert isinstance(llm, OpenAILLM)
        assert llm.model == "gpt-4o"


class TestMemoryAutoResolve:
    def test_memory_auto_creates_llm(self, monkeypatch, tmp_path):
        """ExperimentalMemory(config) with llm_provider set → auto-creates LLM."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        mock_module = types.ModuleType("openai")

        class MockOpenAI:
            def __init__(self, **kwargs):
                pass

        mock_module.OpenAI = MockOpenAI
        monkeypatch.setitem(sys.modules, "openai", mock_module)

        from septmuse.configs.defaults import MemoryConfig
        from septmuse.experimental import ExperimentalMemory

        config = MemoryConfig(db_path=str(tmp_path / "test.db"), llm_provider="openai")
        m = ExperimentalMemory(config=config)
        assert m.llm is not None

    def test_memory_no_llm_when_provider_none(self, tmp_path):
        """ExperimentalMemory(config) with llm_provider=None → llm is None."""
        from septmuse.configs.defaults import MemoryConfig
        from septmuse.experimental import ExperimentalMemory

        config = MemoryConfig(db_path=str(tmp_path / "test.db"), llm_provider=None)
        m = ExperimentalMemory(config=config)
        assert m.llm is None

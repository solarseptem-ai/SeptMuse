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
"""groq / gemini / deepseek 云 LLM provider 测试。

SDK 不可用时用 sys.modules mock 注入, 不调真实 API。
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch


def _make_mock_groq():
    """创建 mock groq 模块。"""
    module = types.ModuleType("groq")

    class MockMessage:
        def __init__(self, content):
            self.content = content

    class MockChoice:
        def __init__(self, content):
            self.message = MockMessage(content)

    class MockResponse:
        def __init__(self, content="groq response"):
            self.choices = [MockChoice(content)]

    class MockCompletions:
        def create(self, **kwargs):
            return MockResponse()

    class MockChat:
        @property
        def completions(self):
            return MockCompletions()

    class MockGroq:
        def __init__(self, api_key=None, **kwargs):
            self.api_key = api_key

        @property
        def chat(self):
            return MockChat()

    module.Groq = MockGroq
    return module


def _make_mock_google_genai():
    """创建 mock google.generativeai 模块。"""
    module = types.ModuleType("google.generativeai")

    class MockResponse:
        def __init__(self, text="gemini response"):
            self.text = text

    class MockModel:
        def __init__(self, model=None):
            self.model = model

        def generate_content(self, content):
            return MockResponse()

    def configure(api_key=None):
        pass

    module.configure = configure
    module.GenerativeModel = MockModel
    return module


def test_groq_complete(monkeypatch):
    """GroqLLM.complete 委托 groq client。"""
    monkeypatch.setitem(sys.modules, "groq", _make_mock_groq())
    from septmuse.llms.groq import GroqLLM

    llm = GroqLLM(api_key="test", model="llama-3.1-70b-versatile")
    result = llm.complete("sys", "user")
    assert result == "groq response"


def test_gemini_complete(monkeypatch):
    """GeminiLLM.complete 委托 google.generativeai。"""
    google_pkg = types.ModuleType("google")
    google_pkg.__path__ = []
    monkeypatch.setitem(sys.modules, "google", google_pkg)
    monkeypatch.setitem(sys.modules, "google.generativeai", _make_mock_google_genai())
    from septmuse.llms.gemini import GeminiLLM

    llm = GeminiLLM(api_key="test", model="gemini-1.5-flash")
    result = llm.complete("sys", "user")
    assert result == "gemini response"


def test_deepseek_complete():
    """DeepSeekLLM.complete 委托 openai client（兼容 API）。"""
    with patch("openai.OpenAI") as mock_openai:
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=MagicMock(content="deepseek response"))]
        mock_client.chat.completions.create.return_value = mock_resp

        from septmuse.llms.deepseek import DeepSeekLLM

        llm = DeepSeekLLM(api_key="test", model="deepseek-chat")
        result = llm.complete("sys", "user")
        assert result == "deepseek response"


def test_groq_config():
    from septmuse.configs.llms.groq import GroqLLMConfig

    config = GroqLLMConfig(api_key="test")
    assert config.backend == "groq"
    assert config.model == "llama-3.1-70b-versatile"


def test_gemini_config():
    from septmuse.configs.llms.gemini import GeminiLLMConfig

    config = GeminiLLMConfig(api_key="test")
    assert config.backend == "gemini"


def test_deepseek_config():
    from septmuse.configs.llms.deepseek import DeepSeekLLMConfig

    config = DeepSeekLLMConfig(api_key="test")
    assert config.backend == "deepseek"

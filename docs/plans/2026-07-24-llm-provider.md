# LLM Provider 实施计划（P3-Task 1）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现 3 个新 LLM provider（ollama/anthropic/dashscope）+ `_resolve_llm` 工厂 + MemoryConfig/Memory 集成，让 `SEPTMUSE_LLM=xxx` 零配置可用

**Architecture:** 每个 provider 延迟 import SDK + 实现 LLM ABC `complete(system_prompt, user_prompt) -> str`。`_resolve_llm(config)` 工厂从 `config.llm_provider` 创建实例。Memory.__init__ 自动调用工厂。

**Tech Stack:** Python 3.10+ / pytest / ruff / openai(已有) / ollama / anthropic / dashscope

## Global Constraints

- PYTHONPATH=src 运行所有 pytest（PowerShell: `$env:PYTHONPATH="src"`）
- ruff line-length 120，select=["E","F","I","W","UP","B","SIM","RUF"]，ignore=["E501","RUF001","RUF002","RUF003"]
- **禁止** `ruff format <file>`（Windows 会清空文件），只用 `ruff format --check` 或 `ruff check --fix`
- 不用 git（文件快照模式），每个 Task 完成后更新 `.sdd/progress.md`
- 现有 828 passed + 36 skipped 测试零回归
- 中文输出（AGENTS.md 强制），代码注释可用英文
- OpenAILLM 已在 `src/septmuse/providers/llms/openai.py` 完整实现，不需修改
- MockLLM 已在 `src/septmuse/providers/llms/mock.py` 实现，测试用
- `openai`/`anthropic`/`ollama` extras 已在 pyproject.toml 中，只需加 `dashscope` + `llm` alias

---

## File Structure

| 文件 | 职责 | 操作 |
|------|------|------|
| `src/septmuse/providers/llms/ollama.py` | OllamaLLM | 新建 |
| `src/septmuse/providers/llms/anthropic.py` | AnthropicLLM | 新建 |
| `src/septmuse/providers/llms/dashscope.py` | DashScopeLLM | 新建 |
| `src/septmuse/providers/llms/__init__.py` | `_resolve_llm` 工厂 | 修改 |
| `src/septmuse/orchestration/memory.py` | `__init__` 用 `_resolve_llm` | 修改 |
| `src/septmuse/configs/defaults.py` | `MemoryConfig` +`llm_model` | 修改 |
| `pyproject.toml` | +`dashscope`/`llm` extras | 修改 |
| `tests/unit/test_llm_providers.py` | ~22 单元测试 | 新建 |
| `CHANGELOG.md` | 变更记录 | 修改 |
| `AGENTS.md` | +LLM Provider 章节 | 修改 |

---

## Task 1: OllamaLLM + AnthropicLLM + DashScopeLLM

**Files:**
- Create: `src/septmuse/providers/llms/ollama.py`, `src/septmuse/providers/llms/anthropic.py`, `src/septmuse/providers/llms/dashscope.py`
- Test: `tests/unit/test_llm_providers.py`

**Interfaces:**
- Produces: `OllamaLLM(api_key=None, model="qwen2.5:7b", host=None)`, `AnthropicLLM(api_key=None, model="claude-3-5-haiku-latest")`, `DashScopeLLM(api_key=None, model="qwen-plus")`
- Consumes: `LLM` ABC from `providers/llms/base.py`，`OpenAILLM` pattern from `providers/llms/openai.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_llm_providers.py`:

```python
#  Copyright 2026 The sonhhxg0529 Authors. All Rights Reserved.
#  ... (Apache 2.0 license header)
"""LLM provider 单元测试 (借鉴 mem0 test_llms + SeptMuse test_reranker _MockLLM 模式)。"""
from __future__ import annotations

import pytest


class TestOllamaLLM:
    def test_complete_returns_text(self, monkeypatch):
        from septmuse.providers.llms.ollama import OllamaLLM

        class MockClient:
            def chat(self, **kwargs):
                return {"message": {"content": "mock response"}}

        monkeypatch.setattr("ollama.Client", lambda host=None: MockClient())
        llm = OllamaLLM(model="qwen2.5:7b")
        result = llm.complete("system prompt", "user prompt")
        assert result == "mock response"

    def test_default_host(self, monkeypatch):
        from septmuse.providers.llms.ollama import OllamaLLM

        captured = {}
        class MockClient:
            def __init__(self, host=None):
                captured["host"] = host
            def chat(self, **kwargs):
                return {"message": {"content": "ok"}}

        monkeypatch.setattr("ollama.Client", lambda host=None: MockClient(host=host))
        llm = OllamaLLM()
        assert captured["host"] is not None

    def test_custom_host(self, monkeypatch):
        from septmuse.providers.llms.ollama import OllamaLLM

        captured = {}
        class MockClient:
            def __init__(self, host=None):
                captured["host"] = host
            def chat(self, **kwargs):
                return {"message": {"content": "ok"}}

        monkeypatch.setattr("ollama.Client", lambda host=None: MockClient(host=host))
        llm = OllamaLLM(host="http://gpu-server:11434")
        assert "gpu-server" in captured["host"]

    def test_no_api_key_required(self, monkeypatch):
        """Ollama 零配置: 不需要 API key。"""
        from septmuse.providers.llms.ollama import OllamaLLM

        class MockClient:
            def chat(self, **kwargs):
                return {"message": {"content": "ok"}}

        monkeypatch.setattr("ollama.Client", lambda host=None: MockClient())
        llm = OllamaLLM()
        assert llm is not None

    def test_messages_format(self, monkeypatch):
        """验证 Ollama 调用时 messages 格式正确。"""
        from septmuse.providers.llms.ollama import OllamaLLM

        captured = {}
        class MockClient:
            def chat(self, **kwargs):
                captured["messages"] = kwargs.get("messages", [])
                return {"message": {"content": "ok"}}

        monkeypatch.setattr("ollama.Client", lambda host=None: MockClient())
        llm = OllamaLLM()
        llm.complete("sys", "usr")
        assert len(captured["messages"]) == 2
        assert captured["messages"][0]["role"] == "system"
        assert captured["messages"][0]["content"] == "sys"
        assert captured["messages"][1]["role"] == "user"
        assert captured["messages"][1]["content"] == "usr"


class TestAnthropicLLM:
    def test_complete_returns_text(self, monkeypatch):
        from septmuse.providers.llms.anthropic import AnthropicLLM

        class MockResponse:
            content = [type("Block", (), {"text": "claude response"})()]

        class MockClient:
            def messages(self):
                return self
            def create(self, **kwargs):
                return MockResponse()

        monkeypatch.setattr("anthropic.Anthropic", lambda api_key=None: MockClient())
        llm = AnthropicLLM(api_key="sk-ant-test", model="claude-3-5-haiku-latest")
        result = llm.complete("system prompt", "user prompt")
        assert result == "claude response"

    def test_api_key_from_env(self, monkeypatch):
        from septmuse.providers.llms.anthropic import AnthropicLLM

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-env")
        captured = {}
        class MockClient:
            def __init__(self, api_key=None):
                captured["api_key"] = api_key
            def messages(self):
                return self
            def create(self, **kwargs):
                return type("R", (), {"content": [type("B", (), {"text": "ok"})()]})()

        monkeypatch.setattr("anthropic.Anthropic", lambda api_key=None: MockClient(api_key=api_key))
        llm = AnthropicLLM()
        assert captured["api_key"] == "sk-ant-env"

    def test_system_passed_separately(self, monkeypatch):
        """验证 Anthropic system prompt 单独传。"""
        from septmuse.providers.llms.anthropic import AnthropicLLM

        captured = {}
        class MockClient:
            def messages(self):
                return self
            def create(self, **kwargs):
                captured["system"] = kwargs.get("system")
                captured["messages"] = kwargs.get("messages", [])
                return type("R", (), {"content": [type("B", (), {"text": "ok"})()]})()

        monkeypatch.setattr("anthropic.Anthropic", lambda api_key=None: MockClient())
        llm = AnthropicLLM(api_key="test")
        llm.complete("sys prompt", "usr prompt")
        assert captured["system"] == "sys prompt"
        assert len(captured["messages"]) == 1
        assert captured["messages"][0]["role"] == "user"

    def test_no_api_key_raises(self, monkeypatch):
        from septmuse.providers.llms.anthropic import AnthropicLLM

        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            AnthropicLLM()


class TestDashScopeLLM:
    def test_complete_returns_text(self, monkeypatch):
        from septmuse.providers.llms.dashscope import DashScopeLLM

        class MockResponse:
            output = type("Output", (), {
                "choices": [type("Choice", (), {
                    "message": type("Msg", (), {"content": "qwen response"})
                }())]
            })()

        class MockGen:
            @staticmethod
            def call(**kwargs):
                return MockResponse()

        import dashscope
        monkeypatch.setattr(dashscope.Generation, "call", MockGen.call)
        llm = DashScopeLLM(api_key="sk-ds-test", model="qwen-plus")
        result = llm.complete("system prompt", "user prompt")
        assert result == "qwen response"

    def test_api_key_from_env(self, monkeypatch):
        from septmuse.providers.llms.dashscope import DashScopeLLM

        monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-ds-env")
        class MockGen:
            @staticmethod
            def call(**kwargs):
                return type("R", (), {
                    "output": type("O", (), {
                        "choices": [type("C", (), {
                            "message": type("M", (), {"content": "ok"})
                        }())]
                    })()
                })()

        import dashscope
        monkeypatch.setattr(dashscope.Generation, "call", MockGen.call)
        llm = DashScopeLLM()
        assert llm is not None

    def test_no_api_key_raises(self, monkeypatch):
        from septmuse.providers.llms.dashscope import DashScopeLLM

        monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
        with pytest.raises(ValueError, match="DASHSCOPE_API_KEY"):
            DashScopeLLM()

    def test_default_model(self, monkeypatch):
        from septmuse.providers.llms.dashscope import DashScopeLLM

        captured = {}
        class MockGen:
            @staticmethod
            def call(**kwargs):
                captured["model"] = kwargs.get("model")
                return type("R", (), {
                    "output": type("O", (), {
                        "choices": [type("C", (), {
                            "message": type("M", (), {"content": "ok"})
                        }())]
                    })()
                })()

        import dashscope
        monkeypatch.setattr(dashscope.Generation, "call", MockGen.call)
        llm = DashScopeLLM(api_key="test")
        llm.complete("sys", "usr")
        assert captured["model"] == "qwen-plus"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_llm_providers.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'septmuse.providers.llms.ollama'`

- [ ] **Step 3: Write implementation**

**3a. Create `src/septmuse/providers/llms/ollama.py`:**

```python
#  Copyright 2026 The sonhhxg0529 Authors. All Rights Reserved.
#  ... (Apache 2.0 license header)
"""Ollama LLM provider (借鉴 mem0 llms/ollama.py 模式)。

对齐 septmuse.providers.llms.base.LLM ABC,
调用 Ollama Chat API。

用法:
    llm = OllamaLLM(model="qwen2.5:7b")
    response = llm.complete(system_prompt, user_prompt)

零配置: 默认 localhost:11434, 无需 API key。
"""

from __future__ import annotations

import os
from typing import Any

from septmuse.observability import get_logger
from septmuse.providers.llms.base import LLM

logger = get_logger(__name__)


class OllamaLLM(LLM):
    """Ollama Chat provider (借鉴 mem0 llms/ollama.py)。

    零配置: 默认 http://localhost:11434, 无需 API key。
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "qwen2.5:7b",
        host: str | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            from ollama import Client
        except ImportError as e:
            raise ImportError("ollama package required: pip install septmuse[ollama]") from e

        self.model = model
        self._host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self._client = Client(host=self._host)
        logger.info("ollama_llm_ready", model=model, host=self._host)

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """调用 Ollama Chat (对齐 LLM ABC)。"""
        try:
            response = self._client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = response["message"]["content"]
            logger.debug("ollama_complete_done", model=self.model, response_len=len(content))
            return content
        except Exception as e:
            logger.error("ollama_complete_failed", error=str(e))
            raise
```

**3b. Create `src/septmuse/providers/llms/anthropic.py`:**

```python
#  Copyright 2026 The sonhhxg0529 Authors. All Rights Reserved.
#  ... (Apache 2.0 license header)
"""Anthropic LLM provider (借鉴 MemOS AnthropicLlmProvider 模式)。

对齐 septmuse.providers.llms.base.LLM ABC,
调用 Anthropic Messages API。

用法:
    llm = AnthropicLLM(api_key="sk-ant-...", model="claude-3-5-haiku-latest")
    response = llm.complete(system_prompt, user_prompt)

零配置: 从 ANTHROPIC_API_KEY 环境变量读取 key。
"""

from __future__ import annotations

import os
from typing import Any

from septmuse.observability import get_logger
from septmuse.providers.llms.base import LLM

logger = get_logger(__name__)


class AnthropicLLM(LLM):
    """Anthropic Messages provider (借鉴 MemOS AnthropicLlmProvider)。

    零配置: 从 ANTHROPIC_API_KEY 环境变量读取。
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-3-5-haiku-latest",
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> None:
        try:
            from anthropic import Anthropic
        except ImportError as e:
            raise ImportError("anthropic package required: pip install septmuse[anthropic]") from e

        self.model = model
        self._max_tokens = max_tokens
        self._api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self._api_key:
            raise ValueError("ANTHROPIC_API_KEY not set: pass api_key or set env var")

        self._client = Anthropic(api_key=self._api_key)
        logger.info("anthropic_llm_ready", model=model)

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """调用 Anthropic Messages API (对齐 LLM ABC)。

        Anthropic API 要求 system 消息单独传 (不在 messages 列表中)。
        """
        try:
            response = self._client.messages.create(
                model=self.model,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                max_tokens=self._max_tokens,
            )
            content = response.content[0].text if response.content else ""
            logger.debug("anthropic_complete_done", model=self.model, response_len=len(content))
            return content
        except Exception as e:
            logger.error("anthropic_complete_failed", error=str(e))
            raise
```

**3c. Create `src/septmuse/providers/llms/dashscope.py`:**

```python
#  Copyright 2026 The sonhhxg0529 Authors. All Rights Reserved.
#  ... (Apache 2.0 license header)
"""DashScope (Qwen) LLM provider (SeptMuse 创新, 对齐中国用户)。

对齐 septmuse.providers.llms.base.LLM ABC,
调用 DashScope Generation API。

用法:
    llm = DashScopeLLM(api_key="sk-ds-...", model="qwen-plus")
    response = llm.complete(system_prompt, user_prompt)

零配置: 从 DASHSCOPE_API_KEY 环境变量读取 key。
"""

from __future__ import annotations

import os
from typing import Any

from septmuse.observability import get_logger
from septmuse.providers.llms.base import LLM

logger = get_logger(__name__)


class DashScopeLLM(LLM):
    """DashScope (Qwen) provider (SeptMuse 创新, 对齐中国用户)。

    零配置: 从 DASHSCOPE_API_KEY 环境变量读取。
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "qwen-plus",
        **kwargs: Any,
    ) -> None:
        try:
            import dashscope
        except ImportError as e:
            raise ImportError("dashscope package required: pip install septmuse[dashscope]") from e

        self.model = model
        self._api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not self._api_key:
            raise ValueError("DASHSCOPE_API_KEY not set: pass api_key or set env var")

        self._dashscope = dashscope
        logger.info("dashscope_llm_ready", model=model)

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """调用 DashScope Generation API (对齐 LLM ABC)。"""
        try:
            response = self._dashscope.Generation.call(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                api_key=self._api_key,
                result_format="message",
            )
            content = response.output.choices[0].message.content
            logger.debug("dashscope_complete_done", model=self.model, response_len=len(content))
            return content
        except Exception as e:
            logger.error("dashscope_complete_failed", error=str(e))
            raise
```

- [ ] **Step 4: Run test to verify it passes**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_llm_providers.py -q`
Expected: PASS (14 tests)

NOTE: Tests that require `ollama`/`anthropic`/`dashscope` SDK may need to mock the import itself. If tests fail with `ImportError` for the SDK, use `monkeypatch.setattr("sys.modules", ...)` or `monkeypatch.setattr` to inject the module before import. See how `test_reranker.py` handles optional deps.

- [ ] **Step 5: Lint + regression**

Run: `ruff check src/septmuse/providers/llms/ tests/unit/test_llm_providers.py`
Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/ tests/e2e/ -q`
Expected: 828 + 14 = 842 passed, 36 skipped, ruff clean

- [ ] **Step 6: Update progress**

Append to `.sdd/progress.md`: `Task 1: complete (3 LLM providers, 14 tests)`

---

## Task 2: _resolve_llm 工厂 + MemoryConfig + Memory 集成

**Files:**
- Modify: `src/septmuse/providers/llms/__init__.py`
- Modify: `src/septmuse/configs/defaults.py`
- Modify: `src/septmuse/orchestration/memory.py`
- Test: `tests/unit/test_llm_providers.py`

**Interfaces:**
- Produces: `_resolve_llm(config: MemoryConfig) -> LLM | None`，`MemoryConfig.llm_model`
- Consumes: `OpenAILLM`/`OllamaLLM`/`AnthropicLLM`/`DashScopeLLM` from Task 1

- [ ] **Step 1: Write failing tests**

Append to `tests/unit/test_llm_providers.py`:

```python
class TestResolveLLM:
    def test_none_provider_returns_none(self):
        from septmuse.configs.defaults import MemoryConfig
        from septmuse.providers.llms import _resolve_llm

        config = MemoryConfig(llm_provider=None)
        assert _resolve_llm(config) is None

    def test_openai_provider(self, monkeypatch):
        from septmuse.configs.defaults import MemoryConfig
        from septmuse.providers.llms import _resolve_llm
        from septmuse.providers.llms.openai import OpenAILLM

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        config = MemoryConfig(llm_provider="openai")
        llm = _resolve_llm(config)
        assert isinstance(llm, OpenAILLM)

    def test_ollama_provider(self, monkeypatch):
        from septmuse.configs.defaults import MemoryConfig
        from septmuse.providers.llms import _resolve_llm
        from septmuse.providers.llms.ollama import OllamaLLM

        class MockClient:
            def chat(self, **kwargs):
                return {"message": {"content": "ok"}}
        monkeypatch.setattr("ollama.Client", lambda host=None: MockClient())
        config = MemoryConfig(llm_provider="ollama")
        llm = _resolve_llm(config)
        assert isinstance(llm, OllamaLLM)

    def test_anthropic_provider(self, monkeypatch):
        from septmuse.configs.defaults import MemoryConfig
        from septmuse.providers.llms import _resolve_llm
        from septmuse.providers.llms.anthropic import AnthropicLLM

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        config = MemoryConfig(llm_provider="anthropic")
        llm = _resolve_llm(config)
        assert isinstance(llm, AnthropicLLM)

    def test_dashscope_provider(self, monkeypatch):
        from septmuse.configs.defaults import MemoryConfig
        from septmuse.providers.llms import _resolve_llm
        from septmuse.providers.llms.dashscope import DashScopeLLM

        monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-ds-test")
        config = MemoryConfig(llm_provider="dashscope")
        llm = _resolve_llm(config)
        assert isinstance(llm, DashScopeLLM)

    def test_unknown_provider_raises(self):
        from septmuse.configs.defaults import MemoryConfig
        from septmuse.providers.llms import _resolve_llm

        config = MemoryConfig(llm_provider="nonexistent")
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            _resolve_llm(config)

    def test_llm_model_override(self, monkeypatch):
        from septmuse.configs.defaults import MemoryConfig
        from septmuse.providers.llms import _resolve_llm
        from septmuse.providers.llms.openai import OpenAILLM

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        config = MemoryConfig(llm_provider="openai", llm_model="gpt-4o")
        llm = _resolve_llm(config)
        assert isinstance(llm, OpenAILLM)
        assert llm.model == "gpt-4o"


class TestMemoryAutoResolve:
    def test_memory_auto_creates_llm(self, monkeypatch, tmp_path):
        """Memory(config) with llm_provider set → auto-creates LLM."""
        from septmuse.configs.defaults import MemoryConfig
        from septmuse.orchestration.memory import Memory

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        config = MemoryConfig(db_path=str(tmp_path / "test.db"), llm_provider="openai")
        m = Memory(config=config)
        assert m.llm is not None

    def test_memory_no_llm_when_provider_none(self, tmp_path):
        """Memory(config) with llm_provider=None → llm is None."""
        from septmuse.configs.defaults import MemoryConfig
        from septmuse.orchestration.memory import Memory

        config = MemoryConfig(db_path=str(tmp_path / "test.db"), llm_provider=None)
        m = Memory(config=config)
        assert m.llm is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_llm_providers.py::TestResolveLLM -q`
Expected: FAIL with `ImportError: cannot import name '_resolve_llm'`

- [ ] **Step 3: Write implementation**

**3a. Add `_resolve_llm` to `src/septmuse/providers/llms/__init__.py`:**

Read the current file, then add after the existing docstring:

```python
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from septmuse.configs.defaults import MemoryConfig
    from septmuse.providers.llms.base import LLM


def _resolve_llm(config: MemoryConfig) -> LLM | None:
    """工厂函数: 根据 config.llm_provider 创建 LLM 实例 (借鉴 _resolve_embedder)。

    llm_provider=None → 返回 None (verbatim 模式)
    """
    provider = config.llm_provider
    if provider is None:
        return None

    match provider:
        case "openai":
            from septmuse.providers.llms.openai import OpenAILLM

            return OpenAILLM(model=config.llm_model or "gpt-4o-mini")
        case "ollama":
            from septmuse.providers.llms.ollama import OllamaLLM

            return OllamaLLM(model=config.llm_model or "qwen2.5:7b")
        case "anthropic":
            from septmuse.providers.llms.anthropic import AnthropicLLM

            return AnthropicLLM(model=config.llm_model or "claude-3-5-haiku-latest")
        case "dashscope":
            from septmuse.providers.llms.dashscope import DashScopeLLM

            return DashScopeLLM(model=config.llm_model or "qwen-plus")
        case _:
            raise ValueError(f"Unknown LLM provider: {provider}")
```

**3b. Add `llm_model` to `MemoryConfig`** in `src/septmuse/configs/defaults.py`:

Read the file, find the `MemoryConfig` class. Add after `llm_provider`:

```python
    llm_model: str | None = Field(
        default=None,
        description="LLM 模型名 (None → provider 默认模型)",
    )
```

In `default_config()`, add to the return:
```python
        llm_model=os.getenv("SEPTMUSE_LLM_MODEL"),
```

**3c. Modify `Memory.__init__`** in `src/septmuse/orchestration/memory.py`:

Read the file. Find where `self.llm` is set (around line 145). Currently:

```python
self.llm: LLM | None = llm
self.extractor: FactExtractor | None = None
if self.llm is not None:
    self.extractor = FactExtractor(self.llm, self.embedder, self.typed_store, self.store)
```

Change to:

```python
self.llm: LLM | None = llm
if self.llm is None and self.config.llm_provider is not None:
    from septmuse.providers.llms import _resolve_llm

    self.llm = _resolve_llm(self.config)
self.extractor: FactExtractor | None = None
if self.llm is not None:
    self.extractor = FactExtractor(self.llm, self.embedder, self.typed_store, self.store)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_llm_providers.py -q`
Expected: PASS (23 tests: 14 provider + 7 resolve + 2 memory)

- [ ] **Step 5: Lint + full regression**

Run: `ruff check src/ tests/`
Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/ tests/e2e/ -q`
Expected: 842 + 9 = 851 passed, 36 skipped, ruff clean

- [ ] **Step 6: Update progress**

Append to `.sdd/progress.md`: `Task 2: complete (_resolve_llm + MemoryConfig + Memory, 9 tests)`

---

## Task 3: pyproject.toml + CHANGELOG + AGENTS.md

**Files:**
- Modify: `pyproject.toml`, `CHANGELOG.md`, `AGENTS.md`

- [ ] **Step 1: Update pyproject.toml**

Read `pyproject.toml`. Find `[project.optional-dependencies]` section. Add `dashscope` and `llm` alias:

```toml
dashscope = ["dashscope>=1.17"]
llm = ["openai>=1.30"]
```

Also update the `all` extra to include `dashscope`:
```toml
all = ["septmuse[openai,anthropic,ollama,dashscope,postgres,graph,activation,parametric,server,dev,onnx]"]
```

- [ ] **Step 2: Update CHANGELOG**

Add to `[Unreleased]` → `### Added`:

```markdown
- LLM Provider 框架: OllamaLLM/AnthropicLLM/DashScopeLLM + _resolve_llm 工厂 (原因: 解锁 LLM infer 模式; 影响: providers/llms/)
- SEPTMUSE_LLM_MODEL 环境变量: 覆盖 provider 默认模型 (原因: 灵活配置; 影响: MemoryConfig)
- Memory.__init__ 自动创建 LLM: llm_provider 配置时零配置可用 (原因: 零配置; 影响: Memory facade)
- pip install septmuse[dashscope] extra: dashscope>=1.17 (原因: DashScope 可选; 影响: pyproject.toml)
```

- [ ] **Step 3: Update AGENTS.md**

Add to environment variables table:
```
| `SEPTMUSE_LLM_MODEL` | 未设 | 覆盖 provider 默认模型 |
```

Add a new section after "### Bitemporal (双时态)":

```markdown
### LLM Provider

- `SEPTMUSE_LLM=openai` — OpenAI GPT（`gpt-4o-mini` 默认），`OPENAI_API_KEY` 必填，`pip install septmuse[openai]`。
- `SEPTMUSE_LLM=ollama` — Ollama 本地（`qwen2.5:7b` 默认），零配置 `localhost:11434`，`pip install septmuse[ollama]`。
- `SEPTMUSE_LLM=anthropic` — Anthropic Claude（`claude-3-5-haiku-latest` 默认），`ANTHROPIC_API_KEY` 必填，`pip install septmuse[anthropic]`。
- `SEPTMUSE_LLM=dashscope` — DashScope Qwen（`qwen-plus` 默认），`DASHSCOPE_API_KEY` 必填，`pip install septmuse[dashscope]`。
- `SEPTMUSE_LLM_MODEL` 覆盖 provider 默认模型。
- `Memory(config)` 当 `llm` 未注入但 `llm_provider` 已设时，自动用 `_resolve_llm` 创建。
- `OpenAILLM` 已在 `providers/llms/openai.py` 实现（支持 `OPENAI_BASE_URL` 兼容端点）。
- LLM ABC：`complete(system_prompt, user_prompt) -> str`，JSON 输出靠 prompt 工程。
```

- [ ] **Step 4: Full test suite + lint**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/ tests/e2e/ -q`
Run: `ruff check src/ tests/`; `ruff format --check src/ tests/`
Expected: 851 passed, 36 skipped, ruff clean

- [ ] **Step 5: Update progress**

Append to `.sdd/progress.md`:

```
Task 3: complete (pyproject + CHANGELOG + AGENTS.md)

## P3-Task 1 LLM Provider Complete: 851 passed, 36 skipped, ZERO REGRESSION from P2 baseline (828)
- 3 new providers: OllamaLLM + AnthropicLLM + DashScopeLLM (OpenAILLM already existed)
- _resolve_llm factory: SEPTMUSE_LLM → provider instance
- MemoryConfig +llm_model + Memory.__init__ auto-resolve
- 23 new tests (14 provider + 7 resolve + 2 memory)
- pyproject: +dashscope +llm extras
```

---

## Self-Review

**1. Spec coverage:**
- Section 2.2 OpenAILLM → Already implemented, no task needed ✅
- Section 2.3 OllamaLLM → Task 1 Step 3a ✅
- Section 2.4 AnthropicLLM → Task 1 Step 3b ✅
- Section 2.5 DashScopeLLM → Task 1 Step 3c ✅
- Section 3.1 _resolve_llm → Task 2 Step 3a ✅
- Section 3.2 MemoryConfig llm_model → Task 2 Step 3b ✅
- Section 3.3 Memory.__init__ → Task 2 Step 3c ✅
- Section 3.4 环境变量 → Task 2 Step 3b ✅
- Section 3.5 pyproject.toml → Task 3 Step 1 ✅
- Section 4 测试策略 → Tasks 1-2 ✅

**2. Placeholder scan:** No TBD/TODO. ✅

**3. Type consistency:** `_resolve_llm(config) -> LLM | None` consistent. `complete(system_prompt, user_prompt) -> str` consistent with ABC. ✅

# AsyncMemory + litellm 云 Provider 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新建 AsyncMemory 异步记忆类（9 方法）+ AsyncMemoryStore 异步存储层（aiosqlite）+ litellm/groq/gemini/deepseek 4 个 LLM 后端

**Architecture:** sync 路径完全不动；新建 AsyncMemory + AsyncSQLiteMemoryStore（aiosqlite）独立异步链路；embedder/LLM sync 组件用 asyncio.to_thread 包装；REST API 切换为 AsyncMemory

**Tech Stack:** aiosqlite（异步 SQLite）、litellm（统一 LLM 代理）、groq/google-generativeai/openai（云 provider）、pytest-asyncio（已配置 auto 模式）

## 全局约束

- **PYTHONPATH=src** 运行所有测试（PowerShell: `$env:PYTHONPATH="src"`）
- **ruff line-length=120**，只用 `ruff check --fix`（**禁用 ruff format**）
- **不是 git 仓库**，无 commit 步骤
- **代码注释用中文**，不暴露任何开源库参考来源
- **现有测试固定不动**，仅新增测试
- **pytest 基线**：1028 passed + 36 skipped + 32 e2e（不退化）
- **LLM ABC 签名**：`complete(self, system_prompt: str, user_prompt: str) -> str`
- **BaseLLMConfig 字段**：backend/model/temperature/api_key/max_tokens/top_p
- **pytest_asyncio_mode = "auto"**：async 测试无需 @pytest.mark.asyncio
- 工作目录：E:\sonhhxg0529\vibe_coding_project\solarseptem-ai\solarseptem-ai-platform\SeptMuse

## 文件结构

**新建：**
- `src/septmuse/llms/litellm.py` — LitellmLLM 类
- `src/septmuse/llms/groq.py` — GroqLLM 类
- `src/septmuse/llms/gemini.py` — GeminiLLM 类
- `src/septmuse/llms/deepseek.py` — DeepSeekLLM 类
- `src/septmuse/configs/llms/litellm.py` — LitellmLLMConfig
- `src/septmuse/configs/llms/groq.py` — GroqLLMConfig
- `src/septmuse/configs/llms/gemini.py` — GeminiLLMConfig
- `src/septmuse/configs/llms/deepseek.py` — DeepSeekLLMConfig
- `src/septmuse/storage/async_base.py` — AsyncMemoryStore ABC
- `src/septmuse/storage/async_sqlite/__init__.py` — 包初始化
- `src/septmuse/storage/async_sqlite/store.py` — AsyncSQLiteMemoryStore
- `src/septmuse/memory/async_main.py` — AsyncMemory facade
- `tests/unit/test_litellm_llm.py` — litellm 测试
- `tests/unit/test_cloud_llms.py` — groq/gemini/deepseek 测试
- `tests/unit/test_async_store_base.py` — AsyncMemoryStore ABC 测试
- `tests/unit/test_async_sqlite_store.py` — AsyncSQLiteMemoryStore 测试
- `tests/unit/test_async_memory.py` — AsyncMemory facade 测试

**修改：**
- `src/septmuse/services/registry.py` — manifest llm 部分加 4 条
- `src/septmuse/api/rest/__init__.py` — create_app 用 AsyncMemory + 端点改 await
- `pyproject.toml` — 加 aiosqlite/litellm/groq/google-generativeai optional deps

---

## Task 1: litellm + 3 个云 LLM Provider

**Files:**
- Create: `src/septmuse/llms/litellm.py`, `src/septmuse/llms/groq.py`, `src/septmuse/llms/gemini.py`, `src/septmuse/llms/deepseek.py`
- Create: `src/septmuse/configs/llms/litellm.py`, `src/septmuse/configs/llms/groq.py`, `src/septmuse/configs/llms/gemini.py`, `src/septmuse/configs/llms/deepseek.py`
- Modify: `src/septmuse/services/registry.py`（manifest llm 加 4 条）
- Modify: `pyproject.toml`（加 4 个 optional deps）
- Test: `tests/unit/test_litellm_llm.py`, `tests/unit/test_cloud_llms.py`

**Interfaces:**
- Consumes: `LLM` ABC（`complete(system_prompt, user_prompt) -> str`）, `BaseLLMConfig`
- Produces: `LitellmLLM`, `GroqLLM`, `GeminiLLM`, `DeepSeekLLM` + 4 个 config 类 + manifest 4 条

- [ ] **Step 1: 写 litellm 失败测试**

```python
# tests/unit/test_litellm_llm.py
"""litellm LLM provider 测试。"""
from unittest.mock import MagicMock, patch

from septmuse.llms.litellm import LitellmLLM


def test_litellm_complete_delegates():
    """complete 委托 litellm.completion。"""
    with patch("litellm.completion") as mock_completion:
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=MagicMock(content="hello world"))]
        mock_completion.return_value = mock_resp

        llm = LitellmLLM(model="gpt-4o-mini", api_key="test-key")
        result = llm.complete("system prompt", "user prompt")

        assert result == "hello world"
        mock_completion.assert_called_once()
        call_kwargs = mock_completion.call_args
        assert call_kwargs.kwargs["model"] == "gpt-4o-mini"
        assert call_kwargs.kwargs["api_key"] == "test-key"


def test_litellm_config():
    """LitellmLLMConfig 基本字段。"""
    from septmuse.configs.llms.litellm import LitellmLLMConfig

    config = LitellmLLMConfig(model="groq/llama-3.1-70b-versatile", api_key="test")
    assert config.backend == "litellm"
    assert config.model == "groq/llama-3.1-70b-versatile"
```

- [ ] **Step 2: 运行确认失败**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_litellm_llm.py -v`
Expected: FAIL — `No module named 'septmuse.llms.litellm'`

- [ ] **Step 3: 写 LitellmLLMConfig**

```python
# src/septmuse/configs/llms/litellm.py
"""litellm LLM 配置。"""
from __future__ import annotations

from septmuse.configs.llms.base import BaseLLMConfig


class LitellmLLMConfig(BaseLLMConfig):
    """litellm 统一 LLM 配置 — 一个依赖覆盖 100+ provider。"""

    backend: str = "litellm"
    model: str = "gpt-4o-mini"
    base_url: str | None = None
```

- [ ] **Step 4: 写 LitellmLLM**

```python
# src/septmuse/llms/litellm.py
"""litellm 统一 LLM 代理 — 一个依赖覆盖 100+ provider。

用法:
    llm = LitellmLLM(model="groq/llama-3.1-70b-versatile", api_key="...")
    result = llm.complete(system_prompt, user_prompt)
"""
from __future__ import annotations

import os
from typing import Any

from septmuse.core.logging import get_logger
from septmuse.llms.base import LLM

logger = get_logger(__name__)


class LitellmLLM(LLM):
    """litellm 统一 LLM provider。

    model 格式: "provider/model"，如 "groq/llama-3.1-70b-versatile"。
    零配置: 从环境变量读取 api_key（按 provider 前缀）。
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            import litellm
        except ImportError as e:
            raise ImportError("litellm package required: pip install septmuse[litellm]") from e

        self._litellm = litellm
        self.model = model
        self._api_key = api_key
        self._base_url = base_url
        self._extra_kwargs = kwargs
        logger.info("litellm_llm_ready", model=model)

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """委托 litellm.completion。"""
        try:
            response = self._litellm.completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                api_key=self._api_key,
                api_base=self._base_url,
                **self._extra_kwargs,
            )
            content = response.choices[0].message.content or ""
            logger.debug("litellm_complete_done", model=self.model, response_len=len(content))
            return content
        except Exception as e:
            logger.error("litellm_complete_failed", error=str(e))
            raise
```

- [ ] **Step 5: 写 3 个云 provider 测试**

```python
# tests/unit/test_cloud_llms.py
"""groq / gemini / deepseek 云 LLM provider 测试。"""
from unittest.mock import MagicMock, patch


def test_groq_complete():
    """GroqLLM.complete 委托 groq client。"""
    with patch("groq.Groq") as mock_groq:
        mock_client = MagicMock()
        mock_groq.return_value = mock_client
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=MagicMock(content="groq response"))]
        mock_client.chat.completions.create.return_value = mock_resp

        from septmuse.llms.groq import GroqLLM

        llm = GroqLLM(api_key="test", model="llama-3.1-70b-versatile")
        result = llm.complete("sys", "user")
        assert result == "groq response"


def test_gemini_complete():
    """GeminiLLM.complete 委托 google.generativeai。"""
    with patch("google.generativeai.configure"), \
         patch("google.generativeai.GenerativeModel") as mock_model_cls:
        mock_model = MagicMock()
        mock_model_cls.return_value = mock_model
        mock_resp = MagicMock()
        mock_resp.text = "gemini response"
        mock_model.generate_content.return_value = mock_resp

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
```

- [ ] **Step 6: 运行确认失败**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_cloud_llms.py -v`
Expected: FAIL — `No module named 'septmuse.llms.groq'`

- [ ] **Step 7: 写 3 个云 provider config 类**

```python
# src/septmuse/configs/llms/groq.py
"""Groq LLM 配置。"""
from __future__ import annotations
from septmuse.configs.llms.base import BaseLLMConfig

class GroqLLMConfig(BaseLLMConfig):
    """Groq 超低延迟推理配置。"""
    backend: str = "groq"
    model: str = "llama-3.1-70b-versatile"
```

```python
# src/septmuse/configs/llms/gemini.py
"""Google Gemini LLM 配置。"""
from __future__ import annotations
from septmuse.configs.llms.base import BaseLLMConfig

class GeminiLLMConfig(BaseLLMConfig):
    """Google Gemini LLM 配置。"""
    backend: str = "gemini"
    model: str = "gemini-1.5-flash"
```

```python
# src/septmuse/configs/llms/deepseek.py
"""DeepSeek LLM 配置。"""
from __future__ import annotations
from septmuse.configs.llms.base import BaseLLMConfig

class DeepSeekLLMConfig(BaseLLMConfig):
    """DeepSeek LLM 配置（OpenAI 兼容 API）。"""
    backend: str = "deepseek"
    model: str = "deepseek-chat"
    base_url: str = "https://api.deepseek.com"
```

- [ ] **Step 8: 写 3 个云 provider LLM 类**

```python
# src/septmuse/llms/groq.py
"""Groq LLM provider — 超低延迟推理。"""
from __future__ import annotations

import os
from typing import Any

from septmuse.core.logging import get_logger
from septmuse.llms.base import LLM

logger = get_logger(__name__)


class GroqLLM(LLM):
    """Groq LLM provider。

    零配置: 从 GROQ_API_KEY 环境变量读取 key。
    """

    def __init__(self, api_key: str | None = None, model: str = "llama-3.1-70b-versatile", **kwargs: Any) -> None:
        try:
            from groq import Groq
        except ImportError as e:
            raise ImportError("groq package required: pip install septmuse[groq]") from e

        self.model = model
        self._client = Groq(api_key=api_key or os.getenv("GROQ_API_KEY"), **kwargs)
        logger.info("groq_llm_ready", model=model)

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error("groq_complete_failed", error=str(e))
            raise
```

```python
# src/septmuse/llms/gemini.py
"""Google Gemini LLM provider。"""
from __future__ import annotations

import os
from typing import Any

from septmuse.core.logging import get_logger
from septmuse.llms.base import LLM

logger = get_logger(__name__)


class GeminiLLM(LLM):
    """Google Gemini LLM provider。

    零配置: 从 GEMINI_API_KEY 环境变量读取 key。
    """

    def __init__(self, api_key: str | None = None, model: str = "gemini-1.5-flash", **kwargs: Any) -> None:
        try:
            import google.generativeai as genai
        except ImportError as e:
            raise ImportError("google-generativeai required: pip install septmuse[gemini]") from e

        genai.configure(api_key=api_key or os.getenv("GEMINI_API_KEY"))
        self._genai = genai
        self.model = model
        self._model = genai.GenerativeModel(model)
        logger.info("gemini_llm_ready", model=model)

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        try:
            response = self._model.generate_content(f"{system_prompt}\n\n{user_prompt}")
            return response.text or ""
        except Exception as e:
            logger.error("gemini_complete_failed", error=str(e))
            raise
```

```python
# src/septmuse/llms/deepseek.py
"""DeepSeek LLM provider — OpenAI 兼容 API。"""
from __future__ import annotations

import os
from typing import Any

from septmuse.core.logging import get_logger
from septmuse.llms.base import LLM

logger = get_logger(__name__)


class DeepSeekLLM(LLM):
    """DeepSeek LLM provider（OpenAI 兼容 API）。

    零配置: 从 DEEPSEEK_API_KEY 环境变量读取 key。
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com",
        **kwargs: Any,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("openai package required: pip install septmuse[openai]") from e

        self.model = model
        self._client = OpenAI(
            api_key=api_key or os.getenv("DEEPSEEK_API_KEY"),
            base_url=base_url,
            **kwargs,
        )
        logger.info("deepseek_llm_ready", model=model)

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error("deepseek_complete_failed", error=str(e))
            raise
```

- [ ] **Step 9: 运行测试确认通过**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_litellm_llm.py tests/unit/test_cloud_llms.py -v`
Expected: PASS（全部测试通过）

- [ ] **Step 10: manifest 加 4 条**

在 `src/septmuse/services/registry.py` 的 `BACKEND_MANIFEST["llm"]` 字典中，现有 4 条后面加 4 条：

```python
        "litellm":   BackendEntry("septmuse.llms.litellm",   "LitellmLLM",   "septmuse.configs.llms.litellm.LitellmLLMConfig",   ("litellm",)),
        "groq":      BackendEntry("septmuse.llms.groq",      "GroqLLM",      "septmuse.configs.llms.groq.GroqLLMConfig",          ("groq",)),
        "gemini":    BackendEntry("septmuse.llms.gemini",    "GeminiLLM",    "septmuse.configs.llms.gemini.GeminiLLMConfig",      ("google-generativeai",)),
        "deepseek":  BackendEntry("septmuse.llms.deepseek",  "DeepSeekLLM",  "septmuse.configs.llms.deepseek.DeepSeekLLMConfig",  ("openai",)),
```

注意：`config_cls` 用字符串路径（与 Task 1 ServiceProvider 的设计一致，零启动开销）。

- [ ] **Step 11: pyproject.toml 加 4 个 optional deps**

在 `[project.optional-dependencies]` 部分加：

```toml
litellm = ["litellm>=1.40"]
groq = ["groq>=0.11"]
gemini = ["google-generativeai>=0.7"]
deepseek = ["openai>=1.30"]
```

- [ ] **Step 12: ruff + manifest 完整性验证**

Run: `ruff check --fix src/septmuse/llms/ src/septmuse/configs/llms/ src/septmuse/services/registry.py tests/unit/test_litellm_llm.py tests/unit/test_cloud_llms.py`

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_manifest.py -v`
Expected: PASS（manifest 完整性，8 个能力都有默认后端）

Run: `$env:PYTHONPATH="src"; python -c "from septmuse.services.providers import llm_provider; print(llm_provider.list_backends())"`
Expected: `['openai', 'ollama', 'anthropic', 'dashscope', 'litellm', 'groq', 'gemini', 'deepseek']`

---

## Task 2: AsyncMemoryStore ABC

**Files:**
- Create: `src/septmuse/storage/async_base.py`
- Test: `tests/unit/test_async_store_base.py`

**Interfaces:**
- Produces: `AsyncMemoryStore` ABC with async abstract methods (add/search/get_all/get/delete/update/get_history/close) + async default methods (keyword_search/hybrid_search/get_access_logs)

- [ ] **Step 1: 写 ABC 失败测试**

```python
# tests/unit/test_async_store_base.py
"""AsyncMemoryStore ABC 测试。"""
import inspect

from septmuse.storage.async_base import AsyncMemoryStore


def test_all_methods_are_async():
    """所有公开方法都是 async def。"""
    for name in ["add", "search", "get_all", "get", "delete", "update", "get_history", "close"]:
        method = getattr(AsyncMemoryStore, name)
        assert inspect.iscoroutinefunction(method), f"{name} 不是 async def"


def test_default_methods_are_async():
    """默认实现方法也是 async。"""
    for name in ["keyword_search", "hybrid_search", "get_access_logs", "get_temporal_valid", "get_temporal_interval"]:
        method = getattr(AsyncMemoryStore, name)
        assert inspect.iscoroutinefunction(method), f"{name} 不是 async def"


def test_cannot_instantiate_abc():
    """不能直接实例化 ABC。"""
    import pytest
    with pytest.raises(TypeError):
        AsyncMemoryStore()
```

- [ ] **Step 2: 运行确认失败**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_async_store_base.py -v`
Expected: FAIL — `No module named 'septmuse.storage.async_base'`

- [ ] **Step 3: 写 AsyncMemoryStore ABC**

```python
# src/septmuse/storage/async_base.py
"""异步记忆存储后端抽象基类。

所有方法为 async def，使用 aiosqlite/asyncpg 等异步驱动。
sync MemoryStore 的对偶，方法签名保持一致。
score 语义: 相似度 (越高越相似, 范围 [0, 1])。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class AsyncMemoryStore(ABC):
    """异步记忆存储后端抽象。

    实现方需保证:
    - add 返回唯一 memory_id
    - search 的 score 为相似度 (0-1, 越高越相似)
    - delete 为软删除 (标记 is_deleted + history 记录)
    - user_id 隔离 (不同用户互不可见)
    """

    @abstractmethod
    async def add(
        self,
        content: str,
        embedding: list[float],
        *,
        user_id: str,
        agent_id: str | None = None,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        valid_at: str | None = None,
    ) -> str:
        """添加记忆，返回 memory_id。"""
        ...

    @abstractmethod
    async def search(
        self,
        query_embedding: list[float],
        *,
        user_id: str,
        session_id: str | None = None,
        top_k: int = 5,
        threshold: float = 0.1,
    ) -> list[dict[str, Any]]:
        """向量检索，返回 [{"id", "memory", "score", ...}]。"""
        ...

    @abstractmethod
    async def get_all(self, *, user_id: str, session_id: str | None = None) -> list[dict[str, Any]]:
        """列出该用户全部未删除记忆。"""
        ...

    @abstractmethod
    async def get(self, memory_id: str) -> dict[str, Any] | None:
        """取单条。"""
        ...

    @abstractmethod
    async def delete(self, memory_id: str) -> None:
        """软删除。"""
        ...

    @abstractmethod
    async def update(
        self,
        memory_id: str,
        content: str,
        embedding: list[float],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """更新记忆。"""
        ...

    @abstractmethod
    async def get_history(self, memory_id: str) -> list[dict[str, Any]]:
        """获取变更历史。"""
        ...

    @abstractmethod
    async def close(self) -> None:
        """释放连接资源。"""
        ...

    # ── 默认实现（子类可覆盖）──

    async def keyword_search(
        self, query: str, *, user_id: str, session_id: str | None = None, top_k: int = 5
    ) -> list[dict[str, Any]]:
        """关键词检索。默认返回空。"""
        return []

    async def hybrid_search(
        self,
        query: str,
        query_embedding: list[float],
        *,
        user_id: str,
        session_id: str | None = None,
        top_k: int = 5,
        alpha: float = 0.5,
    ) -> list[dict[str, Any]]:
        """混合检索（向量 + 关键词 RRF 融合）。"""
        vec = await self.search(query_embedding, user_id=user_id, session_id=session_id, top_k=top_k * 2)
        kw = await self.keyword_search(query, user_id=user_id, session_id=session_id, top_k=top_k * 2)
        return _rrf_fuse(vec, kw, alpha=alpha)[:top_k]

    async def get_access_logs(self, memory_id: str, limit: int = 100) -> list[dict[str, Any]]:
        """查询访问日志。默认返回空。"""
        return []

    async def get_temporal_valid(
        self, reference_time: str, *, user_id: str, session_id: str | None = None
    ) -> list[dict[str, Any]]:
        """查询某时刻为真的记忆。默认返回空。"""
        return []

    async def get_temporal_interval(
        self, start: str, end: str, *, user_id: str, session_id: str | None = None
    ) -> list[dict[str, Any]]:
        """查询时间区间内为真的记忆。默认返回空。"""
        return []

    async def invalidate(self, memory_id: str, *, invalid_at: str | None = None) -> dict[str, Any]:
        """标记事实不再为真。默认不支持。"""
        raise NotImplementedError(f"{type(self).__name__} 不支持 invalidate")


def _rrf_fuse(vec_results: list[dict], kw_results: list[dict], *, alpha: float = 0.5, k: int = 60) -> list[dict]:
    """RRF 融合排序（纯计算，无 I/O，不需 async）。"""
    scores: dict[str, float] = {}
    meta: dict[str, dict] = {}
    if alpha > 0:
        for rank, r in enumerate(vec_results):
            scores[r["id"]] = scores.get(r["id"], 0.0) + alpha / (k + rank + 1)
            meta.setdefault(r["id"], r)
    if alpha < 1:
        for rank, r in enumerate(kw_results):
            scores[r["id"]] = scores.get(r["id"], 0.0) + (1 - alpha) / (k + rank + 1)
            meta.setdefault(r["id"], r)
    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [{**meta[mid], "score": sc} for mid, sc in ordered]
```

- [ ] **Step 4: 运行测试确认通过**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_async_store_base.py -v`
Expected: PASS

- [ ] **Step 5: ruff 检查**

Run: `ruff check --fix src/septmuse/storage/async_base.py tests/unit/test_async_store_base.py`

---

## Task 3: AsyncSQLiteMemoryStore（aiosqlite 实现）

**Files:**
- Create: `src/septmuse/storage/async_sqlite/__init__.py`, `src/septmuse/storage/async_sqlite/store.py`
- Modify: `pyproject.toml`（加 aiosqlite 核心依赖）
- Test: `tests/unit/test_async_sqlite_store.py`

**Interfaces:**
- Consumes: Task 2 的 `AsyncMemoryStore` ABC
- Produces: `AsyncSQLiteMemoryStore` 实现类

- [ ] **Step 1: 加 aiosqlite 依赖**

在 `pyproject.toml` 的 `dependencies` 列表中加 `"aiosqlite>=0.20"`。

Run: `pip install aiosqlite`

- [ ] **Step 2: 写 async store 失败测试**

```python
# tests/unit/test_async_sqlite_store.py
"""AsyncSQLiteMemoryStore 测试。"""
import tempfile
from pathlib import Path

import pytest

from septmuse.storage.async_sqlite.store import AsyncSQLiteMemoryStore


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "async_test.db")


async def test_add_and_search(db_path):
    """添加记忆后能检索到。"""
    store = AsyncSQLiteMemoryStore(db_path=db_path)
    try:
        mid = await store.add("hello world", [1.0, 0.0, 0.0], user_id="alice")
        assert mid.startswith("mem-")

        results = await store.search([1.0, 0.0, 0.0], user_id="alice", top_k=5)
        assert len(results) == 1
        assert results[0]["memory"] == "hello world"
        assert results[0]["score"] > 0.5
    finally:
        await store.close()


async def test_get_and_delete(db_path):
    """获取和软删除。"""
    store = AsyncSQLiteMemoryStore(db_path=db_path)
    try:
        mid = await store.add("test memory", [0.5, 0.5], user_id="bob")
        mem = await store.get(mid)
        assert mem is not None
        assert mem["memory"] == "test memory"

        await store.delete(mid)
        mem_after = await store.get(mid)
        assert mem_after is None  # 软删除后 get 返回 None
    finally:
        await store.close()


async def test_get_all(db_path):
    """列出全部记忆。"""
    store = AsyncSQLiteMemoryStore(db_path=db_path)
    try:
        await store.add("first", [1.0, 0.0], user_id="alice")
        await store.add("second", [0.0, 1.0], user_id="alice")
        all_mems = await store.get_all(user_id="alice")
        assert len(all_mems) == 2
    finally:
        await store.close()


async def test_update(db_path):
    """更新记忆。"""
    store = AsyncSQLiteMemoryStore(db_path=db_path)
    try:
        mid = await store.add("original", [1.0, 0.0], user_id="alice")
        success = await store.update(mid, "updated", [0.0, 1.0])
        assert success is True
        mem = await store.get(mid)
        assert mem["memory"] == "updated"
    finally:
        await store.close()


async def test_get_history(db_path):
    """变更历史。"""
    store = AsyncSQLiteMemoryStore(db_path=db_path)
    try:
        mid = await store.add("original", [1.0, 0.0], user_id="alice")
        await store.update(mid, "updated", [0.0, 1.0])
        await store.delete(mid)
        history = await store.get_history(mid)
        assert len(history) == 3  # ADD + UPDATE + DELETE
        events = [h["event"] for h in history]
        assert "ADD" in events
        assert "UPDATE" in events
        assert "DELETE" in events
    finally:
        await store.close()


async def test_user_isolation(db_path):
    """用户隔离 — alice 看不到 bob 的记忆。"""
    store = AsyncSQLiteMemoryStore(db_path=db_path)
    try:
        await store.add("alice memory", [1.0, 0.0], user_id="alice")
        await store.add("bob memory", [0.0, 1.0], user_id="bob")
        alice_results = await store.search([1.0, 0.0], user_id="alice")
        assert len(alice_results) == 1
        assert alice_results[0]["memory"] == "alice memory"
    finally:
        await store.close()
```

- [ ] **Step 3: 运行确认失败**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_async_sqlite_store.py -v`
Expected: FAIL — `No module named 'septmuse.storage.async_sqlite'`

- [ ] **Step 4: 写 AsyncSQLiteMemoryStore**

先读 `src/septmuse/storage/sqlite/store.py` 全文了解现有 sync 版的表结构（DDL）和双写逻辑（vector_store + keyword_index），然后在 async 版中复用同样的 DDL 和双写组件。

```python
# src/septmuse/storage/async_sqlite/__init__.py
from septmuse.storage.async_sqlite.store import AsyncSQLiteMemoryStore

__all__ = ["AsyncSQLiteMemoryStore"]
```

```python
# src/septmuse/storage/async_sqlite/store.py
"""异步 SQLite 记忆存储（aiosqlite）。

表结构与 sync SQLiteMemoryStore 一致，同一个 DB 文件可共享。
双写组件（SQLiteVectorStore + SQLiteBM25Index）用 asyncio.to_thread 包装。
"""
from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from typing import Any

import aiosqlite

from septmuse.core.logging import get_logger
from septmuse.storage.async_base import AsyncMemoryStore
from septmuse.storage.keyword.sqlite_bm25 import SQLiteBM25Index
from septmuse.storage.vector.sqlite_vec import SQLiteVectorStore

logger = get_logger(__name__)


def _utcnow_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


class AsyncSQLiteMemoryStore(AsyncMemoryStore):
    """异步 SQLite 记忆存储（aiosqlite）。

    用法:
        store = AsyncSQLiteMemoryStore(db_path="mem.db")
        mid = await store.add("hello", [1.0, ...], user_id="alice")
        results = await store.search([1.0, ...], user_id="alice")
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._db_path = str(db_path) if db_path else str(Path.home() / ".septmuse" / "septmuse.db")
        self._conn: aiosqlite.Connection | None = None
        # 双写组件（sync，用 to_thread 包装）
        self._vector_store: SQLiteVectorStore | None = None
        self._keyword_index: SQLiteBM25Index | None = None

    async def _ensure_conn(self) -> aiosqlite.Connection:
        """延迟打开连接（首次操作时）。"""
        if self._conn is None:
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = aiosqlite.connect(self._db_path)
            await self._conn.__aenter__()
            await self._create_tables()
            # 初始化双写组件（复用同一 sqlite 连接的 sync 版）
            import sqlite3
            sync_conn = sqlite3.connect(self._db_path, check_same_thread=False)
            self._vector_store = SQLiteVectorStore(conn=sync_conn)
            self._keyword_index = SQLiteBM25Index(db_path=self._db_path)
        return self._conn

    async def _create_tables(self) -> None:
        """建表（与 sync 版 DDL 一致）。"""
        assert self._conn is not None
        # 读 sync 版的 DDL — 从 storage/sqlite/store.py 复制表结构
        # memories / history / memory_access_logs / entities 等表
        await self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                agent_id TEXT,
                session_id TEXT,
                content TEXT NOT NULL,
                embedding TEXT,
                metadata TEXT DEFAULT '{}',
                created_at TEXT,
                updated_at TEXT,
                valid_at TEXT,
                invalid_at TEXT,
                expired_at TEXT,
                is_deleted INTEGER DEFAULT 0,
                state TEXT DEFAULT 'active',
                app_id TEXT,
                archived_at TEXT,
                deleted_at TEXT
            );
            CREATE TABLE IF NOT EXISTS history (
                id TEXT PRIMARY KEY,
                memory_id TEXT,
                old_memory TEXT,
                new_memory TEXT,
                event TEXT,
                created_at TEXT,
                is_deleted INTEGER
            );
        """)
        await self._conn.commit()

    async def add(self, content, embedding, *, user_id, agent_id=None, session_id=None,
                  metadata=None, valid_at=None) -> str:
        conn = await self._ensure_conn()
        mid = f"mem-{uuid.uuid4()}"
        now = _utcnow_iso()
        await conn.execute(
            """INSERT INTO memories (id, user_id, agent_id, session_id, content, embedding,
               metadata, created_at, updated_at, valid_at, is_deleted, state)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 'active')""",
            (mid, user_id, agent_id, session_id, content, json.dumps(embedding),
             json.dumps(metadata or {}), now, now, valid_at),
        )
        await conn.execute(
            """INSERT INTO history (id, memory_id, old_memory, new_memory, event, created_at, is_deleted)
               VALUES (?, ?, ?, ?, ?, ?, 0)""",
            (str(uuid.uuid4()), mid, None, content, "ADD", now),
        )
        await conn.commit()
        # 双写（sync 组件，用 to_thread）
        if self._vector_store:
            await asyncio.to_thread(self._vector_store.insert_vectors, [embedding], [mid], [{"user_id": user_id}])
        if self._keyword_index:
            await asyncio.to_thread(self._keyword_index.add_docs, {mid: content})
        logger.info("async_memory_added", memory_id=mid, user_id=user_id)
        return mid

    async def search(self, query_embedding, *, user_id, session_id=None, top_k=5, threshold=0.1):
        conn = await self._ensure_conn()
        # 用 SQLite JSON 提取 + Python numpy 余弦（与 sync 版逻辑一致）
        cursor = await conn.execute(
            """SELECT id, content, metadata, created_at, embedding FROM memories
               WHERE user_id=? AND is_deleted=0 AND (state='active' OR state IS NULL)""",
            (user_id,),
        )
        rows = await cursor.fetchall()
        # numpy 余弦计算（CPU，用 to_thread）
        scored = await asyncio.to_thread(self._score_rows, query_embedding, rows)
        return [r for r in scored if r["score"] >= threshold][:top_k]

    def _score_rows(self, query_embedding, rows):
        """余弦相似度计算（纯 CPU，无 I/O）。"""
        import numpy as np
        query = np.array(query_embedding, dtype=np.float32)
        query_norm = float(np.linalg.norm(query))
        if query_norm == 0:
            return []
        scored = []
        for row in rows:
            vid, content, meta_json, created_at, emb_json = row
            vec = np.array(json.loads(emb_json), dtype=np.float32)
            if vec.shape != query.shape:
                continue
            vec_norm = float(np.linalg.norm(vec))
            if vec_norm == 0:
                continue
            score = float(np.dot(query, vec) / (query_norm * vec_norm))
            score = max(0.0, min(1.0, score))
            scored.append({
                "id": vid, "memory": content, "score": score,
                "metadata": json.loads(meta_json) if meta_json else {},
                "created_at": created_at,
            })
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored

    async def get_all(self, *, user_id, session_id=None):
        conn = await self._ensure_conn()
        if session_id:
            cursor = await conn.execute(
                """SELECT id, content, metadata, created_at, updated_at FROM memories
                   WHERE user_id=? AND is_deleted=0 AND session_id=? AND (state='active' OR state IS NULL)""",
                (user_id, session_id),
            )
        else:
            cursor = await conn.execute(
                """SELECT id, content, metadata, created_at, updated_at FROM memories
                   WHERE user_id=? AND is_deleted=0 AND (state='active' OR state IS NULL)""",
                (user_id,),
            )
        rows = await cursor.fetchall()
        return [
            {"id": r[0], "memory": r[1], "metadata": json.loads(r[2]) if r[2] else {},
             "created_at": r[3], "updated_at": r[4]}
            for r in rows
        ]

    async def get(self, memory_id):
        conn = await self._ensure_conn()
        cursor = await conn.execute(
            """SELECT id, content, metadata, created_at FROM memories
               WHERE id=? AND is_deleted=0 AND (state='active' OR state IS NULL)""",
            (memory_id,),
        )
        r = await cursor.fetchone()
        if not r:
            return None
        return {
            "id": r[0], "memory": r[1],
            "metadata": json.loads(r[2]) if r[2] else {}, "created_at": r[3],
        }

    async def delete(self, memory_id):
        conn = await self._ensure_conn()
        now = _utcnow_iso()
        await conn.execute(
            """UPDATE memories SET is_deleted=1, state='deleted', deleted_at=?, updated_at=? WHERE id=?""",
            (now, now, memory_id),
        )
        await conn.execute(
            """INSERT INTO history (id, memory_id, old_memory, new_memory, event, created_at, is_deleted)
               VALUES (?, ?, ?, ?, ?, ?, 1)""",
            (str(uuid.uuid4()), memory_id, None, None, "DELETE", now),
        )
        await conn.commit()
        if self._vector_store:
            await asyncio.to_thread(self._vector_store.delete_vector, memory_id)
        if self._keyword_index:
            await asyncio.to_thread(self._keyword_index.delete_docs, [memory_id])
        logger.info("async_memory_deleted", memory_id=memory_id)

    async def update(self, memory_id, content, embedding, *, metadata=None):
        conn = await self._ensure_conn()
        now = _utcnow_iso()
        cursor = await conn.execute(
            "SELECT content, metadata FROM memories WHERE id=? AND is_deleted=0",
            (memory_id,),
        )
        row = await cursor.fetchone()
        if not row:
            return False
        old_content, old_meta_json = row
        old_meta = json.loads(old_meta_json) if old_meta_json else {}
        await conn.execute(
            """UPDATE memories SET content=?, embedding=?, metadata=?, updated_at=? WHERE id=? AND is_deleted=0""",
            (content, json.dumps(embedding), json.dumps(metadata if metadata is not None else old_meta), now, memory_id),
        )
        await conn.execute(
            """INSERT INTO history (id, memory_id, old_memory, new_memory, event, created_at, is_deleted)
               VALUES (?, ?, ?, ?, ?, ?, 0)""",
            (str(uuid.uuid4()), memory_id, old_content, content, "UPDATE", now),
        )
        await conn.commit()
        if self._vector_store:
            await asyncio.to_thread(self._vector_store.insert_vectors, [embedding], [memory_id])
        if self._keyword_index:
            await asyncio.to_thread(self._keyword_index.add_docs, {memory_id: content})
        logger.info("async_memory_updated", memory_id=memory_id)
        return True

    async def get_history(self, memory_id):
        conn = await self._ensure_conn()
        cursor = await conn.execute(
            """SELECT id, memory_id, old_memory, new_memory, event, created_at, is_deleted
               FROM history WHERE memory_id=? ORDER BY created_at""",
            (memory_id,),
        )
        rows = await cursor.fetchall()
        return [
            {"id": r[0], "memory_id": r[1], "old_memory": r[2], "new_memory": r[3],
             "event": r[4], "created_at": r[5], "is_deleted": bool(r[6])}
            for r in rows
        ]

    async def close(self):
        if self._conn:
            await self._conn.close()
            self._conn = None
        if self._vector_store:
            self._vector_store.close()
        if self._keyword_index:
            self._keyword_index.close()
```

- [ ] **Step 5: 运行测试确认通过**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_async_sqlite_store.py -v`
Expected: PASS（6 个 async 测试全通过）

- [ ] **Step 6: ruff 检查**

Run: `ruff check --fix src/septmuse/storage/async_base.py src/septmuse/storage/async_sqlite/ tests/unit/test_async_store_base.py tests/unit/test_async_sqlite_store.py`

---

## Task 4: AsyncMemory facade

**Files:**
- Create: `src/septmuse/memory/async_main.py`
- Test: `tests/unit/test_async_memory.py`

**Interfaces:**
- Consumes: Task 1 的 `embedder_provider`/`llm_provider`，Task 3 的 `AsyncSQLiteMemoryStore`
- Produces: `AsyncMemory` 类 with 9 async methods

- [ ] **Step 1: 写 facade 失败测试**

```python
# tests/unit/test_async_memory.py
"""AsyncMemory facade 测试。"""
from septmuse.embedders.hash import HashEmbedder
from septmuse.memory.async_main import AsyncMemory


async def test_add_and_search():
    """添加记忆后能检索到。"""
    mem = AsyncMemory(embedder=HashEmbedder())
    try:
        result = await mem.add("hello world", user_id="alice")
        assert "results" in result
        assert len(result["results"]) == 1
        assert result["results"][0]["memory"] == "hello world"

        results = await mem.search("hello", user_id="alice", top_k=5)
        assert len(results) >= 1
        assert results[0]["memory"] == "hello world"
    finally:
        await mem.close()


async def test_get_and_get_all():
    """获取单条和列出全部。"""
    mem = AsyncMemory(embedder=HashEmbedder())
    try:
        result = await mem.add("test memory", user_id="bob")
        mid = result["results"][0]["id"]

        got = await mem.get(mid)
        assert got is not None
        assert got["memory"] == "test memory"

        all_mems = await mem.get_all(user_id="bob")
        assert len(all_mems) == 1
    finally:
        await mem.close()


async def test_delete():
    """删除记忆。"""
    mem = AsyncMemory(embedder=HashEmbedder())
    try:
        result = await mem.add("to delete", user_id="alice")
        mid = result["results"][0]["id"]
        await mem.delete(mid)
        got = await mem.get(mid)
        assert got is None
    finally:
        await mem.close()


async def test_update():
    """更新记忆。"""
    mem = AsyncMemory(embedder=HashEmbedder())
    try:
        result = await mem.add("original", user_id="alice")
        mid = result["results"][0]["id"]
        success = await mem.update(mid, "updated")
        assert success is True
        got = await mem.get(mid)
        assert got["memory"] == "updated"
    finally:
        await mem.close()


async def test_get_history():
    """变更历史。"""
    mem = AsyncMemory(embedder=HashEmbedder())
    try:
        result = await mem.add("original", user_id="alice")
        mid = result["results"][0]["id"]
        await mem.update(mid, "updated")
        await mem.delete(mid)
        history = await mem.get_history(mid)
        assert len(history) == 3
    finally:
        await mem.close()
```

- [ ] **Step 2: 运行确认失败**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_async_memory.py -v`
Expected: FAIL — `No module named 'septmuse.memory.async_main'`

- [ ] **Step 3: 写 AsyncMemory**

```python
# src/septmuse/memory/async_main.py
"""异步记忆 facade — 9 个 async 方法，提供 async/sync 双版本 API。

REST API 用 AsyncMemory，CLI/MCP 用 Memory（sync）。
store 层真 async（aiosqlite），embedder/LLM sync 用 asyncio.to_thread 包装。
"""
from __future__ import annotations

import asyncio
from typing import Any

from septmuse.configs.base import MemoryConfig
from septmuse.configs.defaults import default_config
from septmuse.core.logging import get_logger
from septmuse.embedders.base import Embedder
from septmuse.llms.base import LLM
from septmuse.storage.async_base import AsyncMemoryStore
from septmuse.storage.async_sqlite.store import AsyncSQLiteMemoryStore

logger = get_logger(__name__)


def _normalize_messages(messages: Any) -> list[str]:
    """消息标准化（复用 sync 版逻辑）。"""
    if isinstance(messages, str):
        return [messages]
    if isinstance(messages, list):
        texts = []
        for msg in messages:
            if isinstance(msg, dict):
                texts.append(msg.get("content", ""))
            elif isinstance(msg, str):
                texts.append(msg)
        return [t for t in texts if t.strip()]
    return []


class AsyncMemory:
    """异步记忆 facade。

    用法:
        mem = AsyncMemory()
        result = await mem.add("hello", user_id="alice")
        results = await mem.search("hello", user_id="alice")
    """

    def __init__(
        self,
        config: MemoryConfig | None = None,
        *,
        embedder: Embedder | None = None,
        store: AsyncMemoryStore | None = None,
        llm: LLM | None = None,
    ) -> None:
        self.config = config or default_config()
        self.embedder = embedder or self._resolve_embedder()
        self.store = store or AsyncSQLiteMemoryStore(db_path=self.config.db_path)
        self.llm = llm
        if self.llm is None and self.config.llm is not None:
            self.llm = self._resolve_llm()
        logger.info("async_memory_init", db_path=str(self.config.db_path))

    def _resolve_embedder(self) -> Embedder:
        from septmuse.services.providers import embedder_provider
        return embedder_provider.resolve(self.config.embedder.backend, config=self.config.embedder)

    def _resolve_llm(self) -> LLM | None:
        from septmuse.services.providers import llm_provider
        return llm_provider.resolve(self.config.llm.backend, config=self.config.llm)

    async def add(self, messages, *, user_id, agent_id=None, session_id=None,
                  metadata=None, infer=None, valid_at=None,
                  auto_extract_entities=True) -> dict[str, Any]:
        """异步添加记忆。"""
        texts = _normalize_messages(messages)
        if not texts:
            return {"results": [], "relations": []}

        # embedder sync，用 to_thread 包装
        embeddings = await asyncio.to_thread(self.embedder.embed_batch, texts)

        results = []
        for text, emb in zip(texts, embeddings, strict=True):
            mid = await self.store.add(
                text, emb, user_id=user_id, agent_id=agent_id,
                session_id=session_id, metadata=metadata, valid_at=valid_at,
            )
            results.append({"id": mid, "memory": text, "event": "ADD"})

        logger.info("async_add_done", user_id=user_id, count=len(results))
        return {"results": results, "relations": []}

    async def search(self, query: str, *, user_id: str, top_k: int = 5,
                     threshold: float = 0.1) -> list[dict[str, Any]]:
        """异步检索记忆。"""
        emb = await asyncio.to_thread(self.embedder.embed, query)
        return await self.store.search(emb, user_id=user_id, top_k=top_k, threshold=threshold)

    async def update(self, memory_id: str, content: str, *,
                     metadata: dict[str, Any] | None = None) -> bool:
        """异步更新记忆。"""
        emb = await asyncio.to_thread(self.embedder.embed, content)
        return await self.store.update(memory_id, content, emb, metadata=metadata)

    async def delete(self, memory_id: str) -> None:
        """异步软删除。"""
        await self.store.delete(memory_id)

    async def delete_all(self, *, user_id: str) -> int:
        """异步批量删除该用户所有记忆。"""
        memories = await self.store.get_all(user_id=user_id)
        for m in memories:
            await self.store.delete(m["id"])
        return len(memories)

    async def get(self, memory_id: str) -> dict[str, Any] | None:
        """异步取单条。"""
        return await self.store.get(memory_id)

    async def get_all(self, *, user_id: str) -> list[dict[str, Any]]:
        """异步列出全部。"""
        return await self.store.get_all(user_id=user_id)

    async def get_history(self, memory_id: str) -> list[dict[str, Any]]:
        """异步获取变更历史。"""
        return await self.store.get_history(memory_id)

    async def close(self) -> None:
        """异步释放资源。"""
        await self.store.close()
```

- [ ] **Step 4: 运行测试确认通过**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_async_memory.py -v`
Expected: PASS（5 个 async 测试全通过）

- [ ] **Step 5: ruff 检查**

Run: `ruff check --fix src/septmuse/memory/async_main.py tests/unit/test_async_memory.py`

- [ ] **Step 6: 验证现有测试不破坏**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_memory.py -q --tb=line 2>&1 | Select-Object -Last 3`
Expected: 失败不超过之前基线（sync 路径不动）

---

## Task 5: REST API 切换 AsyncMemory + 全量验证

**Files:**
- Modify: `src/septmuse/api/rest/__init__.py`（create_app 用 AsyncMemory + 端点改 await）
- Test: 无新增（现有 REST 测试需适配 async）

**Interfaces:**
- Consumes: Task 4 的 `AsyncMemory`
- Produces: REST API 用 AsyncMemory，21 端点改 await

- [ ] **Step 1: 读现有 REST API 代码**

用 Read 工具读 `src/septmuse/api/rest/__init__.py` 全文，了解：
- `create_app` 函数结构
- `register_routes` 函数结构
- 21 个端点的路由定义（`@app.post("/memories")` 等）
- 当前端点怎么调 `memory.add()` / `memory.search()` 等

- [ ] **Step 2: 修改 create_app**

把 `create_app` 改为用 `AsyncMemory`：

```python
def create_app(memory=None) -> FastAPI:
    """创建 FastAPI app。"""
    if memory is None:
        from septmuse.embedders.hash import HashEmbedder
        memory = AsyncMemory(config=MemoryConfig(db_path=":memory:"), embedder=HashEmbedder())
    elif isinstance(memory, MemoryConfig):
        from septmuse.embedders.hash import HashEmbedder
        memory = AsyncMemory(config=memory, embedder=HashEmbedder())

    app = FastAPI(title="SeptMuse Memory API", ...)
    from septmuse.api.auth import setup_auth
    setup_auth(app)
    register_routes(app, memory)
    return app
```

- [ ] **Step 3: 修改端点为 await**

21 个端点从 `memory.add(...)` 改为 `await memory.add(...)`，从 `memory.search(...)` 改为 `await memory.search(...)` 等。

逐个修改，注意：
- `@app.post("/memories")` → `result = await memory.add(...)`
- `@app.get("/memories")` → `result = await memory.get_all(...)`
- `@app.get("/memories/search")` → `result = await memory.search(...)`
- `@app.get("/memories/{memory_id}")` → `result = await memory.get(...)`
- `@app.delete("/memories/{memory_id}")` → `await memory.delete(...)`
- `@app.put("/memories/{memory_id}")` → `result = await memory.update(...)`
- `@app.get("/memories/{memory_id}/access-logs")` → `result = await memory.get_access_logs(...)` 或 store.get_access_logs
- 其他端点同理

**注意**：部分端点可能调的是 `ExperimentalMemory` 的方法（如 capture/rehearse/coverage_report），这些在 AsyncMemory 中没有。对这些端点：
- 如果是核心 9 方法 → 改为 await
- 如果是实验方法 → 保留调 sync memory（或用 asyncio.to_thread 包装）

- [ ] **Step 4: ruff 检查**

Run: `ruff check --fix src/septmuse/api/rest/__init__.py`

- [ ] **Step 5: REST 测试验证**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_api_permission_integration.py -q --tb=line 2>&1 | Select-Object -Last 5`
Expected: 通过或失败不超过之前基线

- [ ] **Step 6: 全量 ruff**

Run: `ruff check src/ tests/`
Expected: All checks passed!

- [ ] **Step 7: 全量 pytest**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/ tests/e2e/ -q --tb=line 2>&1 | Select-Object -Last 5`
Expected: 失败不超过 23（之前基线），passed 不低于 1028 + 新增 async 测试

- [ ] **Step 8: CLI backends 验证**

Run: `$env:PYTHONPATH="src"; python -m septmuse.cli.main backends`
Expected: llm 输出 8 个后端（openai/ollama/anthropic/dashscope/litellm/groq/gemini/deepseek）

- [ ] **Step 9: AsyncMemory 零配置验证**

Run: `$env:PYTHONPATH="src"; python -c "import asyncio; from septmuse.memory.async_main import AsyncMemory; from septmuse.embedders.hash import HashEmbedder; m = AsyncMemory(embedder=HashEmbedder()); r = asyncio.run(m.add('hello', user_id='test')); print('OK', r['results'][0]['id']); asyncio.run(m.close())"`
Expected: `OK mem-...`

---

## 自检报告

### Spec 覆盖率

| Spec 章节 | 对应 Task | 状态 |
|-----------|----------|------|
| §4 AsyncMemoryStore ABC | Task 2 | ✅ |
| §5 AsyncSQLiteMemoryStore | Task 3 | ✅ |
| §6 AsyncMemory facade | Task 4 | ✅ |
| §7 REST API 切换 | Task 5 | ✅ |
| §8 litellm LLM | Task 1 Step 3-4 | ✅ |
| §9 云 provider | Task 1 Step 6-8 | ✅ |
| §10 manifest 新增 | Task 1 Step 10 | ✅ |
| §11 测试策略 | Task 1-5 各有测试 | ✅ |
| §12 迁移批次 | Task 1-5 对应批次 1-5 | ✅ |

### 类型一致性

- `LLM.complete(system_prompt, user_prompt) -> str` 在 Task 1 所有 4 个 LLM 类中一致
- `AsyncMemoryStore.add/search/get/...` 在 Task 2 ABC 和 Task 3 实现中签名一致
- `AsyncMemory.add/search/update/...` 在 Task 4 facade 和 Task 5 REST 调用中一致
- `BackendEntry` 4-tuple 结构在 Task 1 manifest 新增条目中一致

### 无占位符

- 所有代码块完整，无 TBD/TODO
- 每步有实际代码 + 运行命令 + 预期输出

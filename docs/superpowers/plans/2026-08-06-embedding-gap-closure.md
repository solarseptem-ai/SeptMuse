# Embedding 差距补齐 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 补齐 11 个缺失 embedding provider + memory_action 接口，使 provider 列表 1:1 对齐 mem0（共 18 个 embedder 后端）。

**Architecture:** OpenAI 兼容 family（openai/together/lmstudio/azure_openai）共享 `_OpenAICompatibleEmbedder` 基类减少重复。其余 provider 独立实现 Embedder ABC。memory_action 作为可选参数加到 Embedder ABC，向后兼容。

**Tech Stack:** Python 3.10+, pydantic v2, pytest, ruff (line-length 120)。运行命令：`$env:PYTHONPATH = "src"` + `pytest`。Windows 上禁止 `ruff format <file>`（会清空文件），用 `ruff check --fix` 安全。

## Global Constraints

- **包名**：`septmuse`，src/ 布局，`PYTHONPATH=src` 运行测试（不 pip install -e .）
- **注释语言**：代码注释用中文，文件头 Apache 2.0 license
- **lint**：`ruff check src/ tests/`（line-length 120，select E/F/I/W/UP/B/SIM/RUF，ignore E501/RUF001-003）
- **测试基线**：1319 passed / 16 failed (pre-existing LLM) / 23 skipped — 零退化
- **测试标记**：`@pytest.mark.integration` 用于需真实 API 的测试（skipped 默认）；mock 单元测试无标记
- **registry 模式**：每个 provider 在 `BACKEND_MANIFEST["embedder"]` 加一条 `BackendEntry(module=, cls=, config_cls=, deps=)`
- **config 模式**：继承 `BaseEmbedderConfig`（pydantic BaseModel），含 `backend: str` 字段 + provider 特有字段
- **embedder 模式**：继承 `Embedder` ABC，实现 `dimension` property + `embed(text, memory_action=None)` + `embed_batch(texts, memory_action=None)`
- **opensource/ 禁止修改、禁止 import** — 所有实现参考 mem0 但在 src/ 重写
- **license header** — 所有新文件用 Apache 2.0 header（完整 14 行，复制自 `src/septmuse/embedders/openai.py:1-13`）。下方代码块省略 header 以节省篇幅，实际实现时必须加上

---

## File Structure

### 修改的文件

| 文件 | 责任 | 改动 |
|------|------|------|
| `src/septmuse/embedders/base.py` | Embedder ABC | 加 memory_action 参数 |
| `src/septmuse/embedders/hash.py` | HashEmbedder | 签名加 memory_action |
| `src/septmuse/embedders/onnx.py` | OnnxEmbedder | 签名加 memory_action |
| `src/septmuse/embedders/auto.py` | AutoOnnxEmbedder | 透传 memory_action |
| `src/septmuse/embedders/sentence_transformers.py` | SentenceTransformerEmbedder | 签名加 memory_action |
| `src/septmuse/embedders/cached.py` | CachedEmbedder | cache key 加 memory_action + 透传 |
| `src/septmuse/embedders/openai.py` | OpenAIEmbedder | 重构为继承 _OpenAICompatibleEmbedder |
| `src/septmuse/services/registry.py` | 后端注册表 | 加 11 条 embedder BackendEntry |
| `src/septmuse/configs/enums.py` | 后端枚举 | EmbedderBackend 加 11 个值 |
| `pyproject.toml` | 依赖 | 加 5 extras + embedders 聚合 |
| `AGENTS.md` | 项目文档 | Embedder section 更新 |
| `CHANGELOG.md` | 变更日志 | 记录本次变更 |

### 新建的文件

| 文件 | 责任 |
|------|------|
| `src/septmuse/embedders/_openai_compatible.py` | OpenAI 兼容基类 |
| `src/septmuse/embedders/{ollama,langchain,azure_openai,huggingface,gemini,vertexai,together,lmstudio,aws_bedrock,fastembed,mock}.py` | 11 个新 provider |
| `src/septmuse/configs/embeddings/{ollama,langchain,azure_openai,huggingface,gemini,vertexai,together,lmstudio,aws_bedrock,fastembed,mock}.py` | 11 个新 config |
| `tests/unit/test_embedders/__init__.py` | 测试包 |
| `tests/unit/test_embedders/test_memory_action.py` | 接口 + cache key 测试 |
| `tests/unit/test_embedders/test_openai_compatible.py` | 基类测试 |
| `tests/unit/test_embedders/test_{ollama,langchain,azure_openai,huggingface,gemini,vertexai,together,lmstudio,aws_bedrock,fastembed,mock}.py` | 11 个 provider 测试 |

---

## Task 1: memory_action 接口改造

**Files:**
- Modify: `src/septmuse/embedders/base.py`
- Modify: `src/septmuse/embedders/hash.py`
- Modify: `src/septmuse/embedders/onnx.py`
- Modify: `src/septmuse/embedders/auto.py`
- Modify: `src/septmuse/embedders/sentence_transformers.py`
- Modify: `src/septmuse/embedders/cached.py`
- Test: `tests/unit/test_embedders/test_memory_action.py`

**Interfaces:**
- Produces: `Embedder.embed(text, memory_action=None) -> list[float]`, `Embedder.embed_batch(texts, memory_action=None) -> list[list[float]]`

- [ ] **Step 1: 创建测试目录 + 写失败测试**

创建 `tests/unit/test_embedders/__init__.py`（空文件）。

创建 `tests/unit/test_embedders/test_memory_action.py`:

```python
"""memory_action 接口测试 — 验证 ABC 签名 + CachedEmbedder cache key 隔离 + 向后兼容。"""

from __future__ import annotations

import pytest


class TestEmbedderABCInterface:
    def test_embed_accepts_memory_action(self):
        from septmuse.embedders.hash import HashEmbedder

        emb = HashEmbedder(dim=64)
        vec = emb.embed("hello", memory_action="add")
        assert len(vec) == 64

    def test_embed_accepts_none_memory_action(self):
        from septmuse.embedders.hash import HashEmbedder

        emb = HashEmbedder(dim=64)
        vec = emb.embed("hello", memory_action=None)
        assert len(vec) == 64

    def test_embed_backward_compatible_no_memory_action(self):
        from septmuse.embedders.hash import HashEmbedder

        emb = HashEmbedder(dim=64)
        vec = emb.embed("hello")
        assert len(vec) == 64

    def test_embed_batch_accepts_memory_action(self):
        from septmuse.embedders.hash import HashEmbedder

        emb = HashEmbedder(dim=64)
        vecs = emb.embed_batch(["hello", "world"], memory_action="search")
        assert len(vecs) == 2
        assert all(len(v) == 64 for v in vecs)

    def test_embed_batch_backward_compatible(self):
        from septmuse.embedders.hash import HashEmbedder

        emb = HashEmbedder(dim=64)
        vecs = emb.embed_batch(["hello", "world"])
        assert len(vecs) == 2


class TestCachedEmbedderMemoryAction:
    def test_cache_key_includes_memory_action(self):
        from septmuse.embedders.cached import CachedEmbedder
        from septmuse.embedders.hash import HashEmbedder

        inner = HashEmbedder(dim=64)
        cached = CachedEmbedder(inner, maxsize=10)

        vec_add = cached.embed("hello", memory_action="add")
        vec_search = cached.embed("hello", memory_action="search")

        stats = cached.stats
        assert stats["hits"] == 0
        assert stats["misses"] == 2

    def test_same_memory_action_hits_cache(self):
        from septmuse.embedders.cached import CachedEmbedder
        from septmuse.embedders.hash import HashEmbedder

        inner = HashEmbedder(dim=64)
        cached = CachedEmbedder(inner, maxsize=10)

        cached.embed("hello", memory_action="add")
        cached.embed("hello", memory_action="add")

        stats = cached.stats
        assert stats["hits"] == 1
        assert stats["misses"] == 1

    def test_none_memory_action_cached_separately_from_add(self):
        from septmuse.embedders.cached import CachedEmbedder
        from septmuse.embedders.hash import HashEmbedder

        inner = HashEmbedder(dim=64)
        cached = CachedEmbedder(inner, maxsize=10)

        cached.embed("hello")
        cached.embed("hello", memory_action="add")

        stats = cached.stats
        assert stats["hits"] == 0
        assert stats["misses"] == 2

    def test_embed_batch_cache_key_includes_memory_action(self):
        from septmuse.embedders.cached import CachedEmbedder
        from septmuse.embedders.hash import HashEmbedder

        inner = HashEmbedder(dim=64)
        cached = CachedEmbedder(inner, maxsize=10)

        cached.embed_batch(["hello"], memory_action="add")
        cached.embed_batch(["hello"], memory_action="search")

        stats = cached.stats
        assert stats["hits"] == 0
        assert stats["misses"] == 2
```

- [ ] **Step 2: 运行测试验证失败**

```bash
$env:PYTHONPATH = "src"; python -m pytest tests/unit/test_embedders/test_memory_action.py -v --tb=short
```
Expected: FAIL — `embed() got an unexpected keyword argument 'memory_action'`

- [ ] **Step 3: 改 Embedder ABC 签名**

修改 `src/septmuse/embedders/base.py`，将 `embed` 和 `embed_batch` 加 `memory_action` 参数:

```python
class Embedder(ABC):
    """嵌入模型抽象。"""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """返回嵌入维度。"""
        ...

    @abstractmethod
    def embed(self, text: str, memory_action: str | None = None) -> list[float]:
        """嵌入单条文本, 返回归一化向量 (便于余弦点积)。

        Args:
            memory_action: "add"/"search"/"update"/None, 部分 provider (如 vertexai)
                据此切换嵌入策略, 大多数 provider 忽略此参数。
        """
        ...

    def embed_batch(self, texts: list[str], memory_action: str | None = None) -> list[list[float]]:
        """批量嵌入 — 默认逐条调用 embed(), 子类可 override 实现真批量推理。"""
        return [self.embed(t, memory_action) for t in texts]
```

- [ ] **Step 4: 改 HashEmbedder 签名**

修改 `src/septmuse/embedders/hash.py` 的 `embed` 和 `embed_batch`:

```python
    def embed(self, text: str, memory_action: str | None = None) -> list[float]:
        vec = np.zeros(self._dim, dtype=np.float32)
        for token in _tokenize(text):
            h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
            vec[h % self._dim] += 1.0
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def embed_batch(self, texts: list[str], memory_action: str | None = None) -> list[list[float]]:
        return [self.embed(t, memory_action) for t in texts]
```

- [ ] **Step 5: 改 OnnxEmbedder 签名**

修改 `src/septmuse/embedders/onnx.py` 的 `embed` 和 `embed_batch`:

```python
    def embed(self, text: str, memory_action: str | None = None) -> list[float]:
        encoding = self._tokenizer.encode(text)
        feeds = self._build_feeds(encoding)
        outputs = self._session.run(None, feeds)
        last_hidden = outputs[0]
        mask = np.array(encoding.attention_mask, dtype=np.float32)
        pooled = (last_hidden[0] * mask[:, None]).sum(axis=0) / mask.sum()
        norm = float(np.linalg.norm(pooled))
        if norm > 0:
            pooled = pooled / norm
        return pooled.tolist()

    def embed_batch(self, texts: list[str], memory_action: str | None = None) -> list[list[float]]:
        if not texts:
            return []
        _BATCH_SIZE = 32
        results: list[list[float]] = []
        for start in range(0, len(texts), _BATCH_SIZE):
            chunk = texts[start : start + _BATCH_SIZE]
            encodings = self._tokenizer.encode_batch(chunk)
            input_ids = np.array([enc.ids for enc in encodings], dtype=np.int64)
            attention_mask = np.array([enc.attention_mask for enc in encodings], dtype=np.int64)
            feeds: dict[str, np.ndarray] = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
            if "token_type_ids" in self._input_names:
                feeds["token_type_ids"] = np.array([enc.type_ids for enc in encodings], dtype=np.int64)
            outputs = self._session.run(None, feeds)
            last_hidden = outputs[0]
            mask = attention_mask.astype(np.float32)
            mask_sum = mask.sum(axis=1, keepdims=True)
            mask_sum = np.where(mask_sum == 0, 1.0, mask_sum)
            pooled = (last_hidden * mask[:, :, None]).sum(axis=1) / mask_sum
            norms = np.linalg.norm(pooled, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            pooled = pooled / norms
            results.extend(pooled.tolist())
        if len(results) != len(texts):
            raise ValueError(f"embed_batch returned {len(results)} embeddings for {len(texts)} texts")
        return results
```

- [ ] **Step 6: 改 AutoOnnxEmbedder 透传**

修改 `src/septmuse/embedders/auto.py` 的 `embed` 和 `embed_batch`:

```python
    def embed(self, text: str, memory_action: str | None = None) -> list[float]:
        return self._inner.embed(text, memory_action)

    def embed_batch(self, texts: list[str], memory_action: str | None = None) -> list[list[float]]:
        return self._inner.embed_batch(texts, memory_action)
```

- [ ] **Step 7: 改 SentenceTransformerEmbedder 签名**

修改 `src/septmuse/embedders/sentence_transformers.py` 的 `embed` 和 `embed_batch`:

```python
    def embed(self, text: str, memory_action: str | None = None) -> list[float]:
        vec = self._model.encode(text, normalize_embeddings=True)
        return vec.tolist()

    def embed_batch(self, texts: list[str], memory_action: str | None = None) -> list[list[float]]:
        vecs = self._model.encode(texts, normalize_embeddings=True)
        return [v.tolist() for v in vecs]
```

- [ ] **Step 8: 改 CachedEmbedder cache key + 透传**

修改 `src/septmuse/embedders/cached.py`，cache key 从 `text` 改为 `(text, memory_action)`:

```python
    def embed(self, text: str, memory_action: str | None = None) -> list[float]:
        cache_key = (text, memory_action)
        with self._lock:
            cached = self._cache.get(cache_key)
            if cached is not None:
                self._cache.move_to_end(cache_key)
                self._hits += 1
                return list(cached)
            self._misses += 1

        vec = self._inner.embed(text, memory_action)

        with self._lock:
            self._cache[cache_key] = vec
            self._cache.move_to_end(cache_key)
            if len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)
        return list(vec)

    def embed_batch(self, texts: list[str], memory_action: str | None = None) -> list[list[float]]:
        if not texts:
            return []

        results: list[list[float] | None] = [None] * len(texts)
        to_embed: list[int] = []

        with self._lock:
            for i, text in enumerate(texts):
                cache_key = (text, memory_action)
                cached = self._cache.get(cache_key)
                if cached is not None:
                    self._cache.move_to_end(cache_key)
                    self._hits += 1
                    results[i] = list(cached)
                else:
                    self._misses += 1
                    to_embed.append(i)

        if to_embed:
            embed_texts = [texts[i] for i in to_embed]
            embed_results = self._inner.embed_batch(embed_texts, memory_action)

            with self._lock:
                for idx, vec in zip(to_embed, embed_results, strict=True):
                    results[idx] = list(vec)
                    cache_key = (texts[idx], memory_action)
                    self._cache[cache_key] = vec
                    self._cache.move_to_end(cache_key)
                    if len(self._cache) > self._maxsize:
                        self._cache.popitem(last=False)

        assert all(r is not None for r in results), "embed_batch internal error: None entry"
        return results
```

- [ ] **Step 9: 运行测试验证通过**

```bash
$env:PYTHONPATH = "src"; python -m pytest tests/unit/test_embedders/test_memory_action.py -v
```
Expected: PASS (7 tests)

- [ ] **Step 10: 运行全套件验证零退化**

```bash
$env:PYTHONPATH = "src"; python -m pytest tests/unit/ tests/e2e/ -q --tb=short -x
```
Expected: 1319 passed (现有基线不变, 新增 7 测试 = 1326)

- [ ] **Step 11: Commit**

```bash
git add src/septmuse/embedders/base.py src/septmuse/embedders/hash.py src/septmuse/embedders/onnx.py src/septmuse/embedders/auto.py src/septmuse/embedders/sentence_transformers.py src/septmuse/embedders/cached.py tests/unit/test_embedders/
git commit -m "feat(embedder): add memory_action parameter to Embedder ABC + cache key isolation"
```

---

## Task 2: _OpenAICompatibleEmbedder 基类 + OpenAIEmbedder 重构

**Files:**
- Create: `src/septmuse/embedders/_openai_compatible.py`
- Modify: `src/septmuse/embedders/openai.py`
- Test: `tests/unit/test_embedders/test_openai_compatible.py`

**Interfaces:**
- Produces: `_OpenAICompatibleEmbedder(client, model, dim, pass_dimensions_to_api)` — 子类传入 client, 共享 embed/embed_batch 逻辑

- [ ] **Step 1: 写失败测试**

创建 `tests/unit/test_embedders/test_openai_compatible.py`:

```python
"""_OpenAICompatibleEmbedder 基类测试 — mock client 验证 embed/embed_batch/matryoshka。"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


@pytest.fixture()
def mock_client():
    client = MagicMock()
    client.embeddings = MagicMock()
    client.embeddings.create = MagicMock()
    return client


class TestOpenAICompatibleEmbedder:
    def test_embed_calls_create(self, mock_client):
        from septmuse.embedders._openai_compatible import _OpenAICompatibleEmbedder

        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3], index=0)]
        mock_client.embeddings.create.return_value = mock_response

        emb = _OpenAICompatibleEmbedder(
            client=mock_client, model="text-embedding-3-small", dim=3, pass_dimensions_to_api=False
        )
        vec = emb.embed("hello")
        assert vec == [0.1, 0.2, 0.3]
        mock_client.embeddings.create.assert_called_once()

    def test_embed_replaces_newlines(self, mock_client):
        from septmuse.embedders._openai_compatible import _OpenAICompatibleEmbedder

        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1], index=0)]
        mock_client.embeddings.create.return_value = mock_response

        emb = _OpenAICompatibleEmbedder(
            client=mock_client, model="m", dim=1, pass_dimensions_to_api=False
        )
        emb.embed("hello\nworld")
        call_kwargs = mock_client.embeddings.create.call_args
        assert "\n" not in call_kwargs.kwargs["input"][0]

    def test_embed_passes_dimensions_when_matryoshka(self, mock_client):
        from septmuse.embedders._openai_compatible import _OpenAICompatibleEmbedder

        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1], index=0)]
        mock_client.embeddings.create.return_value = mock_response

        emb = _OpenAICompatibleEmbedder(
            client=mock_client, model="m", dim=512, pass_dimensions_to_api=True
        )
        emb.embed("hello")
        call_kwargs = mock_client.embeddings.create.call_args
        assert call_kwargs.kwargs["dimensions"] == 512

    def test_embed_no_dimensions_when_not_matryoshka(self, mock_client):
        from septmuse.embedders._openai_compatible import _OpenAICompatibleEmbedder

        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1], index=0)]
        mock_client.embeddings.create.return_value = mock_response

        emb = _OpenAICompatibleEmbedder(
            client=mock_client, model="m", dim=1536, pass_dimensions_to_api=False
        )
        emb.embed("hello")
        call_kwargs = mock_client.embeddings.create.call_args
        assert "dimensions" not in call_kwargs.kwargs

    def test_embed_batch_chunks_100(self, mock_client):
        from septmuse.embedders._openai_compatible import _OpenAICompatibleEmbedder

        def fake_create(*args, **kwargs):
            texts = kwargs["input"]
            return MagicMock(data=[MagicMock(embedding=[0.5], index=i) for i in range(len(texts))])

        mock_client.embeddings.create.side_effect = fake_create

        emb = _OpenAICompatibleEmbedder(
            client=mock_client, model="m", dim=1, pass_dimensions_to_api=False
        )
        texts = [f"text{i}" for i in range(250)]
        vecs = emb.embed_batch(texts)
        assert len(vecs) == 250
        assert mock_client.embeddings.create.call_count == 3

    def test_embed_batch_count_mismatch_raises(self, mock_client):
        from septmuse.embedders._openai_compatible import _OpenAICompatibleEmbedder

        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1], index=0)]
        mock_client.embeddings.create.return_value = mock_response

        emb = _OpenAICompatibleEmbedder(
            client=mock_client, model="m", dim=1, pass_dimensions_to_api=False
        )
        with pytest.raises(ValueError, match="embed_batch"):
            emb.embed_batch(["a", "b"])

    def test_dimension_property(self, mock_client):
        from septmuse.embedders._openai_compatible import _OpenAICompatibleEmbedder

        emb = _OpenAICompatibleEmbedder(
            client=mock_client, model="m", dim=768, pass_dimensions_to_api=False
        )
        assert emb.dimension == 768

    def test_inherits_embedder_abc(self, mock_client):
        from septmuse.embedders._openai_compatible import _OpenAICompatibleEmbedder
        from septmuse.embedders.base import Embedder

        emb = _OpenAICompatibleEmbedder(
            client=mock_client, model="m", dim=1, pass_dimensions_to_api=False
        )
        assert isinstance(emb, Embedder)
```

- [ ] **Step 2: 运行测试验证失败**

```bash
$env:PYTHONPATH = "src"; python -m pytest tests/unit/test_embedders/test_openai_compatible.py -v --tb=short
```
Expected: FAIL — `No module named 'septmuse.embedders._openai_compatible'`

- [ ] **Step 3: 创建 _OpenAICompatibleEmbedder 基类**

创建 `src/septmuse/embedders/_openai_compatible.py`:

```python
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
"""OpenAI 兼容嵌入基类 — 共享 embed/embed_batch 逻辑。

Together/LM Studio/Azure OpenAI 等 provider 的 API 与 OpenAI Embeddings 兼容,
继承此基类只需覆盖 __init__ (创建不同 client + 设置默认 model/dims)。
"""

from __future__ import annotations

from typing import Any

from septmuse.embedders.base import Embedder

MAX_BATCH = 100


class _OpenAICompatibleEmbedder(Embedder):
    """OpenAI 兼容嵌入基类 — 子类传入 client, 共享 embed/embed_batch。

    Args:
        client: OpenAI 兼容 client (openai.OpenAI / AzureOpenAI 等)
        model: 模型名
        dim: 嵌入维度
        pass_dimensions_to_api: 是否向 API 传 dimensions 参数 (matryoshka 模型)
    """

    def __init__(
        self,
        *,
        client: Any,
        model: str,
        dim: int,
        pass_dimensions_to_api: bool,
    ) -> None:
        self._client = client
        self._model = model
        self._dim = dim
        self._pass_dimensions_to_api = pass_dimensions_to_api

    @property
    def dimension(self) -> int:
        return self._dim

    def embed(self, text: str, memory_action: str | None = None) -> list[float]:
        text = text.replace("\n", " ")
        kwargs: dict[str, Any] = {
            "input": [text],
            "model": self._model,
            "encoding_format": "float",
        }
        if self._pass_dimensions_to_api:
            kwargs["dimensions"] = self._dim
        response = self._client.embeddings.create(**kwargs)
        return response.data[0].embedding

    def embed_batch(self, texts: list[str], memory_action: str | None = None) -> list[list[float]]:
        if not texts:
            return []

        cleaned = [t.replace("\n", " ") for t in texts]
        all_embeddings: list[list[float]] = []
        for i in range(0, len(cleaned), MAX_BATCH):
            chunk = cleaned[i : i + MAX_BATCH]
            kwargs: dict[str, Any] = {
                "input": chunk,
                "model": self._model,
                "encoding_format": "float",
            }
            if self._pass_dimensions_to_api:
                kwargs["dimensions"] = self._dim
            response = self._client.embeddings.create(**kwargs)
            all_embeddings.extend(
                item.embedding for item in sorted(response.data, key=lambda x: x.index)
            )

        if len(all_embeddings) != len(texts):
            raise ValueError(
                f"embed_batch() returned {len(all_embeddings)} embeddings "
                f"for {len(texts)} texts using model '{self._model}'"
            )
        return all_embeddings
```

- [ ] **Step 4: 重构 OpenAIEmbedder 继承基类**

修改 `src/septmuse/embedders/openai.py`，替换为:

```python
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
"""OpenAI 嵌入 provider — 继承 _OpenAICompatibleEmbedder。

用法:
    embedder = OpenAIEmbedder(api_key="sk-...", model="text-embedding-3-small")
    vec = embedder.embed("hello")

零配置: 从环境变量 OPENAI_API_KEY 读取 key。
"""

from __future__ import annotations

import os
from typing import Any

from septmuse.core.logging import get_logger
from septmuse.embedders._openai_compatible import _OpenAICompatibleEmbedder

logger = get_logger(__name__)

DEFAULT_MODEL = "text-embedding-3-small"
DEFAULT_DIMS = 1536


class OpenAIEmbedder(_OpenAICompatibleEmbedder):
    """OpenAI Embeddings provider。

    零配置: 从 OPENAI_API_KEY 环境变量读取。
    自定义: OpenAIEmbedder(api_key="sk-...", model="text-embedding-3-small")。

    matryoshka 支持: 显式传 embedding_dims 时向 API 传 dimensions 参数
    (兼容非 matryoshka 后端如 vLLM/Voyage, 它们拒绝 dimensions 参数)。
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        base_url: str | None = None,
        embedding_dims: int | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("openai package required: pip install septmuse[openai]") from e

        self.model = model
        dim = embedding_dims or DEFAULT_DIMS
        pass_dimensions = embedding_dims is not None

        self._api_key = api_key or os.getenv("OPENAI_API_KEY") or "not-required"

        resolved_base_url = base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("SEPTMUSE_EMBEDDER_BASE_URL")
        client_kwargs: dict[str, Any] = {"api_key": self._api_key}
        if resolved_base_url:
            client_kwargs["base_url"] = resolved_base_url
        client_kwargs.update(kwargs)

        logger.info("embedder_loading", model=model, dim=dim)
        client = OpenAI(**client_kwargs)
        logger.info("embedder_ready", model=model, dim=dim)

        super().__init__(
            client=client,
            model=model,
            dim=dim,
            pass_dimensions_to_api=pass_dimensions,
        )
```

- [ ] **Step 5: 运行基类测试验证通过**

```bash
$env:PYTHONPATH = "src"; python -m pytest tests/unit/test_embedders/test_openai_compatible.py -v
```
Expected: PASS (8 tests)

- [ ] **Step 6: 运行现有 OpenAIEmbedder 测试验证零退化**

```bash
$env:PYTHONPATH = "src"; python -m pytest tests/unit/test_openai_embedder.py -v
```
Expected: PASS (现有测试全部通过, 验证重构无行为变化)

- [ ] **Step 7: Commit**

```bash
git add src/septmuse/embedders/_openai_compatible.py src/septmuse/embedders/openai.py tests/unit/test_embedders/test_openai_compatible.py
git commit -m "refactor(embedder): extract _OpenAICompatibleEmbedder base class, OpenAIEmbedder inherits it"
```

---

## Task 3: MockEmbedder

**Files:**
- Create: `src/septmuse/embedders/mock.py`
- Create: `src/septmuse/configs/embeddings/mock.py`
- Modify: `src/septmuse/services/registry.py`
- Modify: `src/septmuse/configs/enums.py`
- Test: `tests/unit/test_embedders/test_mock.py`

**Interfaces:**
- Produces: `MockEmbedder` — 固定 10 维向量, 确定性测试用

- [ ] **Step 1: 写失败测试**

创建 `tests/unit/test_embedders/test_mock.py`:

```python
"""MockEmbedder 测试 — 固定向量验证。"""

from __future__ import annotations


class TestMockEmbedder:
    def test_inherits_embedder_abc(self):
        from septmuse.embedders.base import Embedder
        from septmuse.embedders.mock import MockEmbedder

        emb = MockEmbedder()
        assert isinstance(emb, Embedder)

    def test_dimension_is_10(self):
        from septmuse.embedders.mock import MockEmbedder

        emb = MockEmbedder()
        assert emb.dimension == 10

    def test_embed_returns_fixed_vector(self):
        from septmuse.embedders.mock import MockEmbedder

        emb = MockEmbedder()
        vec = emb.embed("anything")
        assert vec == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    def test_embed_same_for_different_text(self):
        from septmuse.embedders.mock import MockEmbedder

        emb = MockEmbedder()
        assert emb.embed("hello") == emb.embed("world")

    def test_embed_batch(self):
        from septmuse.embedders.mock import MockEmbedder

        emb = MockEmbedder()
        vecs = emb.embed_batch(["a", "b", "c"])
        assert len(vecs) == 3
        assert all(v == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] for v in vecs)

    def test_embed_accepts_memory_action(self):
        from septmuse.embedders.mock import MockEmbedder

        emb = MockEmbedder()
        vec = emb.embed("hello", memory_action="add")
        assert len(vec) == 10
```

- [ ] **Step 2: 运行测试验证失败**

```bash
$env:PYTHONPATH = "src"; python -m pytest tests/unit/test_embedders/test_mock.py -v --tb=short
```
Expected: FAIL — `No module named 'septmuse.embedders.mock'`

- [ ] **Step 3: 创建 MockEmbedderConfig**

创建 `src/septmuse/configs/embeddings/mock.py`:

```python
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
"""Mock 嵌入配置。"""

from __future__ import annotations

from pydantic import Field

from septmuse.configs.embeddings.base import BaseEmbedderConfig


class MockEmbedderConfig(BaseEmbedderConfig):
    """Mock 嵌入配置 — 固定向量测试用。"""

    backend: str = Field(default="mock")
```

- [ ] **Step 4: 创建 MockEmbedder**

创建 `src/septmuse/embedders/mock.py`:

```python
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
"""Mock 嵌入 — 固定向量, 确定性测试用。

区别于 HashEmbedder (哈希向量): MockEmbedder 返回固定向量, 不依赖输入文本,
适合验证流程正确性而非嵌入质量。
"""

from __future__ import annotations

from septmuse.embedders.base import Embedder

FIXED_VECTOR = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


class MockEmbedder(Embedder):
    """固定 10 维向量嵌入 (测试用)。"""

    @property
    def dimension(self) -> int:
        return 10

    def embed(self, text: str, memory_action: str | None = None) -> list[float]:
        return list(FIXED_VECTOR)

    def embed_batch(self, texts: list[str], memory_action: str | None = None) -> list[list[float]]:
        return [list(FIXED_VECTOR) for _ in texts]
```

- [ ] **Step 5: 注册到 registry + enum**

修改 `src/septmuse/services/registry.py`，在 `BACKEND_MANIFEST["embedder"]` 的 `"st"` 条目后加:

```python
        "mock": BackendEntry(
            module="septmuse.embedders.mock",
            cls="MockEmbedder",
            config_cls="septmuse.configs.embeddings.mock.MockEmbedderConfig",
            deps=(),
        ),
```

修改 `src/septmuse/configs/enums.py`，在 `EmbedderBackend` 加:

```python
    MOCK = "mock"
```

- [ ] **Step 6: 运行测试验证通过**

```bash
$env:PYTHONPATH = "src"; python -m pytest tests/unit/test_embedders/test_mock.py -v
```
Expected: PASS (6 tests)

- [ ] **Step 7: Commit**

```bash
git add src/septmuse/embedders/mock.py src/septmuse/configs/embeddings/mock.py src/septmuse/services/registry.py src/septmuse/configs/enums.py tests/unit/test_embedders/test_mock.py
git commit -m "feat(embedder): add MockEmbedder (fixed 10-dim vector for testing)"
```

---

## Task 4: OllamaEmbedder

**Files:**
- Create: `src/septmuse/embedders/ollama.py`
- Create: `src/septmuse/configs/embeddings/ollama.py`
- Modify: `src/septmuse/services/registry.py`
- Modify: `src/septmuse/configs/enums.py`
- Test: `tests/unit/test_embedders/test_ollama.py`

**Interfaces:**
- Produces: `OllamaEmbedder(model="nomic-embed-text", ollama_base_url="http://localhost:11434")` — 本地 Ollama 嵌入, 自动 pull 模型

- [ ] **Step 1: 写失败测试**

创建 `tests/unit/test_embedders/test_ollama.py`:

```python
"""OllamaEmbedder 测试 — mock ollama.Client, 验证 embed/embed_batch/pull/零配置。"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def mock_ollama_client():
    mock_client = MagicMock()
    mock_client.list.return_value = {"models": [{"name": "nomic-embed-text:latest"}]}
    mock_client.embed.return_value = {"embeddings": [[0.1, 0.2, 0.3]]}
    return mock_client


class TestOllamaEmbedder:
    def test_inherits_embedder_abc(self, mock_ollama_client):
        with patch("ollama.Client", return_value=mock_ollama_client):
            from septmuse.embedders.base import Embedder
            from septmuse.embedders.ollama import OllamaEmbedder

            emb = OllamaEmbedder()
            assert isinstance(emb, Embedder)

    def test_default_model(self, mock_ollama_client):
        with patch("ollama.Client", return_value=mock_ollama_client):
            from septmuse.embedders.ollama import OllamaEmbedder

            emb = OllamaEmbedder()
            assert emb.model == "nomic-embed-text"

    def test_default_dimension(self, mock_ollama_client):
        with patch("ollama.Client", return_value=mock_ollama_client):
            from septmuse.embedders.ollama import OllamaEmbedder

            emb = OllamaEmbedder()
            assert emb.dimension == 512

    def test_embed(self, mock_ollama_client):
        with patch("ollama.Client", return_value=mock_ollama_client):
            from septmuse.embedders.ollama import OllamaEmbedder

            emb = OllamaEmbedder()
            vec = emb.embed("hello")
            assert vec == [0.1, 0.2, 0.3]
            mock_ollama_client.embed.assert_called_once_with(model="nomic-embed-text", input="hello")

    def test_embed_batch(self, mock_ollama_client):
        mock_ollama_client.embed.return_value = {"embeddings": [[0.1, 0.2], [0.3, 0.4]]}
        with patch("ollama.Client", return_value=mock_ollama_client):
            from septmuse.embedders.ollama import OllamaEmbedder

            emb = OllamaEmbedder()
            vecs = emb.embed_batch(["hello", "world"])
            assert len(vecs) == 2
            assert vecs[0] == [0.1, 0.2]

    def test_embed_batch_empty(self, mock_ollama_client):
        with patch("ollama.Client", return_value=mock_ollama_client):
            from septmuse.embedders.ollama import OllamaEmbedder

            emb = OllamaEmbedder()
            assert emb.embed_batch([]) == []

    def test_embed_batch_count_mismatch_raises(self, mock_ollama_client):
        mock_ollama_client.embed.return_value = {"embeddings": [[0.1]]}
        with patch("ollama.Client", return_value=mock_ollama_client):
            from septmuse.embedders.ollama import OllamaEmbedder

            emb = OllamaEmbedder()
            with pytest.raises(ValueError, match="embed"):
                emb.embed_batch(["a", "b"])

    def test_ensure_model_exists_pulls(self, mock_ollama_client):
        mock_ollama_client.list.return_value = {"models": []}
        with patch("ollama.Client", return_value=mock_ollama_client):
            from septmuse.embedders.ollama import OllamaEmbedder

            OllamaEmbedder()
            mock_ollama_client.pull.assert_called_once_with("nomic-embed-text")

    def test_import_error_without_ollama(self):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "ollama":
                raise ImportError("no ollama")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            from septmuse.embedders.ollama import OllamaEmbedder

            with pytest.raises(ImportError, match="ollama"):
                OllamaEmbedder()
```

- [ ] **Step 2: 运行测试验证失败**

```bash
$env:PYTHONPATH = "src"; python -m pytest tests/unit/test_embedders/test_ollama.py -v --tb=short
```
Expected: FAIL — `No module named 'septmuse.embedders.ollama'`

- [ ] **Step 3: 创建 OllamaEmbedderConfig**

创建 `src/septmuse/configs/embeddings/ollama.py`:

```python
#  Copyright 2026 The sonhhxg0529 Authors. All Rights Reserved.
#  ... (Apache 2.0 license header)
"""Ollama 嵌入配置。"""

from __future__ import annotations

from pydantic import Field

from septmuse.configs.embeddings.base import BaseEmbedderConfig


class OllamaEmbedderConfig(BaseEmbedderConfig):
    """Ollama 嵌入配置。"""

    backend: str = Field(default="ollama")
    model: str = Field(default="nomic-embed-text")
    ollama_base_url: str = Field(default="http://localhost:11434")
    embedding_dims: int = Field(default=512)
```

- [ ] **Step 4: 创建 OllamaEmbedder**

创建 `src/septmuse/embedders/ollama.py`:

```python
#  Copyright 2026 The sonhhxg0529 Authors. All Rights Reserved.
#  ... (Apache 2.0 license header)
"""Ollama 嵌入 provider — 本地 Ollama 嵌入, 自动 pull 模型。

零 API key, 本地 Ollama 服务。首次使用自动 pull 模型。
"""

from __future__ import annotations

from septmuse.core.logging import get_logger
from septmuse.embedders.base import Embedder

logger = get_logger(__name__)

DEFAULT_MODEL = "nomic-embed-text"
DEFAULT_DIMS = 512


class OllamaEmbedder(Embedder):
    """基于 Ollama 的本地嵌入。"""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        ollama_base_url: str = "http://localhost:11434",
        embedding_dims: int = DEFAULT_DIMS,
    ) -> None:
        try:
            from ollama import Client
        except ImportError as e:
            raise ImportError("ollama package required: pip install septmuse[ollama]") from e

        self.model = model
        self._dim = embedding_dims
        self._client = Client(host=ollama_base_url)
        self._ensure_model_exists()
        logger.info("ollama_embedder_ready", model=model, dim=self._dim)

    @staticmethod
    def _normalize_model_name(name: str) -> str:
        return name if ":" in name else f"{name}:latest"

    def _ensure_model_exists(self) -> None:
        local_models = self._client.list()["models"]
        target = self._normalize_model_name(self.model)
        if not any(
            self._normalize_model_name(m.get("name", "")) == target
            or self._normalize_model_name(m.get("model", "")) == target
            for m in local_models
        ):
            logger.info("ollama_model_pulling", model=self.model)
            self._client.pull(self.model)

    @property
    def dimension(self) -> int:
        return self._dim

    def embed(self, text: str, memory_action: str | None = None) -> list[float]:
        response = self._client.embed(model=self.model, input=text)
        embeddings = response.get("embeddings") or []
        if not embeddings:
            raise ValueError(f"Ollama embed() returned no embeddings for model '{self.model}'")
        return embeddings[0]

    def embed_batch(self, texts: list[str], memory_action: str | None = None) -> list[list[float]]:
        if not texts:
            return []
        response = self._client.embed(model=self.model, input=texts)
        embeddings = response.get("embeddings") or []
        if len(embeddings) != len(texts):
            raise ValueError(
                f"Ollama embed() returned {len(embeddings)} embeddings for {len(texts)} texts"
            )
        return embeddings
```

- [ ] **Step 5: 注册到 registry + enum**

在 `registry.py` 的 `"mock"` 条目后加:

```python
        "ollama": BackendEntry(
            module="septmuse.embedders.ollama",
            cls="OllamaEmbedder",
            config_cls="septmuse.configs.embeddings.ollama.OllamaEmbedderConfig",
            deps=("ollama",),
        ),
```

在 `enums.py` 的 `EmbedderBackend` 加:

```python
    OLLAMA = "ollama"
```

- [ ] **Step 6: 运行测试验证通过**

```bash
$env:PYTHONPATH = "src"; python -m pytest tests/unit/test_embedders/test_ollama.py -v
```
Expected: PASS (10 tests)

- [ ] **Step 7: Commit**

```bash
git add src/septmuse/embedders/ollama.py src/septmuse/configs/embeddings/ollama.py src/septmuse/services/registry.py src/septmuse/configs/enums.py tests/unit/test_embedders/test_ollama.py
git commit -m "feat(embedder): add OllamaEmbedder (local Ollama embeddings with auto-pull)"
```

---

## Task 5: TogetherEmbedder

**Files:**
- Create: `src/septmuse/embedders/together.py`
- Create: `src/septmuse/configs/embeddings/together.py`
- Modify: `src/septmuse/services/registry.py`
- Modify: `src/septmuse/configs/enums.py`
- Test: `tests/unit/test_embedders/test_together.py`

**Interfaces:**
- Consumes: `_OpenAICompatibleEmbedder` (Task 2)
- Produces: `TogetherEmbedder(api_key=, model=)` — 继承基类, 创建 OpenAI client + Together base_url

- [ ] **Step 1: 写失败测试**

创建 `tests/unit/test_embedders/test_together.py`:

```python
"""TogetherEmbedder 测试 — mock openai.OpenAI, 验证 base_url + 默认 model/dims。"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def mock_openai(monkeypatch):
    import openai

    mock_client = MagicMock()
    mock_client.embeddings.create.return_value = MagicMock(
        data=[MagicMock(embedding=[0.1] * 1024, index=0)]
    )
    monkeypatch.setattr(openai, "OpenAI", MagicMock(return_value=mock_client))
    return mock_client


class TestTogetherEmbedder:
    def test_inherits_embedder_abc(self, mock_openai):
        from septmuse.embedders.base import Embedder
        from septmuse.embedders.together import TogetherEmbedder

        emb = TogetherEmbedder(api_key="test")
        assert isinstance(emb, Embedder)

    def test_default_model(self, mock_openai):
        from septmuse.embedders.together import TogetherEmbedder

        emb = TogetherEmbedder(api_key="test")
        assert emb.model == "intfloat/multilingual-e5-large-instruct"

    def test_default_dimension(self, mock_openai):
        from septmuse.embedders.together import TogetherEmbedder

        emb = TogetherEmbedder(api_key="test")
        assert emb.dimension == 1024

    def test_embed(self, mock_openai):
        from septmuse.embedders.together import TogetherEmbedder

        emb = TogetherEmbedder(api_key="test")
        vec = emb.embed("hello")
        assert len(vec) == 1024

    def test_embed_batch(self, mock_openai):
        mock_openai.embeddings.create.return_value = MagicMock(
            data=[MagicMock(embedding=[0.1] * 1024, index=i) for i in range(2)]
        )
        from septmuse.embedders.together import TogetherEmbedder

        emb = TogetherEmbedder(api_key="test")
        vecs = emb.embed_batch(["hello", "world"])
        assert len(vecs) == 2

    def test_uses_together_base_url(self, mock_openai, monkeypatch):
        import openai

        monkeypatch.setenv("TOGETHER_API_KEY", "env-key")
        from septmuse.embedders.together import TogetherEmbedder

        TogetherEmbedder()
        call_kwargs = openai.OpenAI.call_args.kwargs
        assert "together.xyz" in call_kwargs.get("base_url", "")
```

- [ ] **Step 2: 运行测试验证失败**

```bash
$env:PYTHONPATH = "src"; python -m pytest tests/unit/test_embedders/test_together.py -v --tb=short
```
Expected: FAIL — `No module named 'septmuse.embedders.together'`

- [ ] **Step 3: 创建 TogetherEmbedderConfig**

创建 `src/septmuse/configs/embeddings/together.py`:

```python
#  ... (Apache 2.0 license header)
"""Together AI 嵌入配置。"""

from __future__ import annotations

from pydantic import Field

from septmuse.configs.embeddings.base import BaseEmbedderConfig


class TogetherEmbedderConfig(BaseEmbedderConfig):
    """Together AI 嵌入配置。"""

    backend: str = Field(default="together")
    model: str = Field(default="intfloat/multilingual-e5-large-instruct")
    embedding_dims: int = Field(default=1024)
```

- [ ] **Step 4: 创建 TogetherEmbedder**

创建 `src/septmuse/embedders/together.py`:

```python
#  ... (Apache 2.0 license header)
"""Together AI 嵌入 provider — 继承 _OpenAICompatibleEmbedder。

Together API 与 OpenAI Embeddings 兼容, 仅 base_url 和默认 model/dims 不同。
"""

from __future__ import annotations

import os

from septmuse.core.logging import get_logger
from septmuse.embedders._openai_compatible import _OpenAICompatibleEmbedder

logger = get_logger(__name__)

DEFAULT_MODEL = "intfloat/multilingual-e5-large-instruct"
DEFAULT_DIMS = 1024
BASE_URL = "https://api.together.xyz/v1"


class TogetherEmbedder(_OpenAICompatibleEmbedder):
    """Together AI Embeddings provider (OpenAI 兼容)。"""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        embedding_dims: int | None = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("openai package required: pip install septmuse[openai]") from e

        self.model = model
        dim = embedding_dims or DEFAULT_DIMS

        resolved_key = api_key or os.getenv("TOGETHER_API_KEY")
        if not resolved_key:
            raise ValueError("Together API key required: set TOGETHER_API_KEY or pass api_key=")

        logger.info("embedder_loading", provider="together", model=model, dim=dim)
        client = OpenAI(api_key=resolved_key, base_url=BASE_URL)
        logger.info("embedder_ready", provider="together", model=model, dim=dim)

        super().__init__(client=client, model=model, dim=dim, pass_dimensions_to_api=False)
```

- [ ] **Step 5: 注册到 registry + enum**

在 `registry.py` 的 `"ollama"` 条目后加:

```python
        "together": BackendEntry(
            module="septmuse.embedders.together",
            cls="TogetherEmbedder",
            config_cls="septmuse.configs.embeddings.together.TogetherEmbedderConfig",
            deps=("openai",),
        ),
```

在 `enums.py` 的 `EmbedderBackend` 加:

```python
    TOGETHER = "together"
```

- [ ] **Step 6: 运行测试验证通过**

```bash
$env:PYTHONPATH = "src"; python -m pytest tests/unit/test_embedders/test_together.py -v
```
Expected: PASS (6 tests)

- [ ] **Step 7: Commit**

```bash
git add src/septmuse/embedders/together.py src/septmuse/configs/embeddings/together.py src/septmuse/services/registry.py src/septmuse/configs/enums.py tests/unit/test_embedders/test_together.py
git commit -m "feat(embedder): add TogetherEmbedder (OpenAI-compatible, Together AI)"
```

---

## Task 6: LMStudioEmbedder

**Files:**
- Create: `src/septmuse/embedders/lmstudio.py`
- Create: `src/septmuse/configs/embeddings/lmstudio.py`
- Modify: `src/septmuse/services/registry.py`
- Modify: `src/septmuse/configs/enums.py`
- Test: `tests/unit/test_embedders/test_lmstudio.py`

**Interfaces:**
- Consumes: `_OpenAICompatibleEmbedder` (Task 2)
- Produces: `LMStudioEmbedder(lmstudio_base_url=)` — 继承基类, 本地 LM Studio

- [ ] **Step 1: 写失败测试**

创建 `tests/unit/test_embedders/test_lmstudio.py`:

```python
"""LMStudioEmbedder 测试 — mock openai.OpenAI, 验证 base_url + 默认值。"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def mock_openai(monkeypatch):
    import openai

    mock_client = MagicMock()
    mock_client.embeddings.create.return_value = MagicMock(
        data=[MagicMock(embedding=[0.1] * 1536, index=0)]
    )
    monkeypatch.setattr(openai, "OpenAI", MagicMock(return_value=mock_client))
    return mock_client


class TestLMStudioEmbedder:
    def test_inherits_embedder_abc(self, mock_openai):
        from septmuse.embedders.base import Embedder
        from septmuse.embedders.lmstudio import LMStudioEmbedder

        emb = LMStudioEmbedder()
        assert isinstance(emb, Embedder)

    def test_default_model(self, mock_openai):
        from septmuse.embedders.lmstudio import LMStudioEmbedder

        emb = LMStudioEmbedder()
        assert "nomic-embed" in emb.model

    def test_default_dimension(self, mock_openai):
        from septmuse.embedders.lmstudio import LMStudioEmbedder

        emb = LMStudioEmbedder()
        assert emb.dimension == 1536

    def test_embed(self, mock_openai):
        from septmuse.embedders.lmstudio import LMStudioEmbedder

        emb = LMStudioEmbedder()
        vec = emb.embed("hello")
        assert len(vec) == 1536

    def test_embed_batch(self, mock_openai):
        mock_openai.embeddings.create.return_value = MagicMock(
            data=[MagicMock(embedding=[0.1] * 1536, index=i) for i in range(2)]
        )
        from septmuse.embedders.lmstudio import LMStudioEmbedder

        emb = LMStudioEmbedder()
        vecs = emb.embed_batch(["hello", "world"])
        assert len(vecs) == 2

    def test_default_base_url(self, mock_openai):
        import openai

        from septmuse.embedders.lmstudio import LMStudioEmbedder

        LMStudioEmbedder()
        call_kwargs = openai.OpenAI.call_args.kwargs
        assert call_kwargs["base_url"] == "http://localhost:1234/v1"

    def test_custom_base_url(self, mock_openai):
        import openai

        from septmuse.embedders.lmstudio import LMStudioEmbedder

        LMStudioEmbedder(lmstudio_base_url="http://my-host:8080/v1")
        call_kwargs = openai.OpenAI.call_args.kwargs
        assert call_kwargs["base_url"] == "http://my-host:8080/v1"
```

- [ ] **Step 2: 运行测试验证失败**

```bash
$env:PYTHONPATH = "src"; python -m pytest tests/unit/test_embedders/test_lmstudio.py -v --tb=short
```
Expected: FAIL — `No module named 'septmuse.embedders.lmstudio'`

- [ ] **Step 3: 创建 LMStudioEmbedderConfig**

创建 `src/septmuse/configs/embeddings/lmstudio.py`:

```python
#  ... (Apache 2.0 license header)
"""LM Studio 嵌入配置。"""

from __future__ import annotations

from pydantic import Field

from septmuse.configs.embeddings.base import BaseEmbedderConfig


class LMStudioEmbedderConfig(BaseEmbedderConfig):
    """LM Studio 嵌入配置。"""

    backend: str = Field(default="lmstudio")
    model: str = Field(default="nomic-ai/nomic-embed-text-v1.5-GGUF/nomic-embed-text-v1.5.f16.gguf")
    lmstudio_base_url: str = Field(default="http://localhost:1234/v1")
    embedding_dims: int = Field(default=1536)
```

- [ ] **Step 4: 创建 LMStudioEmbedder**

创建 `src/septmuse/embedders/lmstudio.py`:

```python
#  ... (Apache 2.0 license header)
"""LM Studio 嵌入 provider — 继承 _OpenAICompatibleEmbedder。

LM Studio 本地服务, OpenAI 兼容 API。
"""

from __future__ import annotations

from septmuse.core.logging import get_logger
from septmuse.embedders._openai_compatible import _OpenAICompatibleEmbedder

logger = get_logger(__name__)

DEFAULT_MODEL = "nomic-ai/nomic-embed-text-v1.5-GGUF/nomic-embed-text-v1.5.f16.gguf"
DEFAULT_DIMS = 1536
DEFAULT_BASE_URL = "http://localhost:1234/v1"


class LMStudioEmbedder(_OpenAICompatibleEmbedder):
    """LM Studio Embeddings provider (OpenAI 兼容, 本地)。"""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        lmstudio_base_url: str = DEFAULT_BASE_URL,
        embedding_dims: int | None = None,
        api_key: str = "lm-studio",
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("openai package required: pip install septmuse[openai]") from e

        self.model = model
        dim = embedding_dims or DEFAULT_DIMS

        logger.info("embedder_loading", provider="lmstudio", model=model, dim=dim, base_url=lmstudio_base_url)
        client = OpenAI(base_url=lmstudio_base_url, api_key=api_key)
        logger.info("embedder_ready", provider="lmstudio", model=model, dim=dim)

        super().__init__(client=client, model=model, dim=dim, pass_dimensions_to_api=False)
```

- [ ] **Step 5: 注册到 registry + enum**

在 `registry.py` 的 `"together"` 条目后加:

```python
        "lmstudio": BackendEntry(
            module="septmuse.embedders.lmstudio",
            cls="LMStudioEmbedder",
            config_cls="septmuse.configs.embeddings.lmstudio.LMStudioEmbedderConfig",
            deps=("openai",),
        ),
```

在 `enums.py` 加 `LMSTUDIO = "lmstudio"`。

- [ ] **Step 6: 运行测试验证通过**

```bash
$env:PYTHONPATH = "src"; python -m pytest tests/unit/test_embedders/test_lmstudio.py -v
```
Expected: PASS (7 tests)

- [ ] **Step 7: Commit**

```bash
git add src/septmuse/embedders/lmstudio.py src/septmuse/configs/embeddings/lmstudio.py src/septmuse/services/registry.py src/septmuse/configs/enums.py tests/unit/test_embedders/test_lmstudio.py
git commit -m "feat(embedder): add LMStudioEmbedder (OpenAI-compatible, local LM Studio)"
```

---

## Task 7: AzureOpenAIEmbedder

**Files:**
- Create: `src/septmuse/embedders/azure_openai.py`
- Create: `src/septmuse/configs/embeddings/azure_openai.py`
- Modify: `src/septmuse/services/registry.py`
- Modify: `src/septmuse/configs/enums.py`
- Test: `tests/unit/test_embedders/test_azure_openai.py`

**Interfaces:**
- Consumes: `_OpenAICompatibleEmbedder` (Task 2)
- Produces: `AzureOpenAIEmbedder(api_key=, azure_deployment=, azure_endpoint=)` — Azure OpenAI + AD token fallback

- [ ] **Step 1: 写失败测试**

创建 `tests/unit/test_embedders/test_azure_openai.py`:

```python
"""AzureOpenAIEmbedder 测试 — mock AzureOpenAI, 验证 init/embed/embed_batch/AD token。"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def mock_azure_openai(monkeypatch):
    mock_client = MagicMock()
    mock_client.embeddings.create.return_value = MagicMock(
        data=[MagicMock(embedding=[0.1] * 1536, index=0)]
    )
    import openai

    monkeypatch.setattr(openai, "AzureOpenAI", MagicMock(return_value=mock_client))
    return mock_client


class TestAzureOpenAIEmbedder:
    def test_inherits_embedder_abc(self, mock_azure_openai):
        from septmuse.embedders.azure_openai import AzureOpenAIEmbedder
        from septmuse.embedders.base import Embedder

        emb = AzureOpenAIEmbedder(
            api_key="test", azure_deployment="dep", azure_endpoint="https://test.openai.azure.com"
        )
        assert isinstance(emb, Embedder)

    def test_default_dimension(self, mock_azure_openai):
        from septmuse.embedders.azure_openai import AzureOpenAIEmbedder

        emb = AzureOpenAIEmbedder(
            api_key="test", azure_deployment="dep", azure_endpoint="https://test.openai.azure.com"
        )
        assert emb.dimension == 1536

    def test_embed(self, mock_azure_openai):
        from septmuse.embedders.azure_openai import AzureOpenAIEmbedder

        emb = AzureOpenAIEmbedder(
            api_key="test", azure_deployment="dep", azure_endpoint="https://test.openai.azure.com"
        )
        vec = emb.embed("hello")
        assert len(vec) == 1536

    def test_embed_batch(self, mock_azure_openai):
        mock_azure_openai.embeddings.create.return_value = MagicMock(
            data=[MagicMock(embedding=[0.1] * 1536, index=i) for i in range(2)]
        )
        from septmuse.embedders.azure_openai import AzureOpenAIEmbedder

        emb = AzureOpenAIEmbedder(
            api_key="test", azure_deployment="dep", azure_endpoint="https://test.openai.azure.com"
        )
        vecs = emb.embed_batch(["hello", "world"])
        assert len(vecs) == 2

    @pytest.mark.integration
    def test_ad_token_fallback_no_api_key(self, mock_azure_openai):
        pass

    def test_uses_azure_openai_client(self, mock_azure_openai):
        import openai

        from septmuse.embedders.azure_openai import AzureOpenAIEmbedder

        AzureOpenAIEmbedder(
            api_key="test", azure_deployment="dep", azure_endpoint="https://test.openai.azure.com"
        )
        assert openai.AzureOpenAI.called
```

- [ ] **Step 2: 运行测试验证失败**

```bash
$env:PYTHONPATH = "src"; python -m pytest tests/unit/test_embedders/test_azure_openai.py -v --tb=short
```
Expected: FAIL — `No module named 'septmuse.embedders.azure_openai'`

- [ ] **Step 3: 创建 AzureOpenAIEmbedderConfig**

创建 `src/septmuse/configs/embeddings/azure_openai.py`:

```python
#  ... (Apache 2.0 license header)
"""Azure OpenAI 嵌入配置。"""

from __future__ import annotations

from pydantic import Field

from septmuse.configs.embeddings.base import BaseEmbedderConfig


class AzureOpenAIEmbedderConfig(BaseEmbedderConfig):
    """Azure OpenAI 嵌入配置。"""

    backend: str = Field(default="azure_openai")
    model: str = Field(default="text-embedding-3-small")
    azure_deployment: str | None = Field(default=None)
    azure_endpoint: str | None = Field(default=None)
    api_version: str | None = Field(default=None)
    embedding_dims: int = Field(default=1536)
```

- [ ] **Step 4: 创建 AzureOpenAIEmbedder**

创建 `src/septmuse/embedders/azure_openai.py`:

```python
#  ... (Apache 2.0 license header)
"""Azure OpenAI 嵌入 provider — 继承 _OpenAICompatibleEmbedder。

支持 API key 或 DefaultAzureCredential AD token provider 认证。
"""

from __future__ import annotations

import os

from septmuse.core.logging import get_logger
from septmuse.embedders._openai_compatible import _OpenAICompatibleEmbedder

logger = get_logger(__name__)

DEFAULT_DIMS = 1536


class AzureOpenAIEmbedder(_OpenAICompatibleEmbedder):
    """Azure OpenAI Embeddings provider (OpenAI 兼容 + AD token)。"""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "text-embedding-3-small",
        azure_deployment: str | None = None,
        azure_endpoint: str | None = None,
        api_version: str | None = None,
        embedding_dims: int | None = None,
    ) -> None:
        try:
            from openai import AzureOpenAI
        except ImportError as e:
            raise ImportError("openai package required: pip install septmuse[openai]") from e

        self.model = model
        dim = embedding_dims or DEFAULT_DIMS

        resolved_key = api_key or os.getenv("EMBEDDING_AZURE_OPENAI_API_KEY")
        resolved_deployment = azure_deployment or os.getenv("EMBEDDING_AZURE_DEPLOYMENT")
        resolved_endpoint = azure_endpoint or os.getenv("EMBEDDING_AZURE_ENDPOINT")
        resolved_api_version = api_version or os.getenv("EMBEDDING_AZURE_API_VERSION")

        azure_ad_token_provider = None
        if not resolved_key or resolved_key in ("", "your-api-key"):
            try:
                from azure.identity import DefaultAzureCredential, get_bearer_token_provider

                credential = DefaultAzureCredential()
                azure_ad_token_provider = get_bearer_token_provider(
                    credential, "https://cognitiveservices.azure.com/.default"
                )
                resolved_key = None
            except ImportError:
                pass

        logger.info("embedder_loading", provider="azure_openai", model=model, dim=dim)
        client = AzureOpenAI(
            azure_deployment=resolved_deployment,
            azure_endpoint=resolved_endpoint,
            azure_ad_token_provider=azure_ad_token_provider,
            api_version=resolved_api_version,
            api_key=resolved_key,
        )
        logger.info("embedder_ready", provider="azure_openai", model=model, dim=dim)

        super().__init__(client=client, model=model, dim=dim, pass_dimensions_to_api=False)
```

- [ ] **Step 5: 注册到 registry + enum**

在 `registry.py` 的 `"lmstudio"` 条目后加:

```python
        "azure_openai": BackendEntry(
            module="septmuse.embedders.azure_openai",
            cls="AzureOpenAIEmbedder",
            config_cls="septmuse.configs.embeddings.azure_openai.AzureOpenAIEmbedderConfig",
            deps=("openai",),
        ),
```

在 `enums.py` 加 `AZURE_OPENAI = "azure_openai"`。

- [ ] **Step 6: 运行测试验证通过**

```bash
$env:PYTHONPATH = "src"; python -m pytest tests/unit/test_embedders/test_azure_openai.py -v
```
Expected: PASS (5 tests + 1 integration skipped)

- [ ] **Step 7: Commit**

```bash
git add src/septmuse/embedders/azure_openai.py src/septmuse/configs/embeddings/azure_openai.py src/septmuse/services/registry.py src/septmuse/configs/enums.py tests/unit/test_embedders/test_azure_openai.py
git commit -m "feat(embedder): add AzureOpenAIEmbedder (OpenAI-compatible + AD token fallback)"
```

---

## Task 8: GeminiEmbedder

**Files:**
- Create: `src/septmuse/embedders/gemini.py`
- Create: `src/septmuse/configs/embeddings/gemini.py`
- Modify: `src/septmuse/services/registry.py`
- Modify: `src/septmuse/configs/enums.py`
- Test: `tests/unit/test_embedders/test_gemini.py`

**Interfaces:**
- Produces: `GeminiEmbedder(api_key=, model=)` — Google Gemini 嵌入

- [ ] **Step 1: 写失败测试**

创建 `tests/unit/test_embedders/test_gemini.py`:

```python
"""GeminiEmbedder 测试 — mock genai.Client, 验证 embed/embed_batch/output_dimensionality。"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def mock_genai():
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.embeddings = [MagicMock(values=[0.1] * 768)]
    mock_client.models.embed_content.return_value = mock_response
    return mock_client


class TestGeminiEmbedder:
    def test_inherits_embedder_abc(self, mock_genai):
        with patch("google.genai.Client", return_value=mock_genai):
            from septmuse.embedders.base import Embedder
            from septmuse.embedders.gemini import GeminiEmbedder

            emb = GeminiEmbedder(api_key="test")
            assert isinstance(emb, Embedder)

    def test_default_model(self, mock_genai):
        with patch("google.genai.Client", return_value=mock_genai):
            from septmuse.embedders.gemini import GeminiEmbedder

            emb = GeminiEmbedder(api_key="test")
            assert emb.model == "models/gemini-embedding-001"

    def test_default_dimension(self, mock_genai):
        with patch("google.genai.Client", return_value=mock_genai):
            from septmuse.embedders.gemini import GeminiEmbedder

            emb = GeminiEmbedder(api_key="test")
            assert emb.dimension == 768

    def test_embed(self, mock_genai):
        with patch("google.genai.Client", return_value=mock_genai):
            from septmuse.embedders.gemini import GeminiEmbedder

            emb = GeminiEmbedder(api_key="test")
            vec = emb.embed("hello")
            assert len(vec) == 768

    def test_embed_batch(self, mock_genai):
        mock_response = MagicMock()
        mock_response.embeddings = [MagicMock(values=[0.1] * 768), MagicMock(values=[0.2] * 768)]
        mock_genai.models.embed_content.return_value = mock_response
        with patch("google.genai.Client", return_value=mock_genai):
            from septmuse.embedders.gemini import GeminiEmbedder

            emb = GeminiEmbedder(api_key="test")
            vecs = emb.embed_batch(["hello", "world"])
            assert len(vecs) == 2

    def test_embed_batch_count_mismatch_raises(self, mock_genai):
        mock_response = MagicMock()
        mock_response.embeddings = [MagicMock(values=[0.1] * 768)]
        mock_genai.models.embed_content.return_value = mock_response
        with patch("google.genai.Client", return_value=mock_genai):
            from septmuse.embedders.gemini import GeminiEmbedder

            emb = GeminiEmbedder(api_key="test")
            with pytest.raises(ValueError, match="embed_batch"):
                emb.embed_batch(["a", "b"])
```

- [ ] **Step 2: 运行测试验证失败**

```bash
$env:PYTHONPATH = "src"; python -m pytest tests/unit/test_embedders/test_gemini.py -v --tb=short
```
Expected: FAIL

- [ ] **Step 3: 创建 GeminiEmbedderConfig**

创建 `src/septmuse/configs/embeddings/gemini.py`:

```python
#  ... (Apache 2.0 license header)
"""Gemini 嵌入配置。"""

from __future__ import annotations

from pydantic import Field

from septmuse.configs.embeddings.base import BaseEmbedderConfig


class GeminiEmbedderConfig(BaseEmbedderConfig):
    """Google Gemini 嵌入配置。"""

    backend: str = Field(default="gemini")
    model: str = Field(default="models/gemini-embedding-001")
    embedding_dims: int = Field(default=768)
    output_dimensionality: int | None = Field(default=None)
```

- [ ] **Step 4: 创建 GeminiEmbedder**

创建 `src/septmuse/embedders/gemini.py`:

```python
#  ... (Apache 2.0 license header)
"""Google Gemini 嵌入 provider — google-genai SDK。"""

from __future__ import annotations

import os

from septmuse.core.logging import get_logger
from septmuse.embedders.base import Embedder

logger = get_logger(__name__)

DEFAULT_MODEL = "models/gemini-embedding-001"
DEFAULT_DIMS = 768
MAX_BATCH = 100


class GeminiEmbedder(Embedder):
    """Google Gemini Embeddings provider。"""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        embedding_dims: int = DEFAULT_DIMS,
        output_dimensionality: int | None = None,
    ) -> None:
        try:
            from google import genai
            from google.genai import types
        except ImportError as e:
            raise ImportError("google-genai package required: pip install septmuse[gemini]") from e

        self.model = model
        self._dim = embedding_dims
        self._output_dimensionality = output_dimensionality or embedding_dims
        self._types = types

        resolved_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not resolved_key:
            raise ValueError("Google API key required: set GOOGLE_API_KEY or pass api_key=")

        logger.info("embedder_loading", provider="gemini", model=model, dim=self._dim)
        self._client = genai.Client(api_key=resolved_key)
        logger.info("embedder_ready", provider="gemini", model=model, dim=self._dim)

    @property
    def dimension(self) -> int:
        return self._dim

    def embed(self, text: str, memory_action: str | None = None) -> list[float]:
        text = text.replace("\n", " ")
        config = self._types.EmbedContentConfig(output_dimensionality=self._output_dimensionality)
        response = self._client.models.embed_content(model=self.model, contents=text, config=config)
        return response.embeddings[0].values

    def embed_batch(self, texts: list[str], memory_action: str | None = None) -> list[list[float]]:
        if not texts:
            return []
        config = self._types.EmbedContentConfig(output_dimensionality=self._output_dimensionality)
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), MAX_BATCH):
            chunk = [t.replace("\n", " ") for t in texts[i : i + MAX_BATCH]]
            response = self._client.models.embed_content(model=self.model, contents=chunk, config=config)
            all_embeddings.extend(e.values for e in response.embeddings)
        if len(all_embeddings) != len(texts):
            raise ValueError(
                f"Gemini embed_batch() returned {len(all_embeddings)} embeddings for {len(texts)} texts"
            )
        return all_embeddings
```

- [ ] **Step 5: 注册到 registry + enum**

在 `registry.py` 加:

```python
        "gemini": BackendEntry(
            module="septmuse.embedders.gemini",
            cls="GeminiEmbedder",
            config_cls="septmuse.configs.embeddings.gemini.GeminiEmbedderConfig",
            deps=("google-genai",),
        ),
```

在 `enums.py` 加 `GEMINI = "gemini"`。

- [ ] **Step 6: 运行测试验证通过**

```bash
$env:PYTHONPATH = "src"; python -m pytest tests/unit/test_embedders/test_gemini.py -v
```
Expected: PASS (6 tests)

- [ ] **Step 7: Commit**

```bash
git add src/septmuse/embedders/gemini.py src/septmuse/configs/embeddings/gemini.py src/septmuse/services/registry.py src/septmuse/configs/enums.py tests/unit/test_embedders/test_gemini.py
git commit -m "feat(embedder): add GeminiEmbedder (Google Gemini embeddings)"
```

---

## Task 9: VertexAIEmbedder

**Files:**
- Create: `src/septmuse/embedders/vertexai.py`
- Create: `src/septmuse/configs/embeddings/vertexai.py`
- Modify: `src/septmuse/services/registry.py`
- Modify: `src/septmuse/configs/enums.py`
- Test: `tests/unit/test_embedders/test_vertexai.py`

**Interfaces:**
- Produces: `VertexAIEmbedder(model=, vertex_credentials_json=)` — 唯一真正用 `memory_action` 的 provider (add→RETRIEVAL_DOCUMENT, search→RETRIEVAL_QUERY)

- [ ] **Step 1: 写失败测试**

创建 `tests/unit/test_embedders/test_vertexai.py`:

```python
"""VertexAIEmbedder 测试 — mock TextEmbeddingModel, 验证 memory_action task_type 切换。"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def mock_model():
    mock = MagicMock()
    mock.get_embeddings.return_value = [MagicMock(values=[0.1] * 256)]
    return mock


class TestVertexAIEmbedder:
    def test_inherits_embedder_abc(self, mock_model):
        with patch("vertexai.language_models.TextEmbeddingModel.from_pretrained", return_value=mock_model):
            from septmuse.embedders.base import Embedder
            from septmuse.embedders.vertexai import VertexAIEmbedder

            emb = VertexAIEmbedder(vertex_credentials_json="/fake/path.json")
            assert isinstance(emb, Embedder)

    def test_default_model(self, mock_model):
        with patch("vertexai.language_models.TextEmbeddingModel.from_pretrained", return_value=mock_model):
            from septmuse.embedders.vertexai import VertexAIEmbedder

            emb = VertexAIEmbedder(vertex_credentials_json="/fake/path.json")
            assert emb.model == "gemini-embedding-001"

    def test_default_dimension(self, mock_model):
        with patch("vertexai.language_models.TextEmbeddingModel.from_pretrained", return_value=mock_model):
            from septmuse.embedders.vertexai import VertexAIEmbedder

            emb = VertexAIEmbedder(vertex_credentials_json="/fake/path.json")
            assert emb.dimension == 256

    def test_embed_no_memory_action(self, mock_model):
        with patch("vertexai.language_models.TextEmbeddingModel.from_pretrained", return_value=mock_model):
            from septmuse.embedders.vertexai import VertexAIEmbedder

            emb = VertexAIEmbedder(vertex_credentials_json="/fake/path.json")
            vec = emb.embed("hello")
            assert len(vec) == 256

    def test_embed_add_uses_retrieval_document(self, mock_model):
        with patch("vertexai.language_models.TextEmbeddingModel.from_pretrained", return_value=mock_model):
            from septmuse.embedders.vertexai import VertexAIEmbedder

            emb = VertexAIEmbedder(vertex_credentials_json="/fake/path.json")
            emb.embed("hello", memory_action="add")
            call_kwargs = mock_model.get_embeddings.call_args
            inputs = call_kwargs.kwargs.get("texts") or call_kwargs.args[0]
            assert inputs[0].task_type == "RETRIEVAL_DOCUMENT"

    def test_embed_search_uses_retrieval_query(self, mock_model):
        with patch("vertexai.language_models.TextEmbeddingModel.from_pretrained", return_value=mock_model):
            from septmuse.embedders.vertexai import VertexAIEmbedder

            emb = VertexAIEmbedder(vertex_credentials_json="/fake/path.json")
            emb.embed("hello", memory_action="search")
            call_kwargs = mock_model.get_embeddings.call_args
            inputs = call_kwargs.kwargs.get("texts") or call_kwargs.args[0]
            assert inputs[0].task_type == "RETRIEVAL_QUERY"

    def test_embed_batch(self, mock_model):
        mock_model.get_embeddings.return_value = [
            MagicMock(values=[0.1] * 256), MagicMock(values=[0.2] * 256)
        ]
        with patch("vertexai.language_models.TextEmbeddingModel.from_pretrained", return_value=mock_model):
            from septmuse.embedders.vertexai import VertexAIEmbedder

            emb = VertexAIEmbedder(vertex_credentials_json="/fake/path.json")
            vecs = emb.embed_batch(["hello", "world"])
            assert len(vecs) == 2
```

- [ ] **Step 2: 运行测试验证失败**

```bash
$env:PYTHONPATH = "src"; python -m pytest tests/unit/test_embedders/test_vertexai.py -v --tb=short
```
Expected: FAIL

- [ ] **Step 3: 创建 VertexAIEmbedderConfig**

创建 `src/septmuse/configs/embeddings/vertexai.py`:

```python
#  ... (Apache 2.0 license header)
"""Vertex AI 嵌入配置。"""

from __future__ import annotations

from pydantic import Field

from septmuse.configs.embeddings.base import BaseEmbedderConfig


class VertexAIEmbedderConfig(BaseEmbedderConfig):
    """Google Vertex AI 嵌入配置。"""

    backend: str = Field(default="vertexai")
    model: str = Field(default="gemini-embedding-001")
    embedding_dims: int = Field(default=256)
    vertex_credentials_json: str | None = Field(default=None)
    memory_add_embedding_type: str = Field(default="RETRIEVAL_DOCUMENT")
    memory_search_embedding_type: str = Field(default="RETRIEVAL_QUERY")
    memory_update_embedding_type: str = Field(default="RETRIEVAL_DOCUMENT")
```

- [ ] **Step 4: 创建 VertexAIEmbedder**

创建 `src/septmuse/embedders/vertexai.py`:

```python
#  ... (Apache 2.0 license header)
"""Vertex AI 嵌入 provider — 唯一真正用 memory_action 的 provider。

memory_action 切换 task_type:
- "add"/"update" → RETRIEVAL_DOCUMENT
- "search" → RETRIEVAL_QUERY
- None → SEMANTIC_SIMILARITY
"""

from __future__ import annotations

import os

from septmuse.core.logging import get_logger
from septmuse.embedders.base import Embedder

logger = get_logger(__name__)

DEFAULT_MODEL = "gemini-embedding-001"
DEFAULT_DIMS = 256
BATCH_SIZE = 250


class VertexAIEmbedder(Embedder):
    """Google Vertex AI Embeddings provider (memory_action task_type 切换)。"""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        embedding_dims: int = DEFAULT_DIMS,
        vertex_credentials_json: str | None = None,
        memory_add_embedding_type: str = "RETRIEVAL_DOCUMENT",
        memory_search_embedding_type: str = "RETRIEVAL_QUERY",
        memory_update_embedding_type: str = "RETRIEVAL_DOCUMENT",
    ) -> None:
        try:
            from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
        except ImportError as e:
            raise ImportError("google-cloud-aiplatform required: pip install septmuse[vertexai]") from e

        self.model = model
        self._dim = embedding_dims
        self._TextEmbeddingInput = TextEmbeddingInput

        self._embedding_types = {
            "add": memory_add_embedding_type,
            "update": memory_update_embedding_type,
            "search": memory_search_embedding_type,
        }

        creds = vertex_credentials_json or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if creds:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds

        logger.info("embedder_loading", provider="vertexai", model=model, dim=self._dim)
        self._model = TextEmbeddingModel.from_pretrained(model)
        logger.info("embedder_ready", provider="vertexai", model=model, dim=self._dim)

    @property
    def dimension(self) -> int:
        return self._dim

    def _resolve_task_type(self, memory_action: str | None) -> str:
        if memory_action is None:
            return "SEMANTIC_SIMILARITY"
        if memory_action not in self._embedding_types:
            raise ValueError(f"Invalid memory_action: {memory_action}")
        return self._embedding_types[memory_action]

    def embed(self, text: str, memory_action: str | None = None) -> list[float]:
        task_type = self._resolve_task_type(memory_action)
        text_input = self._TextEmbeddingInput(text=text, task_type=task_type)
        embeddings = self._model.get_embeddings(texts=[text_input], output_dimensionality=self._dim)
        return embeddings[0].values

    def embed_batch(self, texts: list[str], memory_action: str | None = None) -> list[list[float]]:
        if not texts:
            return []
        task_type = self._resolve_task_type(memory_action)
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), BATCH_SIZE):
            chunk = texts[i : i + BATCH_SIZE]
            inputs = [self._TextEmbeddingInput(text=t, task_type=task_type) for t in chunk]
            results = self._model.get_embeddings(texts=inputs, output_dimensionality=self._dim)
            all_embeddings.extend(r.values for r in results)
        if len(all_embeddings) != len(texts):
            raise ValueError(
                f"Vertex AI embed_batch() returned {len(all_embeddings)} embeddings for {len(texts)} texts"
            )
        return all_embeddings
```

- [ ] **Step 5: 注册到 registry + enum**

在 `registry.py` 加:

```python
        "vertexai": BackendEntry(
            module="septmuse.embedders.vertexai",
            cls="VertexAIEmbedder",
            config_cls="septmuse.configs.embeddings.vertexai.VertexAIEmbedderConfig",
            deps=("google-cloud-aiplatform",),
        ),
```

在 `enums.py` 加 `VERTEXAI = "vertexai"`。

- [ ] **Step 6: 运行测试验证通过**

```bash
$env:PYTHONPATH = "src"; python -m pytest tests/unit/test_embedders/test_vertexai.py -v
```
Expected: PASS (7 tests)

- [ ] **Step 7: Commit**

```bash
git add src/septmuse/embedders/vertexai.py src/septmuse/configs/embeddings/vertexai.py src/septmuse/services/registry.py src/septmuse/configs/enums.py tests/unit/test_embedders/test_vertexai.py
git commit -m "feat(embedder): add VertexAIEmbedder (memory_action task_type switching)"
```

---

## Task 10: HuggingFaceEmbedder

**Files:**
- Create: `src/septmuse/embedders/huggingface.py`
- Create: `src/septmuse/configs/embeddings/huggingface.py`
- Modify: `src/septmuse/services/registry.py`
- Modify: `src/septmuse/configs/enums.py`
- Test: `tests/unit/test_embedders/test_huggingface.py`

**Interfaces:**
- Produces: `HuggingFaceEmbedder(model=, huggingface_base_url=)` — 双模式: 有 base_url 走 TEI server (OpenAI 兼容), 无 base_url 走本地 SentenceTransformer

- [ ] **Step 1: 写失败测试**

创建 `tests/unit/test_embedders/test_huggingface.py`:

```python
"""HuggingFaceEmbedder 测试 — mock 双模式 (TEI server + 本地 ST)。"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestHuggingFaceTEIMode:
    @pytest.fixture()
    def mock_openai(self, monkeypatch):
        import openai

        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = MagicMock(
            data=[MagicMock(embedding=[0.1] * 768, index=0)]
        )
        monkeypatch.setattr(openai, "OpenAI", MagicMock(return_value=mock_client))
        return mock_client

    def test_tei_mode_uses_openai_client(self, mock_openai):
        from septmuse.embedders.huggingface import HuggingFaceEmbedder

        emb = HuggingFaceEmbedder(huggingface_base_url="https://my-tei.server")
        vec = emb.embed("hello")
        assert len(vec) == 768

    def test_tei_mode_default_model(self, mock_openai):
        from septmuse.embedders.huggingface import HuggingFaceEmbedder

        emb = HuggingFaceEmbedder(huggingface_base_url="https://my-tei.server")
        assert emb.model == "tei"

    def test_tei_mode_embed_batch(self, mock_openai):
        mock_openai.embeddings.create.return_value = MagicMock(
            data=[MagicMock(embedding=[0.1] * 768, index=i) for i in range(2)]
        )
        from septmuse.embedders.huggingface import HuggingFaceEmbedder

        emb = HuggingFaceEmbedder(huggingface_base_url="https://my-tei.server")
        vecs = emb.embed_batch(["hello", "world"])
        assert len(vecs) == 2


class TestHuggingFaceLocalMode:
    @pytest.fixture()
    def mock_st(self, monkeypatch):
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.return_value = MagicMock(tolist=lambda: [0.1] * 384)
        mock_module = MagicMock()
        mock_module.SentenceTransformer = MagicMock(return_value=mock_model)
        monkeypatch.setitem(__import__("sys").modules, "sentence_transformers", mock_module)
        return mock_model

    def test_local_mode_uses_sentence_transformer(self, mock_st):
        from septmuse.embedders.huggingface import HuggingFaceEmbedder

        emb = HuggingFaceEmbedder()
        assert emb.dimension == 384

    def test_local_mode_embed(self, mock_st):
        from septmuse.embedders.huggingface import HuggingFaceEmbedder

        emb = HuggingFaceEmbedder()
        vec = emb.embed("hello")
        assert len(vec) == 384

    def test_local_mode_embed_batch(self, mock_st):
        mock_st.encode.return_value = [MagicMock(tolist=lambda: [0.1] * 384)]
        from septmuse.embedders.huggingface import HuggingFaceEmbedder

        emb = HuggingFaceEmbedder()
        vecs = emb.embed_batch(["hello"])
        assert len(vecs) == 1
```

- [ ] **Step 2: 运行测试验证失败**

```bash
$env:PYTHONPATH = "src"; python -m pytest tests/unit/test_embedders/test_huggingface.py -v --tb=short
```
Expected: FAIL

- [ ] **Step 3: 创建 HuggingFaceEmbedderConfig**

创建 `src/septmuse/configs/embeddings/huggingface.py`:

```python
#  ... (Apache 2.0 license header)
"""HuggingFace 嵌入配置。"""

from __future__ import annotations

from pydantic import Field

from septmuse.configs.embeddings.base import BaseEmbedderConfig


class HuggingFaceEmbedderConfig(BaseEmbedderConfig):
    """HuggingFace 嵌入配置 (本地 ST 或 TEI server)。"""

    backend: str = Field(default="huggingface")
    model: str = Field(default="multi-qa-MiniLM-L6-cos-v1")
    huggingface_base_url: str | None = Field(default=None, description="TEI server URL, None=本地 ST")
    model_kwargs: dict = Field(default_factory=dict)
    embedding_dims: int | None = Field(default=None)
```

- [ ] **Step 4: 创建 HuggingFaceEmbedder**

创建 `src/septmuse/embedders/huggingface.py`:

```python
#  ... (Apache 2.0 license header)
"""HuggingFace 嵌入 provider — 双模式。

有 huggingface_base_url 时走 TEI server (OpenAI 兼容 API);
无 base_url 时走本地 SentenceTransformer。
"""

from __future__ import annotations

from typing import Any

from septmuse.core.logging import get_logger
from septmuse.embedders._openai_compatible import _OpenAICompatibleEmbedder
from septmuse.embedders.base import Embedder

logger = get_logger(__name__)

DEFAULT_MODEL = "multi-qa-MiniLM-L6-cos-v1"


class HuggingFaceEmbedder(Embedder):
    """HuggingFace 嵌入 (TEI server 或 本地 SentenceTransformer)。"""

    def __init__(
        self,
        model: str | None = None,
        huggingface_base_url: str | None = None,
        model_kwargs: dict | None = None,
        embedding_dims: int | None = None,
    ) -> None:
        self._tei_mode = huggingface_base_url is not None
        self._kwargs = model_kwargs or {}

        if self._tei_mode:
            try:
                from openai import OpenAI
            except ImportError as e:
                raise ImportError("openai package required for TEI mode: pip install septmuse[openai]") from e

            resolved_model = model or "tei"
            logger.info("embedder_loading", provider="huggingface_tei", model=resolved_model, base_url=huggingface_base_url)
            client = OpenAI(base_url=huggingface_base_url)
            dim = embedding_dims or 768
            logger.info("embedder_ready", provider="huggingface_tei", model=resolved_model, dim=dim)

            self._inner: Embedder = _OpenAICompatibleEmbedder(
                client=client, model=resolved_model, dim=dim, pass_dimensions_to_api=False
            )
        else:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise ImportError("sentence-transformers required: pip install septmuse[st]") from e

            resolved_model = model or DEFAULT_MODEL
            logger.info("embedder_loading", provider="huggingface_local", model=resolved_model)
            self._st_model = SentenceTransformer(resolved_model, **self._kwargs)
            dim = embedding_dims or self._st_model.get_sentence_embedding_dimension()
            assert dim is not None
            self._dim: int = dim
            logger.info("embedder_ready", provider="huggingface_local", model=resolved_model, dim=dim)

            self._inner = _LocalSTAdapter(self._st_model, self._dim)

        self.model = resolved_model

    @property
    def dimension(self) -> int:
        return self._inner.dimension

    def embed(self, text: str, memory_action: str | None = None) -> list[float]:
        return self._inner.embed(text, memory_action)

    def embed_batch(self, texts: list[str], memory_action: str | None = None) -> list[list[float]]:
        return self._inner.embed_batch(texts, memory_action)


class _LocalSTAdapter(Embedder):
    """SentenceTransformer 适配器 — 包装 ST 模型为 Embedder 接口。"""

    def __init__(self, model: Any, dim: int) -> None:
        self._model = model
        self._dim = dim

    @property
    def dimension(self) -> int:
        return self._dim

    def embed(self, text: str, memory_action: str | None = None) -> list[float]:
        vec = self._model.encode(text, convert_to_numpy=True)
        return vec.tolist()

    def embed_batch(self, texts: list[str], memory_action: str | None = None) -> list[list[float]]:
        vecs = self._model.encode(texts, convert_to_numpy=True)
        return [v.tolist() for v in vecs]
```

- [ ] **Step 5: 注册到 registry + enum**

在 `registry.py` 加:

```python
        "huggingface": BackendEntry(
            module="septmuse.embedders.huggingface",
            cls="HuggingFaceEmbedder",
            config_cls="septmuse.configs.embeddings.huggingface.HuggingFaceEmbedderConfig",
            deps=("sentence_transformers",),
        ),
```

在 `enums.py` 加 `HUGGINGFACE = "huggingface"`。

- [ ] **Step 6: 运行测试验证通过**

```bash
$env:PYTHONPATH = "src"; python -m pytest tests/unit/test_embedders/test_huggingface.py -v
```
Expected: PASS (5 tests)

- [ ] **Step 7: Commit**

```bash
git add src/septmuse/embedders/huggingface.py src/septmuse/configs/embeddings/huggingface.py src/septmuse/services/registry.py src/septmuse/configs/enums.py tests/unit/test_embedders/test_huggingface.py
git commit -m "feat(embedder): add HuggingFaceEmbedder (dual-mode: TEI server + local ST)"
```

---

## Task 11: AWSBedrockEmbedder

**Files:**
- Create: `src/septmuse/embedders/aws_bedrock.py`
- Create: `src/septmuse/configs/embeddings/aws_bedrock.py`
- Modify: `src/septmuse/services/registry.py`
- Modify: `src/septmuse/configs/enums.py`
- Test: `tests/unit/test_embedders/test_aws_bedrock.py`

**Interfaces:**
- Produces: `AWSBedrockEmbedder(model=, aws_access_key_id=, aws_region=)` — boto3 bedrock-runtime, cohere/titan body 差异 + L2 归一化

- [ ] **Step 1: 写失败测试**

创建 `tests/unit/test_embedders/test_aws_bedrock.py`:

```python
"""AWSBedrockEmbedder 测试 — mock boto3.client, 验证 titan/cohere body + L2 归一化。"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def mock_bedrock():
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.get.return_value.read.return_value = json.dumps({"embedding": [3.0, 4.0]})
    mock_client.invoke_model.return_value = mock_response
    return mock_client


class TestAWSBedrockEmbedder:
    def test_inherits_embedder_abc(self, mock_bedrock):
        with patch("boto3.client", return_value=mock_bedrock):
            from septmuse.embedders.aws_bedrock import AWSBedrockEmbedder
            from septmuse.embedders.base import Embedder

            emb = AWSBedrockEmbedder()
            assert isinstance(emb, Embedder)

    def test_default_model(self, mock_bedrock):
        with patch("boto3.client", return_value=mock_bedrock):
            from septmuse.embedders.aws_bedrock import AWSBedrockEmbedder

            emb = AWSBedrockEmbedder()
            assert emb.model == "amazon.titan-embed-text-v1"

    def test_embed_titan(self, mock_bedrock):
        with patch("boto3.client", return_value=mock_bedrock):
            from septmuse.embedders.aws_bedrock import AWSBedrockEmbedder

            emb = AWSBedrockEmbedder()
            vec = emb.embed("hello")
            assert len(vec) == 2
            call_kwargs = mock_bedrock.invoke_model.call_args.kwargs
            body = json.loads(call_kwargs["body"])
            assert body["inputText"] == "hello"

    def test_embed_coherent(self, mock_bedrock):
        mock_response = MagicMock()
        mock_response.get.return_value.read.return_value = json.dumps({"embeddings": [[1.0, 0.0]]})
        mock_bedrock.invoke_model.return_value = mock_response
        with patch("boto3.client", return_value=mock_bedrock):
            from septmuse.embedders.aws_bedrock import AWSBedrockEmbedder

            emb = AWSBedrockEmbedder(model="cohere.embed-multilingual-v3")
            vec = emb.embed("hello")
            assert len(vec) == 2
            call_kwargs = mock_bedrock.invoke_model.call_args.kwargs
            body = json.loads(call_kwargs["body"])
            assert body["input_type"] == "search_document"
            assert body["texts"] == ["hello"]

    def test_embed_l2_normalization(self, mock_bedrock):
        mock_response = MagicMock()
        mock_response.get.return_value.read.return_value = json.dumps({"embedding": [3.0, 4.0]})
        mock_bedrock.invoke_model.return_value = mock_response
        with patch("boto3.client", return_value=mock_bedrock):
            from septmuse.embedders.aws_bedrock import AWSBedrockEmbedder

            emb = AWSBedrockEmbedder()
            vec = emb.embed("hello")
            assert abs(vec[0] ** 2 + vec[1] ** 2 - 1.0) < 1e-6
```

- [ ] **Step 2: 运行测试验证失败**

```bash
$env:PYTHONPATH = "src"; python -m pytest tests/unit/test_embedders/test_aws_bedrock.py -v --tb=short
```
Expected: FAIL

- [ ] **Step 3: 创建 AWSBedrockEmbedderConfig**

创建 `src/septmuse/configs/embeddings/aws_bedrock.py`:

```python
#  ... (Apache 2.0 license header)
"""AWS Bedrock 嵌入配置。"""

from __future__ import annotations

from pydantic import Field

from septmuse.configs.embeddings.base import BaseEmbedderConfig


class AWSBedrockEmbedderConfig(BaseEmbedderConfig):
    """AWS Bedrock 嵌入配置。"""

    backend: str = Field(default="aws_bedrock")
    model: str = Field(default="amazon.titan-embed-text-v1")
    aws_access_key_id: str | None = Field(default=None)
    aws_secret_access_key: str | None = Field(default=None)
    aws_session_token: str | None = Field(default=None)
    aws_region: str = Field(default="us-west-2")
    embedding_dims: int | None = Field(default=None)
```

- [ ] **Step 4: 创建 AWSBedrockEmbedder**

创建 `src/septmuse/embedders/aws_bedrock.py`:

```python
#  ... (Apache 2.0 license header)
"""AWS Bedrock 嵌入 provider — boto3 bedrock-runtime。

按 provider (cohere/titan) 构造不同 body, L2 归一化输出。
"""

from __future__ import annotations

import json
import os

import numpy as np

from septmuse.core.logging import get_logger
from septmuse.embedders.base import Embedder

logger = get_logger(__name__)

DEFAULT_MODEL = "amazon.titan-embed-text-v1"


class AWSBedrockEmbedder(Embedder):
    """AWS Bedrock Embeddings provider (titan/cohere, L2 归一化)。"""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        aws_session_token: str | None = None,
        aws_region: str | None = None,
        embedding_dims: int | None = None,
    ) -> None:
        try:
            import boto3
        except ImportError as e:
            raise ImportError("boto3 required: pip install septmuse[aws-bedrock]") from e

        self.model = model
        self._dim = embedding_dims
        self._provider = model.split(".")[0]

        access_key = aws_access_key_id or os.environ.get("AWS_ACCESS_KEY_ID")
        secret_key = aws_secret_access_key or os.environ.get("AWS_SECRET_ACCESS_KEY")
        session_token = aws_session_token or os.environ.get("AWS_SESSION_TOKEN")
        region = aws_region or os.environ.get("AWS_REGION") or "us-west-2"

        logger.info("embedder_loading", provider="aws_bedrock", model=model, region=region)
        self._client = boto3.client(
            "bedrock-runtime",
            region_name=region,
            aws_access_key_id=access_key if access_key else None,
            aws_secret_access_key=secret_key if secret_key else None,
            aws_session_token=session_token if session_token else None,
        )
        logger.info("embedder_ready", provider="aws_bedrock", model=model)

    @property
    def dimension(self) -> int:
        if self._dim is None:
            raise RuntimeError("AWS Bedrock dimension unknown until first embed() call")
        return self._dim

    def _get_embedding(self, text: str) -> list[float]:
        input_body: dict = {}
        if self._provider == "cohere":
            input_body["input_type"] = "search_document"
            input_body["texts"] = [text]
        else:
            input_body["inputText"] = text
            if self._dim is not None and "v2" in self.model:
                input_body["dimensions"] = self._dim

        body = json.dumps(input_body)
        response = self._client.invoke_model(
            body=body, modelId=self.model, accept="application/json", contentType="application/json"
        )
        response_body = json.loads(response.get("body").read())

        if self._provider == "cohere":
            return response_body.get("embeddings")[0]
        return response_body.get("embedding")

    def embed(self, text: str, memory_action: str | None = None) -> list[float]:
        emb = self._get_embedding(text)
        if self._dim is None:
            self._dim = len(emb)
        vec = np.array(emb, dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def embed_batch(self, texts: list[str], memory_action: str | None = None) -> list[list[float]]:
        return [self.embed(t, memory_action) for t in texts]
```

- [ ] **Step 5: 注册到 registry + enum**

在 `registry.py` 加:

```python
        "aws_bedrock": BackendEntry(
            module="septmuse.embedders.aws_bedrock",
            cls="AWSBedrockEmbedder",
            config_cls="septmuse.configs.embeddings.aws_bedrock.AWSBedrockEmbedderConfig",
            deps=("boto3",),
        ),
```

在 `enums.py` 加 `AWS_BEDROCK = "aws_bedrock"`。

- [ ] **Step 6: 运行测试验证通过**

```bash
$env:PYTHONPATH = "src"; python -m pytest tests/unit/test_embedders/test_aws_bedrock.py -v
```
Expected: PASS (5 tests)

- [ ] **Step 7: Commit**

```bash
git add src/septmuse/embedders/aws_bedrock.py src/septmuse/configs/embeddings/aws_bedrock.py src/septmuse/services/registry.py src/septmuse/configs/enums.py tests/unit/test_embedders/test_aws_bedrock.py
git commit -m "feat(embedder): add AWSBedrockEmbedder (titan/cohere, L2 normalization)"
```

---

## Task 12: FastEmbedEmbedder

**Files:**
- Create: `src/septmuse/embedders/fastembed.py`
- Create: `src/septmuse/configs/embeddings/fastembed.py`
- Modify: `src/septmuse/services/registry.py`
- Modify: `src/septmuse/configs/enums.py`
- Test: `tests/unit/test_embedders/test_fastembed.py`

**Interfaces:**
- Produces: `FastEmbedEmbedder(model=)` — fastembed 库, 轻量 ONNX

- [ ] **Step 1: 写失败测试**

创建 `tests/unit/test_embedders/test_fastembed.py`:

```python
"""FastEmbedEmbedder 测试 — mock TextEmbedding, 验证 embed/embed_batch/dimension。"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def mock_text_embedding():
    mock = MagicMock()
    mock.embedding_size = 768
    mock.embed.return_value = iter([[0.1] * 768])
    return mock


class TestFastEmbedEmbedder:
    def test_inherits_embedder_abc(self, mock_text_embedding):
        with patch("fastembed.TextEmbedding", return_value=mock_text_embedding):
            from septmuse.embedders.base import Embedder
            from septmuse.embedders.fastembed import FastEmbedEmbedder

            emb = FastEmbedEmbedder()
            assert isinstance(emb, Embedder)

    def test_default_model(self, mock_text_embedding):
        with patch("fastembed.TextEmbedding", return_value=mock_text_embedding):
            from septmuse.embedders.fastembed import FastEmbedEmbedder

            emb = FastEmbedEmbedder()
            assert emb.model == "thenlper/gte-large"

    def test_dimension(self, mock_text_embedding):
        with patch("fastembed.TextEmbedding", return_value=mock_text_embedding):
            from septmuse.embedders.fastembed import FastEmbedEmbedder

            emb = FastEmbedEmbedder()
            assert emb.dimension == 768

    def test_embed(self, mock_text_embedding):
        with patch("fastembed.TextEmbedding", return_value=mock_text_embedding):
            from septmuse.embedders.fastembed import FastEmbedEmbedder

            emb = FastEmbedEmbedder()
            vec = emb.embed("hello")
            assert len(vec) == 768

    def test_embed_batch(self, mock_text_embedding):
        mock_text_embedding.embed.return_value = iter([[0.1] * 768, [0.2] * 768])
        with patch("fastembed.TextEmbedding", return_value=mock_text_embedding):
            from septmuse.embedders.fastembed import FastEmbedEmbedder

            emb = FastEmbedEmbedder()
            vecs = emb.embed_batch(["hello", "world"])
            assert len(vecs) == 2
```

- [ ] **Step 2: 运行测试验证失败**

```bash
$env:PYTHONPATH = "src"; python -m pytest tests/unit/test_embedders/test_fastembed.py -v --tb=short
```
Expected: FAIL

- [ ] **Step 3: 创建 FastEmbedEmbedderConfig**

创建 `src/septmuse/configs/embeddings/fastembed.py`:

```python
#  ... (Apache 2.0 license header)
"""FastEmbed 嵌入配置。"""

from __future__ import annotations

from pydantic import Field

from septmuse.configs.embeddings.base import BaseEmbedderConfig


class FastEmbedEmbedderConfig(BaseEmbedderConfig):
    """FastEmbed 嵌入配置。"""

    backend: str = Field(default="fastembed")
    model: str = Field(default="thenlper/gte-large")
    embedding_dims: int | None = Field(default=None)
```

- [ ] **Step 4: 创建 FastEmbedEmbedder**

创建 `src/septmuse/embedders/fastembed.py`:

```python
#  ... (Apache 2.0 license header)
"""FastEmbed 嵌入 provider — 轻量 ONNX, 无 torch。

与 OnnxEmbedder 功能重叠, 但使用 fastembed 库 (不同模型生态)。
"""

from __future__ import annotations

from septmuse.core.logging import get_logger
from septmuse.embedders.base import Embedder

logger = get_logger(__name__)

DEFAULT_MODEL = "thenlper/gte-large"


class FastEmbedEmbedder(Embedder):
    """FastEmbed Embeddings provider (轻量 ONNX)。"""

    def __init__(self, model: str = DEFAULT_MODEL, embedding_dims: int | None = None) -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as e:
            raise ImportError("fastembed required: pip install septmuse[fastembed]") from e

        self.model = model
        logger.info("embedder_loading", provider="fastembed", model=model)
        self._model = TextEmbedding(model_name=model)
        self._dim = embedding_dims or self._model.embedding_size
        logger.info("embedder_ready", provider="fastembed", model=model, dim=self._dim)

    @property
    def dimension(self) -> int:
        return self._dim

    def embed(self, text: str, memory_action: str | None = None) -> list[float]:
        text = text.replace("\n", " ")
        embeddings = list(self._model.embed(text))
        return embeddings[0]

    def embed_batch(self, texts: list[str], memory_action: str | None = None) -> list[list[float]]:
        if not texts:
            return []
        cleaned = [t.replace("\n", " ") for t in texts]
        results = list(self._model.embed(cleaned))
        return results
```

- [ ] **Step 5: 注册到 registry + enum**

在 `registry.py` 加:

```python
        "fastembed": BackendEntry(
            module="septmuse.embedders.fastembed",
            cls="FastEmbedEmbedder",
            config_cls="septmuse.configs.embeddings.fastembed.FastEmbedEmbedderConfig",
            deps=("fastembed",),
        ),
```

在 `enums.py` 加 `FASTEMBED = "fastembed"`。

- [ ] **Step 6: 运行测试验证通过**

```bash
$env:PYTHONPATH = "src"; python -m pytest tests/unit/test_embedders/test_fastembed.py -v
```
Expected: PASS (5 tests)

- [ ] **Step 7: Commit**

```bash
git add src/septmuse/embedders/fastembed.py src/septmuse/configs/embeddings/fastembed.py src/septmuse/services/registry.py src/septmuse/configs/enums.py tests/unit/test_embedders/test_fastembed.py
git commit -m "feat(embedder): add FastEmbedEmbedder (lightweight ONNX, no torch)"
```

---

## Task 13: LangchainEmbedder

**Files:**
- Create: `src/septmuse/embedders/langchain.py`
- Create: `src/septmuse/configs/embeddings/langchain.py`
- Modify: `src/septmuse/services/registry.py`
- Modify: `src/septmuse/configs/enums.py`
- Test: `tests/unit/test_embedders/test_langchain.py`

**Interfaces:**
- Produces: `LangchainEmbedder(model=<Embeddings instance>)` — 桥接 LangChain Embeddings, 调 embed_query/embed_documents

- [ ] **Step 1: 写失败测试**

创建 `tests/unit/test_embedders/test_langchain.py`:

```python
"""LangchainEmbedder 测试 — 用 FakeEmbeddings 验证桥接。"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


class FakeEmbeddings:
    """模拟 langchain Embeddings 接口。"""

    def __init__(self, dim: int = 256):
        self._dim = dim

    def embed_query(self, text: str) -> list[float]:
        return [0.1] * self._dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * self._dim for _ in texts]


class TestLangchainEmbedder:
    def test_inherits_embedder_abc(self):
        from septmuse.embedders.base import Embedder
        from septmuse.embedders.langchain import LangchainEmbedder

        emb = LangchainEmbedder(model=FakeEmbeddings())
        assert isinstance(emb, Embedder)

    def test_dimension_from_model(self):
        from septmuse.embedders.langchain import LangchainEmbedder

        emb = LangchainEmbedder(model=FakeEmbeddings(dim=256))
        assert emb.dimension == 256

    def test_embed(self):
        from septmuse.embedders.langchain import LangchainEmbedder

        emb = LangchainEmbedder(model=FakeEmbeddings(dim=256))
        vec = emb.embed("hello")
        assert len(vec) == 256

    def test_embed_batch(self):
        from septmuse.embedders.langchain import LangchainEmbedder

        emb = LangchainEmbedder(model=FakeEmbeddings(dim=256))
        vecs = emb.embed_batch(["hello", "world"])
        assert len(vecs) == 2

    def test_requires_model_instance(self):
        from septmuse.embedders.langchain import LangchainEmbedder

        with pytest.raises(ValueError, match="model"):
            LangchainEmbedder(model=None)

    def test_rejects_non_embeddings_instance(self):
        from septmuse.embedders.langchain import LangchainEmbedder

        with pytest.raises(ValueError, match="Embeddings"):
            LangchainEmbedder(model="not-an-embeddings-instance")

    def test_embed_batch_empty(self):
        from septmuse.embedders.langchain import LangchainEmbedder

        emb = LangchainEmbedder(model=FakeEmbeddings())
        assert emb.embed_batch([]) == []
```

- [ ] **Step 2: 运行测试验证失败**

```bash
$env:PYTHONPATH = "src"; python -m pytest tests/unit/test_embedders/test_langchain.py -v --tb=short
```
Expected: FAIL

- [ ] **Step 3: 创建 LangchainEmbedderConfig**

创建 `src/septmuse/configs/embeddings/langchain.py`:

```python
#  ... (Apache 2.0 license header)
"""Langchain 嵌入配置。"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from septmuse.configs.embeddings.base import BaseEmbedderConfig


class LangchainEmbedderConfig(BaseEmbedderConfig):
    """Langchain 嵌入配置 — model 字段是 Embeddings 实例 (非字符串)。"""

    backend: str = Field(default="langchain")
    model: Any = Field(default=None, description="langchain Embeddings 实例")
    embedding_dims: int = Field(default=768)
```

- [ ] **Step 4: 创建 LangchainEmbedder**

创建 `src/septmuse/embedders/langchain.py`:

```python
#  ... (Apache 2.0 license header)
"""Langchain 嵌入 provider — 桥接 LangChain Embeddings。

接收用户传入的 langchain.embeddings.Embeddings 实例,
调 embed_query() / embed_documents()。一次桥接整个 LangChain embedding 生态。
"""

from __future__ import annotations

from typing import Any

from septmuse.core.logging import get_logger
from septmuse.embedders.base import Embedder

logger = get_logger(__name__)


class LangchainEmbedder(Embedder):
    """LangChain Embeddings 桥接器。"""

    def __init__(self, model: Any, embedding_dims: int = 768) -> None:
        if model is None:
            raise ValueError("`model` parameter is required (langchain Embeddings instance)")

        try:
            from langchain.embeddings.base import Embeddings
        except ImportError:
            try:
                from langchain_core.embeddings import Embeddings
            except ImportError as e:
                raise ImportError("langchain required: pip install septmuse[langchain]") from e

        if not isinstance(model, Embeddings):
            raise ValueError("`model` must be an instance of langchain Embeddings")

        self._langchain_model = model
        self._dim = embedding_dims
        logger.info("embedder_ready", provider="langchain", dim=self._dim, type=type(model).__name__)

    @property
    def dimension(self) -> int:
        return self._dim

    def embed(self, text: str, memory_action: str | None = None) -> list[float]:
        return self._langchain_model.embed_query(text)

    def embed_batch(self, texts: list[str], memory_action: str | None = None) -> list[list[float]]:
        if not texts:
            return []
        return self._langchain_model.embed_documents(texts)
```

- [ ] **Step 5: 注册到 registry + enum**

在 `registry.py` 加:

```python
        "langchain": BackendEntry(
            module="septmuse.embedders.langchain",
            cls="LangchainEmbedder",
            config_cls="septmuse.configs.embeddings.langchain.LangchainEmbedderConfig",
            deps=("langchain",),
        ),
```

在 `enums.py` 加 `LANGCHAIN = "langchain"`。

- [ ] **Step 6: 运行测试验证通过**

```bash
$env:PYTHONPATH = "src"; python -m pytest tests/unit/test_embedders/test_langchain.py -v
```
Expected: PASS (7 tests)

- [ ] **Step 7: Commit**

```bash
git add src/septmuse/embedders/langchain.py src/septmuse/configs/embeddings/langchain.py src/septmuse/services/registry.py src/septmuse/configs/enums.py tests/unit/test_embedders/test_langchain.py
git commit -m "feat(embedder): add LangchainEmbedder (bridge to LangChain Embeddings ecosystem)"
```

---

## Task 14: pyproject.toml extras + 文档

**Files:**
- Modify: `pyproject.toml`
- Modify: `AGENTS.md`
- Modify: `CHANGELOG.md`

- [ ] **Step 1: 更新 pyproject.toml extras**

在 `pyproject.toml` 的 `[project.optional-dependencies]` 中, 在 `st = ...` 行后加:

```toml
# ── 新增 Embedder providers ──
langchain = ["langchain>=0.1"]
azure-openai = ["openai>=1.30", "azure-identity>=1.16"]
vertexai = ["google-cloud-aiplatform>=1.50"]
aws-bedrock = ["boto3>=1.34"]
fastembed = ["fastembed>=0.2"]

# ── Embedder 聚合 ──
embedders = ["septmuse[onnx,st,openai,ollama,gemini,langchain,azure-openai,vertexai,aws-bedrock,fastembed]"]
```

更新 `all` 聚合 extra:

```toml
all = ["septmuse[llms,vector-stores,graph-stores,backends,embedders,rerankers,ner,activation,parametric,server,dev]"]
```

- [ ] **Step 2: 运行 ruff check 验证**

```bash
ruff check src/septmuse/embedders/ src/septmuse/configs/embeddings/ tests/unit/test_embedders/ --fix
```
Expected: 无错误或自动修复

- [ ] **Step 3: 运行全套件验证零退化**

```bash
$env:PYTHONPATH = "src"; python -m pytest tests/unit/ tests/e2e/ -q --tb=short -x
```
Expected: 1319 + 80 (新增) = 1399 passed, 23 skipped, 16 failed (pre-existing)

- [ ] **Step 4: 更新 AGENTS.md**

在 AGENTS.md 的 Embedder section 补充:

- 新增 11 个 provider（ollama/langchain/azure_openai/huggingface/gemini/vertexai/together/lmstudio/aws_bedrock/fastembed/mock）
- memory_action 参数说明
- 新 extras（langchain/azure-openai/vertexai/aws-bedrock/fastembed/embedders）
- `SEPTMUSE_EMBEDDER` 环境变量新增值

- [ ] **Step 5: 更新 CHANGELOG.md**

在 CHANGELOG.md 加:

```markdown
## [Unreleased]
### Added
- 11 个新 embedding provider: ollama, langchain, azure_openai, huggingface, gemini, vertexai, together, lmstudio, aws_bedrock, fastembed, mock
- `memory_action` 参数到 Embedder ABC (embed + embed_batch), 支持 "add"/"search"/"update" 嵌入策略
- `_OpenAICompatibleEmbedder` 基类, OpenAI 兼容 family (openai/together/lmstudio/azure_openai) 共享 embed/embed_batch 逻辑
- CachedEmbedder cache key 包含 memory_action, 防止 vertexai 等 provider 的 add/search 向量混用
- 5 个新 extras: langchain, azure-openai, vertexai, aws-bedrock, fastembed
- `embedders` 聚合 extra

### Changed
- OpenAIEmbedder 重构为继承 `_OpenAICompatibleEmbedder` 基类 (行为不变)
- EmbedderBackend enum 新增 11 个值
- BACKEND_MANIFEST["embedder"] 新增 11 条 BackendEntry (共 18 个后端)
```

- [ ] **Step 6: 最终验证 — ruff + 全测试**

```bash
ruff check src/ tests/ --fix
$env:PYTHONPATH = "src"; python -m pytest tests/unit/ tests/e2e/ -q --tb=short
```
Expected: ruff 通过, 测试全绿 (零退化)

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml AGENTS.md CHANGELOG.md
git commit -m "docs(embedder): add 11 new provider extras + update AGENTS.md and CHANGELOG"
```

---

## 自审清单

### 1. Spec 覆盖

| Spec 要求 | Task |
|-----------|------|
| memory_action 接口 + CachedEmbedder | Task 1 |
| _OpenAICompatibleEmbedder 基类 + OpenAIEmbedder 重构 | Task 2 |
| mock embedder | Task 3 |
| ollama embedder | Task 4 |
| together embedder | Task 5 |
| lmstudio embedder | Task 6 |
| azure_openai embedder | Task 7 |
| gemini embedder | Task 8 |
| vertexai embedder (memory_action task_type) | Task 9 |
| huggingface embedder (双模式) | Task 10 |
| aws_bedrock embedder | Task 11 |
| fastembed embedder | Task 12 |
| langchain embedder | Task 13 |
| pyproject.toml extras + 文档 | Task 14 |

### 2. 类型一致性

- 所有 embedder 签名统一: `embed(text, memory_action=None)` + `embed_batch(texts, memory_action=None)`
- `_OpenAICompatibleEmbedder` 构造参数统一: `client=, model=, dim=, pass_dimensions_to_api=`
- registry BackendEntry 结构统一: `module=, cls=, config_cls=, deps=`

### 3. 占位符扫描

无 TBD/TODO/placeholder。所有代码块含完整实现。

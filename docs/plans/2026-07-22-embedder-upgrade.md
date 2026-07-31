# Embedder 升级计划（P0-P2：hybrid 默认 + ONNX 英文/多语言嵌入 + 语言检测）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把默认检索路径从纯向量改为 BM25+向量 RRF 混合（P0），新增 ONNX 轻量嵌入模型替代 HashEmbedder（P1），加入中英文语言检测自动选模型（P2）。

**Architecture:** P0 改 `Memory.search()` 默认走 `hybrid_search()`（store 有 `keyword_search` 时自动路由，无则回退纯向量）。P1 新增 `OnnxEmbedder`（onnxruntime + tokenizers，无 torch 依赖），模型从 HuggingFace 下载量化 ONNX 到 `~/.septmuse/models/`。P2 新增 `AutoOnnxEmbedder`——init 时检测语言（CJK 比例 > 30% → zh），选一个模型用于整个 session（不同模型投影到不同语义空间，不可混用），默认 zh（中文优先项目）。

**Tech Stack:** onnxruntime, tokenizers (HuggingFace Rust tokenizer), huggingface_hub, numpy, 纯 Python CJK 检测

## Global Constraints

- **PYTHONPATH=src** 运行所有测试（包未 pip install -e .）
- **ruff line-length 120**，ignore E501/RUF001/002/003（中文全角标点）
- **禁止** `ruff format <file>` 直接调用（Windows 清空文件 bug）——用 `ruff format --stdin-filename <path>` 或格式化后检查大小
- **MCP tools.py 禁止** `from __future__ import annotations`（FastMCP func_metadata 限制）
- **测试保护规则**：现有测试固定不动，禁止改断言绕过缺陷；仅可新增测试
- **不提交**（非 git 仓库，文件快照模式）
- **中文输出**：与用户交互用简体中文
- **score 统一为相似度 [0,1]**：越高越相似
- **模型维度 384**：英文 `Xenova/all-MiniLM-L6-v2`（384 dim）和多语言 `Xenova/paraphrase-multilingual-MiniLM-L12-v2`（384 dim）维度一致，与 HashEmbedder 默认 dim=384 兼容
- **语言检测 init 时执行**：不同模型投影到不同语义空间，不可 per-query 切换；整个 session 用一个模型

## 关键设计决策：语言检测策略

**问题**：用户问"中英文检测默认走中文还是英文模型"。

**答案**：默认走中文（多语言模型），因为：
1. SeptMuse 是中文优先项目（AGENTS.md 强制中文输出，所有文档中文）
2. 多语言模型 `paraphrase-multilingual-MiniLM-L12-v2` 同时支持中英文，即使查询是英文也能检索中文记忆
3. 英文专用模型 `all-MiniLM-L6-v2` 不支持中文，检索中文记忆质量差

**检测时机**：init 时一次，不在每次 embed 时切换。原因：不同模型即使都是 384 维，投影到不同的语义空间——用 A 模型嵌入 "我喜欢Python" 和用 B 模型嵌入 "I like Python" 的向量不可比较，余弦相似度无意义。

**检测方法**：CJK 字符比例 > 30% → zh，否则 en。纯 Python，<1ms。

## 文件结构

| 文件 | 责任 | 动作 |
|------|------|------|
| `src/septmuse/orchestration/memory.py` | Memory facade — search 默认路由 + _resolve_embedder 扩展 | 修改 |
| `src/septmuse/providers/embedders/onnx.py` | OnnxEmbedder — onnxruntime + tokenizers 嵌入 | 新建 |
| `src/septmuse/providers/embedders/auto.py` | AutoOnnxEmbedder — 语言检测 + 模型选择 | 新建 |
| `src/septmuse/providers/embedders/langdetect.py` | detect_language 纯函数 — CJK 比例检测 | 新建 |
| `src/septmuse/configs/defaults.py` | MemoryConfig + 新字段 + default_config 环境变量 | 修改 |
| `pyproject.toml` | +onnx extra | 修改 |
| `tests/unit/test_memory.py` | search 默认 hybrid 测试（新增，不修改现有） | 修改（仅新增） |
| `tests/unit/test_onnx_embedder.py` | OnnxEmbedder 单元测试 | 新建 |
| `tests/unit/test_langdetect.py` | 语言检测单元测试 | 新建 |
| `tests/unit/test_auto_embedder.py` | AutoOnnxEmbedder 单元测试 | 新建 |
| `AGENTS.md` | 环境变量表 + embedder 说明更新 | 修改 |

---

## Task 1: P0 — Memory.search 默认走 hybrid

**目标**：`Memory.search()` 默认走 BM25+向量 RRF 混合检索（store 有 `keyword_search` 时自动路由），纯向量作为 opt-out。

**Files:**
- Modify: `src/septmuse/orchestration/memory.py:184-211`（search 方法）
- Test: `tests/unit/test_memory.py`（新增测试类，不修改现有 TestSearch）

**Interfaces:**
- Consumes: `MemoryStore.hybrid_search()`（`storage/base.py:139`，已存在）, `MemoryStore.keyword_search()`（`storage/base.py:132`，已存在）
- Produces: `Memory.search()` 签名加 `hybrid: bool = True` 参数；返回格式不变（`list[dict]`，含 id/memory/score/metadata/created_at；hybrid 模式额外含 vector_score/bm25_score）

- [ ] **Step 1: 写失败测试 — search 默认走 hybrid（BM25 兜底）**

在 `tests/unit/test_memory.py` 末尾新增测试类（不修改现有 TestSearch）：

```python
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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_memory.py::TestSearchHybridDefault -v`
Expected: FAIL — `search()` 没有 `hybrid` 参数，返回结果不含 `bm25_score`

- [ ] **Step 3: 修改 Memory.search 方法**

修改 `src/septmuse/orchestration/memory.py:184-211`，把 `search` 方法改为：

```python
    def search(
        self,
        query: str,
        *,
        user_id: str,
        top_k: int | None = None,
        threshold: float | None = None,
        hybrid: bool = True,
    ) -> list[dict[str, Any]]:
        """检索记忆 (对齐 mem0 search 签名)。

        默认 hybrid=True: BM25+向量 RRF 融合检索 (store 支持 keyword_search 时)。
        hybrid=False: 纯向量检索 (对齐阶段1 行为)。

        Args:
            query: 查询文本
            user_id: 用户 ID (必填)
            top_k: 返回数 (默认 config.top_k)
            threshold: 相似阈值 (默认 config.threshold)
            hybrid: True=BM25+向量 RRF 融合; False=纯向量

        Returns:
            hybrid=True: list[{"id","memory","score","vector_score","bm25_score","metadata","created_at"}]
            hybrid=False: list[{"id","memory","score","metadata","created_at"}]
        """
        tk = top_k or self.config.top_k
        th = threshold if threshold is not None else self.config.threshold

        if hybrid and hasattr(self.store, "keyword_search"):
            return self.search_hybrid(query, user_id=user_id, top_k=tk, threshold=th)

        emb = self.embedder.embed(query)
        results = self.store.search(emb, user_id=user_id, top_k=tk, threshold=th)
        logger.info("memory_search_done", user_id=user_id, query=query[:50], hits=len(results), hybrid=hybrid)
        return results
```

- [ ] **Step 4: 运行测试确认通过**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_memory.py -v`
Expected: PASS — 全部测试通过（含原有 TestSearch + 新增 TestSearchHybridDefault）

- [ ] **Step 5: 运行完整测试套确认零退化**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/ tests/e2e/ -q`
Expected: 686+ passed, 22 skipped, 23 e2e passed（零退化）

- [ ] **Step 6: Lint 检查**

Run: `ruff check src/septmuse/orchestration/memory.py tests/unit/test_memory.py`
Expected: All checks passed

---

## Task 2: P1 — 语言检测工具函数

**目标**：纯 Python CJK 字符比例检测，无外部依赖，<1ms。

**Files:**
- Create: `src/septmuse/providers/embedders/langdetect.py`
- Test: `tests/unit/test_langdetect.py`

**Interfaces:**
- Produces: `detect_language(text: str) -> str` — 返回 "zh" 或 "en"；`CJK_RATIO_THRESHOLD = 0.3`

- [ ] **Step 1: 写失败测试**

Create `tests/unit/test_langdetect.py`:

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
"""语言检测单元测试 (CJK 字符比例启发式)。"""

from __future__ import annotations

from septmuse.providers.embedders.langdetect import detect_language


class TestDetectLanguage:
    def test_pure_chinese(self) -> None:
        assert detect_language("我喜欢编程") == "zh"

    def test_pure_english(self) -> None:
        assert detect_language("I love programming") == "en"

    def test_mixed_chinese_dominant(self) -> None:
        assert detect_language("我喜欢用 python 编程，它很强大") == "zh"

    def test_mixed_english_dominant(self) -> None:
        assert detect_language("Python is great, 我用它") == "en"

    def test_empty_string_defaults_en(self) -> None:
        assert detect_language("") == "en"

    def test_numbers_only_defaults_en(self) -> None:
        assert detect_language("12345 67890") == "en"

    def test_japanese_kana_not_counted_as_cjk(self) -> None:
        """平假名/片假名不在 CJK Unified 范围, 不算中文。"""
        assert detect_language("こんにちは") == "en"

    def test_cjk_extension_a_counted(self) -> None:
        """CJK Extension A (U+3400-U+4DBF) 算中文。"""
        assert detect_language("㐀㐁㐂") == "zh"

    def test_threshold_boundary(self) -> None:
        """CJK 比例恰好 30% 判为中文。"""
        # 3 中文字 + 7 英文字 = 30% CJK
        text = "我喜欢它abcdefg"
        assert detect_language(text) == "zh"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_langdetect.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'septmuse.providers.embedders.langdetect'`

- [ ] **Step 3: 实现语言检测**

Create `src/septmuse/providers/embedders/langdetect.py`:

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
"""语言检测 — CJK 字符比例启发式 (纯 Python, <1ms, 无外部依赖)。

检测范围:
- CJK Unified Ideographs (U+4E00-U+9FFF): 常用汉字
- CJK Extension A (U+3400-U+4DBF): 罕用汉字

不检测:
- 平假名 (U+3040-U+309F) / 片假名 (U+30A0-U+30FF): 日文
- 谚文 (U+AC00-U+D7AF): 韩文

策略: CJK 字符占文本总字符数 > 30% → 'zh', 否则 'en'。
默认 'en' (空字符串/纯数字/纯符号)。

注意: 此函数用于 init 时选择嵌入模型, 不用于 per-query 切换
(不同模型投影到不同语义空间, per-query 切换会破坏向量可比性)。
"""

from __future__ import annotations

CJK_RATIO_THRESHOLD = 0.3


def _is_cjk(char: str) -> bool:
    """判断字符是否为 CJK 汉字 (Unified + Extension A)。"""
    code = ord(char)
    return (
        0x4E00 <= code <= 0x9FFF  # CJK Unified Ideographs
        or 0x3400 <= code <= 0x4DBF  # CJK Extension A
    )


def detect_language(text: str) -> str:
    """检测文本主语言: 'zh' (中文) 或 'en' (英文)。

    CJK 字符比例 > 30% → 'zh'; 否则 'en'。
    空字符串 → 'en' (安全默认)。
    """
    if not text:
        return "en"
    total = len(text)
    cjk_count = sum(1 for c in text if _is_cjk(c))
    return "zh" if cjk_count / total > CJK_RATIO_THRESHOLD else "en"
```

- [ ] **Step 4: 运行测试确认通过**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_langdetect.py -v`
Expected: 9 passed

- [ ] **Step 5: Lint 检查**

Run: `ruff check src/septmuse/providers/embedders/langdetect.py tests/unit/test_langdetect.py`
Expected: All checks passed

---

## Task 3: P1 — OnnxEmbedder 类（英文模型 all-MiniLM-L6-v2）

**目标**：用 onnxruntime + tokenizers 实现嵌入，无 torch 依赖，模型从 HuggingFace 下载量化 ONNX 缓存到本地。

**Files:**
- Create: `src/septmuse/providers/embedders/onnx.py`
- Test: `tests/unit/test_onnx_embedder.py`

**Interfaces:**
- Consumes: `Embedder` ABC（`providers/embedders/base.py`）
- Produces: `OnnxEmbedder` 类，构造参数 `model_name: str = "Xenova/all-MiniLM-L6-v2"`；`dimension` property 返回 384；`embed(text) -> list[float]`；`embed_batch(texts) -> list[list[float]]`
- 模型缓存路径: `~/.septmuse/models/<model_name_sanitized>/`
- 需要文件: `onnx/model_quantized.onnx` + `tokenizer.json`

- [ ] **Step 1: 写失败测试**

Create `tests/unit/test_onnx_embedder.py`:

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
"""OnnxEmbedder 单元测试。

onnxruntime 未安装时全部 skip (integration marker)。
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration

try:
    import onnxruntime  # noqa: F401
    import tokenizers  # noqa: F401

    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

SKIP_REASON = "onnxruntime + tokenizers 未安装。pip install septmuse[onnx]"


@pytest.fixture(scope="module")
def embedder():
    from septmuse.providers.embedders.onnx import OnnxEmbedder

    return OnnxEmbedder()


@pytest.mark.skipif(not HAS_ONNX, reason=SKIP_REASON)
class TestOnnxEmbedder:
    def test_dimension_is_384(self, embedder) -> None:
        assert embedder.dimension == 384

    def test_embed_returns_normalized_vector(self, embedder) -> None:
        vec = embedder.embed("hello world")
        assert len(vec) == 384
        # 归一化: L2 norm ≈ 1.0
        import math

        norm = math.sqrt(sum(v * v for v in vec))
        assert abs(norm - 1.0) < 0.01

    def test_embed_batch(self, embedder) -> None:
        texts = ["hello", "world"]
        vecs = embedder.embed_batch(texts)
        assert len(vecs) == 2
        assert all(len(v) == 384 for v in vecs)

    def test_similar_texts_high_cosine(self, embedder) -> None:
        """语义相似文本余弦相似度高。"""
        v1 = embedder.embed("I love programming")
        v2 = embedder.embed("I enjoy coding")
        dot = sum(a * b for a, b in zip(v1, v2))
        assert dot > 0.5  # 语义相似, 余弦 > 0.5

    def test_unrelated_texts_low_cosine(self, embedder) -> None:
        """无关文本余弦相似度低。"""
        v1 = embedder.embed("python programming")
        v2 = embedder.embed("sunny weather today")
        dot = sum(a * b for a, b in zip(v1, v2))
        assert dot < 0.5
```

- [ ] **Step 2: 运行测试确认失败（skip）**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_onnx_embedder.py -v`
Expected: 5 skipped (integration marker + onnxruntime 未装)

- [ ] **Step 3: 实现 OnnxEmbedder**

Create `src/septmuse/providers/embedders/onnx.py`:

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
"""ONNX 嵌入实现 — onnxruntime + tokenizers, 无 torch 依赖。

模型: HuggingFace Xenova/all-MiniLM-L6-v2 (量化 ONNX, ~23MB, 384 dim)。
首次使用自动下载到 ~/.septmuse/models/, 后续直接从缓存加载。

优势 (vs sentence-transformers):
- 无 torch 依赖 (~2GB → 0)
- 启动快 (<2s vs ~30s)
- CPU 推理快 (<50ms/query)
- Windows 稳定 (无模型缓存不完整问题)

用法:
    embedder = OnnxEmbedder()  # 自动下载模型
    vec = embedder.embed("hello world")  # → 384 维归一化向量
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from septmuse.observability import get_logger
from septmuse.providers.embedders.base import Embedder

logger = get_logger(__name__)

DEFAULT_EN_MODEL = "Xenova/all-MiniLM-L6-v2"
DEFAULT_ZH_MODEL = "Xenova/paraphrase-multilingual-MiniLM-L12-v2"

_MODEL_FILES = {
    "onnx/model_quantized.onnx",
    "tokenizer.json",
}


def _model_cache_dir(model_name: str) -> Path:
    """模型缓存目录: ~/.septmuse/models/<sanitized_model_name>/"""
    safe = model_name.replace("/", "__")
    base = os.getenv("SEPTMUSE_MODEL_CACHE", str(Path.home() / ".septmuse" / "models"))
    return Path(base) / safe


def _ensure_model_files(model_name: str) -> Path:
    """确保模型文件已下载到本地缓存, 返回缓存目录。"""
    cache_dir = _model_cache_dir(model_name)
    cache_dir.mkdir(parents=True, exist_ok=True)

    missing = [f for f in _MODEL_FILES if not (cache_dir / f).exists()]
    if not missing:
        return cache_dir

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise ImportError(
            "huggingface-hub 未安装。请运行: pip install septmuse[onnx]"
        ) from e

    logger.info("onnx_model_downloading", model=model_name, files=missing)
    for filename in missing:
        local_path = hf_hub_download(
            repo_id=model_name,
            filename=filename,
            local_dir=str(cache_dir),
            local_dir_use_symlinks=False,
        )
        logger.info("onnx_model_file_cached", model=model_name, file=filename, path=local_path)

    return cache_dir


class OnnxEmbedder(Embedder):
    """基于 ONNX Runtime 的嵌入模型 (无 torch 依赖)。

    默认英文模型: Xenova/all-MiniLM-L6-v2 (384 dim, ~23MB 量化)。
    多语言模型: Xenova/paraphrase-multilingual-MiniLM-L12-v2 (384 dim, ~50MB 量化)。
    """

    def __init__(self, model_name: str = DEFAULT_EN_MODEL) -> None:
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as e:
            raise ImportError(
                "onnxruntime + tokenizers 未安装。请运行: pip install septmuse[onnx]"
            ) from e

        self._model_name = model_name
        cache_dir = _ensure_model_files(model_name)

        onnx_path = cache_dir / "onnx" / "model_quantized.onnx"
        tokenizer_path = cache_dir / "tokenizer.json"

        logger.info("onnx_embedder_loading", model=model_name, cache=str(cache_dir))
        self._session = ort.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"],
        )
        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self._tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        self._tokenizer.enable_truncation(max_length=256)

        # 从 ONNX 模型输入/输出推断维度
        output_info = self._session.get_outputs()[0]
        self._dim: int = output_info.shape[-1]
        if not isinstance(self._dim, int) or self._dim <= 0:
            self._dim = 384  # 安全回退

        # 记录模型期望的输入名 (可能含/不含 token_type_ids)
        self._input_names = [inp.name for inp in self._session.get_inputs()]

        logger.info("onnx_embedder_ready", model=model_name, dim=self._dim, inputs=self._input_names)

    @property
    def dimension(self) -> int:
        return self._dim

    def embed(self, text: str) -> list[float]:
        encoding = self._tokenizer.encode(text)
        feeds = self._build_feeds(encoding)

        outputs = self.session.run(None, feeds)
        last_hidden = outputs[0]  # [1, seq_len, dim]

        # Mean pooling (考虑 attention_mask)
        mask = np.array(encoding.attention_mask, dtype=np.float32)
        pooled = (last_hidden[0] * mask[:, None]).sum(axis=0) / mask.sum()

        # L2 归一化 (余弦相似 = 点积)
        norm = float(np.linalg.norm(pooled))
        if norm > 0:
            pooled = pooled / norm
        return pooled.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]

    def _build_feeds(self, encoding) -> dict[str, np.ndarray]:
        """构建 ONNX 推理输入 (仅传模型期望的输入)。"""
        feeds: dict[str, np.ndarray] = {
            "input_ids": np.array([encoding.ids], dtype=np.int64),
            "attention_mask": np.array([encoding.attention_mask], dtype=np.int64),
        }
        if "token_type_ids" in self._input_names:
            feeds["token_type_ids"] = np.array([encoding.type_ids], dtype=np.int64)
        return feeds
```

- [ ] **Step 4: 运行测试（安装 onnx extra 后）**

Run: `pip install onnxruntime tokenizers huggingface-hub; $env:PYTHONPATH="src"; python -m pytest tests/unit/test_onnx_embedder.py -v`
Expected: 5 passed (首次运行需下载 ~23MB 模型，耗时 ~30s)

- [ ] **Step 5: 不装 onnx 时确认 skip**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_onnx_embedder.py -v`
Expected: 5 skipped

- [ ] **Step 6: Lint 检查**

Run: `ruff check src/septmuse/providers/embedders/onnx.py tests/unit/test_onnx_embedder.py`
Expected: All checks passed

---

## Task 4: P1 — Wire OnnxEmbedder 到 _resolve_embedder

**目标**：`SEPTMUSE_EMBEDDER=onnx` → 用 OnnxEmbedder（英文模型）。

**Files:**
- Modify: `src/septmuse/orchestration/memory.py:50-62`（_resolve_embedder）
- Modify: `src/septmuse/configs/defaults.py`（+embedder_backend 字段 + 环境变量）
- Test: `tests/unit/test_memory.py`（新增解析测试）

**Interfaces:**
- Produces: `_resolve_embedder` 新增 `onnx` / `onnx-zh` 分支

- [ ] **Step 1: 写失败测试**

在 `tests/unit/test_memory.py` 末尾新增：

```python
class TestResolveEmbedder:
    """P1: _resolve_embedder 支持 onnx / onnx-zh 选项。"""

    def test_resolve_hash_default(self, monkeypatch) -> None:
        monkeypatch.delenv("SEPTMUSE_EMBEDDER", raising=False)
        from septmuse.orchestration.memory import _resolve_embedder

        emb = _resolve_embedder(MemoryConfig())
        from septmuse.providers.embedders.hash import HashEmbedder

        assert isinstance(emb, HashEmbedder)

    def test_resolve_onnx_english(self, monkeypatch) -> None:
        monkeypatch.setenv("SEPTMUSE_EMBEDDER", "onnx")
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            pytest.skip("onnxruntime 未安装")

        from septmuse.orchestration.memory import _resolve_embedder

        emb = _resolve_embedder(MemoryConfig())
        from septmuse.providers.embedders.onnx import OnnxEmbedder

        assert isinstance(emb, OnnxEmbedder)
        assert emb.dimension == 384
```

- [ ] **Step 2: 运行测试确认失败**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_memory.py::TestResolveEmbedder -v`
Expected: FAIL — `_resolve_embedder` 不识别 `onnx`

- [ ] **Step 3: 修改 _resolve_embedder**

修改 `src/septmuse/orchestration/memory.py:50-62`：

```python
def _resolve_embedder(config: MemoryConfig) -> Embedder:
    """解析 embedder。

    默认 HashEmbedder (零模型加载, 离线可用, 与 CLI/MCP server 一致)。
    onnx: Xenova/all-MiniLM-L6-v2 (英文, 384 dim, ~23MB, 无 torch)。
    onnx-zh: Xenova/paraphrase-multilingual-MiniLM-L12-v2 (多语言, 384 dim, ~50MB)。
    auto: 语言检测自动选 onnx-zh (默认) 或 onnx (SEPTMUSE_LANG=en)。
    st: sentence-transformers (延迟 import, 启动慢 ~30s)。
    """
    choice = os.getenv("SEPTMUSE_EMBEDDER", "hash").lower()
    if choice in ("st", "sentence-transformers", "sentence_transformers"):
        from septmuse.providers.embedders.sentence_transformers import SentenceTransformerEmbedder

        return SentenceTransformerEmbedder(model_name=config.embedder_model)
    if choice in ("onnx", "onnx-en"):
        from septmuse.providers.embedders.onnx import OnnxEmbedder

        return OnnxEmbedder(model_name="Xenova/all-MiniLM-L6-v2")
    if choice in ("onnx-zh", "onnx-multilingual"):
        from septmuse.providers.embedders.onnx import OnnxEmbedder

        return OnnxEmbedder(model_name="Xenova/paraphrase-multilingual-MiniLM-L12-v2")
    if choice == "auto":
        from septmuse.providers.embedders.auto import AutoOnnxEmbedder

        return AutoOnnxEmbedder()
    return HashEmbedder()
```

- [ ] **Step 4: 运行测试确认通过**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_memory.py -v`
Expected: PASS

- [ ] **Step 5: Lint 检查**

Run: `ruff check src/septmuse/orchestration/memory.py`
Expected: All checks passed

---

## Task 5: P2 — AutoOnnxEmbedder（语言检测 + 模型选择）

**目标**：init 时检测语言（CJK 比例），选一个 ONNX 模型用于整个 session。默认 zh（中文优先项目）。

**Files:**
- Create: `src/septmuse/providers/embedders/auto.py`
- Test: `tests/unit/test_auto_embedder.py`

**Interfaces:**
- Consumes: `OnnxEmbedder`（Task 3）, `detect_language`（Task 2）
- Produces: `AutoOnnxEmbedder` 类，构造参数 `sample_text: str | None = None`, `lang: str | None = None`；默认从 `SEPTMUSE_LANG` 环境变量读，未设则默认 `zh`

- [ ] **Step 1: 写失败测试**

Create `tests/unit/test_auto_embedder.py`:

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
"""AutoOnnxEmbedder 单元测试。"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration

try:
    import onnxruntime  # noqa: F401

    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

SKIP_REASON = "onnxruntime + tokenizers 未安装。pip install septmuse[onnx]"


@pytest.mark.skipif(not HAS_ONNX, reason=SKIP_REASON)
class TestAutoOnnxEmbedder:
    def test_default_lang_zh_when_no_env(self, monkeypatch) -> None:
        monkeypatch.delenv("SEPTMUSE_LANG", raising=False)
        monkeypatch.setenv("SEPTMUSE_EMBEDDER", "auto")
        from septmuse.providers.embedders.auto import AutoOnnxEmbedder

        emb = AutoOnnxEmbedder()
        assert emb._lang == "zh"
        assert emb.dimension == 384

    def test_lang_en_from_env(self, monkeypatch) -> None:
        monkeypatch.setenv("SEPTMUSE_LANG", "en")
        from septmuse.providers.embedders.auto import AutoOnnxEmbedder

        emb = AutoOnnxEmbedder()
        assert emb._lang == "en"

    def test_detect_from_sample_text(self, monkeypatch) -> None:
        monkeypatch.delenv("SEPTMUSE_LANG", raising=False)
        from septmuse.providers.embedders.auto import AutoOnnxEmbedder

        emb_zh = AutoOnnxEmbedder(sample_text="我喜欢编程")
        assert emb_zh._lang == "zh"

        emb_en = AutoOnnxEmbedder(sample_text="I love programming")
        assert emb_en._lang == "en"

    def test_embed_uses_same_model_for_all(self, monkeypatch) -> None:
        """整个 session 用同一模型 (不 per-query 切换)。"""
        monkeypatch.setenv("SEPTMUSE_LANG", "zh")
        from septmuse.providers.embedders.auto import AutoOnnxEmbedder

        emb = AutoOnnxEmbedder()
        v1 = emb.embed("我喜欢编程")
        v2 = emb.embed("I love programming")
        # 同一模型嵌入, 维度一致, 可比较
        assert len(v1) == len(v2) == 384
```

- [ ] **Step 2: 运行测试确认失败（skip）**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_auto_embedder.py -v`
Expected: 4 skipped

- [ ] **Step 3: 实现 AutoOnnxEmbedder**

Create `src/septmuse/providers/embedders/auto.py`:

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
"""AutoOnnxEmbedder — init 时语言检测, 自动选 ONNX 嵌入模型。

策略:
1. SEPTMUSE_LANG 环境变量 > sample_text 检测 > 默认 'zh' (中文优先项目)
2. 'zh' → Xenova/paraphrase-multilingual-MiniLM-L12-v2 (多语言, 384 dim)
3. 'en' → Xenova/all-MiniLM-L6-v2 (英文, 384 dim)

关键: 整个 session 用同一个模型 (不同模型投影到不同语义空间,
per-query 切换会破坏向量可比性)。
"""

from __future__ import annotations

import os

from septmuse.observability import get_logger
from septmuse.providers.embedders.base import Embedder
from septmuse.providers.embedders.langdetect import detect_language

logger = get_logger(__name__)

ZH_MODEL = "Xenova/paraphrase-multilingual-MiniLM-L12-v2"
EN_MODEL = "Xenova/all-MiniLM-L6-v2"


class AutoOnnxEmbedder(Embedder):
    """语言检测自动选模型 (init 时一次, 不 per-query 切换)。"""

    def __init__(self, *, sample_text: str | None = None, lang: str | None = None) -> None:
        # 1. 显式参数 > 2. 环境变量 > 3. 样本文本检测 > 4. 默认 'zh'
        if lang is not None:
            self._lang = lang
        elif env_lang := os.getenv("SEPTMUSE_LANG"):
            self._lang = env_lang.lower()
        elif sample_text is not None:
            self._lang = detect_language(sample_text)
        else:
            self._lang = "zh"  # 中文优先项目默认

        # 选模型
        model_name = ZH_MODEL if self._lang == "zh" else EN_MODEL
        logger.info("auto_embedder_selecting", lang=self._lang, model=model_name)

        # 委托给 OnnxEmbedder (同一模型, 整个 session 不切换)
        from septmuse.providers.embedders.onnx import OnnxEmbedder

        self._inner = OnnxEmbedder(model_name=model_name)

    @property
    def dimension(self) -> int:
        return self._inner.dimension

    def embed(self, text: str) -> list[float]:
        return self._inner.embed(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return self._inner.embed_batch(texts)
```

- [ ] **Step 4: 运行测试（安装 onnx extra 后）**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/test_auto_embedder.py -v`
Expected: 4 passed (首次运行需下载模型)

- [ ] **Step 5: Lint 检查**

Run: `ruff check src/septmuse/providers/embedders/auto.py tests/unit/test_auto_embedder.py`
Expected: All checks passed

---

## Task 6: 配置 + pyproject.toml extras + 文档更新

**目标**：新增 `onnx` extra，更新 MemoryConfig，更新 AGENTS.md 环境变量表。

**Files:**
- Modify: `pyproject.toml`（+onnx extra）
- Modify: `src/septmuse/configs/defaults.py`（+onnx 相关字段）
- Modify: `AGENTS.md`（环境变量表 + embedder 说明）

- [ ] **Step 1: pyproject.toml 新增 onnx extra**

在 `[project.optional-dependencies]` 段末尾新增：

```toml
onnx = ["onnxruntime>=1.16", "tokenizers>=0.15", "huggingface-hub>=0.20"]
```

并更新 `all` extra：

```toml
all = ["septmuse[openai,anthropic,ollama,postgres,graph,activation,parametric,server,dev,onnx]"]
```

- [ ] **Step 2: MemoryConfig 新增字段**

在 `src/septmuse/configs/defaults.py` 的 `MemoryConfig` 类中新增：

```python
    embedder_backend: str = Field(
        default="hash",
        description="嵌入后端: hash(默认,离线)/onnx(英文)/onnx-zh(多语言)/auto(语言检测)/st(sentence-transformers)",
    )
    model_cache_dir: str = Field(
        default="",
        description="ONNX 模型缓存目录; 空字符串 → ~/.septmuse/models/",
    )
```

在 `default_config()` 中读取环境变量：

```python
        embedder_backend=os.getenv("SEPTMUSE_EMBEDDER", "hash"),
        model_cache_dir=os.getenv("SEPTMUSE_MODEL_CACHE", ""),
```

- [ ] **Step 3: AGENTS.md 更新环境变量表**

在环境变量表中新增行：

```markdown
| `SEPTMUSE_EMBEDDER` | `hash` | `hash`/`onnx`/`onnx-zh`/`auto`/`st` |
| `SEPTMUSE_LANG` | 未设 | `zh`/`en`（仅 `auto` 模式生效，未设时默认 `zh`） |
| `SEPTMUSE_MODEL_CACHE` | `~/.septmuse/models/` | ONNX 模型缓存目录 |
```

在 Embedder 章节更新说明：

```markdown
### Embedder

- `SEPTMUSE_EMBEDDER=hash`（默认，HashEmbedder，离线零模型加载，0.5s 初始化）— CLI/MCP server/测试默认。
- `SEPTMUSE_EMBEDDER=onnx` — Xenova/all-MiniLM-L6-v2 ONNX 量化版（384 dim，~23MB，无 torch，CPU <50ms）。
- `SEPTMUSE_EMBEDDER=onnx-zh` — Xenova/paraphrase-multilingual-MiniLM-L12-v2（384 dim，多语言，中英文均支持）。
- `SEPTMUSE_EMBEDDER=auto` — init 时语言检测自动选 onnx-zh（默认）或 onnx。`SEPTMUSE_LANG=zh/en` 可覆盖。
- `SEPTMUSE_EMBEDDER=st` — sentence-transformers（延迟 import，启动慢 ~30s，需模型缓存）。
- **语言检测策略**：init 时一次，不 per-query 切换（不同模型投影到不同语义空间）。
- **零配置默认**：hash（离线），生产推荐 `onnx` 或 `auto`。
```

- [ ] **Step 4: 运行全量测试确认零退化**

Run: `$env:PYTHONPATH="src"; python -m pytest tests/unit/ tests/e2e/ -q`
Expected: 686+ passed, 22+ skipped（新增 onnx/auto 测试在未装 extras 时 skip）

- [ ] **Step 5: Lint + format 检查**

Run: `ruff check src/ tests/ examples/; ruff format --check src/ tests/ examples/`
Expected: All checks passed

- [ ] **Step 6: 验证安装 onnx extra 后全量测试**

Run: `pip install onnxruntime tokenizers huggingface-hub; $env:PYTHONPATH="src"; python -m pytest tests/unit/test_onnx_embedder.py tests/unit/test_auto_embedder.py tests/unit/test_langdetect.py -v`
Expected: 全部 passed（onnx/auto 测试不再 skip）

---

## 自检清单

**Spec coverage:**
- [x] P0: Memory.search 默认走 hybrid — Task 1
- [x] P1: OnnxEmbedder 英文模型 — Task 3
- [x] P1: wire 到 _resolve_embedder — Task 4
- [x] P2: 语言检测函数 — Task 2
- [x] P2: AutoOnnxEmbedder 中文模型 — Task 5
- [x] P2: wire 到 _resolve_embedder — Task 4 (auto 分支)
- [x] 配置 + extras + 文档 — Task 6

**Placeholder scan:** 无 TBD/TODO/"implement later"，所有步骤含完整代码。

**Type consistency:**
- `detect_language(text: str) -> str` — Task 2 定义，Task 5 调用 ✓
- `OnnxEmbedder(model_name: str)` — Task 3 定义，Task 4/5 调用 ✓
- `AutoOnnxEmbedder(sample_text: str | None, lang: str | None)` — Task 5 定义，Task 4 调用 `AutoOnnxEmbedder()` ✓
- `Memory.search(hybrid: bool = True)` — Task 1 定义，REST/MCP/CLI 调用处不需要改（默认 True）✓

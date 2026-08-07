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
"""捕获流水线 — SHA256 去重 → 隐私脱敏 → (压缩) → 双索引。

架构文档 §5.1 捕获方式: SHA-256 去重(5min 窗) → 隐私过滤 → 存 raw → LLM 压缩 → 向量化 → 双索引

阶段3 简化:
- 去重: DedupWindow (governance/approval.py)
- 脱敏: PrivacyFilter (governance/privacy.py)
- 压缩: 暂不 LLM 压缩, 原文存 (后续可接 LLM 摘要)
- 双索引: verbatim ORMMemoryStore + (可选) LLM cognify 抽取到 typed_store

详见 docs/specs/agent-memory-architecture.md §5.1 捕获方式。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from septmuse.capture.sanitize import PrivacyFilter
from septmuse.core.logging import get_logger
from septmuse.embedders.base import Embedder
from septmuse.governance.approval import DedupWindow, WriteValidator
from septmuse.llms.base import LLM
from septmuse.storage.base import MemoryStore
from septmuse.storage.relational_stores.typed_store import TypedMemoryStore

logger = get_logger(__name__)


@dataclass
class PipelineResult:
    """捕获流水线输出。"""

    captured: bool = False
    memory_id: str | None = None
    deduped: bool = False
    redacted: bool = False
    original_text: str = ""
    stored_text: str = ""
    text_hash: str = ""
    errors: list[str] = field(default_factory=list)


@dataclass
class PreprocessResult:
    """预处理结果 (去重+脱敏, 不写 store)。"""

    allowed: bool = False
    stored_text: str = ""
    text_hash: str | None = None
    redacted: bool = False
    reason: str | None = None


class CapturePipeline:
    """捕获流水线 (架构文档 §5.1)。

    流程: SHA256 去重 → 隐私脱敏 → (压缩) → 嵌入 → 双索引存储。

    用法:
        pipeline = CapturePipeline(store, embedder)
        result = pipeline.capture("user did something", user_id="alice")
        if result.captured:
            print(f"stored as {result.memory_id}")
    """

    def __init__(
        self,
        store: MemoryStore,
        embedder: Embedder,
        *,
        typed_store: TypedMemoryStore | None = None,
        llm: LLM | None = None,
        privacy_filter: PrivacyFilter | None = None,
        dedup_window: DedupWindow | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.typed_store = typed_store
        self.llm = llm
        self.privacy = privacy_filter or PrivacyFilter()
        self.validator = WriteValidator(dedup_window=dedup_window or DedupWindow())

    def preprocess(
        self,
        text: str,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
    ) -> PreprocessResult:
        """只做去重+脱敏, 不嵌入不写 store (避免与 Memory.add 双写)。

        Steps:
            1. SHA256 去重 (DedupWindow, per-user scope)
            2. 隐私脱敏 (PrivacyFilter)
        返回 PreprocessResult; allowed=True 时 caller 可自行 Memory.add(stored_text)。
        """
        result = PreprocessResult(allowed=False)
        if not text or not text.strip():
            result.reason = "empty text"
            return result

        # Step 1: SHA256 去重 (validate 内部会 add 到窗口, 二次同文本被拒)
        validation = self.validator.validate(text, user_id=user_id, agent_id=agent_id)
        if not validation.allowed:
            result.reason = validation.reason or "duplicate"
            result.text_hash = validation.text_hash or None
            return result
        result.text_hash = validation.text_hash

        # Step 2: 隐私脱敏
        cleaned = self.privacy.redact(text)
        result.redacted = cleaned != text
        result.stored_text = cleaned
        result.allowed = True
        return result

    def capture(
        self,
        text: str,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> PipelineResult:
        """执行捕获流水线。

        session_id: 会话 ID (None=不限)。
        Step 1: SHA256 去重 (DedupWindow, 5min 窗)
        Step 2: 隐私脱敏 (PrivacyFilter)
        Step 3: (压缩 — 暂不实现, 原文存)
        Step 4: 嵌入 (Embedder)
        Step 5: 双索引 (verbatim store + 可选 typed store cognify)
        """
        result = PipelineResult(original_text=text, stored_text=text)
        if not text or not text.strip():
            result.errors.append("empty text")
            return result

        # Step 1: SHA256 去重
        validation = self.validator.validate(text, user_id=user_id, agent_id=agent_id)
        if not validation.allowed:
            result.deduped = validation.dedup
            result.errors.append(validation.reason)
            return result
        result.text_hash = validation.text_hash

        # 校验通过后, 确定 effective_user_id (至少一个 ID 非空)
        effective_user_id = user_id if user_id else agent_id or "default"

        # Step 2: 隐私脱敏
        cleaned = self.privacy.redact(text)
        if cleaned != text:
            result.redacted = True
            result.stored_text = cleaned
        else:
            result.stored_text = cleaned

        # Step 3: (压缩 — 暂不实现, 后续接 LLM 摘要)
        # compressed = self._compress(cleaned) if self.llm else cleaned

        # Step 4: 嵌入
        emb = self.embedder.embed(result.stored_text)

        # Step 5a: verbatim 存储 (ORMMemoryStore)
        mid = self.store.add(
            result.stored_text,
            emb,
            user_id=effective_user_id,
            agent_id=agent_id,
            session_id=session_id,
            metadata={**(metadata or {}), "source": "capture_pipeline", "text_hash": result.text_hash},
        )
        result.memory_id = mid
        result.captured = True

        # Step 5b: (可选) LLM cognify 抽取到 typed_store
        if self.typed_store is not None and self.llm is not None:
            try:
                self._cognify(result.stored_text, user_id=effective_user_id)
            except Exception as e:
                result.errors.append(f"cognify failed: {e}")
                logger.warning("capture_cognify_failed", error=str(e))

        logger.info(
            "capture_pipeline_done",
            user_id=user_id,
            memory_id=mid,
            deduped=result.deduped,
            redacted=result.redacted,
        )
        return result

    def _cognify(self, text: str, *, user_id: str) -> None:
        """LLM cognify 抽取 (后续实现, 当前 placeholder)。"""
        # 后续: 调 FactExtractor.extract_and_store(text, user_id=user_id)
        # 当前: placeholder, 不做抽取
        pass

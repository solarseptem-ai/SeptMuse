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
"""V2 记忆编排入口 — DEPRECATED, 委托 Memory。

V2Memory 已降级为薄层委托, 所有编排逻辑 (remember/recall/forget/improve)
已并入 Memory 类。请直接使用 Memory。

用法:
    from septmuse.memory.main import Memory
    mem = Memory(config=MemoryConfig())
    mem.remember("text", user_id="alice")
    mem.recall("query", user_id="alice")
"""

from __future__ import annotations

import warnings
from typing import Any


def _normalize_messages(messages: Any) -> str:
    """把 messages 归一化为纯文本 (str 或 list[dict] → str)。"""
    if isinstance(messages, str):
        return messages.strip()
    if isinstance(messages, list):
        parts: list[str] = []
        for msg in messages:
            if isinstance(msg, dict):
                parts.append(str(msg.get("content", "")))
            elif isinstance(msg, str):
                parts.append(msg)
        return "\n".join(parts).strip()
    return str(messages).strip()


class V2Memory:
    """DEPRECATED: 委托 Memory.remember/recall/forget/improve。

    所有编排逻辑已并入 Memory 类。此类仅作向后兼容。
    """

    def __init__(self, memory: Any | None = None, *, config: Any | None = None) -> None:
        warnings.warn(
            "V2Memory is deprecated, use Memory directly",
            DeprecationWarning,
            stacklevel=2,
        )
        if memory is not None:
            self.mem = memory
        else:
            from septmuse.memory.main import Memory

            self.mem = Memory(config=config)

        # 兼容: 暴露 mem 的属性 (旧代码可能直接访问 v2.store 等)
        self.store = self.mem.store
        self.embedder = self.mem.embedder
        self.typed_store = self.mem.typed_store
        self.graph_store = self.mem.graph_store
        self.entity_store = self.mem.entity_store
        self.entity_extractor = self.mem.entity_extractor
        self.llm = self.mem.llm
        # 编排组件透传 (旧 V2 测试可能直接访问)
        self.working_memory = getattr(self.mem, "working_memory", None)
        self.semantic = self.mem.semantic
        self.episodic = self.mem.episodic
        self.procedural = self.mem.procedural
        self.capture = getattr(self.mem, "capture", None)
        self.token_budget = getattr(self.mem, "token_budget", None)
        self.meta = getattr(self.mem, "meta", None)
        self.evolution = getattr(self.mem, "evolution", None)
        self.forgetting = getattr(self.mem, "forgetting", None)

        # retrieval: 初始化 Memory 的延迟 HybridRetriever (兼容旧 V2 测试 v2.retrieval)
        if self.mem._retriever is None:
            from septmuse.retrieval.hybrid import HybridRetriever

            self.mem._retriever = HybridRetriever(
                self.mem.store,
                self.mem.embedder,
                entity_extractor=self.mem.entity_extractor,
                entity_store=self.mem.entity_store,
            )
        self.retrieval = self.mem._retriever

        # causal: CausalGraph (兼容旧 V2 测试 v2.causal)
        from septmuse.memory.causal import CausalGraph

        self.causal = CausalGraph(self.mem.typed_store, llm=self.mem.llm)

    def remember(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self.mem.remember(*args, **kwargs)

    def recall(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self.mem.recall(*args, **kwargs)

    def forget(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self.mem.forget(*args, **kwargs)

    def improve(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self.mem.improve(*args, **kwargs)

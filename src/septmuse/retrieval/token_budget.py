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
"""token 预算裁剪 — 检索注入前裁剪记忆到 token 预算内。"""

from __future__ import annotations

from dataclasses import dataclass, field

from septmuse.core.logging import get_logger

logger = get_logger(__name__)

# 默认 token 预算 (对齐 Agent Memory 2000)
DEFAULT_TOKEN_BUDGET = 2000

# 近似 token 比率: ~4 chars/token (对齐 tiktoken 粗估)
CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    """近似 token 计数 (chars/4, 对齐 tiktoken 粗估)。

    空串返回 0; 非串强转 str。
    """
    if not text:
        return 0
    return max(1, len(text) // CHARS_PER_TOKEN)


@dataclass
class BudgetItem:
    """预算裁剪的输入项 (记忆检索结果)。"""

    text: str
    score: float = 0.0
    metadata: dict | None = None
    id: str | None = None  # 原始 memory_id (裁剪后保留, 修 recall id 丢失 bug)


@dataclass
class BudgetResult:
    """预算裁剪输出。"""

    items: list[BudgetItem] = field(default_factory=list)
    used_tokens: int = 0
    budget: int = DEFAULT_TOKEN_BUDGET
    dropped: int = 0

    @property
    def texts(self) -> list[str]:
        """裁剪后的文本列表 (注入 LLM context 用)。"""
        return [item.text for item in self.items]


class TokenBudget:
    """token 预算裁剪器。"""

    def __init__(self, budget: int = DEFAULT_TOKEN_BUDGET) -> None:
        self.budget = budget

    def fit(self, items: list[BudgetItem]) -> BudgetResult:
        """按分数降序贪心填充, 超预算则跳过。"""
        sorted_items = sorted(items, key=lambda x: x.score, reverse=True)
        chosen: list[BudgetItem] = []
        used = 0
        dropped = 0

        for item in sorted_items:
            tokens = estimate_tokens(item.text)
            if used + tokens > self.budget:
                dropped += 1
                continue
            chosen.append(item)
            used += tokens

        logger.info("token_budget_fit", budget=self.budget, used=used, chosen=len(chosen), dropped=dropped)
        return BudgetResult(items=chosen, used_tokens=used, budget=self.budget, dropped=dropped)

    def fit_texts(self, texts: list[str], scores: list[float] | None = None) -> list[str]:
        """直接传文本列表, 返回裁剪后文本列表。"""
        if scores is None:
            scores = [0.0] * len(texts)
        items = [BudgetItem(text=t, score=s) for t, s in zip(texts, scores, strict=True)]
        return self.fit(items).texts

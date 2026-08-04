#  Copyright 2026 The sonhhxg0529 Authors. All Rights Reserved.
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""策略工厂 — 按名称创建策略实例。"""

from __future__ import annotations

from typing import ClassVar

from septmuse.rerankers.strategies.base import BaseRerankerStrategy
from septmuse.rerankers.strategies.full_memory import FullMemoryStrategy
from septmuse.rerankers.strategies.single_turn import SingleTurnStrategy


class RerankerStrategyFactory:
    """按策略名创建实例: full_memory (默认) / single_turn。"""

    _registry: ClassVar[dict[str, type[BaseRerankerStrategy]]] = {
        "full_memory": FullMemoryStrategy,
        "single_turn": SingleTurnStrategy,
    }

    @classmethod
    def create(cls, name: str = "full_memory") -> BaseRerankerStrategy:
        if name not in cls._registry:
            raise ValueError(f"Unknown reranker strategy: {name} (choose from {list(cls._registry)})")
        return cls._registry[name]()

    @classmethod
    def list_strategies(cls) -> list[str]:
        return list(cls._registry)

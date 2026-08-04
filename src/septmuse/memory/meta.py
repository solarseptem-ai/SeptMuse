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
"""V2 元认知子组件 — 聚合 L0 路由 + L1 覆盖报告 + L2 策略自调。

MetacognitionLayer 聚合 meta/ 下三子模块:
- L0 MetaRouter: 查询嵌入 vs 命名空间描述嵌入, 路由到匹配命名空间
- L1 CoverageAnalyzer: 扫描全部命名空间, 生成覆盖报告
- L2 StrategyAdapter: 基于覆盖报告自调检索策略

详见 docs/specs/2026-08-04-v2-memory-architecture.md §4 + §6。
"""

from __future__ import annotations

from septmuse.core.logging import get_logger
from septmuse.embedders.base import Embedder
from septmuse.meta.coverage import CoverageAnalyzer, CoverageReport
from septmuse.meta.router import MetaRouter, RouteResult
from septmuse.meta.strategy import StrategyAdapter, StrategyResult
from septmuse.storage.base import MemoryStore
from septmuse.storage.relational_stores.typed_store import TypedMemoryStore

logger = get_logger(__name__)


class MetacognitionLayer:
    """V2 元认知层 — 聚合 L0 路由 + L1 覆盖 + L2 策略。

    用法:
        meta = MetacognitionLayer(embedder, store, typed_store)
        route = meta.route("what does alice like?")
        report = meta.analyze_coverage(user_id="alice")
        strategy = meta.adapt_strategy(report)
    """

    def __init__(
        self,
        embedder: Embedder,
        store: MemoryStore,
        typed_store: TypedMemoryStore,
    ) -> None:
        self.router = MetaRouter(embedder)
        self.coverage = CoverageAnalyzer(store, typed_store)
        self.strategy = StrategyAdapter()

    def route(self, query: str) -> RouteResult:
        """L0 路由: 查询 → 匹配命名空间。"""
        return self.router.route(query)

    def analyze_coverage(self, *, user_id: str) -> CoverageReport:
        """L1 覆盖报告: 扫描全部命名空间。"""
        return self.coverage.analyze(user_id=user_id)

    def adapt_strategy(self, report: CoverageReport) -> StrategyResult:
        """L2 策略自调: 基于覆盖报告自调检索策略。"""
        return self.strategy.adapt(report)

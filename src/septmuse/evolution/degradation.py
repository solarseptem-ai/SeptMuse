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
"""规则退化策略。"""

from __future__ import annotations

from dataclasses import dataclass

from septmuse.core.logging import get_logger

logger = get_logger(__name__)

# Cass 退化阈值: harmful >= 3 且 harmful > helpful → 废弃
DEFAULT_DEPRECATION_THRESHOLD = 3


@dataclass
class DegradationRecord:
    """退化策略输入 (任意记忆项的 helpful/harmful 状态)。"""

    helpful_count: int = 0
    harmful_count: int = 0
    deprecated: bool = False

    @property
    def confidence(self) -> float:
        """置信度 = helpful / (helpful + harmful) (Cass 模式)。

        无记录时默认 0.5 (中性)。
        """
        total = self.helpful_count + self.harmful_count
        if total == 0:
            return 0.5
        return self.helpful_count / total


class DegradationPolicy:
    """规则退化策略。"""

    def __init__(self, deprecation_threshold: int = DEFAULT_DEPRECATION_THRESHOLD) -> None:
        self.threshold = deprecation_threshold

    def record_outcome(self, record: DegradationRecord, helpful: bool) -> None:
        """记录一次应用结果。"""
        if helpful:
            record.helpful_count += 1
        else:
            record.harmful_count += 1
        # Cass 退化: harmful > helpful 且 harmful >= 阈值 → 废弃
        if record.harmful_count > record.helpful_count and record.harmful_count >= self.threshold:
            record.deprecated = True
        logger.debug(
            "degradation_recorded",
            helpful=record.helpful_count,
            harmful=record.harmful_count,
            deprecated=record.deprecated,
        )

    def should_deprecate(self, record: DegradationRecord) -> bool:
        """判断是否应废弃 (Cass 退化条件)。"""
        return record.harmful_count > record.helpful_count and record.harmful_count >= self.threshold

    def should_inject(self, record: DegradationRecord) -> bool:
        """是否应注入 context (Cass 退化: 废弃规则不注入)。"""
        return not record.deprecated

    def should_inject_with_confidence(self, record: DegradationRecord, min_confidence: float = 0.0) -> bool:
        """是否应注入 + 置信度门槛 (SeptMuse 增量)。"""
        return self.should_inject(record) and record.confidence >= min_confidence

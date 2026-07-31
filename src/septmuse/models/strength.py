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
"""记忆强度模型 — Ebbinghaus 遗忘曲线 (架构文档 §6.2 自研)。

14 家开源均无强度衰减 + 主动复述。SeptMuse 新增:
- MemoryStrength: 每条长时记忆带强度字段
- decay: R = exp(-t / S), strength 随时间衰减, 访问时回升
- final_score = relevance × strength, 低强度记忆自然下沉
- agent idle 时主动复述高价值低强度记忆

公式 (架构文档 §6.2):
    R = exp(-elapsed / (base_value * S_FACTOR))
    strength = min(1.0, strength + REHEARSAL_GAIN)  # rehearse 时回升
"""

from __future__ import annotations

import math
import uuid
from datetime import datetime, timezone

from sqlmodel import Field, SQLModel

# 稳定性因子: base_value * S_FACTOR = 记忆稳定性 (秒)
# 86400 = 1 天: base_value=1.0 → 1 天稳定性; base_value=0.5 → 半天
S_FACTOR = 86400.0

# 每次复述的强度回升量
REHEARSAL_GAIN = 0.2

# 复述候选阈值: strength < 0.3 且 base_value > 0.7
REHEARSE_STRENGTH_THRESHOLD = 0.3
REHEARSE_BASE_VALUE_THRESHOLD = 0.7

# 归档阈值: strength < 0.1
ARCHIVE_THRESHOLD = 0.1


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return f"str-{uuid.uuid4()}"


class MemoryStrength(SQLModel, table=True):
    """记忆强度 — Ebbinghaus 遗忘曲线 (架构文档 §6.2 自研)。

    每条长时记忆对应一条 MemoryStrength 记录,
    跟踪 strength / last_accessed / access_count / base_value。

    strength 随时间衰减 (decay), 访问时回升 (rehearse)。
    final_score = relevance × strength, 低强度记忆自然下沉。
    """

    __tablename__ = "septmuse_strength"  # type: ignore[assignment]

    id: str = Field(default_factory=_new_id, primary_key=True)
    memory_id: str = Field(index=True, description="关联的记忆 ID")
    user_id: str = Field(index=True, description="用户 ID (多租户隔离)")

    strength: float = Field(default=1.0, description="当前强度 [0,1]")
    last_accessed: datetime = Field(default_factory=_utcnow, description="最后访问时间 UTC")
    access_count: int = Field(default=0, description="访问次数")
    base_value: float = Field(default=0.5, description="内禀价值 (规则/事实/偏好)")

    created_at: datetime = Field(default_factory=_utcnow, description="创建时间 UTC")
    is_deleted: bool = Field(default=False, description="软删除")
    archived: bool = Field(default=False, description="是否已归档 (冷存储, 不参与默认检索)")

    def decay(self, now: datetime | None = None) -> float:
        """计算当前衰减后的强度 (Ebbinghaus: R = exp(-t / S))。

        不修改 self.strength, 只返回计算值。
        调用方应在检索排序时用此值。
        """
        if now is None:
            now = _utcnow()
        # SQLite 存储 datetime 可能丢失 tzinfo, 统一为 UTC
        last = self.last_accessed
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        elapsed = max(0.0, (now - last).total_seconds())
        stability = max(1.0, self.base_value * S_FACTOR)
        return min(1.0, math.exp(-elapsed / stability))

    def rehearse(self, now: datetime | None = None) -> None:
        """主动复述: 访问一次, strength 回升 + access_count+1 (架构文档 §6.2)。

        strength = min(1.0, decayed_strength + REHEARSAL_GAIN)
        """
        if now is None:
            now = _utcnow()
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        decayed = self.decay(now)
        self.strength = min(1.0, decayed + REHEARSAL_GAIN)
        self.access_count += 1
        self.last_accessed = now
        # 归档后复述 → 解除归档
        if self.archived and self.strength > ARCHIVE_THRESHOLD:
            self.archived = False

    def should_rehearse(self, now: datetime | None = None) -> bool:
        """是否需要复述 (strength < 阈值 且 base_value > 阈值)。"""
        if now is None:
            now = _utcnow()
        decayed = self.decay(now)
        return (
            decayed < REHEARSE_STRENGTH_THRESHOLD
            and self.base_value > REHEARSE_BASE_VALUE_THRESHOLD
            and not self.is_deleted
        )

    def should_archive(self, now: datetime | None = None) -> bool:
        """是否需要归档 (strength < 0.1 且已归档未触发)。"""
        if now is None:
            now = _utcnow()
        decayed = self.decay(now)
        return decayed < ARCHIVE_THRESHOLD and not self.archived and not self.is_deleted

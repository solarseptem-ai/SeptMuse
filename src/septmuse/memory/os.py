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
"""MemOS 编排入口 (架构文档 §7.2 自研, 借鉴 MemOS mem_os)。

MemOS 是高级编排入口, 封装 MemCube + 源同步器 + 演化触发器,
提供端到端记忆回路: 捕获→存储→检索→演化→同步。

注: cognee 无 MemOS 概念 (探查确认), 本模块为架构文档自研设计。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from septmuse.core.logging import get_logger
from septmuse.memory.cube import MemCube, MemCubeConfig
from septmuse.memory.main import Memory

logger = get_logger(__name__)


@dataclass
class MemOSConfig:
    """MemOS 配置。"""

    auto_evolve: bool = True
    auto_rehearse: bool = False
    auto_sync: bool = True


class MemOS:
    """MemOS 编排入口 (架构文档 §7.2 自研)。

    端到端记忆回路: 捕获→存储→检索→演化→同步。

    用法:
        mos = MemOS.create_zero_config(user_id="alice")
        mos.add("alice likes python", memory_type="semantic")
        results = mos.search("alice likes")
        report = mos.analyze()
    """

    def __init__(self, cube: MemCube, config: MemOSConfig | None = None) -> None:
        self.cube = cube
        self.config = config or MemOSConfig()
        self.memory: Memory = cube.memory

    @classmethod
    def create_zero_config(cls, *, user_id: str, agent_id: str | None = None, **kwargs: Any) -> MemOS:
        """零配置创建 (架构文档 §12.4)。"""
        from septmuse import Memory as Mem
        from septmuse import MemoryConfig

        mem_kwargs = {k: v for k, v in kwargs.items() if k in ("config", "embedder", "store", "llm")}
        if "config" not in mem_kwargs:
            mem_kwargs["config"] = MemoryConfig(db_path=":memory:")
        memory = Mem(**mem_kwargs)
        cube_config = MemCubeConfig(user_id=user_id, agent_id=agent_id)
        cube = MemCube(memory, config=cube_config)
        return cls(cube)

    def add(self, content: str, *, memory_type: str = "verbatim", **kwargs: Any) -> dict[str, Any]:
        """添加记忆 (MemCube 路由) + 自动演化。"""
        result = self.cube.add(content, memory_type=memory_type, **kwargs)

        if self.config.auto_evolve and "id" in result:
            mid = result["id"]
            try:
                self.memory.link_on_add(mid, content, user_id=self.cube.config.user_id)
            except Exception as e:
                logger.warning("memos_auto_evolve_failed", error=str(e))
        elif self.config.auto_evolve and "results" in result:
            for r in result["results"]:
                if "id" in r:
                    try:
                        self.memory.link_on_add(r["id"], content, user_id=self.cube.config.user_id)
                    except Exception as e:
                        logger.warning("memos_auto_evolve_failed", error=str(e))

        return result

    def search(
        self, query: str, *, memory_type: str | None = None, top_k: int = 5, **kwargs: Any
    ) -> list[dict[str, Any]]:
        """检索记忆 (元认知路由 + 多类型)。"""
        return self.cube.search(query, memory_type=memory_type, top_k=top_k, **kwargs)

    def capture(self, text: str, **kwargs: Any) -> dict[str, Any]:
        """PostToolUse 捕获 (架构文档 §5.1)。"""
        uid = kwargs.pop("user_id", self.cube.config.user_id)
        return self.memory.capture(text, user_id=uid, **kwargs)

    def analyze(self) -> dict[str, Any]:
        """元认知分析 (L1 覆盖 + L2 策略)。"""
        return self.cube.analyze()

    def evolve(self) -> dict[str, Any]:
        """手动触发演化 (Dream + Reflect)。"""
        uid = self.cube.config.user_id
        dream_result = self.memory.dream(user_id=uid)
        reflect_result = self.memory.reflect(user_id=uid)
        return {"dream": dream_result, "reflect": reflect_result}

    def rehearse(self) -> dict[str, Any]:
        """主动复述 (遗忘曲线, 架构文档 §6.2)。"""
        return self.cube.rehearse()

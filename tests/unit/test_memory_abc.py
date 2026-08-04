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
"""验证 V2 记忆 ABC 契约 — MemoryABC / ShortTermMemory / LongTermMemory。

详见 docs/specs/2026-08-04-v2-memory-architecture.md §2。
"""

from __future__ import annotations

from abc import ABC

import pytest

from septmuse.memory.base import LongTermMemory, MemoryABC, ShortTermMemory
from septmuse.memory.episodic import EpisodicMemory
from septmuse.memory.procedural import ProceduralMemory
from septmuse.memory.semantic import SemanticMemory
from septmuse.memory.working_memory import WorkingMemory


class TestMemoryABCContract:
    """ABC 契约验证。"""

    def test_memory_abc_is_abc_subclass(self):
        """MemoryABC 继承 ABC。"""
        assert issubclass(MemoryABC, ABC)

    def test_short_term_memory_is_memory_abc(self):
        """ShortTermMemory 继承 MemoryABC。"""
        assert issubclass(ShortTermMemory, MemoryABC)

    def test_long_term_memory_is_memory_abc(self):
        """LongTermMemory 继承 MemoryABC。"""
        assert issubclass(LongTermMemory, MemoryABC)

    def test_short_term_memory_abstract_methods(self):
        """ShortTermMemory 有 3 个 abstractmethod。"""
        abstract_methods = ShortTermMemory.__abstractmethods__
        assert "compile_to_prompt" in abstract_methods
        assert "get_limit" in abstract_methods
        assert "evict_overflow" in abstract_methods

    def test_long_term_memory_abstract_methods(self):
        """LongTermMemory 有 3 个 abstractmethod。"""
        abstract_methods = LongTermMemory.__abstractmethods__
        assert "invalidate" in abstract_methods
        assert "get_history" in abstract_methods
        assert "get_all" in abstract_methods

    def test_cannot_instantiate_abc_directly(self):
        """不能直接实例化 ABC。"""
        with pytest.raises(TypeError):
            ShortTermMemory()  # type: ignore[abstract]
        with pytest.raises(TypeError):
            LongTermMemory()  # type: ignore[abstract]


class TestV2SubcomponentABC:
    """V2 子组件 ABC 注册验证。"""

    def test_working_memory_is_short_term(self):
        """WorkingMemory 继承 ShortTermMemory。"""
        assert issubclass(WorkingMemory, ShortTermMemory)

    def test_semantic_is_long_term(self):
        """SemanticMemory 继承 LongTermMemory。"""
        assert issubclass(SemanticMemory, LongTermMemory)

    def test_episodic_is_long_term(self):
        """EpisodicMemory 继承 LongTermMemory。"""
        assert issubclass(EpisodicMemory, LongTermMemory)

    def test_procedural_is_long_term(self):
        """ProceduralMemory 继承 LongTermMemory。"""
        assert issubclass(ProceduralMemory, LongTermMemory)

    def test_isinstance_short_vs_long(self, tmp_path, monkeypatch):
        """isinstance 判断短期 vs 长期。"""
        monkeypatch.setenv("SEPTMUSE_DB_PATH", str(tmp_path / "abc_test.db"))
        from septmuse.configs import default_config
        from septmuse.memory import V2Memory

        v2 = V2Memory(config=default_config())
        assert isinstance(v2.working_memory, ShortTermMemory)
        assert isinstance(v2.working_memory, MemoryABC)
        assert not isinstance(v2.working_memory, LongTermMemory)

        assert isinstance(v2.semantic, LongTermMemory)
        assert isinstance(v2.semantic, MemoryABC)
        assert not isinstance(v2.semantic, ShortTermMemory)

        assert isinstance(v2.episodic, LongTermMemory)
        assert isinstance(v2.procedural, LongTermMemory)

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
"""Block 工作记忆单元测试 — 固化对齐 letta BasicBlockMemory 的行为。

覆盖:
- 默认块集 (human + persona)
- core_memory_append: current + "\\n" + content
- core_memory_replace: 精确匹配替换, 不存在 raise ValueError
- get_block: 不存在 raise KeyError
- set_block: 替换同 label, 否则新增
- update_block_value: 非字符串 raise ValueError
- compile_to_xml: 格式正确
"""

from __future__ import annotations

import pytest

from septmuse.models.block import Block, WorkingMemory


@pytest.fixture()
def wm() -> WorkingMemory:
    return WorkingMemory(agent_id="agent-test")


class TestDefaultBlocks:
    def test_default_labels(self, wm: WorkingMemory) -> None:
        assert wm.list_block_labels() == ["human", "persona"]

    def test_default_values_empty(self, wm: WorkingMemory) -> None:
        assert wm.get_block("human").value == ""
        assert wm.get_block("persona").value == ""


class TestCoreMemoryAppend:
    def test_append_adds_newline(self, wm: WorkingMemory) -> None:
        wm.core_memory_append("human", "Name: Alice")
        assert wm.get_block("human").value == "\nName: Alice"

    def test_append_multiple(self, wm: WorkingMemory) -> None:
        wm.core_memory_append("human", "Name: Alice")
        wm.core_memory_append("human", "Likes: Python")
        assert wm.get_block("human").value == "\nName: Alice\nLikes: Python"

    def test_append_to_unknown_label_raises(self, wm: WorkingMemory) -> None:
        with pytest.raises(KeyError):
            wm.core_memory_append("unknown", "x")


class TestCoreMemoryReplace:
    def test_replace_exact_match(self, wm: WorkingMemory) -> None:
        wm.core_memory_append("human", "Likes: Python, hiking")
        wm.core_memory_replace("human", "hiking", "skiing, coffee")
        assert wm.get_block("human").value == "\nLikes: Python, skiing, coffee"

    def test_replace_delete_by_empty(self, wm: WorkingMemory) -> None:
        wm.core_memory_append("human", "Likes: Python, hiking")
        wm.core_memory_replace("human", "Likes: Python, hiking", "")
        assert wm.get_block("human").value == "\n"

    def test_replace_old_not_found_raises(self, wm: WorkingMemory) -> None:
        wm.core_memory_append("human", "Name: Alice")
        with pytest.raises(ValueError, match="Old content '不存在' not found"):
            wm.core_memory_replace("human", "不存在", "x")


class TestGetBlock:
    def test_get_existing(self, wm: WorkingMemory) -> None:
        assert wm.get_block("human").label == "human"

    def test_get_unknown_raises_keyerror(self, wm: WorkingMemory) -> None:
        with pytest.raises(KeyError, match="does not exist"):
            wm.get_block("unknown")


class TestUpdateBlockValue:
    def test_update_existing(self, wm: WorkingMemory) -> None:
        wm.update_block_value("human", "new value")
        assert wm.get_block("human").value == "new value"

    def test_update_non_string_raises(self, wm: WorkingMemory) -> None:
        with pytest.raises(ValueError, match="must be a string"):
            wm.update_block_value("human", 123)  # type: ignore[arg-type]

    def test_update_unknown_raises(self, wm: WorkingMemory) -> None:
        with pytest.raises(ValueError, match="does not exist"):
            wm.update_block_value("unknown", "x")


class TestSetBlock:
    def test_set_new_block(self, wm: WorkingMemory) -> None:
        wm.set_block(Block(agent_id="agent-test", label="task", value="t1"))
        assert "task" in wm.list_block_labels()
        assert wm.get_block("task").value == "t1"

    def test_set_replace_existing(self, wm: WorkingMemory) -> None:
        wm.set_block(Block(agent_id="agent-test", label="task", value="t1"))
        wm.set_block(Block(agent_id="agent-test", label="task", value="t2"))
        assert wm.get_block("task").value == "t2"
        assert wm.list_block_labels().count("task") == 1


class TestCompileToXml:
    def test_xml_structure(self, wm: WorkingMemory) -> None:
        wm.core_memory_append("human", "Name: Alice")
        xml = wm.compile_to_xml()
        assert xml.startswith("<memory>")
        assert xml.endswith("</memory>")
        assert '<block label="human">' in xml
        assert '<block label="persona">' in xml
        assert "Name: Alice" in xml

    def test_xml_escapes_special_chars(self, wm: WorkingMemory) -> None:
        wm.core_memory_append("human", "a < b & c > d")
        xml = wm.compile_to_xml()
        assert "a &lt; b &amp; c &gt; d" in xml
        assert "< b" not in xml  # 原文不应直接出现

    def test_empty_values(self, wm: WorkingMemory) -> None:
        xml = wm.compile_to_xml()
        assert '<block label="human"></block>' in xml


class TestToDict:
    def test_to_dict_has_compiled(self, wm: WorkingMemory) -> None:
        d = wm.to_dict()
        assert d["agent_id"] == "agent-test"
        assert "compiled" in d
        assert len(d["blocks"]) == 2

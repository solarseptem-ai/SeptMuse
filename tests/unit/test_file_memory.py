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
"""文件记忆单元测试 — Markdown + wikilinks + backlinks。

固化 (架构文档 §5, 源码参考 ReMe FileNode/FileLink):
- write/read roundtrip (frontmatter + body + wikilinks)
- extract_wikilinks
- find_backlinks (图遍历)
- list_files / delete
"""

from __future__ import annotations

import pytest

from septmuse.storage.file.markdown import (
    FileMemoryStore,
    extract_wikilinks,
    parse_front_matter,
)


@pytest.fixture()
def store(tmp_path) -> FileMemoryStore:
    return FileMemoryStore(tmp_path)


class TestFrontMatter:
    def test_parse_with_frontmatter(self) -> None:
        content = "---\nname: doc\ndescription: test\n---\n# body"
        fm, body = parse_front_matter(content)
        assert fm.name == "doc"
        assert fm.description == "test"
        assert body == "# body"

    def test_parse_without_frontmatter(self) -> None:
        content = "# just body"
        fm, body = parse_front_matter(content)
        assert fm.name == ""
        assert body == "# just body"

    def test_parse_extras(self) -> None:
        content = "---\nname: doc\ncustom_key: custom_value\n---\nbody"
        fm, _ = parse_front_matter(content)
        assert fm.extras.get("custom_key") == "custom_value"


class TestWikilinks:
    def test_extract_simple(self) -> None:
        links = extract_wikilinks("see [[target]] here", "src.md")
        assert len(links) == 1
        assert links[0].target_path == "target"

    def test_extract_with_anchor(self) -> None:
        links = extract_wikilinks("[[target#section]]", "src.md")
        assert links[0].target_path == "target"
        assert links[0].target_anchor == "section"

    def test_extract_multiple(self) -> None:
        links = extract_wikilinks("[[a]] and [[b]] and [[c]]", "src.md")
        assert len(links) == 3

    def test_extract_none(self) -> None:
        assert extract_wikilinks("no links here", "src.md") == []


class TestFileStoreWriteRead:
    def test_write_read_roundtrip(self, store: FileMemoryStore) -> None:
        node = store.write("test.md", "body text [[link]]", name="Test", description="d")
        assert len(node.links) == 1

        read = store.read("test.md")
        assert read is not None
        assert read.front_matter.name == "Test"
        assert "body text" in store.workspace.joinpath("test.md").read_text()

    def test_read_nonexistent(self, store: FileMemoryStore) -> None:
        assert store.read("nope.md") is None

    def test_list_files(self, store: FileMemoryStore) -> None:
        store.write("a.md", "a")
        store.write("b.md", "b")
        assert len(store.list_files()) == 2

    def test_delete(self, store: FileMemoryStore) -> None:
        store.write("del.md", "x")
        assert store.delete("del.md") is True
        assert store.delete("del.md") is False

    def test_path_escape_rejected(self, store: FileMemoryStore) -> None:
        with pytest.raises(ValueError, match="escapes workspace"):
            store.write("../escape.md", "x")


class TestBacklinks:
    def test_find_backlinks(self, store: FileMemoryStore) -> None:
        store.write("a.md", "refs [[b]]")
        store.write("c.md", "also refs [[b]]")
        backlinks = store.find_backlinks("b")
        assert len(backlinks) == 2
        assert {bl.source_path for bl in backlinks} == {"a.md", "c.md"}

    def test_no_backlinks(self, store: FileMemoryStore) -> None:
        store.write("a.md", "no links")
        assert store.find_backlinks("target") == []

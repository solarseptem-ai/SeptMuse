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
"""文件记忆后端 — Markdown + frontmatter + wikilinks (人可读真相源)。"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from septmuse.core.logging import get_logger

logger = get_logger(__name__)

WIKILINK_RE = re.compile(r"\[\[([^\]]+?)]]")


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Schema (源码参考 ReMe file_front_matter/file_link/file_node)
# ---------------------------------------------------------------------------


class FileFrontMatter(BaseModel):
    """Markdown front matter，未知键保留为 extras。"""

    model_config = ConfigDict(extra="allow")

    name: str = Field(default="", description="文档名")
    description: str = Field(default="", description="文档描述")

    @property
    def extras(self) -> dict[str, Any]:
        return self.__pydantic_extra__ or {}


class FileLink(BaseModel):
    """文件间 wikilink 关系。"""

    model_config = ConfigDict(extra="forbid")

    source_path: str = Field(description="源文件相对路径")
    target_path: str = Field(description="目标文件相对路径")
    target_anchor: str | None = Field(default=None, description="标题/锚点 (# 后)")
    predicate: str | None = Field(default=None, description="datavogy typed-link 谓词")


class FileNode(BaseModel):
    """文件节点: path + front_matter + links。"""

    path: str = Field(description="相对工作区路径")
    st_mtime: float = Field(description="文件系统 mtime")
    front_matter: FileFrontMatter = Field(default_factory=FileFrontMatter)
    links: list[FileLink] = Field(default_factory=list, description="出向 wikilinks")


# ---------------------------------------------------------------------------
# Markdown 读写 (Basic Memory 双向同步模式)
# ---------------------------------------------------------------------------


def parse_front_matter(content: str) -> tuple[FileFrontMatter, str]:
    """解析 Markdown frontmatter，返回 (front_matter, body)。"""
    fm = FileFrontMatter()
    if not content.startswith("---"):
        return fm, content
    parts = content.split("---", 2)
    if len(parts) < 3:
        return fm, content
    yaml_block = parts[1].strip()
    body = parts[2].lstrip("\n")
    for line in yaml_block.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key in ("name", "description"):
            setattr(fm, key, value)
        else:
            fm.__pydantic_extra__ = fm.__pydantic_extra__ or {}
            fm.__pydantic_extra__[key] = value
    return fm, body


def extract_wikilinks(body: str, source_path: str) -> list[FileLink]:
    """从正文提取 [[wikilinks]] (源码参考 ReMe FileLink 解析)。"""
    links: list[FileLink] = []
    for match in WIKILINK_RE.finditer(body):
        target = match.group(1)
        anchor = None
        if "#" in target:
            target, anchor = target.split("#", 1)
        links.append(
            FileLink(
                source_path=source_path,
                target_path=target.strip(),
                target_anchor=anchor.strip() if anchor else None,
                predicate=None,
            )
        )
    return links


def render_markdown(node: FileNode, body: str) -> str:
    """渲染 FileNode + body 为 Markdown (frontmatter + 正文)。"""
    fm_lines = ["---"]
    fm_lines.append(f"name: {node.front_matter.name}")
    fm_lines.append(f"description: {node.front_matter.description}")
    for k, v in node.front_matter.extras.items():
        fm_lines.append(f"{k}: {v}")
    fm_lines.append("---")
    fm_lines.append("")
    return "\n".join(fm_lines) + body


# ---------------------------------------------------------------------------
# FileMemoryStore (Basic Memory: Markdown=真相源, 索引可重建)
# ---------------------------------------------------------------------------


class FileMemoryStore:
    """文件记忆存储 (Markdown 为真相源，索引可重建)。"""

    def __init__(self, workspace: str | Path) -> None:
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)
        logger.info("file_store_ready", workspace=str(self.workspace))

    def _resolve_path(self, rel_path: str) -> Path:
        """解析相对工作区路径，防越界。"""
        p = (self.workspace / rel_path).resolve()
        if not str(p).startswith(str(self.workspace.resolve())):
            raise ValueError(f"path escapes workspace: {rel_path}")
        return p

    def write(
        self,
        rel_path: str,
        body: str,
        *,
        name: str = "",
        description: str = "",
        extras: dict[str, Any] | None = None,
    ) -> FileNode:
        """写入 Markdown 文件。"""
        path = self._resolve_path(rel_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        fm = FileFrontMatter(name=name or rel_path, description=description)
        if extras:
            fm.__pydantic_extra__ = extras

        node = FileNode(
            path=rel_path,
            st_mtime=0.0,  # 写入后更新
            front_matter=fm,
            links=extract_wikilinks(body, rel_path),
        )
        content = render_markdown(node, body)
        path.write_text(content, encoding="utf-8")
        node.st_mtime = path.stat().st_mtime
        logger.info("file_written", path=rel_path, links=len(node.links))
        return node

    def read(self, rel_path: str) -> FileNode | None:
        """读取并解析 Markdown 文件。"""
        path = self._resolve_path(rel_path)
        if not path.exists():
            return None
        content = path.read_text(encoding="utf-8")
        fm, body = parse_front_matter(content)
        links = extract_wikilinks(body, rel_path)
        return FileNode(
            path=rel_path,
            st_mtime=path.stat().st_mtime,
            front_matter=fm,
            links=links,
        )

    def list_files(self, pattern: str = "*.md") -> list[FileNode]:
        """列出工作区文件。"""
        nodes: list[FileNode] = []
        for path in self.workspace.rglob(pattern):
            rel = str(path.relative_to(self.workspace)).replace("\\", "/")
            node = self.read(rel)
            if node:
                nodes.append(node)
        return nodes

    def search_by_predicate(self, predicate: str) -> list[FileLink]:
        """按 predicate 搜索关系。"""
        results: list[FileLink] = []
        for node in self.list_files():
            results.extend(link for link in node.links if link.predicate == predicate)
        return results

    def find_backlinks(self, target_path: str) -> list[FileLink]:
        """查找反向链接。"""
        results: list[FileLink] = []
        for node in self.list_files():
            for link in node.links:
                if link.target_path == target_path:
                    results.append(link)
        return results

    def delete(self, rel_path: str) -> bool:
        """删除文件。"""
        path = self._resolve_path(rel_path)
        if path.exists():
            path.unlink()
            logger.info("file_deleted", path=rel_path)
            return True
        return False

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
"""CLI 命令测试 (init/add/search/dump/serve)。"""

from __future__ import annotations

import json
import sys


def _run_cli(argv, monkeypatch, capsys):
    """辅助: 设置 sys.argv 并调 main()。"""
    monkeypatch.setattr(sys, "argv", ["septmuse", *argv])
    from septmuse.cli.main import main

    rc = main()
    out = capsys.readouterr()
    return rc, out.out, out.err


class TestVersion:
    def test_version(self, monkeypatch, capsys):
        rc, out, _ = _run_cli(["version"], monkeypatch, capsys)
        assert rc == 0
        assert "0.1.0" in out


class TestInit:
    def test_init_creates_db(self, tmp_path, monkeypatch, capsys):
        db = tmp_path / "test.db"
        rc, out, _ = _run_cli(["init", "--user", "alice", "--db-path", str(db)], monkeypatch, capsys)
        assert rc == 0
        assert db.exists()
        assert "initialized" in out.lower()

    def test_init_default_db(self, tmp_path, monkeypatch, capsys):
        db = tmp_path / "default.db"
        monkeypatch.setenv("SEPTMUSE_DB_PATH", str(db))
        rc, _, _ = _run_cli(["init", "--user", "bob"], monkeypatch, capsys)
        assert rc == 0
        assert db.exists()

    def test_init_creates_parent_dir(self, tmp_path, monkeypatch, capsys):
        db = tmp_path / "nested" / "deep" / "test.db"
        rc, _, _ = _run_cli(["init", "--user", "alice", "--db-path", str(db)], monkeypatch, capsys)
        assert rc == 0
        assert db.exists()


class TestAdd:
    def test_add_verbatim(self, tmp_path, monkeypatch, capsys):
        db = tmp_path / "test.db"
        _run_cli(["init", "--user", "alice", "--db-path", str(db)], monkeypatch, capsys)
        rc, out, _ = _run_cli(
            ["add", "我喜欢 Python", "--user", "alice", "--db-path", str(db)],
            monkeypatch,
            capsys,
        )
        assert rc == 0
        data = json.loads(out)
        assert "memory_id" in data or "id" in data

    def test_add_semantic(self, tmp_path, monkeypatch, capsys):
        db = tmp_path / "test.db"
        _run_cli(["init", "--user", "alice", "--db-path", str(db)], monkeypatch, capsys)
        rc, out, _ = _run_cli(
            ["add", "alice likes python", "--user", "alice", "--type", "semantic", "--db-path", str(db)],
            monkeypatch,
            capsys,
        )
        assert rc == 0
        data = json.loads(out)
        assert "id" in data

    def test_add_episodic(self, tmp_path, monkeypatch, capsys):
        db = tmp_path / "test.db"
        _run_cli(["init", "--user", "alice", "--db-path", str(db)], monkeypatch, capsys)
        rc, out, _ = _run_cli(
            ["add", "用户登录", "--user", "alice", "--type", "episodic", "--db-path", str(db)],
            monkeypatch,
            capsys,
        )
        assert rc == 0
        data = json.loads(out)
        assert "id" in data

    def test_add_procedural(self, tmp_path, monkeypatch, capsys):
        db = tmp_path / "test.db"
        _run_cli(["init", "--user", "alice", "--db-path", str(db)], monkeypatch, capsys)
        rc, out, _ = _run_cli(
            ["add", "先检查权限再执行", "--user", "alice", "--type", "procedural", "--db-path", str(db)],
            monkeypatch,
            capsys,
        )
        assert rc == 0
        data = json.loads(out)
        assert "id" in data


class TestSearch:
    def test_search_returns_json_array(self, tmp_path, monkeypatch, capsys):
        db = tmp_path / "test.db"
        _run_cli(["init", "--user", "alice", "--db-path", str(db)], monkeypatch, capsys)
        _run_cli(["add", "我喜欢 Python", "--user", "alice", "--db-path", str(db)], monkeypatch, capsys)
        rc, out, _ = _run_cli(
            ["search", "喜欢什么", "--user", "alice", "--db-path", str(db)],
            monkeypatch,
            capsys,
        )
        assert rc == 0
        results = json.loads(out)
        assert isinstance(results, list)
        assert len(results) > 0

    def test_search_no_results(self, tmp_path, monkeypatch, capsys):
        db = tmp_path / "test.db"
        _run_cli(["init", "--user", "alice", "--db-path", str(db)], monkeypatch, capsys)
        rc, out, _ = _run_cli(
            ["search", "完全不相关的内容xyz", "--user", "alice", "--db-path", str(db), "--threshold", "0.99"],
            monkeypatch,
            capsys,
        )
        assert rc == 0
        results = json.loads(out)
        assert isinstance(results, list)
        assert len(results) == 0

    def test_search_top_k(self, tmp_path, monkeypatch, capsys):
        db = tmp_path / "test.db"
        _run_cli(["init", "--user", "alice", "--db-path", str(db)], monkeypatch, capsys)
        _run_cli(["add", "测试1", "--user", "alice", "--db-path", str(db)], monkeypatch, capsys)
        _run_cli(["add", "测试2", "--user", "alice", "--db-path", str(db)], monkeypatch, capsys)
        rc, out, _ = _run_cli(
            ["search", "测试", "--user", "alice", "--db-path", str(db), "--top-k", "1"],
            monkeypatch,
            capsys,
        )
        assert rc == 0
        results = json.loads(out)
        assert len(results) <= 1


class TestDump:
    def test_dump_json_stdout(self, tmp_path, monkeypatch, capsys):
        db = tmp_path / "test.db"
        _run_cli(["init", "--user", "alice", "--db-path", str(db)], monkeypatch, capsys)
        _run_cli(["add", "测试记忆", "--user", "alice", "--db-path", str(db)], monkeypatch, capsys)
        rc, out, _ = _run_cli(
            ["dump", "--user", "alice", "--db-path", str(db)],
            monkeypatch,
            capsys,
        )
        assert rc == 0
        data = json.loads(out)
        assert "results" in data
        assert len(data["results"]) >= 1

    def test_dump_markdown(self, tmp_path, monkeypatch, capsys):
        db = tmp_path / "test.db"
        _run_cli(["init", "--user", "alice", "--db-path", str(db)], monkeypatch, capsys)
        _run_cli(["add", "测试记忆", "--user", "alice", "--db-path", str(db)], monkeypatch, capsys)
        rc, out, _ = _run_cli(
            ["dump", "--user", "alice", "--format", "markdown", "--db-path", str(db)],
            monkeypatch,
            capsys,
        )
        assert rc == 0
        assert "- **" in out

    def test_dump_to_file(self, tmp_path, monkeypatch, capsys):
        db = tmp_path / "test.db"
        out_file = tmp_path / "dump.json"
        _run_cli(["init", "--user", "alice", "--db-path", str(db)], monkeypatch, capsys)
        _run_cli(["add", "测试记忆", "--user", "alice", "--db-path", str(db)], monkeypatch, capsys)
        rc, _, _ = _run_cli(
            ["dump", "--user", "alice", "--db-path", str(db), "--output", str(out_file)],
            monkeypatch,
            capsys,
        )
        assert rc == 0
        assert out_file.exists()
        data = json.loads(out_file.read_text(encoding="utf-8"))
        assert "results" in data


class TestServe:
    def test_serve_no_uvicorn(self, tmp_path, monkeypatch, capsys):
        """uvicorn 未安装时报友好错误。"""
        import sys as _sys

        monkeypatch.setitem(_sys.modules, "uvicorn", None)
        db = str(tmp_path / "test.db")
        rc, _, err = _run_cli(
            ["serve", "--db-path", db],
            monkeypatch,
            capsys,
        )
        assert rc == 1
        assert "uvicorn" in err.lower()

    def test_serve_argparse(self, tmp_path, monkeypatch, capsys):
        """serve 参数解析正确 (不真正启动 server)。"""
        from septmuse.cli.main import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["serve", "--host", "0.0.0.0", "--port", "9999", "--with-rest"])
        assert args.host == "0.0.0.0"
        assert args.port == 9999
        assert args.with_rest is True

    def test_serve_default_args(self, tmp_path, monkeypatch, capsys):
        """serve 默认参数。"""
        from septmuse.cli.main import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["serve"])
        assert args.host == "127.0.0.1"
        assert args.port == 8000
        assert args.with_rest is False


class TestCliUpdate:
    def test_update(self, tmp_path, monkeypatch, capsys):
        db = tmp_path / "test.db"
        _run_cli(["init", "--user", "alice", "--db-path", str(db)], monkeypatch, capsys)
        _add_rc, add_out, _ = _run_cli(["add", "旧内容", "--user", "alice", "--db-path", str(db)], monkeypatch, capsys)
        mid = json.loads(add_out).get("memory_id")
        rc, out, _ = _run_cli(
            ["update", mid, "新内容", "--db-path", str(db)],
            monkeypatch,
            capsys,
        )
        assert rc == 0
        data = json.loads(out)
        assert data["event"] == "UPDATE"

    def test_update_not_found(self, tmp_path, monkeypatch, capsys):
        db = tmp_path / "test.db"
        _run_cli(["init", "--user", "alice", "--db-path", str(db)], monkeypatch, capsys)
        rc, out, _ = _run_cli(
            ["update", "nonexistent", "x", "--db-path", str(db)],
            monkeypatch,
            capsys,
        )
        assert rc == 0
        data = json.loads(out)
        assert data["event"] == "NOT_FOUND"


class TestCliBlock:
    def test_block_set_and_list(self, tmp_path, monkeypatch, capsys):
        db = tmp_path / "test.db"
        _run_cli(["init", "--user", "alice", "--db-path", str(db)], monkeypatch, capsys)
        rc, _, _ = _run_cli(
            ["block", "set", "agent-1", "human", "Name: Alice", "--db-path", str(db)],
            monkeypatch,
            capsys,
        )
        assert rc == 0
        rc, out, _ = _run_cli(
            ["block", "list", "agent-1", "--db-path", str(db)],
            monkeypatch,
            capsys,
        )
        assert rc == 0
        data = json.loads(out)
        labels = [b["label"] for b in data]
        assert "human" in labels


class TestCliEntities:
    def test_entities_search_finds_entity(self, tmp_path, monkeypatch, capsys):
        db = tmp_path / "test.db"
        _run_cli(["init", "--user", "alice", "--db-path", str(db)], monkeypatch, capsys)
        _run_cli(["add", "Alice works at Google", "--user", "alice", "--db-path", str(db)], monkeypatch, capsys)
        rc, out, _ = _run_cli(
            ["entities", "Google", "--user-id", "alice", "--db-path", str(db)],
            monkeypatch,
            capsys,
        )
        assert rc == 0
        assert "Google" in out

    def test_entity_list_returns_entities(self, tmp_path, monkeypatch, capsys):
        db = tmp_path / "test.db"
        _run_cli(["init", "--user", "alice", "--db-path", str(db)], monkeypatch, capsys)
        _run_cli(["add", "Alice works at Google", "--user", "alice", "--db-path", str(db)], monkeypatch, capsys)
        rc, out, _ = _run_cli(
            ["entity-list", "--user-id", "alice", "--db-path", str(db)],
            monkeypatch,
            capsys,
        )
        assert rc == 0
        assert "Alice" in out or "Google" in out

    def test_entities_argparse(self):
        from septmuse.cli.main import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["entities", "query", "--user-id", "u1", "--top-k", "3"])
        assert args.query == "query"
        assert args.user_id == "u1"
        assert args.top_k == 3

    def test_entity_list_argparse(self):
        from septmuse.cli.main import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["entity-list", "--user-id", "u1", "--entity-type", "PROPER", "--limit", "10"])
        assert args.user_id == "u1"
        assert args.entity_type == "PROPER"
        assert args.limit == 10

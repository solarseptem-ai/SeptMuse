"""CLI backends / config show 自省命令测试。"""
import os
import subprocess
import sys


def _run_cli(args: list[str]) -> str:
    """运行 CLI 命令，返回 stdout。"""
    env = {**os.environ, "PYTHONPATH": "src"}
    result = subprocess.run(
        [sys.executable, "-m", "septmuse.cli.main", *args],
        capture_output=True, text=True, env=env, timeout=30,
    )
    return result.stdout


def test_backends_lists_all_capabilities():
    out = _run_cli(["backends"])
    assert "vector_store" in out
    assert "embedder" in out
    assert "llm" in out
    assert "reranker" in out
    assert "search_recipe" in out


def test_backends_shows_availability():
    out = _run_cli(["backends"])
    # sqlite/hash 是零依赖，应该可用
    assert "sqlite" in out
    assert "hash" in out


def test_config_show_outputs_current_config():
    out = _run_cli(["config", "show"])
    assert "embedder" in out
    assert "backend" in out or "hash" in out

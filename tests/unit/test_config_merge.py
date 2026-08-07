"""配置三层合并测试: env > yaml > code_default。"""

from septmuse.configs.base import MemoryConfig


def test_zero_config_defaults(monkeypatch):
    """无 env 无 yaml 时用代码默认。"""
    monkeypatch.delenv("SEPTMUSE_EMBEDDER", raising=False)
    monkeypatch.delenv("SEPTMUSE_EMBEDDING_DIMS", raising=False)
    config = MemoryConfig()
    assert config.embedder.backend == "bge-zh"
    assert config.vector_store.backend == "qdrant"


def test_yaml_overrides_default(tmp_path, monkeypatch):
    """YAML 覆盖代码默认。"""
    # 确保没有环境变量干扰
    monkeypatch.delenv("SEPTMUSE_EMBEDDER", raising=False)
    monkeypatch.delenv("SEPTMUSE_EMBEDDER__BACKEND", raising=False)
    yaml_content = "embedder:\n  backend: onnx\n"
    yaml_file = tmp_path / "septmuse.yaml"
    yaml_file.write_text(yaml_content, encoding="utf-8")

    config = MemoryConfig(_yaml_file=str(yaml_file))
    assert config.embedder.backend == "onnx"


def test_env_overrides_yaml(tmp_path, monkeypatch):
    """环境变量覆盖 YAML。"""
    yaml_content = "embedder:\n  backend: onnx\n"
    yaml_file = tmp_path / "septmuse.yaml"
    yaml_file.write_text(yaml_content, encoding="utf-8")

    monkeypatch.setenv("SEPTMUSE_EMBEDDER", "st")
    config = MemoryConfig(_yaml_file=str(yaml_file))
    assert config.embedder.backend == "st"

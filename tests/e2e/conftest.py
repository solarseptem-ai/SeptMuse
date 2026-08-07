import pytest


@pytest.fixture(autouse=True)
def _isolate_chroma(tmp_path, monkeypatch):
    """每个 e2e 测试用独立的 Chroma 路径 + hash embedder (dim=128), 避免维度冲突和模型下载。"""
    monkeypatch.setenv("SEPTMUSE_CHROMA_PERSIST_PATH", str(tmp_path / "chroma"))
    monkeypatch.setenv("SEPTMUSE_EMBEDDER", "hash")
    monkeypatch.setenv("SEPTMUSE_EMBEDDING_DIMS", "128")

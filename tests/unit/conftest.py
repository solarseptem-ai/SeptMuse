import pytest


@pytest.fixture(autouse=True)
def _isolate_chroma(tmp_path, monkeypatch):
    """每个测试用独立的 Chroma/Qdrant 持久化路径 + hash embedder (dim=128), 避免维度冲突和模型下载。"""
    monkeypatch.setenv("SEPTMUSE_CHROMA_PERSIST_PATH", str(tmp_path / "chroma"))
    monkeypatch.setenv("SEPTMUSE_QDRANT_PATH", str(tmp_path / "qdrant"))
    monkeypatch.setenv("SEPTMUSE_EMBEDDER", "hash")
    monkeypatch.setenv("SEPTMUSE_EMBEDDING_DIMS", "128")
    # 测试环境强制正则分词 (保持现有 BM25 断言行为不变)
    monkeypatch.setenv("SEPTMUSE_TOKENIZER", "space")

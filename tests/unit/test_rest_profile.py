"""REST GET /agents/{user_id}/profile 端点测试."""
from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path):
    os.environ["SEPTMUSE_EMBEDDER"] = "hash"
    os.environ["SEPTMUSE_TOKENIZER"] = "space"
    os.environ["SEPTMUSE_DB_PATH"] = str(tmp_path / "test.db")
    from septmuse.api.rest import create_app
    app = create_app()
    return TestClient(app)


@pytest.fixture
def setup_facts(client):
    """通过 typed_store 存事实 (共享 REST app 的 Memory)."""
    import os

    from sqlalchemy import create_engine

    from septmuse.storage.relational_stores.typed_store import TypedMemoryStore

    db_path = os.environ["SEPTMUSE_DB_PATH"]
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    ts = TypedMemoryStore(engine=engine)
    ts.add_fact("user", "name", "Alice", user_id="alice", confidence=0.9)
    ts.add_fact("user", "occupation", "Engineer", user_id="alice")
    ts.add_fact("user", "likes", "Python", user_id="alice")
    return ts


def test_rest_get_profile(client, setup_facts):
    resp = client.get("/agents/alice/profile")
    assert resp.status_code == 200
    data = resp.json()
    assert data["user_id"] == "alice"
    assert data["attributes"]["name"]["value"] == "Alice"
    assert data["preferences"]["likes"]["value"] == "Python"


def test_rest_get_profile_temporal(client, setup_facts):
    """?include_temporal=true 返回 temporal_summary."""
    resp = client.get("/agents/alice/profile?include_temporal=true")
    assert resp.status_code == 200
    data = resp.json()
    assert "temporal_summary" in data
    assert data["temporal_summary"]["total"] >= 3


def test_rest_get_profile_empty_user(client):
    """无记忆用户 → 200 + 空画像."""
    resp = client.get("/agents/nobody/profile")
    assert resp.status_code == 200
    data = resp.json()
    assert data["user_id"] == "nobody"
    assert len(data["attributes"]) == 0

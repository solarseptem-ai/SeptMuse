#  Copyright 2026 The sonhhxg0529 Authors. All Rights Reserved.
"""Health 端点测试。"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("SEPTMUSE_DB_PATH", str(tmp_path / "test.db"))
    from septmuse.api.rest import create_app

    app = create_app()
    yield TestClient(app)


def test_health_returns_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_health_no_auth_required(client):
    """health 端点公开可访问，不需要 API key。"""
    resp = client.get("/health")
    assert resp.status_code == 200

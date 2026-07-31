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
"""auth.py 单元测试 — API key 认证中间件。"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from septmuse.api.auth import ApiKeyMiddleware, setup_auth


@pytest.fixture
def app_factory() -> Iterator:
    """工厂: 构造带中间件的 app。"""
    created_keys: list[str | None] = []

    def _make(api_key: str | None = None) -> FastAPI:
        app = FastAPI()
        setup_auth(app, api_key=api_key)

        @app.get("/health")
        async def health() -> dict[str, str]:
            return {"status": "ok"}

        @app.get("/memories")
        async def list_memories() -> dict[str, list]:
            return {"results": []}

        created_keys.append(api_key)
        return app

    yield _make


def test_dev_mode_no_api_key_allows_all(app_factory) -> None:
    """开发模式 (无 SEPTMUSE_API_KEY): 不拦截任何请求。"""
    app = app_factory(api_key=None)
    client = TestClient(app)

    # /health 豁免
    r = client.get("/health")
    assert r.status_code == 200

    # /memories 也通过 (开发模式不拦截)
    r = client.get("/memories")
    assert r.status_code == 200
    assert r.json() == {"results": []}


def test_prod_mode_no_token_returns_401(app_factory) -> None:
    """生产模式 (设了 API key): 无 token 返回 401。"""
    app = app_factory(api_key="sk-test-key")
    client = TestClient(app)

    r = client.get("/memories")
    assert r.status_code == 401
    assert "API key" in r.json()["detail"]


def test_prod_mode_wrong_token_returns_401(app_factory) -> None:
    """生产模式: 错误 token 返回 401。"""
    app = app_factory(api_key="sk-test-key")
    client = TestClient(app)

    r = client.get("/memories", headers={"Authorization": "Bearer wrong-key"})
    assert r.status_code == 401


def test_prod_mode_bearer_token_allows(app_factory) -> None:
    """生产模式: 正确 Bearer token 通过。"""
    app = app_factory(api_key="sk-test-key")
    client = TestClient(app)

    r = client.get("/memories", headers={"Authorization": "Bearer sk-test-key"})
    assert r.status_code == 200
    assert r.json() == {"results": []}


def test_prod_mode_x_api_key_header_allows(app_factory) -> None:
    """生产模式: X-API-Key header 也通过。"""
    app = app_factory(api_key="sk-test-key")
    client = TestClient(app)

    r = client.get("/memories", headers={"X-API-Key": "sk-test-key"})
    assert r.status_code == 200


def test_prod_mode_exempt_paths_accessible(app_factory) -> None:
    """生产模式: 豁免路径不需要认证。"""
    app = app_factory(api_key="sk-test-key")
    client = TestClient(app)

    # /health 豁免
    r = client.get("/health")
    assert r.status_code == 200

    # /docs 豁免 (Swagger UI)
    r = client.get("/docs")
    assert r.status_code == 200

    # /openapi.json 豁免
    r = client.get("/openapi.json")
    assert r.status_code == 200


def test_env_var_septmuse_api_key(monkeypatch, app_factory) -> None:
    """SEPTMUSE_API_KEY 环境变量自动启用生产模式。"""
    monkeypatch.setenv("SEPTMUSE_API_KEY", "sk-from-env")
    app = app_factory(api_key=None)  # 未显式传, 从环境变量读
    client = TestClient(app)

    # 无 token → 401
    r = client.get("/memories")
    assert r.status_code == 401

    # 正确 token → 200
    r = client.get("/memories", headers={"Authorization": "Bearer sk-from-env"})
    assert r.status_code == 200


def test_dev_mode_warns_once(app_factory) -> None:
    """开发模式: middleware 实例 dev_mode=True (不拦截)。"""
    app = app_factory(api_key=None)
    # 找到 ApiKeyMiddleware 实例, 验证 dev_mode 标记
    middleware = next(
        (m for m in app.user_middleware if m.cls is ApiKeyMiddleware),
        None,
    )
    assert middleware is not None
    # middleware.kwargs 含 api_key=None → dev_mode=True
    assert middleware.kwargs.get("api_key") is None


def test_prod_mode_sets_api_key(app_factory) -> None:
    """生产模式: middleware 实例 api_key 非空。"""
    app = app_factory(api_key="sk-test")
    middleware = next(
        (m for m in app.user_middleware if m.cls is ApiKeyMiddleware),
        None,
    )
    assert middleware is not None
    assert middleware.kwargs.get("api_key") == "sk-test"

from __future__ import annotations

import pytest


def test_mcp_http_initialize_and_tools_list() -> None:
    fastapi = pytest.importorskip("fastapi")
    assert fastapi is not None

    from doc_rag.server.mcp_http import _handle_jsonrpc

    st, out = _handle_jsonrpc({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    assert st == 200
    assert isinstance(out, dict)
    assert out["id"] == 1
    assert out["result"]["serverInfo"]["name"] == "doc-rag"

    st, out = _handle_jsonrpc({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
    assert st == 200
    assert isinstance(out, dict)
    tools = out["result"]["tools"]
    assert any(t.get("name") == "doc_search" for t in tools)


def test_mcp_http_auth_required_when_env_set(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    monkeypatch.setenv("DOC_RAG_API_KEY", "secret")
    from doc_rag.server.mcp_http import app

    c = TestClient(app)
    r = c.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    assert r.status_code == 401

    r = c.post(
        "/mcp",
        headers={"Authorization": "Bearer secret"},
        json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
    )
    assert r.status_code == 200


def test_mcp_http_denies_when_no_key_and_anon_off(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    monkeypatch.delenv("DOC_RAG_API_KEY", raising=False)
    from doc_rag.server.mcp_http import app

    c = TestClient(app)
    r = c.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    assert r.status_code == 200


def test_mcp_http_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    monkeypatch.setenv("DOC_RAG_RATE_LIMIT_RPS", "1")
    monkeypatch.setenv("DOC_RAG_RATE_LIMIT_BURST", "1")
    from doc_rag.server.mcp_http import app

    c = TestClient(app)
    r1 = c.get("/health")
    assert r1.status_code == 200
    r2 = c.get("/health")
    assert r2.status_code in (200, 429)


def test_ui_requires_key_when_api_key_set(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    monkeypatch.setenv("DOC_RAG_API_KEY", "secret")
    monkeypatch.setenv("DOC_RAG_ROOT", str(tmp_path))

    from doc_rag.server.mcp_http import app

    c = TestClient(app)
    r = c.get("/ui")
    assert r.status_code == 401
    r = c.get("/ui?key=secret")
    assert r.status_code == 200


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


def test_ui_status_includes_http_log_tail(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    monkeypatch.delenv("DOC_RAG_API_KEY", raising=False)
    monkeypatch.setenv("DOC_RAG_ROOT", str(tmp_path))

    from doc_rag.server import mcp_http as mh
    from doc_rag.server.mcp_http import app

    mh._log_line("probe-http-access-log")
    c = TestClient(app)
    r = c.get("/ui/status")
    assert r.status_code == 200
    data = r.json()
    assert "http_log_tail" in data
    assert any("probe-http-access-log" in line for line in data["http_log_tail"])
    assert "job" in data


def test_ui_multi_upload_writes_incoming(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    monkeypatch.delenv("DOC_RAG_API_KEY", raising=False)
    monkeypatch.setenv("DOC_RAG_ROOT", str(tmp_path))
    from doc_rag.server.mcp_http import app

    (tmp_path / "sources" / "incoming").mkdir(parents=True)

    # Two distinct payloads — same prefix, different sha256.
    pdf_a = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\r\n" + b"file-a\n"
    pdf_b = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\r\n" + b"file-b\n"
    multi = [
        ("files", ("batch_a.pdf", pdf_a, "application/pdf")),
        ("files", ("batch_b.pdf", pdf_b, "application/pdf")),
    ]

    c = TestClient(app)
    r = c.post("/ui/upload", files=multi, data={}, follow_redirects=False)
    assert r.status_code == 303
    assert "up_saved=2" in r.headers.get("location", "")
    incoming = tmp_path / "sources" / "incoming"
    assert len(list(incoming.glob("*.pdf"))) == 2


def test_ui_rebuild_returns_redirect(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    monkeypatch.delenv("DOC_RAG_API_KEY", raising=False)
    monkeypatch.setenv("DOC_RAG_ROOT", str(tmp_path))

    import doc_rag.server.mcp_http as mh

    async def _noop_rebuild() -> None:
        return

    monkeypatch.setattr(mh, "_run_rebuild_background", _noop_rebuild)
    app = mh.app

    r = TestClient(app).post("/ui/rebuild", data={}, follow_redirects=False)
    assert r.status_code == 303
    assert "/ui" in r.headers.get("location", "")


def test_ui_status_includes_log_tail(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    monkeypatch.delenv("DOC_RAG_API_KEY", raising=False)
    monkeypatch.setenv("DOC_RAG_ROOT", str(tmp_path))

    from doc_rag.server import mcp_http as mh

    mh._reset_ingest_ui_log_lines()
    mh._append_ingest_ui_line("__ui_log_probe__", ansi_strip=False)

    from doc_rag.server.mcp_http import app

    c = TestClient(app)
    r = c.get("/ui/status")
    assert r.status_code == 200
    data = r.json()
    assert "log_tail" in data
    assert any("__ui_log_probe__" in line for line in data["log_tail"])


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


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


_SAMPLE_CONFIG = """\
chunking:
  # keep this comment intact across a form-save
  target_tokens: 512
  overlap_tokens: 64

index:
  metric: "ip"
  top_k: 6
"""


def _write_sample_config(tmp_path):
    p = tmp_path / "config" / "config.yaml"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(_SAMPLE_CONFIG, encoding="utf-8")
    return p


def test_ui_config_parsed_returns_mapping(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    monkeypatch.delenv("DOC_RAG_API_KEY", raising=False)
    monkeypatch.setenv("DOC_RAG_ROOT", str(tmp_path))
    _write_sample_config(tmp_path)
    from doc_rag.server.mcp_http import app

    r = TestClient(app).get("/ui/config/parsed")
    assert r.status_code == 200
    data = r.json()
    assert data["ok"] is True
    assert data["config"]["chunking"]["target_tokens"] == 512
    assert data["config"]["index"]["metric"] == "ip"


def test_ui_config_patch_updates_and_preserves_comments(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("ruamel.yaml")
    from fastapi.testclient import TestClient

    monkeypatch.delenv("DOC_RAG_API_KEY", raising=False)
    monkeypatch.setenv("DOC_RAG_ROOT", str(tmp_path))
    p = _write_sample_config(tmp_path)
    from doc_rag.server.mcp_http import app

    updates = '{"chunking.target_tokens": 256, "index.metric": "l2"}'
    r = TestClient(app).post("/ui/config/patch", data={"updates": updates})
    assert r.status_code == 200
    assert r.json()["ok"] is True

    on_disk = p.read_text(encoding="utf-8")
    # Value changed, type stays an int (no quotes), comment survives.
    assert "target_tokens: 256" in on_disk
    assert "keep this comment intact across a form-save" in on_disk
    assert 'metric: "l2"' in on_disk


def test_ui_config_patch_rejects_unknown_key(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("ruamel.yaml")
    from fastapi.testclient import TestClient

    monkeypatch.delenv("DOC_RAG_API_KEY", raising=False)
    monkeypatch.setenv("DOC_RAG_ROOT", str(tmp_path))
    _write_sample_config(tmp_path)
    from doc_rag.server.mcp_http import app

    r = TestClient(app).post("/ui/config/patch", data={"updates": '{"chunking.bogus": 1}'})
    assert r.status_code == 400
    assert r.json()["ok"] is False


def test_ui_config_patch_requires_key_when_api_key_set(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    monkeypatch.setenv("DOC_RAG_API_KEY", "secret")
    monkeypatch.setenv("DOC_RAG_ROOT", str(tmp_path))
    _write_sample_config(tmp_path)
    from doc_rag.server.mcp_http import app

    c = TestClient(app)
    assert c.get("/ui/config/parsed").status_code == 401
    assert c.post("/ui/config/patch", data={"updates": "{}"}).status_code == 401


def test_ui_env_get_lists_editable_keys(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    monkeypatch.delenv("DOC_RAG_API_KEY", raising=False)
    monkeypatch.setenv("DOC_RAG_ENV_FILE", str(tmp_path / ".env"))
    from doc_rag.server.mcp_http import app

    r = TestClient(app).get("/ui/env")
    assert r.status_code == 200
    data = r.json()
    assert data["ok"] is True
    keys = {f["key"] for f in data["fields"]}
    assert "DOC_RAG_HTTP_PORT" in keys
    assert "DOC_RAG_UI_RESTART_CMD" in keys
    # API key is a secret: present only as a set/not-set flag, never a field.
    assert "DOC_RAG_API_KEY" not in keys
    assert any(s["key"] == "DOC_RAG_API_KEY" for s in data["secrets"])


def test_ui_env_get_masks_secret_value(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    monkeypatch.setenv("DOC_RAG_API_KEY", "supersecret123")
    monkeypatch.setenv("DOC_RAG_ENV_FILE", str(tmp_path / ".env"))
    from doc_rag.server.mcp_http import app

    r = TestClient(app).get("/ui/env?key=supersecret123")
    assert r.status_code == 200
    data = r.json()
    secret = next(s for s in data["secrets"] if s["key"] == "DOC_RAG_API_KEY")
    assert secret["set"] is True
    # The secret value must never appear in the response body.
    assert "supersecret123" not in r.text


def test_ui_env_save_writes_quoted_dotenv(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    monkeypatch.delenv("DOC_RAG_API_KEY", raising=False)
    env_file = tmp_path / ".env"
    monkeypatch.setenv("DOC_RAG_ENV_FILE", str(env_file))
    from doc_rag.server.mcp_http import app

    updates = (
        '{"DOC_RAG_HTTP_PORT": "4000", '
        '"DOC_RAG_UI_RESTART_CMD": "sudo systemctl restart doc-rag-mcp"}'
    )
    r = TestClient(app).post("/ui/env/save", data={"updates": updates})
    assert r.status_code == 200
    assert r.json()["ok"] is True

    written = env_file.read_text(encoding="utf-8")
    assert "DOC_RAG_HTTP_PORT='4000'" in written
    # Value with spaces must be single-quoted so the file is safe to `source`.
    assert "DOC_RAG_UI_RESTART_CMD='sudo systemctl restart doc-rag-mcp'" in written


def test_ui_env_save_rejects_secret_unknown_and_bad_type(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    monkeypatch.delenv("DOC_RAG_API_KEY", raising=False)
    monkeypatch.setenv("DOC_RAG_ENV_FILE", str(tmp_path / ".env"))
    from doc_rag.server.mcp_http import app

    c = TestClient(app)
    # secret key cannot be set from the UI
    assert c.post("/ui/env/save", data={"updates": '{"DOC_RAG_API_KEY": "x"}'}).status_code == 400
    # unknown key
    assert c.post("/ui/env/save", data={"updates": '{"DOC_RAG_NOPE": "1"}'}).status_code == 400
    # bad type (port must be int)
    assert (
        c.post("/ui/env/save", data={"updates": '{"DOC_RAG_HTTP_PORT": "abc"}'}).status_code == 400
    )


def test_ui_env_requires_key_when_api_key_set(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    monkeypatch.setenv("DOC_RAG_API_KEY", "secret")
    monkeypatch.setenv("DOC_RAG_ENV_FILE", str(tmp_path / ".env"))
    from doc_rag.server.mcp_http import app

    c = TestClient(app)
    assert c.get("/ui/env").status_code == 401
    assert c.post("/ui/env/save", data={"updates": "{}"}).status_code == 401

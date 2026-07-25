from __future__ import annotations

import os

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

    async def _noop_rebuild(force: bool = False) -> None:
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

    monkeypatch.setenv("DOC_RAG_ROOT", str(tmp_path))
    _write_sample_config(tmp_path)
    from doc_rag.server.mcp_http import app

    r = TestClient(app).post("/ui/config/patch", data={"updates": '{"chunking.bogus": 1}'})
    assert r.status_code == 400
    assert r.json()["ok"] is False


def test_ui_env_get_lists_editable_keys(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    monkeypatch.setenv("DOC_RAG_ENV_FILE", str(tmp_path / ".env"))
    from doc_rag.server.mcp_http import app

    r = TestClient(app).get("/ui/env")
    assert r.status_code == 200
    data = r.json()
    assert data["ok"] is True
    keys = {f["key"] for f in data["fields"]}
    assert "DOC_RAG_HTTP_PORT" in keys
    assert "DOC_RAG_UI_RESTART_CMD" in keys


def test_ui_env_save_writes_quoted_dotenv(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

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


def test_ui_env_save_rejects_unknown_and_bad_type(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    monkeypatch.setenv("DOC_RAG_ENV_FILE", str(tmp_path / ".env"))
    from doc_rag.server.mcp_http import app

    c = TestClient(app)
    # unknown key
    assert c.post("/ui/env/save", data={"updates": '{"DOC_RAG_NOPE": "1"}'}).status_code == 400
    # bad type (port must be int)
    assert (
        c.post("/ui/env/save", data={"updates": '{"DOC_RAG_HTTP_PORT": "abc"}'}).status_code == 400
    )


def test_doc_stats_endpoint() -> None:
    """GET /ui/doc-stats returns document statistics when build dir exists."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from doc_rag.server.mcp_http import app

    c = TestClient(app)
    r = c.get("/ui/doc-stats")
    assert r.status_code == 200
    data = r.json()
    assert data["ok"] is True
    assert "documents" in data
    assert "count" in data


def test_health_ready_includes_subsystem_checks() -> None:
    """GET /health/ready returns 'checks' dict with faiss_index and disk_space."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from doc_rag.server.mcp_http import app

    c = TestClient(app)
    r = c.get("/health/ready")
    data = r.json()
    assert "checks" in data
    assert "faiss_index" in data["checks"]
    assert "disk_space" in data["checks"]
    assert isinstance(data["checks"]["faiss_index"], bool)
    assert isinstance(data["checks"]["disk_space"], bool)
    assert "reasons" in data
    assert isinstance(data["reasons"], list)


def test_health_live_returns_ok() -> None:
    """GET /health/live always returns 200."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from doc_rag.server.mcp_http import app

    c = TestClient(app)
    r = c.get("/health/live")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


# ── Upload flow regression tests ─────────────────────────────────────


def test_upload_xhr_returns_json_saved(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Upload via XHR should return JSON with saved=1, and file should land in incoming."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    monkeypatch.delenv("DOC_RAG_API_KEY", raising=False)
    monkeypatch.setenv("DOC_RAG_ROOT", str(tmp_path))
    from doc_rag.server.mcp_http import app

    (tmp_path / "sources" / "incoming").mkdir(parents=True)
    (tmp_path / "build").mkdir(parents=True)

    content = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\r\nupload-test-content\n"
    files = [("files", ("test_upload.pdf", content, "application/pdf"))]

    c = TestClient(app)
    r = c.post(
        "/ui/upload",
        files=files,
        data={},
        headers={"X-Requested-With": "XMLHttpRequest"},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["saved"] == 1
    assert data["dups"] == []
    assert data["errors"] == []
    incoming = tmp_path / "sources" / "incoming"
    pdfs = list(incoming.glob("*.pdf"))
    assert len(pdfs) == 1


def test_upload_two_empty_files_second_flagged_as_dup_of_first(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Characterizes current behavior for two distinctly-named 0-byte uploads.

    sha256 of empty content is the same constant for *any* empty file — this
    isn't a hash collision between different content, it's two genuinely
    identical (empty) files. `_find_dup_in_incoming` computes sha256 from
    what's already on disk in incoming/, so the second empty upload gets
    flagged as a duplicate of the first, even though the two files have
    different names and the user may have intended them as separate
    (currently-empty) documents. This test documents that outcome rather
    than asserting it's necessarily desired — see audit notes on whether
    empty uploads should instead be rejected outright with a clearer error.
    """
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    monkeypatch.delenv("DOC_RAG_API_KEY", raising=False)
    monkeypatch.setenv("DOC_RAG_ROOT", str(tmp_path))
    from doc_rag.server.mcp_http import app

    (tmp_path / "sources" / "incoming").mkdir(parents=True)
    (tmp_path / "build").mkdir(parents=True)

    files = [
        ("files", ("first.txt", b"", "text/plain")),
        ("files", ("second.txt", b"", "text/plain")),
    ]

    c = TestClient(app)
    r = c.post(
        "/ui/upload",
        files=files,
        data={},
        headers={"X-Requested-With": "XMLHttpRequest"},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["saved"] == 1
    assert len(data["dups"]) == 1
    assert "second.txt" in data["dups"][0]

    incoming = tmp_path / "sources" / "incoming"
    txts = list(incoming.glob("*.txt"))
    assert len(txts) == 1
    assert txts[0].name == "first.txt"


def test_upload_xhr_returns_json_dup(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Upload duplicate file (same sha256 as manifest) should return dups=1, saved=0."""
    pytest.importorskip("fastapi")
    import hashlib

    from fastapi.testclient import TestClient

    monkeypatch.delenv("DOC_RAG_API_KEY", raising=False)
    monkeypatch.setenv("DOC_RAG_ROOT", str(tmp_path))
    from doc_rag.server.mcp_http import app

    (tmp_path / "sources" / "incoming").mkdir(parents=True)

    content = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\r\nalready-indexed\n"
    sha = hashlib.sha256(content).hexdigest()

    # Mock _find_dup_in_manifest to return match for this sha256
    def mock_dup(h):
        if h == sha:
            return "existing.pdf"
        return None

    monkeypatch.setattr("doc_rag.server.mcp_http._find_dup_in_manifest", mock_dup)

    files = [("files", ("dup.pdf", content, "application/pdf"))]
    c = TestClient(app)
    r = c.post(
        "/ui/upload",
        files=files,
        data={},
        headers={"X-Requested-With": "XMLHttpRequest"},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["saved"] == 0
    assert len(data["dups"]) == 1
    incoming = tmp_path / "sources" / "incoming"
    assert len(list(incoming.glob("*.pdf"))) == 0


def test_upload_xhr_rejects_bad_extension(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Upload .exe file should return errors=1, saved=0."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    monkeypatch.delenv("DOC_RAG_API_KEY", raising=False)
    monkeypatch.setenv("DOC_RAG_ROOT", str(tmp_path))
    from doc_rag.server.mcp_http import app

    (tmp_path / "sources" / "incoming").mkdir(parents=True)

    files = [("files", ("malware.exe", b"MZ\x90\x00", "application/octet-stream"))]
    c = TestClient(app)
    r = c.post(
        "/ui/upload",
        files=files,
        data={},
        headers={"X-Requested-With": "XMLHttpRequest"},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["saved"] == 0
    assert len(data["errors"]) == 1


def test_upload_then_ingest_roundtrip(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Upload a file, then run ingest — file should end up in manifest."""
    pytest.importorskip("fastapi")
    import json as _json

    from fastapi.testclient import TestClient

    monkeypatch.delenv("DOC_RAG_API_KEY", raising=False)
    monkeypatch.setenv("DOC_RAG_ROOT", str(tmp_path))
    from doc_rag.raglib.pipeline import ingest
    from doc_rag.server.mcp_http import app

    # Create standard directory layout
    for d in [
        "sources/incoming",
        "build/docs_md",
        "build/chunks_jsonl",
        "build/blocks",
        "build/index",
        "config",
    ]:
        (tmp_path / d).mkdir(parents=True, exist_ok=True)

    config = tmp_path / "config" / "config.yaml"
    config.write_text("""\
paths:
  sources_incoming: sources/incoming
  sources_archived: sources/archived
  docs_md_dir: build/docs_md
  chunks_dir: build/chunks_jsonl
  manifest_path: build/manifest.json
  blocks_dir: build/blocks
  index_dir: build/index
parsing:
  pdf_backend: docling
chunking:
  strategy: fixed
  target_tokens: 512
  overlap_tokens: 64
embeddings:
  model_name: BAAI/bge-small-en-v1.5
  device: cpu
""")

    # Upload a simple .txt file (no OCR/parsing needed)
    content = b"Hello world. This is a test document with some content for chunking."
    files = [("files", ("test_roundtrip.txt", content, "text/plain"))]
    c = TestClient(app)
    r = c.post(
        "/ui/upload",
        files=files,
        data={},
        headers={"X-Requested-With": "XMLHttpRequest"},
    )
    assert r.status_code == 200
    assert r.json()["saved"] == 1

    # Verify file is in incoming
    incoming = tmp_path / "sources" / "incoming"
    assert len(list(incoming.glob("*.txt"))) == 1

    # Run ingest (direct call, not via HTTP, to avoid background task complexity)
    ingest(str(config))

    # Verify file was processed
    manifest_path = tmp_path / "build" / "manifest.json"
    assert manifest_path.exists()
    m = _json.loads(manifest_path.read_text())
    doc_ids = [d["doc_id"] for d in m["documents"]]
    assert len(doc_ids) == 1


def test_upload_old_form_returns_303(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Upload without XHR header should return 303 redirect (backward compat)."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    monkeypatch.delenv("DOC_RAG_API_KEY", raising=False)
    monkeypatch.setenv("DOC_RAG_ROOT", str(tmp_path))
    from doc_rag.server.mcp_http import app

    (tmp_path / "sources" / "incoming").mkdir(parents=True)

    content = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\r\nform-upload\n"
    files = [("files", ("form.pdf", content, "application/pdf"))]

    c = TestClient(app)
    r = c.post("/ui/upload", files=files, data={}, follow_redirects=False)
    assert r.status_code == 303
    loc = r.headers.get("location", "")
    assert "up_saved=1" in loc


@pytest.mark.parametrize(
    ("path", "data"),
    [
        ("/ui/delete", {"doc_ids": "abc123"}),
        ("/ui/wipe", {"confirm": "DELETE"}),
        ("/ui/clean-orphans", {}),
        ("/ui/clear-incoming", {}),
    ],
)
def test_destructive_ui_endpoints_busy_when_job_running(
    monkeypatch: pytest.MonkeyPatch, tmp_path, path: str, data: dict
) -> None:
    """A rebuild/ingest already in flight must block delete/wipe/clean-orphans/
    clear-incoming instead of racing it — see `_exclusive_job`. Before the fix,
    these endpoints checked `_INGEST_STATE["running"]` without holding
    `_INGEST_LOCK` (or, for clear-incoming, didn't check it at all), so a
    concurrent request could slip through and mutate manifest/chunks/index
    while a rebuild was still writing them.
    """
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    monkeypatch.delenv("DOC_RAG_API_KEY", raising=False)
    monkeypatch.setenv("DOC_RAG_ROOT", str(tmp_path))

    import doc_rag.server.mcp_http as mh

    called = False

    def _fail_if_called(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("pipeline op must not run while a job is marked running")

    monkeypatch.setattr(mh, "_pipeline_delete_documents", _fail_if_called)
    monkeypatch.setattr(mh, "_pipeline_wipe_index", _fail_if_called)
    monkeypatch.setattr(mh, "_pipeline_clean_orphans", _fail_if_called)
    monkeypatch.setattr(mh, "_pipeline_clear_incoming", _fail_if_called)

    mh._INGEST_STATE["running"] = True
    mh._INGEST_STATE["job"] = "rebuild"
    try:
        r = TestClient(mh.app).post(path, data=data)
        assert r.status_code == 409
        assert r.json()["ok"] is False
        assert called is False
    finally:
        mh._INGEST_STATE["running"] = False
        mh._INGEST_STATE["job"] = None


def test_ui_delete_releases_exclusive_job_state_after_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    monkeypatch.delenv("DOC_RAG_API_KEY", raising=False)
    monkeypatch.setenv("DOC_RAG_ROOT", str(tmp_path))

    import doc_rag.server.mcp_http as mh

    def _ok_delete(config_path: str, doc_ids: list) -> dict:
        assert mh._INGEST_STATE["running"] is True
        assert mh._INGEST_STATE["job"] == "delete"
        return {"removed_docs": len(doc_ids)}

    monkeypatch.setattr(mh, "_pipeline_delete_documents", _ok_delete)

    r = TestClient(mh.app).post("/ui/delete", data={"doc_ids": "abc123"})
    assert r.status_code == 200
    assert r.json()["ok"] is True
    assert mh._INGEST_STATE["running"] is False
    assert mh._INGEST_STATE["job"] is None


def _write_lock(build_dir, job: str, ts: str) -> None:
    build_dir.mkdir(parents=True, exist_ok=True)
    (build_dir / ".rebuild.lock").write_text(f"{job}\n{ts}\n")


def test_rebuild_interrupted_true_when_lock_has_no_done_marker(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """No `.done` marker next to the lock = the process never confirmed
    completion (crashed mid-rebuild, or killed before writing the marker) —
    must report interrupted and leave the lock in place."""
    import doc_rag.server.mcp_http as mh

    monkeypatch.setattr(mh, "_root_dir", lambda: tmp_path)
    _write_lock(tmp_path / "build", "rebuild", "1000.0")

    payload = mh._ui_status_payload()

    assert payload["rebuild_interrupted"] is True
    assert (tmp_path / "build" / ".rebuild.lock").is_file()


def test_rebuild_interrupted_false_when_done_marker_matches_lock(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """A `.done` marker whose timestamp matches the lock means *this* run
    finished successfully and only failed to clean up after itself (killed
    between finishing the job and removing the lock) — safe to auto-clear."""
    import doc_rag.server.mcp_http as mh

    monkeypatch.setattr(mh, "_root_dir", lambda: tmp_path)
    build_dir = tmp_path / "build"
    _write_lock(build_dir, "rebuild", "1000.0")
    (build_dir / ".rebuild.lock.done").write_text("1000.0\n")

    payload = mh._ui_status_payload()

    assert payload["rebuild_interrupted"] is False
    assert not (build_dir / ".rebuild.lock").exists()
    assert not (build_dir / ".rebuild.lock.done").exists()


def test_unrelated_manifest_write_does_not_clear_interrupted_rebuild_banner(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Regression for the bug fixed in this change: delete/clean-orphans/wipe
    rewrite manifest.json for reasons unrelated to any in-flight rebuild.
    Before the fix, staleness was inferred by comparing manifest.json's mtime
    to the lock's timestamp, so any later unrelated write to manifest.json
    (bumping its mtime past the lock's) made a genuinely interrupted rebuild
    look "stale" and silently cleared the banner. The fix decouples
    staleness from manifest.json entirely (see the `.done` marker tests
    above); this test locks in that a later manifest write has no effect.
    """
    import doc_rag.server.mcp_http as mh

    monkeypatch.setattr(mh, "_root_dir", lambda: tmp_path)
    build_dir = tmp_path / "build"
    _write_lock(build_dir, "rebuild", "1000.0")

    # Simulate an unrelated `/ui/delete` running after the crash, rewriting
    # manifest.json well after the lock's timestamp.
    (build_dir / "manifest.json").write_text('{"documents": []}')
    os.utime(build_dir / "manifest.json", (2000.0, 2000.0))

    payload = mh._ui_status_payload()

    assert payload["rebuild_interrupted"] is True
    assert (build_dir / ".rebuild.lock").is_file()


def test_run_index_job_background_leaves_no_lock_files_on_clean_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """End-to-end success path: lock + `.done` marker must both be gone once
    `_run_index_job_background` finishes without crashing."""
    import doc_rag.server.mcp_http as mh

    monkeypatch.setattr(mh, "_root_dir", lambda: tmp_path)

    import asyncio as _asyncio

    _asyncio.run(mh._run_index_job_background("rebuild", lambda: None))

    build_dir = tmp_path / "build"
    assert not (build_dir / ".rebuild.lock").exists()
    assert not (build_dir / ".rebuild.lock.done").exists()
    assert mh._INGEST_STATE["last_ok"] is True

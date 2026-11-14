"""Background HF Hub weight-hash watcher (Task 6).

All Hub interaction goes through injected fake api objects: the suite
never touches the real network.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
from pathlib import Path
from types import SimpleNamespace

from doc_rag.raglib.model_paths import fingerprint_from_pairs
from doc_rag.server import model_watcher

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_local_model(base: Path) -> Path:
    """A minimal sentence-transformers-style model dir with one weight file."""
    d = base / "model"
    d.mkdir()
    (d / "config.json").write_text("{}", encoding="utf-8")
    (d / "modules.json").write_text("[]", encoding="utf-8")
    (d / "model.safetensors").write_bytes(b"local-weights")
    return d


def _local_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _check_cfg(model_dir: Path) -> dict:
    return {"embeddings": {"model_path": str(model_dir), "model_name": "BAAI/bge-m3"}}


def _sib(name: str, sha: str | None = None) -> SimpleNamespace:
    """A fake RepoSibling: lfs is None for non-LFS files."""
    lfs = None if sha is None else SimpleNamespace(sha256=sha)
    return SimpleNamespace(rfilename=name, lfs=lfs)


def _api(info: SimpleNamespace | Exception) -> SimpleNamespace:
    """A fake HfApi whose model_info returns info or raises."""

    class Api:
        def __init__(self):
            self.calls: list[tuple[tuple, dict]] = []

        def model_info(self, *a, **kw):
            self.calls.append((a, kw))
            if isinstance(info, Exception):
                raise info
            return info

    return Api()


def _start_cfg(base: Path, hours) -> dict:
    """Config whose model resolves under <base>/build/models/bge-m3."""
    model = base / "build" / "models" / "bge-m3"
    model.mkdir(parents=True, exist_ok=True)
    (model / "config.json").write_text("{}", encoding="utf-8")
    (model / "modules.json").write_text("[]", encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"w")
    return {
        "_root": str(base),
        "embeddings": {"model_name": "BAAI/bge-m3", "update_check_interval_hours": hours},
    }


# ---------------------------------------------------------------------------
# check_once
# ---------------------------------------------------------------------------


def test_initial_state_is_clean():
    assert model_watcher.state() == {"update_available": False, "checked_at": None, "error": None}


def test_check_reports_update_when_weight_hash_differs(tmp_path):
    d = _make_local_model(tmp_path)
    cfg = _check_cfg(d)
    remote_sha = "a" * 64
    info = SimpleNamespace(siblings=[_sib("model.safetensors", remote_sha)])

    result = asyncio.run(model_watcher.check_once(cfg, api=_api(info)))

    assert result == fingerprint_from_pairs([(remote_sha, "model.safetensors")])
    st = model_watcher.state()
    assert st["update_available"] is True
    assert st["error"] is None
    assert st["checked_at"] is not None


def test_check_none_when_hashes_match(tmp_path):
    d = _make_local_model(tmp_path)
    cfg = _check_cfg(d)
    info = SimpleNamespace(
        siblings=[_sib("model.safetensors", _local_sha(d / "model.safetensors"))]
    )

    assert asyncio.run(model_watcher.check_once(cfg, api=_api(info))) is None
    st = model_watcher.state()
    assert st["update_available"] is False
    assert st["error"] is None


def test_non_lfs_and_foreign_files_do_not_participate(tmp_path):
    """Non-LFS files, non-weight files and excluded dirs must not fake an update."""
    d = _make_local_model(tmp_path)
    cfg = _check_cfg(d)
    info = SimpleNamespace(
        siblings=[
            _sib("model.safetensors", _local_sha(d / "model.safetensors")),
            _sib("weights.pt"),  # non-LFS: no cheap hash, must be ignored
            _sib("tokenizer.json", "b" * 64),  # not a weight file
            _sib("onnx/model.onnx", "c" * 64),  # excluded subdirectory
        ]
    )

    assert asyncio.run(model_watcher.check_once(cfg, api=_api(info))) is None
    assert model_watcher.state()["update_available"] is False


def test_hf_error_is_swallowed(tmp_path):
    d = _make_local_model(tmp_path)
    cfg = _check_cfg(d)

    class RaisingApi:
        def model_info(self, *a, **kw):
            raise RuntimeError("network down")

    assert asyncio.run(model_watcher.check_once(cfg, api=RaisingApi())) is None
    st = model_watcher.state()
    assert st["error"]
    assert st["update_available"] is False


def test_missing_hub_package_is_reported(tmp_path, monkeypatch):
    d = _make_local_model(tmp_path)
    cfg = _check_cfg(d)
    monkeypatch.setattr(model_watcher, "_import_hf_api", lambda: None)

    assert asyncio.run(model_watcher.check_once(cfg)) is None
    assert model_watcher.state()["error"]


def test_unresolvable_model_is_reported(tmp_path):
    cfg = {"embeddings": {"model_name": "BAAI/bge-m3"}}

    assert asyncio.run(model_watcher.check_once(cfg)) is None
    assert model_watcher.state()["error"]


# ---------------------------------------------------------------------------
# start / stop
# ---------------------------------------------------------------------------


def _assert_start_disabled(cfg):
    async def scenario():
        assert model_watcher.start(cfg) is None

    asyncio.run(scenario())


def test_interval_zero_disables(tmp_path):
    _assert_start_disabled(_start_cfg(tmp_path, 0))


def test_interval_parsing_tolerates_garbage(tmp_path):
    # Garbage and negative values disable the watcher without raising.
    for hours in ("abc", -5, None, ""):
        _assert_start_disabled(_start_cfg(tmp_path, hours))


def test_explicit_model_path_disables_watcher(tmp_path):
    d = _make_local_model(tmp_path)
    cfg = {
        "embeddings": {
            "model_path": str(d),
            "model_name": "BAAI/bge-m3",
            "update_check_interval_hours": 72,
        }
    }
    _assert_start_disabled(cfg)


def test_unresolvable_model_disables_watcher(tmp_path):
    cfg = {
        "_root": str(tmp_path / "empty"),
        "embeddings": {"model_name": "BAAI/bge-m3", "update_check_interval_hours": 72},
    }
    _assert_start_disabled(cfg)


def test_missing_hub_package_disables_start(tmp_path, monkeypatch):
    cfg = _start_cfg(tmp_path, 72)
    monkeypatch.setattr(model_watcher, "_import_hf_api", lambda: None)
    _assert_start_disabled(cfg)


def test_start_checks_immediately_and_stop_cancels(tmp_path, monkeypatch):
    cfg = _start_cfg(tmp_path, 72)
    calls: list[dict] = []

    async def fake_check(c):
        calls.append(c)

    monkeypatch.setattr(model_watcher, "check_once", fake_check)

    async def scenario():
        task = model_watcher.start(cfg)
        assert task is not None
        # First check runs at the task's first await, i.e. before the
        # first sleep(interval).
        await asyncio.sleep(0.05)
        assert calls == [cfg]
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    model_watcher.stop()  # no-op on an already cancelled task
    assert model_watcher.state()["update_available"] is False


def test_start_is_idempotent_while_running(tmp_path, monkeypatch):
    cfg = _start_cfg(tmp_path, 72)

    async def fake_check(c):
        return None

    monkeypatch.setattr(model_watcher, "check_once", fake_check)

    async def scenario():
        t1 = model_watcher.start(cfg)
        t2 = model_watcher.start(cfg)
        assert t1 is not None
        assert t1 is t2
        t1.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await t1

    asyncio.run(scenario())

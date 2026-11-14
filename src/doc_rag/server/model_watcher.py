"""Background check that the local embedding weights match the HF Hub.

Compares the local weight fingerprint (model_paths.model_fingerprint)
against the LFS sha256 hashes published on the Hub for the configured
model. Read-only: nothing is downloaded and the loaded model is never
swapped. The result only feeds the UI/status flag
``model_update_available``.

Design constraints:

- huggingface_hub is imported lazily INSIDE check_once/start: the server
  must stay importable for --minimal installs without the embeddings
  extra. This module itself never imports it at module level.
- Only LFS siblings carry a cheap content hash (lfs.sha256). Non-LFS
  files (e.g. weights shipped as plain *.pt) cannot be hashed without
  downloading, so they must NOT participate in the comparison —
  otherwise every non-LFS deployment would report a permanent false
  "update available".
- The loop is a plain asyncio task: check immediately after startup,
  then sleep(interval) between checks. Task cancellation is the only
  stop mechanism (stop()).
"""

from __future__ import annotations

import asyncio
import fnmatch
import logging
import time
from collections.abc import Iterable
from typing import Any

from doc_rag.raglib.model_paths import (
    _WEIGHT_EXCLUDED_DIRS,
    _WEIGHT_PATTERNS,
    fingerprint_from_pairs,
    model_fingerprint,
    resolve_model_path,
)

log = logging.getLogger(__name__)

_DEFAULT_INTERVAL_HOURS = 72
_HUB_TIMEOUT_S = 10

# Weight-name filter shared with model_paths: only files that the local
# fingerprint also counts may take part in the remote comparison.
_WEIGHT_PATTERNS = tuple(_WEIGHT_PATTERNS)
_WEIGHT_EXCLUDED_DIRS = set(_WEIGHT_EXCLUDED_DIRS)

_state: dict[str, Any] = {
    "update_available": False,
    "checked_at": None,
    "error": None,
}
_task: asyncio.Task | None = None


def _import_hf_api():
    """Return the HfApi class, or None when huggingface_hub is absent.

    Indirection exists so tests can simulate a missing embeddings extra
    via monkeypatch without uninstalling the real package.
    """
    try:
        from huggingface_hub import HfApi
    except Exception:
        return None
    return HfApi


def _is_weight(rel: str) -> bool:
    """True when relpath matches the local weight patterns and is not
    inside an excluded subdirectory (mirrors model_paths.weight_files)."""
    parts = rel.split("/")
    if any(part in _WEIGHT_EXCLUDED_DIRS for part in parts[:-1]):
        return False
    name = parts[-1]
    return any(fnmatch.fnmatch(name, pat) for pat in _WEIGHT_PATTERNS)


def _parse_interval(cfg: dict[str, Any]) -> float:
    """embeddings.update_check_interval_hours as hours, 0.0 when disabled.

    Tolerates garbage: unparseable values and negatives disable the
    watcher instead of raising.
    """
    raw = (cfg.get("embeddings") or {}).get("update_check_interval_hours")
    try:
        hours = float(raw)
    except (TypeError, ValueError):
        return 0.0
    if hours <= 0:
        return 0.0
    return hours


def state() -> dict[str, Any]:
    """Watcher outcome: update_available, checked_at, error."""
    return dict(_state)


def _set_state(**updates: Any) -> None:
    _state.update(updates)


def reset_for_tests() -> None:
    """Clear state and drop any running task; test isolation only."""
    global _task
    if _task is not None and not _task.done():
        _task.cancel()
    _task = None
    _state.clear()
    _state.update({"update_available": False, "checked_at": None, "error": None})


def _remote_pairs(siblings: Iterable[Any]) -> list[tuple[str, str]]:
    """(sha256, relpath) pairs from Hub siblings, LFS weight files only."""
    pairs: list[tuple[str, str]] = []
    for sib in siblings or []:
        rel = getattr(sib, "rfilename", None)
        lfs = getattr(sib, "lfs", None)
        if not rel or lfs is None:
            continue
        sha = getattr(lfs, "sha256", None)
        if not sha or not _is_weight(str(rel)):
            continue
        pairs.append((str(sha), str(rel)))
    return pairs


async def check_once(cfg: dict[str, Any], *, api: Any | None = None) -> str | None:
    """Run one Hub comparison and record the outcome in state().

    Returns the remote aggregate fingerprint when it differs from the
    local one, else None (including any case where the comparison cannot
    be performed: no Hub package, unresolvable model, network error).
    ``api`` is an injection point for tests; by default a lazily imported
    HfApi() is used. The Hub call itself runs in a worker thread so the
    event loop is never blocked by the synchronous HTTP request.
    """
    try:
        return await asyncio.to_thread(_check_once_inner, cfg, api)
    except Exception as exc:
        _set_state(error=f"{type(exc).__name__}: {exc}")
        log.warning(
            "Сверка весов с HF Hub не выполнена (%s: %s).",
            type(exc).__name__,
            exc,
        )
        return None


def _check_once_inner(cfg: dict[str, Any], api: Any | None) -> str | None:
    hf_api_cls = api if api is not None else _import_hf_api()
    # Injection may also pass a bare api object; None only means "no hub".
    if hf_api_cls is None:
        _set_state(error="huggingface_hub is not installed")
        return None

    model_dir = resolve_model_path(cfg)
    if model_dir is None:
        _set_state(error="local embedding model directory not found")
        return None
    local_fp = model_fingerprint(model_dir)
    if local_fp is None:
        _set_state(error="model directory has no weight files")
        return None

    repo_id = str((cfg.get("embeddings") or {}).get("model_name") or "")
    if not repo_id:
        _set_state(error="embeddings.model_name is not configured")
        return None

    if api is None:
        api = hf_api_cls()
    info = api.model_info(repo_id, files_metadata=True, timeout=_HUB_TIMEOUT_S)
    remote_fp = fingerprint_from_pairs(_remote_pairs(getattr(info, "siblings", None)))
    _set_state(checked_at=time.time(), error=None)

    if remote_fp == local_fp:
        _set_state(update_available=False)
        log.info("Веса модели совпадают с HF Hub (%s).", repo_id)
        return None

    _set_state(update_available=True)
    log.warning(
        "На HF Hub доступна обновлённая ревизия модели %s "
        "(локальный фингерпринт %s, удалённый %s). "
        "Обновите модель: scripts/fetch_model.py, затем перезапустите сервис и выполните "
        "doc-rag rebuild.",
        repo_id,
        local_fp,
        remote_fp,
    )
    return remote_fp


async def _loop(cfg: dict[str, Any], interval_s: float) -> None:
    """check immediately, then every interval_s seconds."""
    while True:
        await check_once(cfg)
        await asyncio.sleep(interval_s)


def start(cfg: dict[str, Any]) -> asyncio.Task | None:
    """Start the background check task, or None when it must not run.

    Disabled (returns None) when the interval is 0/unparseable, an
    explicit embeddings.model_path is set (the user manages the model
    manually, no Hub repo is implied), the model directory cannot be
    resolved, or huggingface_hub is unavailable. When a task is already
    running it is returned unchanged (idempotent).
    """
    global _task
    if _task is not None and not _task.done():
        return _task

    if _parse_interval(cfg) == 0.0:
        return None
    embed = cfg.get("embeddings") or {}
    if embed.get("model_path"):
        return None
    if resolve_model_path(cfg) is None:
        return None
    if _import_hf_api() is None:
        return None

    interval_s = _parse_interval(cfg) * 3600.0
    _task = asyncio.create_task(_loop(cfg, interval_s))
    return _task


def stop() -> None:
    """Cancel the background task, if running."""
    global _task
    if _task is not None and not _task.done():
        _task.cancel()
    _task = None

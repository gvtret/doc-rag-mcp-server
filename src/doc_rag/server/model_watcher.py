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
  downloading, so they must NOT participate in the comparison --
  otherwise every non-LFS deployment would report a permanent false
  "update available". An empty remote LFS set (nothing comparable)
  is likewise never reported as an update.
- The loop is a plain asyncio task: check immediately after startup,
  then sleep(interval) between checks. Task cancellation is the only
  stop mechanism (stop()).
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Iterable
from typing import Any

from doc_rag.raglib.model_paths import (
    fingerprint_from_pairs,
    is_weight_relpath,
    model_fingerprint,
    resolve_model_path,
)

log = logging.getLogger(__name__)

# Used when embeddings.update_check_interval_hours is absent from the
# config; an explicit 0/garbage value still disables the watcher.
_DEFAULT_INTERVAL_HOURS = 72.0
_HUB_TIMEOUT_S = 10

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


def _parse_interval(cfg: dict[str, Any]) -> float:
    """embeddings.update_check_interval_hours as hours, 0.0 when disabled.

    Tolerates garbage: unparseable values and negatives disable the
    watcher instead of raising. A missing key falls back to the default
    interval; an explicit value always wins.
    """
    embed = cfg.get("embeddings") or {}
    if "update_check_interval_hours" not in embed:
        return _DEFAULT_INTERVAL_HOURS
    raw = embed.get("update_check_interval_hours")
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
        if not sha or not is_weight_relpath(str(rel)):
            continue
        pairs.append((str(sha), str(rel)))
    return pairs


def _log_check_failure(exc: Exception) -> None:
    """Record a failed check and log it at INFO.

    A network problem or a missing package is an ordinary condition for a
    background checker (the brief reserves WARN for a found update); the
    error string stays visible in state().
    """
    _set_state(error=f"{type(exc).__name__}: {exc}")
    log.info("Сверка весов с HF Hub не выполнена (%s: %s).", type(exc).__name__, exc)


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
        _log_check_failure(exc)
        return None


def _check_once_inner(cfg: dict[str, Any], api: Any | None) -> str | None:
    # ``api`` is either a ready-to-call object (test injection) or None;
    # in the latter case a lazily imported HfApi class is instantiated.
    hf = api if api is not None else _import_hf_api()
    if hf is None:
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
        hf = hf()
    info = hf.model_info(repo_id, files_metadata=True, timeout=_HUB_TIMEOUT_S)
    remote_pairs = _remote_pairs(getattr(info, "siblings", None))
    if not remote_pairs:
        # Nothing comparable (all siblings non-LFS, e.g. *.pt weights, or
        # an unexpected repo layout). fingerprint_from_pairs([]) would be
        # sha256("") and could never match the local aggregate, so this
        # must NOT be reported as an update.
        _set_state(checked_at=time.time(), error="no LFS weight files on the Hub")
        log.info(
            "На HF Hub для %s нет LFS-файлов весов, сопоставление невозможно.",
            repo_id,
        )
        return None

    remote_fp = fingerprint_from_pairs(remote_pairs)
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
    running it is returned unchanged (idempotent). A stale task left
    over from a dead event loop does not block a new one.
    """
    global _task
    if _task is not None and not _task.done() and _task.get_loop() is asyncio.get_running_loop():
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

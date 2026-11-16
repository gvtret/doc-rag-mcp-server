"""Single thread-safe creation point for the sentence-transformers embedder.

All former load sites (vector_index._load_embedder, the legacy FAISS
builder now in doc_rag.raglib.vectorstore.faiss_store, and the legacy
FAISS path in server/retrieval.py) delegate here. The loader
prefers a local model directory resolved by model_paths (embeddings.model_path,
then <root>/build/models/bge-m3, then <root>/models/bge-m3) and always loads
it with local_files_only=True. When no local directory exists but
embeddings.model_name is configured, the model is loaded from the HF cache by
name, also offline (source "hf_cache"), so existing installations keep working.

Behaviour:
- guarded module-level import via _import_st(): catches Exception, not
  just ImportError, so a broken torch/CUDA (OSError at import time)
  leaves the module importable; tests monkeypatch
  doc_rag.raglib.model_loader.SentenceTransformer;
- one threading.Lock guards the cache check + load (single-flight):
  concurrent callers with the same (model, device) key share one instance
  and the constructor runs exactly once;
- a failed attempt on the selected device is retried once on "cpu"; when
  that also fails the loader returns None and records the error in
  status();
- only successful loads are cached; failures are retried on the next call,
  so a fixed environment recovers without a process restart.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from doc_rag.raglib.model_paths import fingerprint_for_cfg, resolve_model_path


def _import_st():
    """Import sentence_transformers, returning None on any failure.

    Exception (not just ImportError): a broken CUDA/torch installation
    can raise OSError (e.g. missing libcudart.so) at import time, and the
    module must stay importable so callers degrade to lexical search.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except Exception:  # pragma: no cover - covered via _import_st unit test
        return None
    return SentenceTransformer


SentenceTransformer = _import_st()

_lock = threading.Lock()
_cache: dict[tuple[str, str], Any] = {}
_current: Any | None = None
_EMPTY_STATUS: dict[str, Any] = {
    "ready": False,
    "path": None,
    "fingerprint": None,
    "error": None,
    "source": None,
}
_status: dict[str, Any] = dict(_EMPTY_STATUS)


def _load_ollama_embedder(cfg: dict[str, Any]) -> Any | None:
    """Return an OllamaEmbedder when cfg opts in and Ollama is reachable.

    Lazy import keeps module importable when the helper module is absent
    (e.g. partial deployments). A cheap GET /api/tags probe gates the
    "ready" status: without it the loader would mark the service ready
    while every search fails after retries instead of degrading to
    lexical fallback (the documented degradation contract). The probe
    result is cached for _OLLAMA_PROBE_TTL seconds so the service
    recovers on its own once Ollama comes back, without a restart.
    """
    try:
        from doc_rag.raglib.ollama_embedder import build_ollama_embedder
    except Exception as exc:  # pragma: no cover - defensive
        _status.update({"ready": False, "path": None, "fingerprint": None,
                        "error": f"ollama_embedder import failed: {exc}", "source": None})
        return None
    try:
        emb = build_ollama_embedder(cfg)
    except Exception as exc:
        # Keep the ollama-specific reason visible: the caller (load_embedder)
        # must not fall through to the local branch and overwrite status.error.
        _status.update({"ready": False, "path": None, "fingerprint": None,
                        "error": f"ollama embedder config error: {exc}", "source": None})
        return None
    if emb is None:
        return None
    try:
        from doc_rag.raglib.model_paths import fingerprint_for_cfg

        fingerprint = fingerprint_for_cfg(cfg)
    except Exception:  # pragma: no cover - defensive: fingerprint is advisory
        fingerprint = None
    if not _ollama_probe_ok(emb):
        _status.update({"ready": False, "path": f"ollama:{emb.model}",
                        "fingerprint": fingerprint,
                        "error": f"ollama unreachable at {emb.api_base}",
                        "source": "ollama"})
        return None
    _status.update({"ready": True, "path": f"ollama:{emb.model}",
                    "fingerprint": fingerprint, "error": None, "source": "ollama"})
    return emb


_OLLAMA_PROBE_TTL = 30.0
_OLLAMA_PROBE_STATE: dict[str, tuple[bool, float]] = {}
_OLLAMA_PROBE_LOCK = threading.Lock()


def _ollama_probe_ok(emb: Any) -> bool:
    """Cached reachability probe for the Ollama server (TTL-gated)."""
    from doc_rag.raglib.ollama_embedder import ollama_reachable

    now = time.monotonic()
    with _OLLAMA_PROBE_LOCK:
        cached = _OLLAMA_PROBE_STATE.get(emb.api_base)
        if cached is not None and now - cached[1] < _OLLAMA_PROBE_TTL:
            return cached[0]
    ok = False
    try:
        ok = bool(ollama_reachable(emb.api_base, timeout=2.0))
    except Exception:
        ok = False
    with _OLLAMA_PROBE_LOCK:
        _OLLAMA_PROBE_STATE[emb.api_base] = (ok, time.monotonic())
    return ok


def _resolve_device(cfg: dict[str, Any]) -> str:
    """Normalize embeddings.device; '', 'auto' and 'gpu' pick cuda when available."""
    dev = str((cfg.get("embeddings") or {}).get("device") or "auto").strip().lower()
    if dev in ("", "auto", "gpu"):
        try:
            import torch

            has_cuda = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
        except Exception:
            has_cuda = False
        return "cuda" if has_cuda else "cpu"
    return dev


def load_embedder(cfg: dict[str, Any]) -> Any | None:
    """Return the embedder for cfg, creating it once per (model, device) key.

    Cache key is (model_dir as str, or model_name when no local directory
    exists, device). Failures are not cached: the next call retries.
    Note: max_seq_length / normalize are not part of the key; changing
    them in config requires a process restart to take effect.
    """
    ollama_configured = str((cfg.get("embeddings") or {}).get("provider") or "").strip().lower() == "ollama"
    ollama = _load_ollama_embedder(cfg)
    if ollama is not None:
        return ollama
    if ollama_configured:
        # Fail-closed: with provider=ollama a dead/misconfigured server must
        # NOT silently fall through to the local sentence-transformers branch
        # -- that would mask the real error in status() and try to load a
        # model directory that (by design of the ollama deployment) has no
        # weights. Returning None lets callers degrade to lexical search.
        return None
    device = _resolve_device(cfg)
    model_dir = resolve_model_path(cfg)
    model_name = str((cfg.get("embeddings") or {}).get("model_name") or "")
    identity = str(model_dir) if model_dir is not None else model_name
    key = (identity, device)
    with _lock:
        cached = _cache.get(key)
        if cached is not None:
            return cached
        model = _build(cfg, model_dir, model_name, device)
        if model is None:
            return None
        _cache[key] = model
        return model


def _build(
    cfg: dict[str, Any],
    model_dir: Any | None,
    model_name: str,
    device: str,
) -> Any | None:
    """Construct the model and update status. Caller holds the lock."""
    global _current
    if SentenceTransformer is None:
        _status.update(
            {
                "ready": False,
                "path": None,
                "fingerprint": None,
                "error": "sentence-transformers is not installed",
                "source": None,
            }
        )
        return None
    if model_dir is None and not model_name:
        _status.update(
            {
                "ready": False,
                "path": None,
                "fingerprint": None,
                "error": "no local model directory and no embeddings.model_name",
                "source": None,
            }
        )
        return None

    path = str(model_dir) if model_dir is not None else None
    source = "local" if model_dir is not None else "hf_cache"

    model = None
    error: str | None = None
    try:
        if model_dir is not None:
            model = SentenceTransformer(str(model_dir), device=device, local_files_only=True)
        else:
            model = SentenceTransformer(model_name, device=device, local_files_only=True)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    if model is None and device != "cpu":
        # Some stacks reject the selected device (e.g. broken cuda setup):
        # retry once on cpu before degrading to "no embedder".
        try:
            if model_dir is not None:
                model = SentenceTransformer(str(model_dir), device="cpu", local_files_only=True)
            else:
                model = SentenceTransformer(model_name, device="cpu", local_files_only=True)
        except Exception as exc:
            error = error or f"{type(exc).__name__}: {exc}"
        else:
            error = None

    if model is None:
        _status.update(
            {
                "ready": False,
                "path": path,
                "fingerprint": None,
                "error": error or "embedder construction failed",
                "source": None,
            }
        )
        return None

    # Fingerprint only after a successful load: fingerprint_for_cfg can
    # hash gigabytes of weights, which must not run on a request path
    # that fails repeatedly under the loader lock.
    fingerprint = fingerprint_for_cfg(cfg) if model_dir is not None else None

    # Cap the sequence length to the configured value (chunks are ~512
    # tokens; bge-m3 defaults to 8192, which only wastes memory/time).
    max_seq = (cfg.get("embeddings") or {}).get("max_seq_length")
    if max_seq:
        try:
            model.max_seq_length = int(max_seq)
        except Exception:
            pass

    _current = model
    _status.update(
        {"ready": True, "path": path, "fingerprint": fingerprint, "error": None, "source": source}
    )
    return model


def current_embedder() -> Any | None:
    """The most recently loaded embedder instance, or None before any success."""
    return _current


def status() -> dict[str, Any]:
    """Last load attempt outcome: ready, path, fingerprint, error, source."""
    return dict(_status)


def reset_for_tests() -> None:
    """Drop cached models and status; test isolation only."""
    global _current
    with _lock:
        _cache.clear()
        _current = None
        _status.clear()
        _status.update(_EMPTY_STATUS)

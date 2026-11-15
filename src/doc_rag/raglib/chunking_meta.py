"""Chunking / embedding-text config fingerprint.

sha256 over the chunking parameters, the schema version constants and the
tokenizer.json hash. Persisted in index meta as ``config_fingerprint``:
changing any of these inputs changes the fingerprint, which forces a full
index rebuild (vectors and texts would otherwise silently desynchronize
from the configuration that produced them).

Also hosts ``count_tokens`` -- the real bge-m3 token count used by the
chunkers for sizing (replaces the len//4 estimate). The tokenizer is
loaded lazily once per process and shared across threads.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from pathlib import Path
from typing import Any

# Bump when anything that determines chunk boundaries OR chunk text changes
# for the same config values (1 = len//4 sizing, 2 = postprocessing v2,
# 3 = postprocessing full-set rebuild).
# 4 = chunk sizing moved to the real bge-m3 tokenizer with a hard 504-token
# cap: chunk boundaries change for the same config values.
CHUNKING_SCHEMA_VERSION = "4"
# Bump when the embedding-text scheme changes (prefixes, metadata injection,
# trimming rules), even if chunk boundaries stay the same.
EMBEDDING_TEXT_VERSION = "1"

#: Hard cap on real (bge-m3) tokens per chunk, including tables. The
#: embedding model truncates inputs at 512 tokens; 8 tokens of headroom
#: absorb special tokens ([CLS]/[SEP]/[EOS]-style) added by the encoder.
CHUNK_MAX_TOKENS = 504

logger = logging.getLogger(__name__)

# Latches so each one-time WARN below is logged once, not on every call.
_warned_no_tokenizer = False
_warned_bad_chunking_values = False


def _warn_once(flag: str, message: str) -> None:
    """Log `message` at WARN at most once per process for `flag`."""
    global _warned_no_tokenizer, _warned_bad_chunking_values
    if flag == "no_tokenizer":
        already = _warned_no_tokenizer
        _warned_no_tokenizer = True
    else:
        already = _warned_bad_chunking_values
        _warned_bad_chunking_values = True
    if not already:
        logger.warning(message)


# --------------------------------------------------------------------------
# count_tokens: real bge-m3 token counting with lazy tokenizer singleton
# --------------------------------------------------------------------------

_tokenizer_lock = threading.Lock()
_tokenizer_cache: dict[str, Any] = {}  # path -> tokenizers.Tokenizer or None
_MISS = object()  # sentinel: key absent from _tokenizer_cache
# Resolved default tokenizer.json path. Seeded by configure_tokenizer(cfg)
# when the pipeline loads its config (so embeddings.model_path is
# honoured); falls back to the cfg-less DOC_RAG_ROOT / repo discovery.
_configured_tokenizer_path: str | None = None


def configure_tokenizer(cfg: dict[str, Any]) -> None:
    """Seed the default tokenizer path from a loaded config.

    Resolves via the same model-path rules as config_fingerprint
    (embeddings.model_path / build/models / models) and caches the
    result, so later count_tokens calls skip path discovery entirely.
    Safe to call repeatedly; re-resolves each time (config may change).
    """
    global _configured_tokenizer_path
    tok_path = resolve_tokenizer_path(cfg)
    with _tokenizer_lock:
        _configured_tokenizer_path = str(tok_path) if tok_path is not None else None


def _load_tokenizer(tokenizer_path: str) -> Any:
    """Load and cache a Tokenizer for `tokenizer_path` (or None on failure).

    The double-checked-lock singleton: the first caller loads the file
    while holding the lock; concurrent callers block and then reuse the
    cached object. A failed load (missing file / no `tokenizers` package)
    is cached as None so we do not retry on every call.
    """
    # Fast path without the lock: the tokenizer for this path is already
    # cached (hit on every call after the first).
    cached = _tokenizer_cache.get(tokenizer_path, _MISS)
    if cached is not _MISS:
        return cached
    with _tokenizer_lock:
        if tokenizer_path in _tokenizer_cache:
            return _tokenizer_cache[tokenizer_path]
        tok = None
        try:
            from tokenizers import Tokenizer

            tok = Tokenizer.from_file(tokenizer_path)
        except FileNotFoundError:
            _warn_once(
                "no_tokenizer",
                "tokenizer.json not found: count_tokens falls back to "
                "len(text)//4 (chunk sizes become estimates)",
            )
        except ImportError:
            _warn_once(
                "no_tokenizer",
                "tokenizers package not installed: count_tokens falls back "
                "to len(text)//4 (chunk sizes become estimates)",
            )
        except Exception as e:  # malformed tokenizer.json and the like
            _warn_once(
                "no_tokenizer",
                f"tokenizer load failed ({e}): count_tokens falls back to "
                "len(text)//4 (chunk sizes become estimates)",
            )
        _tokenizer_cache[tokenizer_path] = tok
        return tok


def _default_tokenizer_file() -> str | None:
    """Default tokenizer.json path for count_tokens.

    Order: the path seeded by configure_tokenizer(cfg) (respects
    embeddings.model_path), then the cfg-less discovery via
    resolve_model_path({}) which honours DOC_RAG_ROOT / repo root.
    Returns None when no model dir (or no tokenizer.json in it) is
    found. The resolved path is cached so repeated calls do not stat
    the candidate directories on every chunk.
    """
    global _configured_tokenizer_path
    if _configured_tokenizer_path is not None:
        return _configured_tokenizer_path

    from doc_rag.raglib.model_paths import resolve_model_path

    model_dir = resolve_model_path({})
    if model_dir is None:
        return None
    candidate = model_dir / "tokenizer.json"
    if not candidate.is_file():
        return None
    resolved = str(candidate)
    with _tokenizer_lock:
        if _configured_tokenizer_path is None:
            _configured_tokenizer_path = resolved
    return resolved


def count_tokens(text: str, tokenizer_path: str | None = None) -> int:
    """Real bge-m3 token count for `text`.

    Uses the model's tokenizer.json (path resolved via the shared
    model-path discovery when `tokenizer_path` is None). Without a
    usable tokenizer, falls back to max(1, len(text)//4) with a WARN
    logged once per process.
    """
    if not text:
        return 0
    if tokenizer_path is None:
        tokenizer_path = _default_tokenizer_file()
    if tokenizer_path is None:
        _warn_once(
            "no_tokenizer",
            "no model dir / tokenizer.json found: count_tokens falls back "
            "to len(text)//4 (chunk sizes become estimates)",
        )
        return max(1, len(text) // 4)
    tok = _load_tokenizer(tokenizer_path)
    if tok is None:
        return max(1, len(text) // 4)
    return len(tok.encode(text).ids)


def reset_count_tokens_for_tests() -> None:
    """Drop the tokenizer cache, seeded path and WARN latches (tests only)."""
    global _warned_no_tokenizer, _configured_tokenizer_path
    with _tokenizer_lock:
        _tokenizer_cache.clear()
        _configured_tokenizer_path = None
    _warned_no_tokenizer = False


def _tokenizer_sha256(tokenizer_path: Path | None) -> str:
    """sha256 of tokenizer.json, or "" when it is missing/unreadable.

    An excluded hash keeps the fingerprint computable for deployments
    without a local tokenizer file; the fingerprint still changes on the
    other inputs.
    """
    if tokenizer_path is None or not tokenizer_path.is_file():
        return ""
    try:
        h = hashlib.sha256()
        with tokenizer_path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError as e:
        logger.warning("could not hash tokenizer %s (%s): excluded from fingerprint", e)
        return ""


def resolve_tokenizer_path(cfg: dict[str, Any]) -> Path | None:
    """Path to the embedding model's tokenizer.json, or None.

    Uses the shared model-path resolution (model_path / build/models /
    models) and looks for tokenizer.json inside it.
    """
    from doc_rag.raglib.model_paths import resolve_model_path

    model_dir = resolve_model_path(cfg)
    if model_dir is None:
        return None
    candidate = model_dir / "tokenizer.json"
    return candidate if candidate.is_file() else None


def config_fingerprint(cfg: dict[str, Any], tokenizer_path: Path | None = None) -> str:
    """Aggregate sha256 over the chunking-relevant config inputs.

    Inputs: chunking strategy / target_tokens / overlap_tokens, both
    schema version constants and sha256(tokenizer.json). When
    tokenizer_path is None it is resolved from cfg; a missing
    tokenizer.json simply excludes the tokenizer hash (WARN logged once).
    """
    if tokenizer_path is None:
        tokenizer_path = resolve_tokenizer_path(cfg)

    tok_sha = _tokenizer_sha256(tokenizer_path)
    if not tok_sha:
        _warn_once(
            "no_tokenizer",
            "tokenizer.json not found: its hash is excluded from the config fingerprint",
        )

    chunking = cfg.get("chunking") or {}
    try:
        target_tokens = int(chunking.get("target_tokens", 512))
        overlap_tokens = int(chunking.get("overlap_tokens", 64))
    except (TypeError, ValueError):
        _warn_once(
            "bad_chunking",
            "non-integer chunking target_tokens/overlap_tokens: "
            "falling back to defaults (512/64) for the config fingerprint",
        )
        target_tokens, overlap_tokens = 512, 64
    payload = {
        "chunking_schema_version": CHUNKING_SCHEMA_VERSION,
        "embedding_text_version": EMBEDDING_TEXT_VERSION,
        "strategy": str(chunking.get("strategy", "")),
        "target_tokens": target_tokens,
        "overlap_tokens": overlap_tokens,
        "tokenizer_sha256": tok_sha,
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

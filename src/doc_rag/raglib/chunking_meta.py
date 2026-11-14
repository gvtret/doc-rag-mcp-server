"""Chunking / embedding-text config fingerprint.

sha256 over the chunking parameters, the schema version constants and the
tokenizer.json hash. Persisted in index meta as ``config_fingerprint``:
changing any of these inputs changes the fingerprint, which forces a full
index rebuild (vectors and texts would otherwise silently desynchronize
from the configuration that produced them).

Stdlib-only: imported by vector_index and migrate_vectors, which must not
pull in heavy deps just to compute a hash.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

# Bump when the chunker's behaviour changes in a way that changes chunk
# boundaries for the same config values (1 = current len//4 behaviour).
CHUNKING_SCHEMA_VERSION = "2"
# Bump when the embedding-text scheme changes (prefixes, metadata injection,
# trimming rules), even if chunk boundaries stay the same.
EMBEDDING_TEXT_VERSION = "1"

logger = logging.getLogger(__name__)


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
    tokenizer.json simply excludes the tokenizer hash (logged once per
    call at WARN).
    """
    if tokenizer_path is None:
        tokenizer_path = resolve_tokenizer_path(cfg)

    tok_sha = _tokenizer_sha256(tokenizer_path)
    if not tok_sha:
        logger.warning("tokenizer.json not found: its hash is excluded from the config fingerprint")

    chunking = cfg.get("chunking") or {}
    payload = {
        "chunking_schema_version": CHUNKING_SCHEMA_VERSION,
        "embedding_text_version": EMBEDDING_TEXT_VERSION,
        "strategy": str(chunking.get("strategy", "")),
        "target_tokens": int(chunking.get("target_tokens", 0)),
        "overlap_tokens": int(chunking.get("overlap_tokens", 0)),
        "tokenizer_sha256": tok_sha,
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

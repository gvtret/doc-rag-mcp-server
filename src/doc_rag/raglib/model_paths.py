"""Local embedding-model discovery and weights fingerprinting.

Stdlib-only on purpose: both the server and the offline model download
script import this module, and neither must pull in torch or
sentence-transformers just to know where the weights live.

The fingerprint (aggregate sha256 over weight files) is shared by the
local side and the remote verification side, so both must compute it
from the same normalisation rules.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

# Marker files that identify a directory as a (sentence-transformers
# style) embedding model, not just a random folder of weights. Shared
# with scripts/fetch_model.py, which validates a freshly downloaded
# snapshot against the same tuple.
MODEL_MARKER_FILES = ("config.json", "modules.json")

# Weight file patterns and subdirectories excluded from the fingerprint.
_WEIGHT_PATTERNS = ("pytorch_model*.bin", "*.safetensors", "*.pt")
_WEIGHT_EXCLUDED_DIRS = {"onnx", "imgs"}

_WEIGHTS_SHA256_NAME = "WEIGHTS.sha256"

# Sidecar next to WEIGHTS.sha256: records the weight-file inventory
# (relpath + size + mtime_ns) the manifest was computed from, so a
# repeated model_fingerprint call can skip re-hashing gigabytes.
_INVENTORY_NAME = "WEIGHTS.sha256.meta.json"


def _default_root() -> Path | None:
    """Repository root inferred from this file's location.

    src/doc_rag/raglib/model_paths.py -> parents[3] is the repo root.
    The root counts as known only when config/config.yaml exists there.
    """
    candidate = Path(__file__).resolve().parents[3]
    if (candidate / "config" / "config.yaml").is_file():
        return candidate
    return None


def _looks_like_model(d: Path) -> bool:
    return d.is_dir() and all((d / name).is_file() for name in MODEL_MARKER_FILES)


def resolve_model_path(cfg: dict[str, Any], root: str | None = None) -> Path | None:
    """Resolve the local embedding-model directory.

    Order: ``embeddings.model_path`` from cfg, then ``<root>/build/models/bge-m3``,
    then ``<root>/models/bge-m3``. A directory qualifies as a model only when
    both marker files (config.json, modules.json) are present.

    ``root`` falls back to cfg["_root"], then the DOC_RAG_ROOT environment
    variable, then the repository root inferred from this file's location.
    Returns None when nothing resolves.
    """
    cfg_embed = cfg.get("embeddings") or {}
    model_path = cfg_embed.get("model_path")
    if model_path:
        d = Path(os.path.abspath(Path(model_path).expanduser()))
        if _looks_like_model(d):
            return d

    if root:
        base: Path | None = Path(root)
    else:
        cfg_root = cfg.get("_root")
        if cfg_root:
            base = Path(cfg_root)
        else:
            env_root = os.environ.get("DOC_RAG_ROOT")
            if env_root:
                base = Path(os.path.abspath(Path(env_root).expanduser()))
            else:
                base = _default_root()

    if base is None:
        return None

    for candidate in (base / "build" / "models" / "bge-m3", base / "models" / "bge-m3"):
        if _looks_like_model(candidate):
            return candidate
    return None


def weight_files(model_dir: Path) -> list[Path]:
    """Weight files inside model_dir, sorted by path relative to model_dir.

    Matches pytorch_model*.bin, *.safetensors and *.pt at any depth,
    skipping the onnx/ and imgs/ subdirectories.
    """
    out: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(model_dir):
        dirnames[:] = [d for d in dirnames if d not in _WEIGHT_EXCLUDED_DIRS]
        for fn in filenames:
            if any(fnmatch.fnmatch(fn, pat) for pat in _WEIGHT_PATTERNS):
                out.append(Path(dirpath) / fn)
    return sorted(out, key=lambda p: p.relative_to(model_dir).as_posix())


def _pairs_from_weight_files(model_dir: Path) -> list[tuple[str, str]]:
    return [(_sha256_file(p), p.relative_to(model_dir).as_posix()) for p in weight_files(model_dir)]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint_from_pairs(pairs: Iterable[tuple[str, str]]) -> str:
    """Aggregate sha256 over (sha256, relpath) pairs.

    Normalisation: lines of the form ``<sha256>  <relpath>`` are sorted
    by relpath and hashed as sha256 over their concatenation, each line
    terminated by ``\\n`` (i.e. exactly the content of a sha256sum-format
    file). Both the local and the remote side must use this same function.
    """
    lines = [f"{sha}  {rel}" for sha, rel in sorted(pairs, key=lambda kv: kv[1])]
    h = hashlib.sha256()
    for line in lines:
        h.update(line.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def write_weights_sha256(model_dir: Path) -> Path:
    """Write WEIGHTS.sha256 (sha256sum format, sorted by relpath)."""
    lines = [f"{sha}  {rel}" for sha, rel in _pairs_from_weight_files(model_dir)]
    out = model_dir / _WEIGHTS_SHA256_NAME
    out.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    return out


def _write_manifest_from_pairs(model_dir: Path, pairs: list[tuple[str, str]]) -> None:
    """Persist WEIGHTS.sha256 from already-computed (sha, relpath) pairs.

    Avoids the re-hash that calling write_weights_sha256 would do.
    """
    lines = [f"{sha}  {rel}" for sha, rel in pairs]
    out = model_dir / _WEIGHTS_SHA256_NAME
    out.write_text("".join(line + "\n" for line in lines), encoding="utf-8")


def _write_inventory_meta(model_dir: Path, meta_path: Path) -> None:
    meta = {"inventory": _inventory(model_dir)}
    meta_path.write_text(json.dumps(meta), encoding="utf-8")


def _parse_sha256_file(path: Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) == 2 and len(parts[0]) == 64:
            pairs.append((parts[0], parts[1].lstrip("*").strip()))
    return pairs


def _inventory(model_dir: Path) -> list[dict[str, Any]]:
    """Cheap per-file inventory used as the manifest staleness gate.

    Sorted by relpath (weight_files already sorts); captures size and
    mtime_ns so any content rewrite is noticed without hashing.
    """
    inv: list[dict[str, Any]] = []
    for p in weight_files(model_dir):
        st = p.stat()
        inv.append(
            {
                "rel": p.relative_to(model_dir).as_posix(),
                "size": st.st_size,
                "mtime_ns": st.st_mtime_ns,
            }
        )
    return inv


def _load_inventory_meta(meta_path: Path) -> dict[str, Any] | None:
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return meta if isinstance(meta, dict) else None


def model_fingerprint(model_dir: Path) -> str | None:
    """Aggregate weights fingerprint for the model directory.

    Fast path: when WEIGHTS.sha256 exists together with its inventory
    sidecar and the current weight files still match that inventory
    (same relpaths, sizes, mtimes), the manifest aggregate is returned
    without re-hashing the weights.

    Slow path: full re-hash, then persist WEIGHTS.sha256 and the
    sidecar (this also migrates model_path deployments that have no
    manifest yet). A read-only model dir keeps its computed fingerprint
    even when the files cannot be written. Returns None only when there
    are no weight files at all.
    """
    manifest = model_dir / _WEIGHTS_SHA256_NAME
    meta_path = model_dir / _INVENTORY_NAME

    meta = _load_inventory_meta(meta_path)
    if manifest.is_file() and meta is not None:
        try:
            gate_matches = _inventory(model_dir) == meta.get("inventory")
        except OSError:
            gate_matches = False
        if gate_matches:
            pairs = _parse_sha256_file(manifest)
            if pairs:
                return fingerprint_from_pairs(pairs)
            # Empty/inconsistent manifest: fall through and recompute
            # (which also returns None when there are no weight files).

    pairs = _pairs_from_weight_files(model_dir)
    if not pairs:
        return None
    try:
        _write_manifest_from_pairs(model_dir, pairs)
        _write_inventory_meta(model_dir, meta_path)
    except OSError:
        # Read-only model dir: the fingerprint is still valid, it just
        # cannot be persisted for the fast path next time.
        pass
    return fingerprint_from_pairs(pairs)


def fingerprint_for_cfg(cfg: dict[str, Any]) -> str | None:
    """Resolve the model directory from cfg and return its fingerprint."""
    model_dir = resolve_model_path(cfg)
    if model_dir is None:
        return None
    return model_fingerprint(model_dir)

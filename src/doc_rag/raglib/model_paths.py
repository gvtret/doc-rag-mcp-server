"""Local embedding-model discovery and weights fingerprinting.

Stdlib-only on purpose: both the server and the offline model download
script import this module, and neither must pull in torch or
sentence-transformers just to know where the weights live.

The fingerprint (aggregate sha256 over weight files) is shared by the
local side and the remote verification side, so both must compute it
from the same normalisation rules.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

# Marker files that identify a directory as a (sentence-transformers
# style) embedding model, not just a random folder of weights.
_MODEL_MARKER_FILES = ("config.json", "modules.json")

# Weight file patterns and subdirectories excluded from the fingerprint.
_WEIGHT_PATTERNS = ("pytorch_model*.bin", "*.safetensors", "*.pt")
_WEIGHT_EXCLUDED_DIRS = {"onnx", "imgs"}

_WEIGHTS_SHA256_NAME = "WEIGHTS.sha256"


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
    return d.is_dir() and all((d / name).is_file() for name in _MODEL_MARKER_FILES)


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
        d = Path(model_path)
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
                base = Path(env_root)
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
    import fnmatch

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
    and hashed as sha256 over their concatenation, each line terminated
    by ``\\n`` (i.e. exactly the content of a sha256sum-format file).
    Both the local and the remote side must use this same function.
    """
    lines = sorted(f"{sha}  {rel}" for sha, rel in pairs)
    h = hashlib.sha256()
    for line in lines:
        h.update(line.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def write_weights_sha256(model_dir: Path) -> Path:
    """Write WEIGHTS.sha256 (sha256sum format, sorted) into model_dir."""
    lines = sorted(f"{sha}  {rel}" for sha, rel in _pairs_from_weight_files(model_dir))
    out = model_dir / _WEIGHTS_SHA256_NAME
    out.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    return out


def _parse_sha256_file(path: Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) == 2 and len(parts[0]) == 64:
            pairs.append((parts[0], parts[1].lstrip("*").strip()))
    return pairs


def _manifest_matches(model_dir: Path, manifest: Path) -> bool:
    """True when WEIGHTS.sha256 lines still describe the actual weights.

    Guards against a stale manifest left behind after a partial update:
    a fingerprint pinned to stale lines would never notice new weights.
    """
    listed = _parse_sha256_file(manifest)
    actual = _pairs_from_weight_files(model_dir)
    return sorted(listed) == sorted(actual)


def model_fingerprint(model_dir: Path) -> str | None:
    """Aggregate weights fingerprint for the model directory.

    Reuses WEIGHTS.sha256 when present and consistent with the actual
    weight files. Otherwise computes the fingerprint from the weight
    files and rewrites WEIGHTS.sha256 (this also migrates model_path
    deployments that have no manifest yet). Returns None only when
    there are no weight files at all.
    """
    manifest = model_dir / _WEIGHTS_SHA256_NAME
    if manifest.is_file() and _manifest_matches(model_dir, manifest):
        return fingerprint_from_pairs(_parse_sha256_file(manifest))

    pairs = _pairs_from_weight_files(model_dir)
    if not pairs:
        return None
    write_weights_sha256(model_dir)
    return fingerprint_from_pairs(pairs)


def fingerprint_for_cfg(cfg: dict[str, Any]) -> str | None:
    """Resolve the model directory from cfg and return its fingerprint."""
    model_dir = resolve_model_path(cfg)
    if model_dir is None:
        return None
    return model_fingerprint(model_dir)

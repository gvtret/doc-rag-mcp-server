"""FAISS index builder/updater.

This module is intentionally defensive:
- FAISS and embeddings are optional dependencies.
- If they are not available, callers should gracefully fall back to lexical search.

Index layout (under cfg.paths.index_dir):
- faiss.index: FAISS index file
- index_meta.json: metadata that maps FAISS vector order -> chunk_id
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class IndexMeta:
    """Metadata for the vector index."""

    protocol_version: str
    model_name: str
    metric: str
    normalize: bool
    dim: int
    chunk_ids: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "model_name": self.model_name,
            "metric": self.metric,
            "normalize": bool(self.normalize),
            "dim": int(self.dim),
            "chunk_ids": list(self.chunk_ids),
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "IndexMeta":
        return IndexMeta(
            protocol_version=str(d.get("protocol_version", "1")),
            model_name=str(d.get("model_name", "")),
            metric=str(d.get("metric", "ip")),
            normalize=bool(d.get("normalize", True)),
            dim=int(d.get("dim", 0)),
            chunk_ids=[str(x) for x in d.get("chunk_ids", [])],
        )


def _try_import_faiss():
    try:
        import faiss  # type: ignore
        return faiss
    except Exception:
        return None


def _try_import_sentence_transformers():
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        return SentenceTransformer
    except Exception:
        return None



def _resolve_device(device: str) -> str:
    """Normalize device string for torch/sentence-transformers.

    Supported: cpu, cuda, cuda:0, mps, etc.
    Special value 'auto' selects cuda if available, otherwise cpu.
    """
    dev = (device or "").strip().lower()
    if dev == "" or dev == "auto":
        try:
            import torch  # type: ignore
            return "cuda" if getattr(torch, "cuda", None) and torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    # Some people write 'gpu' expecting cuda.
    if dev == "gpu":
        try:
            import torch  # type: ignore
            return "cuda" if getattr(torch, "cuda", None) and torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return device.strip()

def _load_embedder(cfg: Dict[str, Any]):
    SentenceTransformer = _try_import_sentence_transformers()
    if SentenceTransformer is None:
        return None
    model = cfg.get("embeddings", {}).get("model_name", "")
    device = _resolve_device(str(cfg.get("embeddings", {}).get("device", "auto")))
    try:
        return SentenceTransformer(model, device=device)
    except Exception:
        # Fallback: some stacks reject unknown device strings (e.g. 'auto').
        try:
            return SentenceTransformer(model, device="cpu")
        except Exception:
            return None



def _read_chunks_jsonl(chunks_jsonl_path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    if not chunks_jsonl_path.exists():
        return items
    with chunks_jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _paths_from_cfg(cfg: Dict[str, Any]) -> Tuple[Path, Path, Path]:
    root = Path(cfg.get("_root", "."))
    index_dir = root / cfg["paths"]["index_dir"]
    chunks_path = root / cfg["paths"]["chunks_dir"] / "chunks.jsonl"
    index_file = index_dir / "faiss.index"
    meta_file = index_dir / "index_meta.json"
    return chunks_path, index_file, meta_file


def ensure_faiss_index(cfg: Dict[str, Any], *, force_rebuild: bool = False, log=print) -> bool:
    """Ensure that FAISS index exists and is up to date with chunks.jsonl.

    Returns True if semantic index is available (built/updated or already OK).
    Returns False if FAISS/embeddings are missing or build failed.
    """
    faiss = _try_import_faiss()
    if faiss is None:
        log("[doc-rag][index] FAISS not installed -> semantic search disabled")
        return False

    chunks_path, index_file, meta_file = _paths_from_cfg(cfg)
    _ensure_dir(index_file.parent)

    chunks = _read_chunks_jsonl(chunks_path)
    if not chunks:
        log("[doc-rag][index] No chunks yet -> skipping index")
        return False

    metric = str(cfg.get("index", {}).get("metric", "ip")).lower()
    normalize = bool(cfg.get("embeddings", {}).get("normalize", True))
    model_name = str(cfg.get("embeddings", {}).get("model_name", ""))
    protocol_version = str(cfg.get("pipeline_version", "1"))

    meta: Optional[IndexMeta] = None
    if meta_file.exists() and not force_rebuild:
        try:
            meta = IndexMeta.from_dict(json.loads(meta_file.read_text("utf-8")))
        except Exception:
            meta = None

    embedder = _load_embedder(cfg)
    if embedder is None:
        log("[doc-rag][index] sentence-transformers not installed -> semantic search disabled")
        return False

    if force_rebuild:
        meta = None

    all_chunk_ids = [str(c.get("chunk_id")) for c in chunks if c.get("chunk_id")]
    if not all_chunk_ids:
        log("[doc-rag][index] chunks.jsonl has no chunk_id -> cannot build index")
        return False

    if meta is not None:
        if meta.model_name != model_name or meta.metric != metric or meta.normalize != normalize:
            log("[doc-rag][index] Index settings changed -> forcing rebuild")
            meta = None
        elif meta.chunk_ids and len(meta.chunk_ids) > len(all_chunk_ids):
            log("[doc-rag][index] Index meta longer than chunks -> forcing rebuild")
            meta = None

    if meta is None or not index_file.exists():
        log("[doc-rag][index] Building FAISS index from scratch...")
        texts = [str(c.get("text", "")) for c in chunks]
        batch_size = int(cfg.get("embeddings", {}).get("batch_size", 32))
        vecs = embedder.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=normalize,
        )
        dim = int(vecs.shape[1])
        index = faiss.IndexFlatIP(dim) if metric == "ip" else faiss.IndexFlatL2(dim)
        index.add(vecs.astype("float32"))
        faiss.write_index(index, str(index_file))
        meta = IndexMeta(
            protocol_version=protocol_version,
            model_name=model_name,
            metric=metric,
            normalize=normalize,
            dim=dim,
            chunk_ids=all_chunk_ids,
        )
        meta_file.write_text(json.dumps(meta.to_dict(), ensure_ascii=False, indent=2), "utf-8")
        log(f"[doc-rag][index] Built index: ntotal={index.ntotal} dim={dim}")
        return True

    assert meta is not None
    already = len(meta.chunk_ids)
    if already >= len(all_chunk_ids):
        log("[doc-rag][index] Index up to date")
        return True

    log(f"[doc-rag][index] Updating FAISS index incrementally: +{len(all_chunk_ids) - already} chunks")
    index = faiss.read_index(str(index_file))
    if getattr(index, "d", None) and int(index.d) != int(meta.dim):
        log("[doc-rag][index] Dimension mismatch -> forcing rebuild")
        return ensure_faiss_index(cfg, force_rebuild=True, log=log)

    new_chunks = chunks[already:]
    texts = [str(c.get("text", "")) for c in new_chunks]
    batch_size = int(cfg.get("embeddings", {}).get("batch_size", 32))
    vecs = embedder.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=normalize,
    )
    index.add(vecs.astype("float32"))
    faiss.write_index(index, str(index_file))
    meta.chunk_ids.extend(all_chunk_ids[already:])
    meta_file.write_text(json.dumps(meta.to_dict(), ensure_ascii=False, indent=2), "utf-8")
    log(f"[doc-rag][index] Updated index: ntotal={index.ntotal}")
    return True

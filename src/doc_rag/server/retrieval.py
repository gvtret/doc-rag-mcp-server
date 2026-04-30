from __future__ import annotations

"""Shared chunk retrieval (lexical fallback + optional FAISS) for MCP HTTP and debug HTTP."""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from doc_rag.raglib.indexer import ensure_faiss_index


_MANIFEST_CACHE: Dict[str, Any] = {"root": None, "mtime": None, "map": None}
_CFG_CACHE: Dict[str, Any] = {"root": None, "mtime": None, "cfg": None}
_CHUNKS_CACHE: Dict[str, Any] = {"path": None, "mtime": None, "chunks": None}
_SEMANTIC_CACHE: Dict[str, Any] = {"index_path": None, "index_mtime": None, "model_key": None, "index": None, "embedder": None}


def project_root() -> str:
    """Repository root for config/build paths.

    Override with DOC_RAG_ROOT when the package is installed away from project files.
    """
    env = (os.environ.get("DOC_RAG_ROOT") or "").strip()
    if env:
        return os.path.abspath(os.path.expanduser(env))
    here = os.path.abspath(os.path.dirname(__file__))
    return os.path.abspath(os.path.join(here, "..", "..", ".."))


def load_config() -> Dict[str, Any]:
    import yaml

    root = project_root()
    cfg_path = os.path.join(root, "config", "config.yaml")
    try:
        st = os.stat(cfg_path)
    except Exception:
        st = None

    if (
        _CFG_CACHE.get("root") == root
        and st is not None
        and _CFG_CACHE.get("mtime") == st.st_mtime
        and isinstance(_CFG_CACHE.get("cfg"), dict)
    ):
        return _CFG_CACHE["cfg"]

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f) or {}
    cfg["_root"] = root

    if st is not None:
        _CFG_CACHE["root"] = root
        _CFG_CACHE["mtime"] = st.st_mtime
        _CFG_CACHE["cfg"] = cfg
    return cfg


def load_chunks(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    root = str(cfg.get("_root", project_root()))
    paths = cfg.get("paths", {}) or {}
    chunks_dir = paths.get("chunks_dir", "build/chunks_jsonl")
    chunks_path = os.path.join(root, chunks_dir, "chunks.jsonl")
    try:
        st = os.stat(chunks_path)
    except Exception:
        return []

    if (
        _CHUNKS_CACHE.get("path") == chunks_path
        and _CHUNKS_CACHE.get("mtime") == st.st_mtime
        and isinstance(_CHUNKS_CACHE.get("chunks"), list)
    ):
        return _CHUNKS_CACHE["chunks"]

    out: List[Dict[str, Any]] = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue

    _CHUNKS_CACHE["path"] = chunks_path
    _CHUNKS_CACHE["mtime"] = st.st_mtime
    _CHUNKS_CACHE["chunks"] = out
    return out


def _load_manifest_source_map(cfg: Dict[str, Any]) -> Dict[str, str]:
    """Map doc_id -> source_file from manifest (best-effort)."""
    root = str(cfg.get("_root", project_root()))
    paths = cfg.get("paths", {}) or {}
    manifest_rel = str(paths.get("manifest_path", "build/manifest.json"))
    manifest_path = os.path.join(root, manifest_rel)

    try:
        st = os.stat(manifest_path)
    except Exception:
        return {}

    if (
        _MANIFEST_CACHE.get("root") == root
        and _MANIFEST_CACHE.get("mtime") == st.st_mtime
        and isinstance(_MANIFEST_CACHE.get("map"), dict)
    ):
        return _MANIFEST_CACHE["map"]

    out: Dict[str, str] = {}
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        docs = data.get("documents", []) if isinstance(data, dict) else []
        if isinstance(docs, list):
            for d in docs:
                if not isinstance(d, dict):
                    continue
                doc_id = d.get("doc_id")
                source_file = d.get("source_file")
                if isinstance(doc_id, str) and isinstance(source_file, str) and doc_id and source_file:
                    out[doc_id] = source_file
    except Exception:
        out = {}

    _MANIFEST_CACHE["root"] = root
    _MANIFEST_CACHE["mtime"] = st.st_mtime
    _MANIFEST_CACHE["map"] = out
    return out


def _enrich_results_with_source_file(cfg: Dict[str, Any], results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not results:
        return results
    m = _load_manifest_source_map(cfg)
    if not m:
        return results
    for r in results:
        if not isinstance(r, dict):
            continue
        if r.get("source_file"):
            continue
        doc_id = r.get("doc_id")
        if isinstance(doc_id, str) and doc_id in m:
            src = m[doc_id]
            expose = (os.environ.get("DOC_RAG_EXPOSE_SOURCE_PATHS") or "").strip() in ("1", "true", "yes")
            r["source_file"] = src if expose else os.path.basename(src)
    return results


def lexical_search(chunks: List[Dict[str, Any]], query: str, top_k: int) -> List[Dict[str, Any]]:
    q = (query or "").strip().lower()
    if not q:
        return []
    scored: List[Tuple[int, Dict[str, Any]]] = []
    for ch in chunks:
        txt = str(ch.get("text", ""))
        hay = txt.lower()
        score = 0
        if q in hay:
            score += 10 + hay.count(q)
        for token in q.split():
            if token and token in hay:
                score += 1
        if score > 0:
            item = dict(ch)
            item["score"] = float(score)
            scored.append((score, item))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [it for _, it in scored[: max(1, top_k)]]


def semantic_search(cfg: Dict[str, Any], chunks: List[Dict[str, Any]], query: str, top_k: int) -> Optional[List[Dict[str, Any]]]:
    root = str(cfg.get("_root", project_root()))
    paths = cfg.get("paths", {}) or {}
    index_dir = paths.get("index_dir", "build/index")
    index_path = os.path.join(root, index_dir, "faiss.index")
    if not os.path.exists(index_path):
        try:
            ok = ensure_faiss_index(cfg, force_rebuild=False, log=lambda m: None)
        except Exception:
            ok = False
        if not ok or not os.path.exists(index_path):
            return None

    emb = cfg.get("embeddings", {}) or {}
    model_name = emb.get("model_name")
    if not model_name:
        return None

    try:
        import numpy as np  # type: ignore
        import faiss  # type: ignore
        import torch  # type: ignore
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception:
        return None

    device = emb.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model_key = f"{model_name}|{device}"
    try:
        st = os.stat(index_path)
        index_mtime = st.st_mtime
    except Exception:
        index_mtime = None

    if (
        _SEMANTIC_CACHE.get("index_path") == index_path
        and _SEMANTIC_CACHE.get("index_mtime") == index_mtime
        and _SEMANTIC_CACHE.get("model_key") == model_key
        and _SEMANTIC_CACHE.get("index") is not None
        and _SEMANTIC_CACHE.get("embedder") is not None
    ):
        index = _SEMANTIC_CACHE["index"]
        model = _SEMANTIC_CACHE["embedder"]
    else:
        index = faiss.read_index(index_path)
        model = SentenceTransformer(model_name, device=device)
        _SEMANTIC_CACHE["index_path"] = index_path
        _SEMANTIC_CACHE["index_mtime"] = index_mtime
        _SEMANTIC_CACHE["model_key"] = model_key
        _SEMANTIC_CACHE["index"] = index
        _SEMANTIC_CACHE["embedder"] = model

    vec = model.encode([query], normalize_embeddings=True)
    vec = np.asarray(vec, dtype=np.float32)

    D, I = index.search(vec, max(1, int(top_k)))

    out: List[Dict[str, Any]] = []
    for rank, idx in enumerate(I[0]):
        ii = int(idx)
        if 0 <= ii < len(chunks):
            item = dict(chunks[ii])
            item["score"] = float(D[0][rank])
            out.append(item)
    return out


def doc_search(query: str, top_k: int) -> List[Dict[str, Any]]:
    cfg = load_config()
    chunks = load_chunks(cfg)
    top_k = max(1, min(50, int(top_k) if top_k else 6))
    mode = str((cfg.get("mcp", {}) or {}).get("retrieval_mode", "semantic")).lower()
    if mode == "semantic":
        res = semantic_search(cfg, chunks, query, top_k)
        if res is not None:
            return _enrich_results_with_source_file(cfg, res)
    return _enrich_results_with_source_file(cfg, lexical_search(chunks, query, top_k))

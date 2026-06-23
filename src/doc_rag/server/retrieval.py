from __future__ import annotations

"""Shared chunk retrieval (lexical fallback + optional vector store) for MCP HTTP and debug HTTP."""

import json
import os
import re
from typing import Any

_MANIFEST_CACHE: dict[str, Any] = {"root": None, "mtime": None, "map": None}
_CFG_CACHE: dict[str, Any] = {"root": None, "mtime": None, "cfg": None}
_CHUNKS_CACHE: dict[str, Any] = {"path": None, "mtime": None, "chunks": None}
_SEMANTIC_CACHE: dict[str, Any] = {
    "index_path": None,
    "index_mtime": None,
    "model_key": None,
    "index": None,
    "embedder": None,
}
_VECTOR_INDEX_CACHE: dict[str, Any] = {"cfg_root": None, "index": None}


def project_root() -> str:
    """Repository root for config/build paths.

    Override with DOC_RAG_ROOT when the package is installed away from project files.
    """
    env = (os.environ.get("DOC_RAG_ROOT") or "").strip()
    if env:
        return os.path.abspath(os.path.expanduser(env))
    here = os.path.abspath(os.path.dirname(__file__))
    return os.path.abspath(os.path.join(here, "..", "..", ".."))


def load_config() -> dict[str, Any]:
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

    with open(cfg_path, encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f) or {}
    cfg["_root"] = root

    if st is not None:
        _CFG_CACHE["root"] = root
        _CFG_CACHE["mtime"] = st.st_mtime
        _CFG_CACHE["cfg"] = cfg
    return cfg


def load_chunks(cfg: dict[str, Any]) -> list[dict[str, Any]]:
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

    out: list[dict[str, Any]] = []
    with open(chunks_path, encoding="utf-8") as f:
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


def _load_manifest_source_map(cfg: dict[str, Any]) -> dict[str, str]:
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

    out: dict[str, str] = {}
    try:
        with open(manifest_path, encoding="utf-8") as f:
            data = json.load(f)
        docs = data.get("documents", []) if isinstance(data, dict) else []
        if isinstance(docs, list):
            for d in docs:
                if not isinstance(d, dict):
                    continue
                doc_id = d.get("doc_id")
                source_file = d.get("source_file")
                if (
                    isinstance(doc_id, str)
                    and isinstance(source_file, str)
                    and doc_id
                    and source_file
                ):
                    out[doc_id] = source_file
    except Exception:
        out = {}

    _MANIFEST_CACHE["root"] = root
    _MANIFEST_CACHE["mtime"] = st.st_mtime
    _MANIFEST_CACHE["map"] = out
    return out


def _enrich_results_with_source_file(
    cfg: dict[str, Any], results: list[dict[str, Any]]
) -> list[dict[str, Any]]:
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
            expose = (os.environ.get("DOC_RAG_EXPOSE_SOURCE_PATHS") or "").strip() in (
                "1",
                "true",
                "yes",
            )
            r["source_file"] = src if expose else os.path.basename(src)
    return results


def load_manifest_json() -> dict[str, Any] | None:
    """Return the on-disk manifest dict, or None if missing/unreadable."""
    cfg = load_config()
    root = str(cfg.get("_root", project_root()))
    paths = cfg.get("paths", {}) or {}
    manifest_rel = str(paths.get("manifest_path", "build/manifest.json"))
    manifest_path = os.path.join(root, manifest_rel)
    try:
        with open(manifest_path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def indexed_catalog() -> dict[str, Any]:
    """Summarize manifest + artifact presence for UI (indexed documents, search readiness)."""
    cfg = load_config()
    root = str(cfg.get("_root", project_root()))
    paths = cfg.get("paths", {}) or {}
    manifest_rel = str(paths.get("manifest_path", "build/manifest.json"))
    index_dir_rel = str(paths.get("index_dir", "build/index"))
    chunks_dir_rel = str(paths.get("chunks_dir", "build/chunks_jsonl"))

    manifest_path = os.path.join(root, manifest_rel)
    faiss_path = os.path.join(root, index_dir_rel, "faiss.index")
    meta_path = os.path.join(root, index_dir_rel, "index_meta.json")
    chunks_path = os.path.join(root, chunks_dir_rel, "chunks.jsonl")

    chunks_present = os.path.isfile(chunks_path)
    semantic_index_present = os.path.isfile(faiss_path) and os.path.isfile(meta_path)

    documents: list[dict[str, Any]] = []
    generated_at: str | None = None
    manifest_present = False
    corpus_content_sha256: str | None = None
    pipeline_version: str | None = None
    try:
        with open(manifest_path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            manifest_present = True
            gen = data.get("generated_at_utc")
            if gen is not None:
                generated_at = str(gen)
            cs = data.get("corpus_content_sha256")
            if isinstance(cs, str) and cs.strip():
                corpus_content_sha256 = cs.strip()
            pv = data.get("pipeline_version")
            if isinstance(pv, str) and pv.strip():
                pipeline_version = pv.strip()
            raw_docs = data.get("documents")
            if isinstance(raw_docs, list):
                for d in raw_docs:
                    if not isinstance(d, dict):
                        continue
                    sf = d.get("source_file")
                    sf_s = str(sf) if sf is not None else ""
                    cov = d.get("coverage")
                    cov_d: dict[str, Any] = cov if isinstance(cov, dict) else {}
                    th = d.get("title_hint")
                    ey = d.get("edition_year")
                    ey_out: int | None = None
                    if isinstance(ey, int):
                        ey_out = ey
                    elif isinstance(ey, str) and ey.strip().isdigit():
                        try:
                            ey_out = int(ey.strip())
                        except ValueError:
                            ey_out = None
                    documents.append(
                        {
                            "doc_id": d.get("doc_id"),
                            "source_file": sf_s,
                            "basename": os.path.basename(sf_s) if sf_s else "",
                            "chunk_count": d.get("chunk_count"),
                            "sha256": d.get("sha256"),
                            "title_hint": th if isinstance(th, str) else None,
                            "edition_year": ey_out,
                            "coverage": cov_d,
                        }
                    )
    except Exception:
        pass

    documents.sort(
        key=lambda x: ((str(x.get("basename") or "").lower()), str(x.get("doc_id") or ""))
    )

    doc_n = len(documents)
    lexical_ready = chunks_present and doc_n > 0
    semantic_ready = semantic_index_present and chunks_present and doc_n > 0

    return {
        "manifest_present": manifest_present,
        "manifest_path": manifest_rel,
        "manifest_generated_at_utc": generated_at,
        "corpus_content_sha256": corpus_content_sha256,
        "pipeline_version": pipeline_version,
        "document_count": doc_n,
        "documents": documents,
        "chunks_jsonl_present": chunks_present,
        "semantic_index_present": semantic_index_present,
        "lexical_search_ready": lexical_ready,
        "semantic_search_ready": semantic_ready,
    }


def annotation_from_markdown(md: str, *, max_chars: int = 720) -> tuple[str, str]:
    """Derive a short title and plain-text preview from normalized markdown (build/docs_md/*.md)."""
    text = (md or "").strip()
    if not text:
        return ("", "(Пустой документ.)")

    title = ""
    body_text = text
    first_line = text.split("\n", 1)[0].strip()
    m = re.match(r"^#{1,6}\s+(.+)$", first_line)
    if m:
        title = m.group(1).strip()
        body_text = text.split("\n", 1)[1] if "\n" in text else ""

    lines_out: list[str] = []
    for line in body_text.splitlines():
        ls = line.strip()
        if re.match(r"^#{1,6}\s", ls):
            continue
        lines_out.append(line)
    body = "\n".join(lines_out).strip()
    body_one = re.sub(r"\s+", " ", body)
    if len(body_one) > max_chars:
        body_one = body_one[:max_chars].rsplit(" ", 1)[0] + "…"

    if not title:
        for line in body_text.splitlines():
            s = line.strip()
            if s and not s.startswith("#"):
                title = s[:117] + "…" if len(s) > 120 else s
                break
    if not title:
        title = "Документ"

    if not body_one:
        body_one = "(Нет текста для аннотации — только заголовки или пусто.)"
    return title, body_one


def document_preview(doc_id: str) -> dict[str, Any]:
    """Load markdown for a manifest doc_id and return title + preview for UI."""
    doc_id = (doc_id or "").strip()
    if not doc_id:
        return {"ok": False, "error": "empty doc_id"}

    cfg = load_config()
    root = str(cfg.get("_root", project_root()))
    paths = cfg.get("paths", {}) or {}
    manifest_path = os.path.join(root, str(paths.get("manifest_path", "build/manifest.json")))
    docs_md_dir = str(paths.get("docs_md_dir", "build/docs_md"))

    try:
        with open(manifest_path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        return {"ok": False, "error": f"manifest: {exc}"}

    doc_entry: dict[str, Any] | None = None
    docs_list = data.get("documents") if isinstance(data, dict) else None
    if isinstance(docs_list, list):
        for d in docs_list:
            if isinstance(d, dict) and d.get("doc_id") == doc_id:
                doc_entry = d
                break
    if not doc_entry:
        return {"ok": False, "error": "document not in manifest"}

    md_rel = doc_entry.get("md_path")
    if isinstance(md_rel, str) and md_rel.strip():
        md_abs = os.path.join(root, md_rel)
    else:
        md_abs = os.path.join(root, docs_md_dir, f"{doc_id}.md")

    try:
        with open(md_abs, encoding="utf-8") as f:
            raw = f.read(400_000)
    except Exception as exc:
        return {"ok": False, "error": f"read markdown: {exc}"}

    title, preview = annotation_from_markdown(raw)
    src = doc_entry.get("source_file")
    src_s = str(src) if src is not None else ""
    ey = doc_entry.get("edition_year")
    ey_out: int | None = None
    if isinstance(ey, int):
        ey_out = ey
    elif isinstance(ey, str) and ey.strip().isdigit():
        try:
            ey_out = int(ey.strip())
        except ValueError:
            ey_out = None
    return {
        "ok": True,
        "doc_id": doc_id,
        "title": title,
        "preview": preview,
        "source_file": src_s,
        "basename": os.path.basename(src_s) if src_s else "",
        "edition_year": ey_out,
    }


def lexical_search(chunks: list[dict[str, Any]], query: str, top_k: int) -> list[dict[str, Any]]:
    q = (query or "").strip().lower()
    if not q:
        return []
    scored: list[tuple[int, dict[str, Any]]] = []
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


def semantic_search(
    cfg: dict[str, Any],
    chunks: list[dict[str, Any]],
    query: str,
    top_k: int,
    namespace: str = "default",
    filters: dict[str, Any] | None = None,
) -> list[dict[str, Any]] | None:
    # Try VectorIndex first (pluggable backends)
    try:
        from doc_rag.raglib.vector_index import VectorIndex

        root = str(cfg.get("_root", project_root()))
        cache_key = f"{root}:{namespace}"
        if _VECTOR_INDEX_CACHE.get("cfg_root") != cache_key:
            _VECTOR_INDEX_CACHE["cfg_root"] = cache_key
            _VECTOR_INDEX_CACHE["index"] = None

        vi = _VECTOR_INDEX_CACHE.get("index")
        if vi is None:
            vi = VectorIndex(cfg, namespace=namespace)
            _VECTOR_INDEX_CACHE["index"] = vi

        if not vi.is_available():
            return None

        if filters:
            dists, ids, _payloads = vi.search_with_filter(query, top_k, filters)
        else:
            dists, ids = vi.search(query, top_k)
        if not ids:
            return None

        id_to_chunk = {c.get("chunk_id"): c for c in chunks}
        out: list[dict[str, Any]] = []
        for dist, cid in zip(dists, ids, strict=False):
            chunk = id_to_chunk.get(cid)
            if chunk:
                item = dict(chunk)
                item["score"] = float(dist)
                out.append(item)
        return out
    except Exception:
        pass

    # Fallback: legacy FAISS path (backward compatibility)
    root = str(cfg.get("_root", project_root()))
    paths = cfg.get("paths", {}) or {}
    index_dir = paths.get("index_dir", "build/index")
    index_path = os.path.join(root, index_dir, "faiss.index")
    if not os.path.exists(index_path):
        return None

    emb = cfg.get("embeddings", {}) or {}
    model_name = emb.get("model_name")
    if not model_name:
        return None

    try:
        import faiss  # type: ignore
        import numpy as np  # type: ignore
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

    distances, indices = index.search(vec, max(1, int(top_k)))

    out: list[dict[str, Any]] = []
    for rank, idx in enumerate(indices[0]):
        ii = int(idx)
        if 0 <= ii < len(chunks):
            item = dict(chunks[ii])
            item["score"] = float(distances[0][rank])
            out.append(item)
    return out


def _rrf_fusion(
    *ranked_lists: list[dict[str, Any]],
    k: int = 60,
) -> list[dict[str, Any]]:
    """Reciprocal Rank Fusion of multiple ranked lists.

    Each list is ordered best-first. Chunks are identified by chunk_id.
    Final score = sum(1 / (k + rank_i)) across all lists containing
    the chunk. Returns merged list sorted by fused score descending.
    """
    scores: dict[str, float] = {}
    chunks_by_id: dict[str, dict[str, Any]] = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked):
            cid = item.get("chunk_id", "")
            if not cid:
                continue
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
            if cid not in chunks_by_id:
                chunks_by_id[cid] = dict(item)

    merged = []
    for cid, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        item = chunks_by_id[cid]
        item["score"] = round(score, 6)
        merged.append(item)
    return merged


def hybrid_search(
    cfg: dict[str, Any],
    chunks: list[dict[str, Any]],
    query: str,
    top_k: int,
    namespace: str = "default",
    filters: dict[str, Any] | None = None,
    lexical_weight: float = 0.5,
    semantic_weight: float = 0.5,
) -> list[dict[str, Any]]:
    """Hybrid search: combine lexical + semantic via RRF."""
    lex_results = lexical_search(chunks, query, top_k * 2)
    sem_results = semantic_search(
        cfg, chunks, query, top_k * 2, namespace=namespace, filters=filters
    )
    lists: list[list[dict[str, Any]]] = []
    if sem_results is not None:
        lists.append(sem_results)
    if lex_results:
        lists.append(lex_results)
    if not lists:
        return []
    if len(lists) == 1:
        return lists[0][:top_k]
    return _rrf_fusion(*lists)[:top_k]


def doc_search(
    query: str,
    top_k: int,
    namespace: str = "default",
    filters: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    cfg = load_config()
    chunks = load_chunks(cfg)
    top_k = max(1, min(50, int(top_k) if top_k else 6))
    mode = str((cfg.get("mcp", {}) or {}).get("retrieval_mode", "semantic")).lower()
    if mode == "hybrid":
        return _enrich_results_with_source_file(
            cfg, hybrid_search(cfg, chunks, query, top_k, namespace=namespace, filters=filters)
        )
    if mode == "semantic":
        res = semantic_search(cfg, chunks, query, top_k, namespace=namespace, filters=filters)
        if res is not None:
            return _enrich_results_with_source_file(cfg, res)
    return _enrich_results_with_source_file(cfg, lexical_search(chunks, query, top_k))

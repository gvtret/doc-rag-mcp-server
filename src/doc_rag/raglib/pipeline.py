
from __future__ import annotations
import hashlib
import json
import os
import re
import shutil
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

from doc_rag.raglib.audit_log import record_event as _audit

from doc_rag.raglib.edition_year import resolve_edition_year
from doc_rag.raglib.parsers import parse_document
from doc_rag.raglib.utils import ensure_dir, list_files_recursive, safe_slug
from doc_rag.raglib.indexer import ensure_faiss_index


SUPPORTED_EXTS = {".pdf", ".docx", ".doc", ".md", ".txt"}


def load_config(config_path: str) -> Dict[str, Any]:
    cfg = yaml.safe_load(open(config_path, "r", encoding="utf-8"))
    root = os.path.abspath(os.path.join(os.path.dirname(config_path), ".."))
    cfg["_root"] = root
    return cfg


def _hash_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _chunk_text(text: str, target_tokens: int, overlap_tokens: int) -> List[str]:
    if not text.strip():
        return []
    chars_per_token = 4
    window = target_tokens * chars_per_token
    overlap = overlap_tokens * chars_per_token
    step = max(1, window - overlap)

    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + window)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start += step
    return chunks



def _dedup_chunks(
    chunks: List[Dict[str, Any]],
    doc_id_to_source: Dict[str, str],
    threshold: float,
) -> Tuple[List[Dict[str, Any]], int]:
    """Remove near-duplicate chunks that originate from different documents.

    Uses word-bigram Jaccard similarity. When two chunks are near-duplicates,
    keeps the one from the higher-priority source (.pdf > .docx > other).

    Returns (deduped_chunks, n_dropped).
    """
    if threshold <= 0.0 or not chunks:
        return chunks, 0

    def _norm(text: str) -> str:
        t = re.sub(r"[^\w\s]", " ", text.lower())
        return re.sub(r"\s+", " ", t).strip()

    def _bigrams(text: str) -> frozenset:
        words = text.split()
        if not words:
            return frozenset()
        if len(words) < 2:
            return frozenset({(words[0],)})
        return frozenset(zip(words, words[1:]))

    def _jaccard(a: frozenset, b: frozenset) -> float:
        if not a and not b:
            return 1.0
        u = len(a | b)
        return len(a & b) / u if u else 0.0

    def _priority(doc_id: str) -> int:
        """Lower value = kept over near-duplicates."""
        ext = os.path.splitext(doc_id_to_source.get(doc_id, "").lower())[1]
        if ext == ".pdf":
            return 0
        if ext == ".docx":
            return 1
        return 2

    # Process higher-priority chunks first so they are retained over near-dupes
    ordered = sorted(chunks, key=lambda c: _priority(c.get("doc_id", "")))
    bsets = [_bigrams(_norm(c.get("text", ""))) for c in ordered]

    kept_indices: List[int] = []
    kept_bsets: List[frozenset] = []
    kept_doc_ids: List[str] = []

    for i, chunk in enumerate(ordered):
        doc_id_i = chunk.get("doc_id", "")
        bs_i = bsets[i]
        is_dup = False
        for j, bs_j in enumerate(kept_bsets):
            # Only flag as duplicate if from a *different* document
            if kept_doc_ids[j] == doc_id_i:
                continue
            if _jaccard(bs_i, bs_j) >= threshold:
                is_dup = True
                break
        if not is_dup:
            kept_indices.append(i)
            kept_bsets.append(bs_i)
            kept_doc_ids.append(doc_id_i)

    dropped = len(chunks) - len(kept_indices)
    return [ordered[i] for i in kept_indices], dropped


def _log(level: str, msg: str) -> None:
    print(f"[doc-rag][{level}] {msg}", file=os.sys.stderr)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def compute_corpus_fingerprint(documents: List[dict]) -> str:
    """Deterministic fingerprint: SHA-256 over sorted per-document content hashes (manifest `corpus_content_sha256`)."""
    hashes: List[str] = []
    for d in documents:
        if not isinstance(d, dict):
            continue
        h = d.get("sha256")
        if isinstance(h, str) and h.strip():
            hashes.append(h.strip())
    hashes.sort()
    x = hashlib.sha256()
    for h in hashes:
        x.update(h.encode("utf-8"))
        x.update(b"\n")
    return x.hexdigest()


#: Current on-disk schema version for build/manifest.json. Bump only when
#: the layout changes in a way that an older `doc-rag` would misread.
MANIFEST_SCHEMA_VERSION = 1


def _manifest_shell(cfg: Dict[str, Any], documents: List[dict]) -> Dict[str, Any]:
    pv = cfg.get("pipeline_version")
    if not isinstance(pv, str) or not pv.strip():
        pv = "1.4.0"
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_at_utc": _utc_now_iso(),
        "pipeline_version": pv.strip(),
        "corpus_content_sha256": compute_corpus_fingerprint(documents),
        "documents": documents,
    }


class ManifestSchemaTooNew(RuntimeError):
    """Manifest was written by a newer version of doc-rag than this one."""

    def __init__(self, found: int, supported: int) -> None:
        super().__init__(
            f"manifest schema_version={found} is newer than this build supports "
            f"(supported: {supported}). Upgrade doc-rag or run `doc-rag migrate`."
        )
        self.found = found
        self.supported = supported


def _check_manifest_schema(data: Dict[str, Any]) -> None:
    """Raise ManifestSchemaTooNew if the manifest is from a future build.

    Missing `schema_version` is treated as 0, i.e. legacy and still
    readable — we only refuse explicitly higher values.
    """
    raw = data.get("schema_version", 0)
    try:
        found = int(raw)
    except (TypeError, ValueError):
        return
    if found > MANIFEST_SCHEMA_VERSION:
        raise ManifestSchemaTooNew(found=found, supported=MANIFEST_SCHEMA_VERSION)


def _archive_or_dedup_sources(cfg: Dict[str, Any], root: str, processed_sources: List[str]) -> None:
    """Deduplicate/move processed source files from incoming to archived.

    Policy (as requested):
    - If archived contains a file with the same relative name AND sha256 matches:
        delete the incoming file.
    - Else:
        move incoming -> archived.
        If destination exists with different hash, do NOT overwrite; append a suffix.

    Also removes empty directories left behind in sources/incoming.
    """
    incoming_rel = cfg["paths"]["sources_incoming"]
    archived_rel = cfg["paths"].get("sources_archived")
    if not archived_rel:
        _log("WARN", "paths.sources_archived is not set; skipping archiving.")
        return

    incoming_dir = os.path.join(root, incoming_rel)
    archived_dir = os.path.join(root, archived_rel)
    ensure_dir(archived_dir)

    for src in processed_sources:
        rel_from_incoming = os.path.relpath(src, incoming_dir)
        dst = os.path.join(archived_dir, rel_from_incoming)
        ensure_dir(os.path.dirname(dst))

        try:
            src_hash = _hash_file(src)
        except Exception as e:
            _log("ERROR", f"hash failed for incoming '{rel_from_incoming}': {e}")
            continue

        if os.path.exists(dst):
            try:
                dst_hash = _hash_file(dst)
            except Exception as e:
                _log("WARN", f"hash failed for archived '{os.path.relpath(dst, archived_dir)}': {e}; will not overwrite")
                dst_hash = None

            if dst_hash is not None and dst_hash == src_hash:
                # Same content: drop incoming copy
                try:
                    os.remove(src)
                    _log("INFO", f"dedup: incoming '{rel_from_incoming}' removed (same as archived).")
                except Exception as e:
                    _log("ERROR", f"failed to remove incoming '{rel_from_incoming}': {e}")
                continue

            # Different content: keep both, avoid overwrite
            base, ext = os.path.splitext(dst)
            suffix = src_hash[:10]
            dst = f"{base}__{suffix}{ext}"
            _log("WARN", f"name collision: archived has '{rel_from_incoming}' with different hash; moving as '{os.path.relpath(dst, archived_dir)}'.")

        try:
            shutil.move(src, dst)
            _log("INFO", f"archived: '{rel_from_incoming}' -> '{os.path.relpath(dst, archived_dir)}'.")
        except Exception as e:
            _log("ERROR", f"failed to move '{rel_from_incoming}' to archived: {e}")

    # Cleanup empty dirs in incoming
    for dirpath, dirnames, filenames in os.walk(incoming_dir, topdown=False):
        if not dirnames and not filenames:
            try:
                os.rmdir(dirpath)
            except OSError:
                pass


def _ingest_sources(cfg: Dict[str, Any], root: str, sources: List[str], docs_md: str, chunk_target: int, chunk_overlap: int) -> tuple[list[dict], list[dict], list[str]]:
    """Parse sources into MD + chunk list.

    Returns:
    - manifest_documents: list of manifest entries
    - all_chunks: list of chunk entries
    - processed_sources: subset of sources that were successfully processed
    """
    manifest_documents: list[dict] = []
    all_chunks: list[dict] = []
    processed_sources: list[str] = []

    for src in sources:
        rel_src = os.path.relpath(src, root)
        _log("INFO", f"parse: {rel_src}")
        try:
            file_hash = _hash_file(src)
            base = os.path.basename(src)
            doc_id = safe_slug(os.path.splitext(base)[0])[:80] + "_" + file_hash[:10]

            parsed = parse_document(cfg, src)
            md_text = parsed["markdown"]

            md_path = os.path.join(docs_md, f"{doc_id}.md")
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(md_text)

            chunks = _chunk_text(md_text, chunk_target, chunk_overlap)
            for idx, c in enumerate(chunks):
                all_chunks.append({
                    "doc_id": doc_id,
                    "chunk_id": f"{doc_id}:{idx}",
                    "text": c,
                })

            cov = parsed.get("stats") if isinstance(parsed.get("stats"), dict) else {}
            ed_y = resolve_edition_year(cfg, abs_path=src, rel_path=rel_src, sha256_hex=file_hash)
            manifest_documents.append({
                "doc_id": doc_id,
                "source_file": rel_src,
                "md_path": os.path.relpath(md_path, root),
                "sha256": file_hash,
                "chunk_count": len(chunks),
                "title_hint": os.path.splitext(base)[0],
                "edition_year": ed_y,
                "coverage": cov,
            })

            processed_sources.append(src)
            _log("INFO", f"ok: {rel_src} -> {os.path.relpath(md_path, root)} (chunks={len(chunks)})")
        except Exception as e:
            _log("ERROR", f"failed: {rel_src}: {e}")

    return manifest_documents, all_chunks, processed_sources


def _load_existing_manifest(manifest_path: str) -> dict:
    if not os.path.exists(manifest_path):
        return {"generated_at_utc": None, "documents": []}
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {"generated_at_utc": None, "documents": []}
        _check_manifest_schema(data)
        docs = data.get("documents")
        if not isinstance(docs, list):
            docs = []
        return {"generated_at_utc": data.get("generated_at_utc"), "documents": docs}
    except Exception as e:
        _log("WARN", f"failed to read existing manifest '{manifest_path}': {e}; starting fresh")
        return {"generated_at_utc": None, "documents": []}


def ingest(config_path: str) -> None:
    cfg = load_config(config_path)
    root = cfg["_root"]

    incoming_dir = os.path.join(root, cfg["paths"]["sources_incoming"])
    docs_md = os.path.join(root, cfg["paths"]["docs_md_dir"])
    chunks_dir = os.path.join(root, cfg["paths"]["chunks_dir"])
    manifest_path = os.path.join(root, cfg["paths"]["manifest_path"])

    ensure_dir(docs_md)
    ensure_dir(chunks_dir)

    sources = list_files_recursive(incoming_dir, exts=SUPPORTED_EXTS)
    _log("INFO", f"ingest: found {len(sources)} file(s) in {os.path.relpath(incoming_dir, root)}")

    target = int(cfg.get("chunking", {}).get("target_tokens", 512))
    overlap = int(cfg.get("chunking", {}).get("overlap_tokens", 64))

    archive_enabled = bool(cfg.get("sources", {}).get("archive_after_ingest", True))
    incremental = bool(cfg.get("sources", {}).get("incremental_ingest", True))

    chunks_path = os.path.join(chunks_dir, "chunks.jsonl")

    processed_for_archive: list[str] = []

    if not incremental:
        manifest_documents, all_chunks, processed = _ingest_sources(cfg, root, sources, docs_md, target, overlap)

        dedup_thresh = float(cfg.get("chunking", {}).get("dedup_similarity_threshold", 0.0))
        if dedup_thresh > 0.0:
            doc_src_map = {d["doc_id"]: d.get("source_file", "") for d in manifest_documents}
            all_chunks, n_dropped = _dedup_chunks(all_chunks, doc_src_map, dedup_thresh)
            _log("INFO", f"dedup: removed {n_dropped} near-duplicate chunks (threshold={dedup_thresh})")

        manifest = _manifest_shell(cfg, manifest_documents)

        with open(chunks_path, "w", encoding="utf-8") as f:
            for it in all_chunks:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        _log("INFO", f"written: {os.path.relpath(chunks_path, root)} (chunks={len(all_chunks)})")
        _log("INFO", f"written: {os.path.relpath(manifest_path, root)} (docs={len(manifest_documents)})")

        processed_for_archive = processed
    else:
        existing = _load_existing_manifest(manifest_path)
        documents: list[dict] = existing.get("documents", [])
        existing_hashes = set()
        existing_doc_ids = set()
        for d in documents:
            if isinstance(d, dict):
                h = d.get("sha256")
                if isinstance(h, str):
                    existing_hashes.add(h)
                did = d.get("doc_id")
                if isinstance(did, str):
                    existing_doc_ids.add(did)

        new_docs: list[dict] = []
        appended_chunks = 0
        skipped = 0
        failed = 0

        # Append-only chunks file
        ensure_dir(os.path.dirname(chunks_path))
        with open(chunks_path, "a", encoding="utf-8") as chunks_f:
            for src in sources:
                rel_src = os.path.relpath(src, root)
                try:
                    file_hash = _hash_file(src)
                except Exception as e:
                    failed += 1
                    _log("ERROR", f"hash failed: {rel_src}: {e}")
                    continue

                if file_hash in existing_hashes:
                    skipped += 1
                    processed_for_archive.append(src)
                    _log("INFO", f"skip: {rel_src} (already in manifest by sha256)")
                    continue

                _log("INFO", f"parse: {rel_src}")
                try:
                    base = os.path.basename(src)
                    doc_id = safe_slug(os.path.splitext(base)[0])[:80] + "_" + file_hash[:10]
                    if doc_id in existing_doc_ids:
                        # Extremely unlikely, but avoid duplicates
                        skipped += 1
                        processed_for_archive.append(src)
                        _log("WARN", f"skip: {rel_src} (doc_id collision: {doc_id})")
                        continue

                    parsed = parse_document(cfg, src)
                    md_text = parsed["markdown"]

                    md_path = os.path.join(docs_md, f"{doc_id}.md")
                    with open(md_path, "w", encoding="utf-8") as f:
                        f.write(md_text)

                    chunks = _chunk_text(md_text, target, overlap)
                    for idx, c in enumerate(chunks):
                        chunks_f.write(json.dumps({
                            "doc_id": doc_id,
                            "chunk_id": f"{doc_id}:{idx}",
                            "text": c,
                        }, ensure_ascii=False) + "\n")
                    appended_chunks += len(chunks)

                    cov = parsed.get("stats") if isinstance(parsed.get("stats"), dict) else {}
                    ed_y = resolve_edition_year(cfg, abs_path=src, rel_path=rel_src, sha256_hex=file_hash)
                    entry = {
                        "doc_id": doc_id,
                        "source_file": rel_src,
                        "md_path": os.path.relpath(md_path, root),
                        "sha256": file_hash,
                        "chunk_count": len(chunks),
                        "title_hint": os.path.splitext(base)[0],
                        "edition_year": ed_y,
                        "coverage": cov,
                    }
                    new_docs.append(entry)

                    existing_hashes.add(file_hash)
                    existing_doc_ids.add(doc_id)
                    processed_for_archive.append(src)

                    _log("INFO", f"ok: {rel_src} -> {os.path.relpath(md_path, root)} (chunks={len(chunks)})")
                except Exception as e:
                    failed += 1
                    _log("ERROR", f"failed: {rel_src}: {e}")

        if new_docs:
            documents.extend(new_docs)

        manifest = _manifest_shell(cfg, documents)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        _log("INFO", f"updated: {os.path.relpath(chunks_path, root)} (appended_chunks={appended_chunks}, total_docs={len(documents)})")
        _log("INFO", f"updated: {os.path.relpath(manifest_path, root)} (new_docs={len(new_docs)}, skipped={skipped}, failed={failed})")

    # Best-effort semantic index update. If FAISS/embeddings are missing, we keep lexical search.
    try:
        ensure_faiss_index(cfg, force_rebuild=False, log=print)
    except Exception as e:
        _log("WARN", f"index update skipped: {e}")

    # Archive/dedup processed sources
    if archive_enabled and processed_for_archive:
        _archive_or_dedup_sources(cfg, root, processed_for_archive)
    elif not archive_enabled:
        _log("INFO", "archiving disabled (sources.archive_after_ingest=false).")
def rebuild(config_path: str) -> None:
    """Full rebuild from archived first, then ingest incoming."""
    cfg = load_config(config_path)
    root = cfg["_root"]

    archived_rel = cfg["paths"].get("sources_archived")
    if not archived_rel:
        _log("WARN", "paths.sources_archived is not set; rebuild will behave like ingest().")
        ingest(config_path)
        return

    archived_dir = os.path.join(root, archived_rel)
    incoming_dir = os.path.join(root, cfg["paths"]["sources_incoming"])
    docs_md = os.path.join(root, cfg["paths"]["docs_md_dir"])
    chunks_dir = os.path.join(root, cfg["paths"]["chunks_dir"])
    manifest_path = os.path.join(root, cfg["paths"]["manifest_path"])

    ensure_dir(docs_md)
    ensure_dir(chunks_dir)

    # Wipe previous generated outputs (but keep sources)
    for p in [docs_md, chunks_dir]:
        if os.path.isdir(p):
            shutil.rmtree(p)
        ensure_dir(p)

    target = int(cfg.get("chunking", {}).get("target_tokens", 512))
    overlap = int(cfg.get("chunking", {}).get("overlap_tokens", 64))

    archived_sources = list_files_recursive(archived_dir, exts=SUPPORTED_EXTS)
    _log("INFO", f"rebuild: archived pass ({len(archived_sources)} file(s))")
    man_a, chunks_a, _ = _ingest_sources(cfg, root, archived_sources, docs_md, target, overlap)

    incoming_sources = list_files_recursive(incoming_dir, exts=SUPPORTED_EXTS)
    _log("INFO", f"rebuild: incoming ingest pass ({len(incoming_sources)} file(s))")
    man_i, chunks_i, processed_i = _ingest_sources(cfg, root, incoming_sources, docs_md, target, overlap)

    all_docs = man_a + man_i
    all_chunks = chunks_a + chunks_i

    dedup_thresh = float(cfg.get("chunking", {}).get("dedup_similarity_threshold", 0.0))
    if dedup_thresh > 0.0:
        doc_src_map = {d["doc_id"]: d.get("source_file", "") for d in all_docs}
        all_chunks, n_dropped = _dedup_chunks(all_chunks, doc_src_map, dedup_thresh)
        _log("INFO", f"dedup: removed {n_dropped} near-duplicate chunks (threshold={dedup_thresh})")

    manifest = _manifest_shell(cfg, all_docs)

    chunks_path = os.path.join(chunks_dir, "chunks.jsonl")
    with open(chunks_path, "w", encoding="utf-8") as f:
        for it in all_chunks:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    _log("INFO", f"written: {os.path.relpath(chunks_path, root)} (chunks={len(all_chunks)})")
    _log("INFO", f"written: {os.path.relpath(manifest_path, root)} (docs={len(manifest['documents'])})")

    archive_enabled = bool(cfg.get("sources", {}).get("archive_after_ingest", True))
    if archive_enabled and processed_i:
        _archive_or_dedup_sources(cfg, root, processed_i)
    elif not archive_enabled:
        _log("INFO", "archiving disabled (sources.archive_after_ingest=false).")

    # Delete stale index files before encoding so that an interrupted rebuild
    # (SIGHUP, OOM, etc.) doesn't leave index_meta.json mismatched against
    # the freshly written chunks.jsonl — next ingest/rebuild will start clean.
    _index_dir = os.path.join(root, cfg["paths"]["index_dir"])
    for _fname in ("faiss.index", "index_meta.json"):
        _p = os.path.join(_index_dir, _fname)
        if os.path.exists(_p):
            try:
                os.remove(_p)
            except OSError as _e:
                _log("WARN", f"could not remove stale index file {_fname}: {_e}")

    # Force rebuild index so semantic search matches rebuilt chunks (best-effort)
    try:
        ensure_faiss_index(cfg, force_rebuild=True, log=print)
    except Exception as e:
        _log("WARN", f"index rebuild skipped: {e}")


def _rebuild_faiss_after_delete(cfg: Dict[str, Any], deleted_doc_ids: set) -> Dict[str, Any]:
    """Rebuild FAISS index by reconstructing kept vectors (no re-encoding).

    Reads the old index + meta, drops vectors whose chunk_id belongs to a
    deleted doc, and writes a fresh index with the surviving vectors. If
    the index/meta files are missing it silently returns.
    """
    root = cfg["_root"]
    index_dir = os.path.join(root, cfg["paths"]["index_dir"])
    index_file = os.path.join(index_dir, "faiss.index")
    meta_file = os.path.join(index_dir, "index_meta.json")
    stats = {"removed_vectors": 0, "kept_vectors": 0, "had_index": False}

    if not (os.path.exists(index_file) and os.path.exists(meta_file)):
        return stats
    stats["had_index"] = True

    try:
        import faiss  # type: ignore
    except Exception as e:
        _log("WARN", f"faiss unavailable; cannot prune index: {e}")
        return stats

    try:
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception as e:
        _log("WARN", f"failed to read index_meta.json: {e}")
        return stats

    old_chunk_ids: List[str] = list(meta.get("chunk_ids") or [])
    kept_positions: List[int] = []
    kept_chunk_ids: List[str] = []
    for i, cid in enumerate(old_chunk_ids):
        doc_id = cid.rsplit(":", 1)[0] if ":" in cid else cid
        if doc_id in deleted_doc_ids:
            continue
        kept_positions.append(i)
        kept_chunk_ids.append(cid)

    stats["removed_vectors"] = len(old_chunk_ids) - len(kept_positions)
    stats["kept_vectors"] = len(kept_positions)

    if stats["removed_vectors"] == 0:
        return stats

    if not kept_positions:
        for p in (index_file, meta_file):
            try:
                os.remove(p)
            except OSError:
                pass
        return stats

    try:
        old_index = faiss.read_index(index_file)
        dim = int(getattr(old_index, "d", 0)) or int(meta.get("dim", 0))
        if dim <= 0:
            _log("WARN", "index has unknown dimension; skipping prune")
            return stats
        metric = str(meta.get("metric", "ip")).lower()
        new_index = faiss.IndexFlatIP(dim) if metric == "ip" else faiss.IndexFlatL2(dim)
        vecs = np.zeros((len(kept_positions), dim), dtype=np.float32)
        for new_i, old_pos in enumerate(kept_positions):
            vecs[new_i] = old_index.reconstruct(int(old_pos))
        new_index.add(vecs)
        faiss.write_index(new_index, index_file)
        meta["chunk_ids"] = kept_chunk_ids
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    except Exception as e:
        _log("WARN", f"FAISS prune failed, removing index for clean rebuild: {e}")
        for p in (index_file, meta_file):
            try:
                os.remove(p)
            except OSError:
                pass
    return stats


def delete_documents(config_path: str, doc_ids: List[str]) -> Dict[str, Any]:
    """Remove documents from the index by doc_id.

    Side effects:
    - Removes archived source file(s)
    - Deletes build/docs_md/<doc_id>.md
    - Rewrites chunks.jsonl, manifest.json without the deleted entries
    - Prunes the FAISS index in-place by reconstructing kept vectors
    """
    cfg = load_config(config_path)
    root = cfg["_root"]
    target = {d for d in doc_ids if d}
    if not target:
        return {"requested": 0, "deleted": 0, "missing": 0}

    manifest_path = os.path.join(root, cfg["paths"]["manifest_path"])
    chunks_path = os.path.join(root, cfg["paths"]["chunks_dir"], "chunks.jsonl")

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception as e:
        raise RuntimeError(f"cannot read manifest: {e}") from e

    if isinstance(manifest, dict):
        _check_manifest_schema(manifest)
    docs = manifest.get("documents") if isinstance(manifest, dict) else None
    if not isinstance(docs, list):
        raise RuntimeError("manifest.documents is malformed")

    to_delete = [d for d in docs if isinstance(d, dict) and d.get("doc_id") in target]
    to_keep = [d for d in docs if isinstance(d, dict) and d.get("doc_id") not in target]
    found_ids = {d["doc_id"] for d in to_delete if d.get("doc_id")}
    missing = target - found_ids

    for d in to_delete:
        sf_rel = d.get("source_file") or ""
        if sf_rel:
            # The manifest may point to sources/incoming/ even after the file was
            # moved to sources/archived/. Try the recorded path, then both common
            # roots with the same basename.
            candidates = {sf_rel}
            bn = os.path.basename(sf_rel)
            for dir_key in ("sources_archived", "sources_incoming"):
                d_rel = cfg["paths"].get(dir_key)
                if d_rel:
                    candidates.add(os.path.join(d_rel, bn))
            for cand in candidates:
                cand_abs = os.path.join(root, cand)
                if os.path.isfile(cand_abs):
                    try:
                        os.remove(cand_abs)
                    except OSError as e:
                        _log("WARN", f"failed to remove source '{cand}': {e}")
        md_rel = d.get("md_path") or ""
        if md_rel:
            md_abs = os.path.join(root, md_rel)
            if os.path.isfile(md_abs):
                try:
                    os.remove(md_abs)
                except OSError as e:
                    _log("WARN", f"failed to remove md '{md_rel}': {e}")

    removed_chunks = 0
    if os.path.exists(chunks_path):
        tmp = chunks_path + ".tmp"
        with open(chunks_path, "r", encoding="utf-8") as src, open(tmp, "w", encoding="utf-8") as dst:
            for line in src:
                s = line.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                except Exception:
                    continue
                if obj.get("doc_id") in target:
                    removed_chunks += 1
                    continue
                dst.write(json.dumps(obj, ensure_ascii=False) + "\n")
        os.replace(tmp, chunks_path)

    index_stats = _rebuild_faiss_after_delete(cfg, target)

    manifest["documents"] = to_keep
    manifest["generated_at_utc"] = _utc_now_iso()
    manifest["corpus_content_sha256"] = compute_corpus_fingerprint(to_keep)
    manifest["schema_version"] = MANIFEST_SCHEMA_VERSION
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    deleted_names = [os.path.basename(d.get("source_file") or d.get("doc_id") or "") for d in to_delete]
    _log("INFO", f"delete: removed {len(to_delete)} doc(s), {removed_chunks} chunk(s), {index_stats['removed_vectors']} vector(s)")
    _audit(
        root,
        "delete",
        doc_ids=sorted(found_ids),
        counts={
            "requested": len(target),
            "deleted": len(to_delete),
            "missing": len(missing),
            "removed_chunks": removed_chunks,
            "removed_vectors": index_stats.get("removed_vectors", 0),
        },
    )
    return {
        "requested": len(target),
        "deleted": len(to_delete),
        "missing": sorted(missing),
        "removed_chunks": removed_chunks,
        "index": index_stats,
        "deleted_names": deleted_names,
    }


def wipe_index(config_path: str) -> Dict[str, Any]:
    """Delete everything: sources/archived/*, build/*, manifest.json, FAISS index.

    sources/incoming is left alone (user-controlled inbox).
    """
    cfg = load_config(config_path)
    root = cfg["_root"]
    paths = cfg["paths"]
    targets = [
        os.path.join(root, paths["sources_archived"]),
        os.path.join(root, paths["docs_md_dir"]),
        os.path.join(root, paths["chunks_dir"]),
        os.path.join(root, paths["index_dir"]),
    ]
    removed = 0
    for d in targets:
        if not os.path.isdir(d):
            continue
        for entry in os.listdir(d):
            p = os.path.join(d, entry)
            try:
                if os.path.isfile(p) or os.path.islink(p):
                    os.remove(p)
                    removed += 1
                elif os.path.isdir(p):
                    shutil.rmtree(p)
                    removed += 1
            except OSError as e:
                _log("WARN", f"wipe: could not remove '{p}': {e}")

    manifest_path = os.path.join(root, paths["manifest_path"])
    if os.path.isfile(manifest_path):
        try:
            os.remove(manifest_path)
            removed += 1
        except OSError as e:
            _log("WARN", f"wipe: could not remove manifest: {e}")

    _log("INFO", f"wipe: removed {removed} entries (sources/archived, build/, manifest)")
    _audit(root, "wipe", counts={"removed_entries": removed})
    return {"removed_entries": removed}


def clean_orphans(config_path: str) -> Dict[str, Any]:
    """Remove artifacts not referenced by the manifest.

    - md files in build/docs_md/ whose doc_id is absent from manifest
    - chunks.jsonl lines whose doc_id is absent from manifest
    - rebuilds FAISS index in-place to drop orphan vectors
    """
    cfg = load_config(config_path)
    root = cfg["_root"]
    paths = cfg["paths"]

    manifest_path = os.path.join(root, paths["manifest_path"])
    known_ids: set = set()
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for d in data.get("documents", []):
            did = d.get("doc_id") if isinstance(d, dict) else None
            if isinstance(did, str):
                known_ids.add(did)
    except Exception:
        pass

    orphan_md = 0
    docs_md = os.path.join(root, paths["docs_md_dir"])
    if os.path.isdir(docs_md):
        for fn in os.listdir(docs_md):
            if not fn.endswith(".md"):
                continue
            doc_id = fn[:-3]
            if doc_id not in known_ids:
                try:
                    os.remove(os.path.join(docs_md, fn))
                    orphan_md += 1
                except OSError as e:
                    _log("WARN", f"clean_orphans: cannot remove {fn}: {e}")

    chunks_path = os.path.join(root, paths["chunks_dir"], "chunks.jsonl")
    orphan_chunks = 0
    orphan_doc_ids: set = set()
    if os.path.exists(chunks_path):
        tmp = chunks_path + ".tmp"
        with open(chunks_path, "r", encoding="utf-8") as src, open(tmp, "w", encoding="utf-8") as dst:
            for line in src:
                s = line.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                except Exception:
                    continue
                did = obj.get("doc_id")
                if did not in known_ids:
                    orphan_chunks += 1
                    if isinstance(did, str):
                        orphan_doc_ids.add(did)
                    continue
                dst.write(json.dumps(obj, ensure_ascii=False) + "\n")
        os.replace(tmp, chunks_path)

    index_stats = _rebuild_faiss_after_delete(cfg, orphan_doc_ids) if orphan_doc_ids else {"removed_vectors": 0, "kept_vectors": 0, "had_index": False}

    _log("INFO", f"clean_orphans: removed {orphan_md} md, {orphan_chunks} chunk(s), {index_stats['removed_vectors']} vector(s)")
    _audit(
        root,
        "clean_orphans",
        doc_ids=sorted(orphan_doc_ids) if orphan_doc_ids else None,
        counts={
            "orphan_md_removed": orphan_md,
            "orphan_chunks_removed": orphan_chunks,
            "removed_vectors": index_stats.get("removed_vectors", 0),
        },
    )
    return {
        "orphan_md_removed": orphan_md,
        "orphan_chunks_removed": orphan_chunks,
        "orphan_doc_ids": sorted(orphan_doc_ids),
        "index": index_stats,
    }


def clear_incoming(config_path: str) -> Dict[str, Any]:
    """Delete all files inside sources/incoming/ (does not touch the index)."""
    cfg = load_config(config_path)
    root = cfg["_root"]
    incoming = os.path.join(root, cfg["paths"]["sources_incoming"])
    removed = 0
    if os.path.isdir(incoming):
        for entry in os.listdir(incoming):
            p = os.path.join(incoming, entry)
            try:
                if os.path.isfile(p) or os.path.islink(p):
                    os.remove(p)
                    removed += 1
                elif os.path.isdir(p):
                    shutil.rmtree(p)
                    removed += 1
            except OSError as e:
                _log("WARN", f"clear_incoming: cannot remove '{entry}': {e}")
    _log("INFO", f"clear_incoming: removed {removed} entries")
    _audit(root, "clear_incoming", counts={"removed": removed})
    return {"removed": removed}

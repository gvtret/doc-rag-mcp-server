
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

from doc_rag.raglib.edition_year import resolve_edition_year
from doc_rag.raglib.parsers import parse_document
from doc_rag.raglib.utils import ensure_dir, list_files_recursive, safe_slug
from doc_rag.raglib.indexer import ensure_faiss_index


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


def _manifest_shell(cfg: Dict[str, Any], documents: List[dict]) -> Dict[str, Any]:
    pv = cfg.get("pipeline_version")
    if not isinstance(pv, str) or not pv.strip():
        pv = "1.0.0"
    return {
        "generated_at_utc": _utc_now_iso(),
        "pipeline_version": pv.strip(),
        "corpus_content_sha256": compute_corpus_fingerprint(documents),
        "documents": documents,
    }


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

    sources = list_files_recursive(incoming_dir, exts={".pdf", ".docx"})
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

    archived_sources = list_files_recursive(archived_dir, exts={".pdf", ".docx"})
    _log("INFO", f"rebuild: archived pass ({len(archived_sources)} file(s))")
    man_a, chunks_a, _ = _ingest_sources(cfg, root, archived_sources, docs_md, target, overlap)

    incoming_sources = list_files_recursive(incoming_dir, exts={".pdf", ".docx"})
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

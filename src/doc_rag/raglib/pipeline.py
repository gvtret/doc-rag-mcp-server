from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from datetime import datetime, timezone
from typing import Any

import yaml

from doc_rag.raglib.audit_log import record_event as _audit
from doc_rag.raglib.blocks import BLOCKS_SCHEMA_VERSION, Block, dump_blocks
from doc_rag.raglib.edition_year import resolve_edition_year
from doc_rag.raglib.parsers import parse_document
from doc_rag.raglib.utils import ensure_dir, list_files_recursive, safe_slug

# VectorIndex cache for the pipeline
_VECTOR_INDEX: Any = None


def _get_vector_index(cfg: dict[str, Any]) -> Any:
    """Get or create a VectorIndex instance (per-root cache)."""
    global _VECTOR_INDEX
    root = cfg.get("_root", "")
    if _VECTOR_INDEX is not None and getattr(_VECTOR_INDEX, "_cfg", {}).get("_root") != root:
        _VECTOR_INDEX = None
    if _VECTOR_INDEX is None:
        from doc_rag.raglib.vector_index import VectorIndex

        _VECTOR_INDEX = VectorIndex(cfg)
    return _VECTOR_INDEX


SUPPORTED_EXTS = {".pdf", ".docx", ".doc", ".md", ".txt"}


def load_config(config_path: str) -> dict[str, Any]:
    cfg = yaml.safe_load(open(config_path, encoding="utf-8"))
    root = os.path.abspath(os.path.join(os.path.dirname(config_path), ".."))
    cfg["_root"] = root
    return cfg


def _hash_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _chunk_text(text: str, target_tokens: int, overlap_tokens: int) -> list[str]:
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
    chunks: list[dict[str, Any]],
    doc_id_to_source: dict[str, str],
    threshold: float,
) -> tuple[list[dict[str, Any]], int]:
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
        return frozenset(zip(words, words[1:], strict=False))

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

    kept_indices: list[int] = []
    kept_bsets: list[frozenset] = []
    kept_doc_ids: list[str] = []

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


def compute_corpus_fingerprint(documents: list[dict]) -> str:
    """Deterministic fingerprint: SHA-256 over sorted per-document content hashes (manifest `corpus_content_sha256`)."""
    hashes: list[str] = []
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


def _manifest_shell(cfg: dict[str, Any], documents: list[dict]) -> dict[str, Any]:
    pv = cfg.get("pipeline_version")
    if not isinstance(pv, str) or not pv.strip():
        pv = "1.4.0"
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "blocks_schema_version": BLOCKS_SCHEMA_VERSION,
        "generated_at_utc": _utc_now_iso(),
        "pipeline_version": pv.strip(),
        "corpus_content_sha256": compute_corpus_fingerprint(documents),
        "documents": documents,
    }


def _blocks_dir(cfg: dict[str, Any]) -> str:
    """Absolute path to `build/blocks/` (default), where one
    `<doc_id>.jsonl` is written per document.
    """
    root = cfg["_root"]
    rel = cfg.get("paths", {}).get("blocks_dir", "build/blocks")
    return os.path.join(root, rel)


def _persist_blocks(blocks_dir: str, root: str, doc_id: str, blocks: list[Block]) -> str:
    """Save blocks for one document and return the manifest-style relative
    path (`build/blocks/<doc_id>.jsonl`).

    Block IDs coming from `parse_document` use the placeholder `tmp:NNNN`
    prefix because the parser does not yet know the eventual `doc_id`.
    Here we rewrite them in place to the canonical `<doc_id>:NNNN` form
    so the saved file is self-consistent.
    """
    os.makedirs(blocks_dir, exist_ok=True)
    out_path = os.path.join(blocks_dir, f"{doc_id}.jsonl")
    for i, b in enumerate(blocks):
        b.block_id = f"{doc_id}:{i:04d}"
        b.doc_id = doc_id
    dump_blocks(out_path, blocks)
    return os.path.relpath(out_path, root)


class ManifestSchemaTooNew(RuntimeError):
    """Manifest was written by a newer version of doc-rag than this one."""

    def __init__(self, found: int, supported: int) -> None:
        super().__init__(
            f"manifest schema_version={found} is newer than this build supports "
            f"(supported: {supported}). Upgrade doc-rag or run `doc-rag migrate`."
        )
        self.found = found
        self.supported = supported


def _check_manifest_schema(data: dict[str, Any]) -> None:
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


def _archive_or_dedup_sources(cfg: dict[str, Any], root: str, processed_sources: list[str]) -> None:
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
                _log(
                    "WARN",
                    f"hash failed for archived '{os.path.relpath(dst, archived_dir)}': {e}; will not overwrite",
                )
                dst_hash = None

            if dst_hash is not None and dst_hash == src_hash:
                # Same content: drop incoming copy
                try:
                    os.remove(src)
                    _log(
                        "INFO", f"dedup: incoming '{rel_from_incoming}' removed (same as archived)."
                    )
                except Exception as e:
                    _log("ERROR", f"failed to remove incoming '{rel_from_incoming}': {e}")
                continue

            # Different content: keep both, avoid overwrite
            base, ext = os.path.splitext(dst)
            suffix = src_hash[:10]
            dst = f"{base}__{suffix}{ext}"
            _log(
                "WARN",
                f"name collision: archived has '{rel_from_incoming}' with different hash; moving as '{os.path.relpath(dst, archived_dir)}'.",
            )

        try:
            shutil.move(src, dst)
            _log(
                "INFO",
                f"archived: '{rel_from_incoming}' -> '{os.path.relpath(dst, archived_dir)}'.",
            )
        except Exception as e:
            _log("ERROR", f"failed to move '{rel_from_incoming}' to archived: {e}")

    # Cleanup empty dirs in incoming
    for dirpath, dirnames, filenames in os.walk(incoming_dir, topdown=False):
        if not dirnames and not filenames:
            try:
                os.rmdir(dirpath)
            except OSError:
                pass


def _ingest_sources(
    cfg: dict[str, Any],
    root: str,
    sources: list[str],
    docs_md: str,
    chunk_target: int,
    chunk_overlap: int,
) -> tuple[list[dict], list[dict], list[str]]:
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
            blocks = parsed.get("blocks") or []

            md_path = os.path.join(docs_md, f"{doc_id}.md")
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(md_text)

            blocks_rel = _persist_blocks(_blocks_dir(cfg), root, doc_id, blocks)

            quality_report = None
            if blocks:
                from doc_rag.raglib.quality import persist_quality_report, run_quality_checks

                quality_dir = os.path.join(root, "build", "quality")
                quality_report = run_quality_checks(doc_id, blocks)
                persist_quality_report(quality_dir, quality_report)

            chunking_strategy = (cfg.get("chunking") or {}).get("strategy", "fixed")
            if chunking_strategy == "recursive" and blocks:
                from doc_rag.raglib.chunker_recursive import chunk_recursive

                rc_chunks = chunk_recursive(blocks, target_tokens=chunk_target)
                for idx, c in enumerate(rc_chunks):
                    entry: dict[str, Any] = {
                        "doc_id": doc_id,
                        "chunk_id": f"{doc_id}:{idx}",
                        "text": c["text"],
                    }
                    if c.get("section_path"):
                        entry["section_path"] = c["section_path"]
                    if c.get("is_table"):
                        entry["is_table"] = True
                    if c.get("is_table_summary"):
                        entry["is_table_summary"] = True
                    all_chunks.append(entry)
                chunks = [c["text"] for c in rc_chunks]
            else:
                chunks = _chunk_text(md_text, chunk_target, chunk_overlap)
                for idx, c in enumerate(chunks):
                    all_chunks.append(
                        {
                            "doc_id": doc_id,
                            "chunk_id": f"{doc_id}:{idx}",
                            "text": c,
                        }
                    )

            cov = parsed.get("stats") if isinstance(parsed.get("stats"), dict) else {}
            ed_y = resolve_edition_year(cfg, abs_path=src, rel_path=rel_src, sha256_hex=file_hash)
            manifest_documents.append(
                {
                    "doc_id": doc_id,
                    "source_file": rel_src,
                    "md_path": os.path.relpath(md_path, root),
                    "blocks_path": blocks_rel,
                    "sha256": file_hash,
                    "chunk_count": len(chunks),
                    "title_hint": os.path.splitext(base)[0],
                    "edition_year": ed_y,
                    "coverage": cov,
                    **(
                        {
                            "quality_score": quality_report.score,
                            "quality_warning_count": len(quality_report.warnings),
                        }
                        if quality_report is not None
                        else {}
                    ),
                }
            )

            processed_sources.append(src)
            _log(
                "INFO", f"ok: {rel_src} -> {os.path.relpath(md_path, root)} (chunks={len(chunks)})"
            )
        except Exception as e:
            _log("ERROR", f"failed: {rel_src}: {e}")

    return manifest_documents, all_chunks, processed_sources


def _load_existing_manifest(manifest_path: str) -> dict:
    if not os.path.exists(manifest_path):
        return {"generated_at_utc": None, "documents": []}
    try:
        with open(manifest_path, encoding="utf-8") as f:
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
        manifest_documents, all_chunks, processed = _ingest_sources(
            cfg, root, sources, docs_md, target, overlap
        )

        dedup_thresh = float(cfg.get("chunking", {}).get("dedup_similarity_threshold", 0.0))
        if dedup_thresh > 0.0:
            doc_src_map = {d["doc_id"]: d.get("source_file", "") for d in manifest_documents}
            all_chunks, n_dropped = _dedup_chunks(all_chunks, doc_src_map, dedup_thresh)
            _log(
                "INFO",
                f"dedup: removed {n_dropped} near-duplicate chunks (threshold={dedup_thresh})",
            )

        manifest = _manifest_shell(cfg, manifest_documents)

        with open(chunks_path, "w", encoding="utf-8") as f:
            for it in all_chunks:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        _log("INFO", f"written: {os.path.relpath(chunks_path, root)} (chunks={len(all_chunks)})")
        _log(
            "INFO",
            f"written: {os.path.relpath(manifest_path, root)} (docs={len(manifest_documents)})",
        )

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
                    blocks = parsed.get("blocks") or []

                    md_path = os.path.join(docs_md, f"{doc_id}.md")
                    with open(md_path, "w", encoding="utf-8") as f:
                        f.write(md_text)

                    blocks_rel = _persist_blocks(_blocks_dir(cfg), root, doc_id, blocks)

                    chunks = _chunk_text(md_text, target, overlap)
                    for idx, c in enumerate(chunks):
                        chunks_f.write(
                            json.dumps(
                                {
                                    "doc_id": doc_id,
                                    "chunk_id": f"{doc_id}:{idx}",
                                    "text": c,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                    appended_chunks += len(chunks)

                    # Per-file index update
                    try:
                        vi = _get_vector_index(cfg)
                        file_chunk_dicts = [
                            {"doc_id": doc_id, "chunk_id": f"{doc_id}:{idx}", "text": c}
                            for idx, c in enumerate(chunks)
                        ]
                        vi.build(file_chunk_dicts, force=False, log=print)
                    except Exception:
                        pass

                    cov = parsed.get("stats") if isinstance(parsed.get("stats"), dict) else {}
                    ed_y = resolve_edition_year(
                        cfg, abs_path=src, rel_path=rel_src, sha256_hex=file_hash
                    )
                    entry = {
                        "doc_id": doc_id,
                        "source_file": rel_src,
                        "md_path": os.path.relpath(md_path, root),
                        "blocks_path": blocks_rel,
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

                    _log(
                        "INFO",
                        f"ok: {rel_src} -> {os.path.relpath(md_path, root)} (chunks={len(chunks)})",
                    )
                except Exception as e:
                    failed += 1
                    _log("ERROR", f"failed: {rel_src}: {e}")

        if new_docs:
            documents.extend(new_docs)

        manifest = _manifest_shell(cfg, documents)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        _log(
            "INFO",
            f"updated: {os.path.relpath(chunks_path, root)} (appended_chunks={appended_chunks}, total_docs={len(documents)})",
        )
        _log(
            "INFO",
            f"updated: {os.path.relpath(manifest_path, root)} (new_docs={len(new_docs)}, skipped={skipped}, failed={failed})",
        )

    # Best-effort semantic index update. If embeddings are missing, we keep lexical search.
    try:
        vi = _get_vector_index(cfg)
        # Load chunks and build/update index
        chunks_path = os.path.join(root, cfg["paths"]["chunks_dir"], "chunks.jsonl")
        if os.path.exists(chunks_path):
            chunks = []
            with open(chunks_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        chunks.append(json.loads(line))
            if chunks:
                vi.build(chunks, force=False, log=print)
    except Exception as e:
        _log("WARN", f"index update skipped: {e}")

    # Archive/dedup processed sources
    if archive_enabled and processed_for_archive:
        _archive_or_dedup_sources(cfg, root, processed_for_archive)
    elif not archive_enabled:
        _log("INFO", "archiving disabled (sources.archive_after_ingest=false).")


def _load_existing_chunks(chunks_path: str) -> list[dict[str, Any]]:
    """Load existing chunks from chunks.jsonl."""
    chunks: list[dict[str, Any]] = []
    if not os.path.exists(chunks_path):
        return chunks
    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def _write_chunks(chunks_path: str, chunks: list[dict[str, Any]]) -> None:
    with open(chunks_path, "w", encoding="utf-8") as f:
        for it in chunks:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def _append_chunks(chunks_path: str, chunks: list[dict[str, Any]]) -> None:
    """Append chunks to chunks.jsonl (create if missing)."""
    ensure_dir(os.path.dirname(chunks_path))
    with open(chunks_path, "a", encoding="utf-8") as f:
        for it in chunks:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def rebuild(config_path: str, *, force: bool = False) -> None:
    """Rebuild chunks/embeddings/index.

    By default (incremental), skips files already present in the manifest
    whose .md file still exists on disk — so an interrupted rebuild
    resumes where it left off.  Pass ``force=True`` (or CLI ``--force``)
    for a full wipe-and-reparse.
    """
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

    existing = _load_existing_manifest(manifest_path)
    existing_docs: list[dict] = existing.get("documents", [])
    existing_by_hash: dict[str, dict] = {}
    for d in existing_docs:
        if isinstance(d, dict) and isinstance(d.get("sha256"), str):
            existing_by_hash[d["sha256"]] = d

    archived_sources = list_files_recursive(archived_dir, exts=SUPPORTED_EXTS)
    incoming_sources = list_files_recursive(incoming_dir, exts=SUPPORTED_EXTS)
    all_sources = archived_sources + incoming_sources

    # Build hash → source mapping
    source_by_hash: dict[str, str] = {}
    for src in all_sources:
        try:
            source_by_hash[_hash_file(src)] = src
        except Exception as e:
            _log("WARN", f"hash failed: {os.path.relpath(src, root)}: {e}")

    if force:
        _log("INFO", "rebuild (force): wiping docs_md + chunks, re-parsing everything")
        for p in [docs_md, chunks_dir]:
            if os.path.isdir(p):
                shutil.rmtree(p)
            ensure_dir(p)
        to_skip_hashes: set[str] = set()
    else:
        to_skip_hashes = set()
        for h, d in existing_by_hash.items():
            md_rel = d.get("md_path", "")
            md_abs = os.path.join(root, md_rel) if md_rel else ""
            if md_abs and os.path.isfile(md_abs):
                to_skip_hashes.add(h)

    # Determine which sources need parsing
    sources_to_parse: list[str] = []
    skipped = 0
    for h, src in source_by_hash.items():
        if h in to_skip_hashes:
            skipped += 1
            _log("INFO", f"skip: {os.path.relpath(src, root)} (already processed)")
        else:
            sources_to_parse.append(src)

    _log(
        "INFO",
        f"rebuild: {len(all_sources)} source(s) total, "
        f"{skipped} skipped, {len(sources_to_parse)} to parse",
    )

    # Sort: light files first (md, txt), then heavier (docx, pdf, doc)
    _PARSE_ORDER = {".md": 0, ".txt": 1, ".docx": 2, ".doc": 3, ".pdf": 4}
    sources_to_parse.sort(key=lambda s: _PARSE_ORDER.get(os.path.splitext(s)[1].lower(), 99))

    target = int(cfg.get("chunking", {}).get("target_tokens", 512))
    overlap = int(cfg.get("chunking", {}).get("overlap_tokens", 64))
    chunks_path = os.path.join(chunks_dir, "chunks.jsonl")

    # Phase 1: Parse files one-by-one, persist chunks incrementally
    new_docs: list[dict] = []
    new_chunks: list[dict] = []
    processed: list[str] = []

    for src in sources_to_parse:
        rel_src = os.path.relpath(src, root)
        try:
            file_hash = _hash_file(src)
            base = os.path.basename(src)
            doc_id = safe_slug(os.path.splitext(base)[0])[:80] + "_" + file_hash[:10]

            _log("INFO", f"parse: {rel_src}")
            parsed = parse_document(cfg, src)
            md_text = parsed["markdown"]
            blocks = parsed.get("blocks") or []

            md_path = os.path.join(docs_md, f"{doc_id}.md")
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(md_text)

            blocks_rel = _persist_blocks(_blocks_dir(cfg), root, doc_id, blocks)

            quality_report = None
            if blocks:
                from doc_rag.raglib.quality import persist_quality_report, run_quality_checks

                quality_dir = os.path.join(root, "build", "quality")
                quality_report = run_quality_checks(doc_id, blocks)
                persist_quality_report(quality_dir, quality_report)

            chunking_strategy = (cfg.get("chunking") or {}).get("strategy", "fixed")
            if chunking_strategy == "recursive" and blocks:
                from doc_rag.raglib.chunker_recursive import chunk_recursive

                rc_chunks = chunk_recursive(blocks, target_tokens=target)
                file_chunks: list[dict] = []
                for idx, c in enumerate(rc_chunks):
                    entry: dict[str, Any] = {
                        "doc_id": doc_id,
                        "chunk_id": f"{doc_id}:{idx}",
                        "text": c["text"],
                    }
                    if c.get("section_path"):
                        entry["section_path"] = c["section_path"]
                    if c.get("is_table"):
                        entry["is_table"] = True
                    if c.get("is_table_summary"):
                        entry["is_table_summary"] = True
                    file_chunks.append(entry)
                chunks_text = [c["text"] for c in rc_chunks]
            else:
                chunks_text = _chunk_text(md_text, target, overlap)
                file_chunks = [
                    {"doc_id": doc_id, "chunk_id": f"{doc_id}:{idx}", "text": c}
                    for idx, c in enumerate(chunks_text)
                ]

            # Per-file persistence: append chunks immediately
            _append_chunks(chunks_path, file_chunks)

            cov = parsed.get("stats") if isinstance(parsed.get("stats"), dict) else {}
            ed_y = resolve_edition_year(cfg, abs_path=src, rel_path=rel_src, sha256_hex=file_hash)
            new_docs.append({
                "doc_id": doc_id,
                "source_file": rel_src,
                "md_path": os.path.relpath(md_path, root),
                "blocks_path": blocks_rel,
                "sha256": file_hash,
                "chunk_count": len(file_chunks),
                "title_hint": os.path.splitext(base)[0],
                "edition_year": ed_y,
                "coverage": cov,
                **(
                    {
                        "quality_score": quality_report.score,
                        "quality_warning_count": len(quality_report.warnings),
                    }
                    if quality_report is not None
                    else {}
                ),
            })

            new_chunks.extend(file_chunks)
            processed.append(src)
            _log(
                "INFO",
                f"ok: {rel_src} -> {os.path.relpath(md_path, root)} (chunks={len(file_chunks)})",
            )
        except Exception as e:
            _log("ERROR", f"failed: {rel_src}: {e}")

    # Phase 2: Merge with existing (force=False path)
    kept_docs = [d for d in existing_by_hash.values() if d.get("sha256") in to_skip_hashes]
    all_docs = kept_docs + new_docs

    # Chunks are already in chunks.jsonl from per-file appends.
    # Just load and dedup — no need to merge new_chunks again.
    if force or not kept_docs:
        all_chunks = new_chunks
    else:
        all_chunks = _load_existing_chunks(chunks_path)
    original_chunk_count = len(all_chunks)

    # Phase 3: Dedup
    seen_ids: set[str] = set()
    deduped_docs: list[dict] = []
    for d in reversed(all_docs):
        did = d.get("doc_id", "")
        if did and did not in seen_ids:
            seen_ids.add(did)
            deduped_docs.append(d)
    if len(deduped_docs) != len(all_docs):
        _log("WARN", f"rebuild: removed {len(all_docs) - len(deduped_docs)} duplicate doc entries")
    all_docs = list(reversed(deduped_docs))

    seen_cids: set[str] = set()
    deduped_chunks: list[dict] = []
    for c in reversed(all_chunks):
        cid = c.get("chunk_id", "")
        if cid and cid not in seen_cids:
            seen_cids.add(cid)
            deduped_chunks.append(c)
    if len(deduped_chunks) != len(all_chunks):
        _log("WARN", f"rebuild: removed {len(all_chunks) - len(deduped_chunks)} duplicate chunk entries")
    all_chunks = list(reversed(deduped_chunks))

    dedup_thresh = float(cfg.get("chunking", {}).get("dedup_similarity_threshold", 0.0))
    if dedup_thresh > 0.0:
        doc_src_map = {d["doc_id"]: d.get("source_file", "") for d in all_docs}
        all_chunks, n_dropped = _dedup_chunks(all_chunks, doc_src_map, dedup_thresh)
        _log("INFO", f"dedup: removed {n_dropped} near-duplicate chunks (threshold={dedup_thresh})")

    # If dedup changed the set, rewrite chunks.jsonl
    if len(all_chunks) != original_chunk_count:
        _write_chunks(chunks_path, all_chunks)

    # Phase 4: Write manifest
    manifest = _manifest_shell(cfg, all_docs)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    _log("INFO", f"written: {os.path.relpath(chunks_path, root)} (chunks={len(all_chunks)})")
    _log(
        "INFO",
        f"written: {os.path.relpath(manifest_path, root)} (docs={len(manifest['documents'])})",
    )

    # Phase 5: Archive
    archive_enabled = bool(cfg.get("sources", {}).get("archive_after_ingest", True))
    if archive_enabled and processed:
        _archive_or_dedup_sources(cfg, root, processed)
    elif not archive_enabled:
        _log("INFO", "archiving disabled (sources.archive_after_ingest=false).")

    # Phase 6: Update vector index
    try:
        vi = _get_vector_index(cfg)
        if force:
            # force=True: full rebuild from scratch
            _index_dir = os.path.join(root, cfg["paths"]["index_dir"])
            for _fname in ("faiss.index", "index_meta.json"):
                _p = os.path.join(_index_dir, _fname)
                if os.path.exists(_p):
                    try:
                        os.remove(_p)
                    except OSError as _e:
                        _log("WARN", f"could not remove stale index file {_fname}: {_e}")
            vi.reset()
            if os.path.exists(chunks_path):
                chunks = _load_existing_chunks(chunks_path)
                if chunks:
                    vi.build(chunks, force=True, log=print)
        elif new_chunks:
            # force=False with new chunks: incremental update
            vi.build(new_chunks, force=False, log=print)
        else:
            _log("INFO", "rebuild: no new chunks — index unchanged")
    except Exception as e:
        _log("WARN", f"index rebuild skipped: {e}")


def rebuild_docs(config_path: str, doc_ids: list[str], filenames: list[str] | None = None) -> dict[str, Any]:
    """Rebuild only specific documents by doc_id or filename.

    Removes their .md and chunks, re-parses the source files, then
    merges everything back. Returns a summary dict.
    """
    cfg = load_config(config_path)
    root = cfg["_root"]

    manifest_path = os.path.join(root, cfg["paths"]["manifest_path"])
    chunks_path = os.path.join(root, cfg["paths"]["chunks_dir"], "chunks.jsonl")
    docs_md = os.path.join(root, cfg["paths"]["docs_md_dir"])
    chunks_dir = os.path.join(root, cfg["paths"]["chunks_dir"])

    ensure_dir(docs_md)
    ensure_dir(chunks_dir)

    existing = _load_existing_manifest(manifest_path)
    existing_docs: list[dict] = existing.get("documents", [])

    target_ids = {d for d in doc_ids if d}
    target_filenames = {f for f in (filenames or []) if f}

    # Phase 1: find indexed docs by doc_id — delete their .md, resolve source files
    to_rebuild_indexed = [d for d in existing_docs if isinstance(d, dict) and d.get("doc_id") in target_ids]
    sources_to_parse: list[str] = []

    for d in to_rebuild_indexed:
        # Delete old .md
        md_rel = d.get("md_path", "")
        if md_rel:
            md_abs = os.path.join(root, md_rel)
            if os.path.isfile(md_abs):
                try:
                    os.remove(md_abs)
                except OSError:
                    pass
        # Resolve source file
        sf_rel = d.get("source_file", "")
        if not sf_rel:
            continue
        candidates = [sf_rel]
        bn = os.path.basename(sf_rel)
        for dir_key in ("sources_archived", "sources_incoming"):
            d_rel = cfg["paths"].get(dir_key)
            if d_rel:
                candidates.append(os.path.join(d_rel, bn))
        for cand in candidates:
            cand_abs = os.path.join(root, cand)
            if os.path.isfile(cand_abs):
                sources_to_parse.append(cand_abs)
                break

    # Phase 2: find pending files by filename — add directly to sources_to_parse
    if target_filenames:
        archived_rel = cfg["paths"].get("sources_archived", "sources/archived")
        incoming_rel = cfg["paths"].get("sources_incoming", "sources/incoming")
        for dir_rel in [archived_rel, incoming_rel]:
            dir_path = os.path.join(root, dir_rel)
            if not os.path.isdir(dir_path):
                continue
            for p in list_files_recursive(dir_path, exts=SUPPORTED_EXTS):
                if os.path.basename(p) in target_filenames:
                    sources_to_parse.append(p)
                    _log("INFO", f"rebuild-docs: found pending file {os.path.basename(p)}")

    # Remove duplicates
    sources_to_parse = list(dict.fromkeys(sources_to_parse))

    _log("INFO", f"rebuild-docs: {len(to_rebuild_indexed)} indexed + {len(sources_to_parse) - len(to_rebuild_indexed)} pending = {len(sources_to_parse)} to parse")

    if not sources_to_parse:
        return {"requested": len(target_ids) + len(target_filenames), "rebuilt": 0, "missing": len(target_ids) - len(to_rebuild_indexed)}

    target = int(cfg.get("chunking", {}).get("target_tokens", 512))
    overlap = int(cfg.get("chunking", {}).get("overlap_tokens", 64))

    _PARSE_ORDER = {".md": 0, ".txt": 1, ".docx": 2, ".doc": 3, ".pdf": 4}
    sources_to_parse.sort(key=lambda s: _PARSE_ORDER.get(os.path.splitext(s)[1].lower(), 99))

    # Phase 1: Parse files one-by-one, persist chunks incrementally
    new_docs: list[dict] = []
    new_chunks: list[dict] = []

    for src in sources_to_parse:
        rel_src = os.path.relpath(src, root)
        try:
            file_hash = _hash_file(src)
            base = os.path.basename(src)
            doc_id = safe_slug(os.path.splitext(base)[0])[:80] + "_" + file_hash[:10]

            _log("INFO", f"parse: {rel_src}")
            parsed = parse_document(cfg, src)
            md_text = parsed["markdown"]
            blocks = parsed.get("blocks") or []

            md_path = os.path.join(docs_md, f"{doc_id}.md")
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(md_text)

            blocks_rel = _persist_blocks(_blocks_dir(cfg), root, doc_id, blocks)

            quality_report = None
            if blocks:
                from doc_rag.raglib.quality import persist_quality_report, run_quality_checks

                quality_dir = os.path.join(root, "build", "quality")
                quality_report = run_quality_checks(doc_id, blocks)
                persist_quality_report(quality_dir, quality_report)

            chunking_strategy = (cfg.get("chunking") or {}).get("strategy", "fixed")
            if chunking_strategy == "recursive" and blocks:
                from doc_rag.raglib.chunker_recursive import chunk_recursive

                rc_chunks = chunk_recursive(blocks, target_tokens=target)
                file_chunks: list[dict] = []
                for idx, c in enumerate(rc_chunks):
                    entry: dict[str, Any] = {
                        "doc_id": doc_id,
                        "chunk_id": f"{doc_id}:{idx}",
                        "text": c["text"],
                    }
                    if c.get("section_path"):
                        entry["section_path"] = c["section_path"]
                    if c.get("is_table"):
                        entry["is_table"] = True
                    if c.get("is_table_summary"):
                        entry["is_table_summary"] = True
                    file_chunks.append(entry)
                chunks_text = [c["text"] for c in rc_chunks]
            else:
                chunks_text = _chunk_text(md_text, target, overlap)
                file_chunks = [
                    {"doc_id": doc_id, "chunk_id": f"{doc_id}:{idx}", "text": c}
                    for idx, c in enumerate(chunks_text)
                ]

            # Per-file persistence: append chunks immediately
            _append_chunks(chunks_path, file_chunks)

            cov = parsed.get("stats") if isinstance(parsed.get("stats"), dict) else {}
            ed_y = resolve_edition_year(cfg, abs_path=src, rel_path=rel_src, sha256_hex=file_hash)
            new_docs.append({
                "doc_id": doc_id,
                "source_file": rel_src,
                "md_path": os.path.relpath(md_path, root),
                "blocks_path": blocks_rel,
                "sha256": file_hash,
                "chunk_count": len(file_chunks),
                "title_hint": os.path.splitext(base)[0],
                "edition_year": ed_y,
                "coverage": cov,
                **(
                    {
                        "quality_score": quality_report.score,
                        "quality_warning_count": len(quality_report.warnings),
                    }
                    if quality_report is not None
                    else {}
                ),
            })

            # Incremental manifest update: write after each file so UI sees progress
            _kept = [d for d in existing_docs if isinstance(d, dict) and d.get("doc_id") not in target_ids]
            _inc_manifest = _manifest_shell(cfg, _kept + new_docs)
            with open(manifest_path, "w", encoding="utf-8") as _mf:
                json.dump(_inc_manifest, _mf, ensure_ascii=False, indent=2)

            new_chunks.extend(file_chunks)
            _log(
                "INFO",
                f"ok: {rel_src} -> {os.path.relpath(md_path, root)} (chunks={len(file_chunks)})",
            )
        except Exception as e:
            _log("ERROR", f"failed: {rel_src}: {e}")

    # Phase 2: Merge with existing
    kept_docs = [d for d in existing_docs if isinstance(d, dict) and d.get("doc_id") not in target_ids]
    all_docs = kept_docs + new_docs

    _seen: set[str] = set()
    _deduped: list[dict] = []
    for d in reversed(all_docs):
        _did = d.get("doc_id", "")
        if _did and _did not in _seen:
            _seen.add(_did)
            _deduped.append(d)
    if len(_deduped) != len(all_docs):
        _log("WARN", f"rebuild-docs: removed {len(all_docs) - len(_deduped)} duplicate doc entries")
    all_docs = list(reversed(_deduped))

    kept_doc_ids = {d["doc_id"] for d in kept_docs if d.get("doc_id")}
    # Chunks already in chunks.jsonl from per-file appends — just load and dedup
    all_chunks = _load_existing_chunks(chunks_path)
    original_chunk_count = len(all_chunks)

    _sc: set[str] = set()
    _dc: list[dict] = []
    for c in reversed(all_chunks):
        _cid = c.get("chunk_id", "")
        if _cid and _cid not in _sc:
            _sc.add(_cid)
            _dc.append(c)
    if len(_dc) != len(all_chunks):
        _log("WARN", f"rebuild-docs: removed {len(all_chunks) - len(_dc)} duplicate chunk entries")
    all_chunks = list(reversed(_dc))

    dedup_thresh = float(cfg.get("chunking", {}).get("dedup_similarity_threshold", 0.0))
    if dedup_thresh > 0.0:
        doc_src_map = {d["doc_id"]: d.get("source_file", "") for d in all_docs}
        all_chunks, n_dropped = _dedup_chunks(all_chunks, doc_src_map, dedup_thresh)

    # If dedup changed the set, rewrite chunks.jsonl
    if len(all_chunks) != original_chunk_count:
        _write_chunks(chunks_path, all_chunks)

    manifest = _manifest_shell(cfg, all_docs)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    _log("INFO", f"written: chunks={len(all_chunks)}, docs={len(all_docs)}")

    # Rebuild vector index from all chunks for consistency
    try:
        vi = _get_vector_index(cfg)
        if all_chunks:
            _index_dir = os.path.join(root, cfg["paths"]["index_dir"])
            for _fname in ("faiss.index", "index_meta.json"):
                _p = os.path.join(_index_dir, _fname)
                if os.path.exists(_p):
                    try:
                        os.remove(_p)
                    except OSError as _e:
                        _log("WARN", f"could not remove stale index file {_fname}: {_e}")
            vi.reset()
            vi.build(all_chunks, force=True, log=print)
        else:
            _log("INFO", "rebuild-docs: no chunks — index unchanged")
    except Exception as e:
        _log("WARN", f"index rebuild skipped: {e}")

    return {"requested": len(target_ids) + len(target_filenames), "rebuilt": len(new_docs), "missing": 0}


def _rebuild_faiss_after_delete(cfg: dict[str, Any], deleted_doc_ids: set) -> dict[str, Any]:
    """Rebuild index after document deletion using VectorIndex.

    Drops vectors whose chunk_id belongs to a deleted doc. If the index
    is unavailable it silently returns.
    """
    stats = {"removed_vectors": 0, "kept_vectors": 0, "had_index": False}

    try:
        vi = _get_vector_index(cfg)
        if not vi.is_available():
            return stats
        stats["had_index"] = True
        removed = vi.delete(deleted_doc_ids)
        stats["removed_vectors"] = removed
        stats["kept_vectors"] = vi.size
        return stats
    except Exception as e:
        _log("WARN", f"vector index prune failed: {e}")
        return stats


def delete_documents(config_path: str, doc_ids: list[str]) -> dict[str, Any]:
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
        with open(manifest_path, encoding="utf-8") as f:
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
        # v1.5: also drop the per-document blocks file. Both the
        # manifest-recorded path and the conventional
        # `build/blocks/<doc_id>.jsonl` are tried so older manifests
        # without `blocks_path` still get their stragglers cleaned.
        candidates = set()
        bl_rel = d.get("blocks_path") or ""
        if bl_rel:
            candidates.add(bl_rel)
        did = d.get("doc_id") or ""
        if did:
            blocks_dir_rel = cfg.get("paths", {}).get("blocks_dir", "build/blocks")
            candidates.add(os.path.join(blocks_dir_rel, f"{did}.jsonl"))
        for cand in candidates:
            cand_abs = os.path.join(root, cand)
            if os.path.isfile(cand_abs):
                try:
                    os.remove(cand_abs)
                except OSError as e:
                    _log("WARN", f"failed to remove blocks '{cand}': {e}")

    removed_chunks = 0
    if os.path.exists(chunks_path):
        tmp = chunks_path + ".tmp"
        with open(chunks_path, encoding="utf-8") as src, open(tmp, "w", encoding="utf-8") as dst:
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

    deleted_names = [
        os.path.basename(d.get("source_file") or d.get("doc_id") or "") for d in to_delete
    ]
    _log(
        "INFO",
        f"delete: removed {len(to_delete)} doc(s), {removed_chunks} chunk(s), {index_stats['removed_vectors']} vector(s)",
    )
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


def wipe_index(config_path: str) -> dict[str, Any]:
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
        os.path.join(root, paths.get("blocks_dir", "build/blocks")),
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


def clean_orphans(config_path: str) -> dict[str, Any]:
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
        with open(manifest_path, encoding="utf-8") as f:
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

    orphan_blocks = 0
    blocks_dir = os.path.join(root, paths.get("blocks_dir", "build/blocks"))
    if os.path.isdir(blocks_dir):
        for fn in os.listdir(blocks_dir):
            if not fn.endswith(".jsonl"):
                continue
            doc_id = fn[:-6]
            if doc_id not in known_ids:
                try:
                    os.remove(os.path.join(blocks_dir, fn))
                    orphan_blocks += 1
                except OSError as e:
                    _log("WARN", f"clean_orphans: cannot remove blocks {fn}: {e}")

    chunks_path = os.path.join(root, paths["chunks_dir"], "chunks.jsonl")
    orphan_chunks = 0
    orphan_doc_ids: set = set()
    if os.path.exists(chunks_path):
        tmp = chunks_path + ".tmp"
        with open(chunks_path, encoding="utf-8") as src, open(tmp, "w", encoding="utf-8") as dst:
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

    index_stats = (
        _rebuild_faiss_after_delete(cfg, orphan_doc_ids)
        if orphan_doc_ids
        else {"removed_vectors": 0, "kept_vectors": 0, "had_index": False}
    )

    _log(
        "INFO",
        f"clean_orphans: removed {orphan_md} md, {orphan_blocks} blocks, "
        f"{orphan_chunks} chunk(s), {index_stats['removed_vectors']} vector(s)",
    )
    _audit(
        root,
        "clean_orphans",
        doc_ids=sorted(orphan_doc_ids) if orphan_doc_ids else None,
        counts={
            "orphan_md_removed": orphan_md,
            "orphan_blocks_removed": orphan_blocks,
            "orphan_chunks_removed": orphan_chunks,
            "removed_vectors": index_stats.get("removed_vectors", 0),
        },
    )
    return {
        "orphan_md_removed": orphan_md,
        "orphan_blocks_removed": orphan_blocks,
        "orphan_chunks_removed": orphan_chunks,
        "orphan_doc_ids": sorted(orphan_doc_ids),
        "index": index_stats,
    }


def clear_incoming(config_path: str) -> dict[str, Any]:
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

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

from doc_rag.raglib.pipeline import (
    MANIFEST_SCHEMA_VERSION,
    clean_orphans,
    clear_incoming,
    delete_documents,
    ingest,
    load_config,
    rebuild,
    rebuild_docs,
    rebuild_index,
    wipe_index,
)
from doc_rag.server.http_server import run as run_http


def _resolve_server_url(cfg: dict) -> str:
    """Resolve base URL from config, defaulting to localhost."""
    host = cfg.get("server", {}).get("host", "0.0.0.0")
    port = cfg.get("server", {}).get("port", 3333)
    if host in ("0.0.0.0", ""):
        host = "127.0.0.1"
    return f"http://{host}:{port}"


def _server_running(cfg: dict) -> bool:
    """Check if the HTTP server is reachable on localhost."""
    try:
        url = f"{_resolve_server_url(cfg)}/health/live"
        urllib.request.urlopen(url, timeout=2)
        return True
    except Exception:
        return False


def _api_post(cfg: dict, path: str, body: dict | None = None) -> dict:
    """POST to the running server. Returns parsed JSON or a stub dict."""
    url = f"{_resolve_server_url(cfg)}{path}"
    data = json.dumps(body or {}).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(req, timeout=30)
    raw = resp.read()
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return {"ok": resp.status < 400}


def _wait_for_job(cfg: dict, timeout: int = 3600) -> None:
    """Poll /ui/status until the background job finishes."""
    url = f"{_resolve_server_url(cfg)}/ui/status"
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = urllib.request.urlopen(url, timeout=5)
            data = json.loads(resp.read())
            if not data.get("running"):
                return
            job = data.get("job", "")
            done = data.get("docs_done", 0)
            total = data.get("docs_total")
            total_str = str(total) if total else "?"
            print(f"\r  {job}: {done}/{total_str}", end="", flush=True)
        except Exception:
            pass
        time.sleep(2)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="doc-rag", description="Universal local document RAG pipeline"
    )
    parser.add_argument("--config", default="config/config.yaml", help="Path to config.yaml")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("ingest", help="Incremental ingest from sources/incoming")
    p_rebuild = sub.add_parser("rebuild", help="Rebuild chunks/embeddings/index from build/docs_md")
    p_rebuild.add_argument(
        "--force",
        action="store_true",
        help="Full rebuild: wipe docs_md + chunks, reparse everything",
    )
    p_rebuild.add_argument(
        "--skip-index",
        action="store_true",
        help="Skip vector index rebuild (for local GPU processing)",
    )

    p_rebuild_docs = sub.add_parser("rebuild-docs", help="Rebuild specific documents by doc_id")
    p_rebuild_docs.add_argument("doc_ids", nargs="+", help="doc_id(s) to rebuild")

    sub.add_parser("rebuild-index", help="Rebuild vector index from existing chunks.jsonl")

    sub.add_parser("serve", help="Start local HTTP retrieval server (debug)")

    p_delete = sub.add_parser("delete", help="Remove one or more documents by doc_id")
    p_delete.add_argument("doc_ids", nargs="+", help="doc_id(s) to delete")

    p_wipe = sub.add_parser(
        "wipe", help="Delete EVERYTHING (sources/archived, build/, manifest, index)"
    )
    p_wipe.add_argument("--confirm", default="", help="Must equal 'DELETE' to proceed")

    sub.add_parser("clean-orphans", help="Drop md/chunks/vectors not referenced by manifest")
    sub.add_parser("clear-incoming", help="Delete every file in sources/incoming/")

    p_migrate = sub.add_parser(
        "migrate-vectors",
        help="Migrate vectors between backends (faiss→qdrant, faiss→pgvector, etc.)",
    )
    p_migrate.add_argument("--from", dest="source_backend", default="faiss", help="Source backend")
    p_migrate.add_argument("--to", dest="target_backend", required=True, help="Target backend")
    p_migrate.add_argument("--collection", default="default", help="Target collection name")

    sub.add_parser(
        "migrate",
        help="Upgrade build/manifest.json to the current schema (no-op when already current)",
    )

    args = parser.parse_args()
    cfg_path = os.path.normpath(args.config)

    if args.cmd == "ingest":
        cfg = load_config(cfg_path)
        if _server_running(cfg):
            print("Сервер запущен — отправляю ingest через API…")
            _api_post(cfg, "/ui/ingest")
            _wait_for_job(cfg)
            print("Готово.")
        else:
            ingest(cfg_path)
        return
    if args.cmd == "rebuild":
        cfg = load_config(cfg_path)
        if _server_running(cfg) and not args.skip_index:
            print("Сервер запущен — отправляю rebuild через API…")
            _api_post(cfg, "/ui/rebuild")
            _wait_for_job(cfg)
            print("Готово.")
        else:
            rebuild(cfg_path, force=args.force, skip_index=args.skip_index)
        return
    if args.cmd == "rebuild-docs":
        cfg = load_config(cfg_path)
        if _server_running(cfg):
            print(
                f"Сервер запущен — отправляю rebuild-docs ({len(args.doc_ids)} doc(s)) через API…"
            )
            _api_post(cfg, "/ui/rebuild-docs", {"doc_ids": args.doc_ids})
            _wait_for_job(cfg)
            print("Готово.")
        else:
            result = rebuild_docs(cfg_path, args.doc_ids)
            json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
            sys.stdout.write("\n")
        return
    if args.cmd == "rebuild-index":
        rebuild_index(cfg_path)
        return
    if args.cmd == "serve":
        cfg = load_config(cfg_path)
        server_cfg = cfg.get("server", {})
        host = server_cfg.get("host", "0.0.0.0")
        port = int(server_cfg.get("port", 3333))
        run_http(host, port)
        return
    if args.cmd == "delete":
        result = delete_documents(cfg_path, args.doc_ids)
        json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
        return
    if args.cmd == "wipe":
        if args.confirm != "DELETE":
            sys.stderr.write("Refusing to wipe: pass --confirm DELETE to proceed.\n")
            sys.exit(2)
        result = wipe_index(cfg_path)
        json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
        return
    if args.cmd == "clean-orphans":
        result = clean_orphans(cfg_path)
        json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
        return
    if args.cmd == "clear-incoming":
        result = clear_incoming(cfg_path)
        json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
        return
    if args.cmd == "migrate":
        # Stub for forward compatibility. The handler exists so users and
        # downstream tooling can rely on `doc-rag migrate` being a valid
        # invocation; concrete migrations land here as the schema evolves.
        cfg = load_config(cfg_path)
        root = cfg["_root"]
        manifest_path = os.path.join(root, cfg["paths"]["manifest_path"])
        found_version = None
        if os.path.isfile(manifest_path):
            try:
                with open(manifest_path, encoding="utf-8") as fh:
                    found_version = json.load(fh).get("schema_version", 0)
            except Exception:
                found_version = None
        result = {
            "supported_schema_version": MANIFEST_SCHEMA_VERSION,
            "found_schema_version": found_version,
            "migrations_applied": [],
            "message": "no migrations defined for this version",
        }
        json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
        return
    if args.cmd == "migrate-vectors":
        from doc_rag.raglib.migrate_vectors import migrate_vectors

        cfg = load_config(cfg_path)
        result = migrate_vectors(
            cfg,
            source_backend=args.source_backend,
            target_backend=args.target_backend,
            collection=args.collection,
        )
        json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
        return


if __name__ == "__main__":
    main()

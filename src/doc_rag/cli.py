from __future__ import annotations

import argparse
import json
import os
import sys

from doc_rag.raglib.pipeline import (
    MANIFEST_SCHEMA_VERSION,
    clean_orphans,
    clear_incoming,
    delete_documents,
    ingest,
    load_config,
    rebuild,
    wipe_index,
)
from doc_rag.server.http_server import run as run_http


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="doc-rag", description="Universal local document RAG pipeline"
    )
    parser.add_argument("--config", default="config/config.yaml", help="Path to config.yaml")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("ingest", help="Incremental ingest from sources/incoming")
    sub.add_parser("rebuild", help="Rebuild chunks/embeddings/index from build/docs_md")
    sub.add_parser("serve", help="Start local HTTP retrieval server (debug)")

    p_delete = sub.add_parser("delete", help="Remove one or more documents by doc_id")
    p_delete.add_argument("doc_ids", nargs="+", help="doc_id(s) to delete")

    p_wipe = sub.add_parser(
        "wipe", help="Delete EVERYTHING (sources/archived, build/, manifest, index)"
    )
    p_wipe.add_argument("--confirm", default="", help="Must equal 'DELETE' to proceed")

    sub.add_parser("clean-orphans", help="Drop md/chunks/vectors not referenced by manifest")
    sub.add_parser("clear-incoming", help="Delete every file in sources/incoming/")
    sub.add_parser(
        "migrate",
        help="Upgrade build/manifest.json to the current schema (no-op when already current)",
    )

    args = parser.parse_args()
    cfg_path = os.path.normpath(args.config)

    if args.cmd == "ingest":
        ingest(cfg_path)
        return
    if args.cmd == "rebuild":
        rebuild(cfg_path)
        return
    if args.cmd == "serve":
        cfg = load_config(cfg_path)
        run_http(cfg["server"]["host"], int(cfg["server"]["port"]))
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


if __name__ == "__main__":
    main()

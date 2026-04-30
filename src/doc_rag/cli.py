from __future__ import annotations
import argparse
import os

from doc_rag.raglib.pipeline import ingest, rebuild, load_config
from doc_rag.server.http_server import run as run_http


def main() -> None:
    parser = argparse.ArgumentParser(prog="doc-rag", description="Universal local document RAG pipeline")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config.yaml")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("ingest", help="Incremental ingest from sources/incoming")
    sub.add_parser("rebuild", help="Rebuild chunks/embeddings/index from build/docs_md")
    sub.add_parser("serve", help="Start local HTTP retrieval server (debug)")

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


if __name__ == "__main__":
    main()

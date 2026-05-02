#!/usr/bin/env python3
"""Сверка manifest с удалённого doc-rag (GET /api/v1/manifest) с эталоном для CI.

Пример:
  export DOC_RAG_MANIFEST_URL=https://docs.example.com/api/v1/manifest
  export DOC_RAG_API_KEY=secret  # если задан DOC_RAG_API_KEY на сервере
  python scripts/check_manifest_remote.py --expected ci/expected_manifest.json

Формат expected (все поля опциональны, кроме того что вы хотите проверить):
  {
    "corpus_content_sha256": "…",
    "document_sha256_allowlist": ["abc…", "def…"],
    "document_count": 3
  }

Код выхода: 0 — ок, 1 — несоответствие, 2 — сеть/HTTP/файл.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Set


def _load_expected(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("expected file must be a JSON object")
    return data


def _fetch_manifest(url: str, api_key: str) -> Dict[str, Any]:
    headers = {"Accept": "application/json", "User-Agent": "doc-rag-check-manifest/1.0"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, headers=headers, method="GET")
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = resp.read().decode("utf-8")
    data = json.loads(body)
    if not isinstance(data, dict):
        raise ValueError("remote manifest is not a JSON object")
    return data


def _remote_sha_set(manifest: Dict[str, Any]) -> Set[str]:
    out: Set[str] = set()
    docs = manifest.get("documents")
    if not isinstance(docs, list):
        return out
    for d in docs:
        if not isinstance(d, dict):
            continue
        h = d.get("sha256")
        if isinstance(h, str) and h.strip():
            out.add(h.strip().lower())
    return out


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Compare remote /api/v1/manifest to expected JSON.")
    p.add_argument(
        "--url",
        default=(os.environ.get("DOC_RAG_MANIFEST_URL") or "").strip(),
        help="Full URL to manifest API (or DOC_RAG_MANIFEST_URL)",
    )
    p.add_argument(
        "--expected",
        required=True,
        help="Path to JSON with corpus_content_sha256 / document_sha256_allowlist / document_count",
    )
    p.add_argument(
        "--api-key",
        default=(os.environ.get("DOC_RAG_API_KEY") or "").strip(),
        help="Bearer token if server uses DOC_RAG_API_KEY (or DOC_RAG_API_KEY env)",
    )
    args = p.parse_args(argv)

    if not args.url:
        print("check_manifest_remote: missing --url or DOC_RAG_MANIFEST_URL", file=sys.stderr)
        return 2

    try:
        expected = _load_expected(args.expected)
    except Exception as e:
        print(f"check_manifest_remote: failed to read expected: {e}", file=sys.stderr)
        return 2

    try:
        remote = _fetch_manifest(args.url, args.api_key)
    except urllib.error.HTTPError as e:
        print(f"check_manifest_remote: HTTP {e.code} {e.reason}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"check_manifest_remote: request failed: {e}", file=sys.stderr)
        return 2

    errors: List[str] = []

    exp_fp = expected.get("corpus_content_sha256")
    if isinstance(exp_fp, str) and exp_fp.strip():
        got = remote.get("corpus_content_sha256")
        if str(got).strip().lower() != exp_fp.strip().lower():
            errors.append(f"corpus_content_sha256: expected {exp_fp!r}, got {got!r}")

    allow = expected.get("document_sha256_allowlist")
    if isinstance(allow, list) and allow:
        need = {str(x).strip().lower() for x in allow if isinstance(x, str) and x.strip()}
        have = _remote_sha_set(remote)
        missing = sorted(need - have)
        if missing:
            errors.append(f"manifest missing sha256 (first few): {missing[:8]}")

    exp_n = expected.get("document_count")
    if exp_n is not None:
        try:
            want = int(exp_n)
        except Exception:
            want = -1
        docs = remote.get("documents")
        got_n = len(docs) if isinstance(docs, list) else 0
        if want >= 0 and got_n != want:
            errors.append(f"document_count: expected {want}, got {got_n}")

    if errors:
        for e in errors:
            print(f"check_manifest_remote: FAIL: {e}", file=sys.stderr)
        return 1

    print("check_manifest_remote: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

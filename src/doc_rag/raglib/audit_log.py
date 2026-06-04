"""Append-only audit log for destructive operations.

Every operation that removes or rewrites user-visible state — `delete`,
`wipe`, `clean-orphans`, `clear-incoming` — appends one JSON record to
`build/audit.log` (line-delimited JSON). The file is never compacted or
rotated by this module; an operator can size it with `journalctl` style
rotation if they care to.

Schema, version 1:

    {
      "ts": "2026-06-04T10:11:12+0300",
      "op": "delete" | "wipe" | "clean_orphans" | "clear_incoming",
      "principal": null | "api-key" | "<custom>",
      "doc_ids": [...]            # optional, when relevant
      "counts": { ... }           # operation-specific summary
      "schema_version": 1
    }

The schema is intentionally tiny: this file is read by humans when
something goes wrong, not by another machine. If automation needs more,
parse `op` and `counts` — we promise those keys are stable.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
_AUDIT_FILENAME = "audit.log"


def _audit_log_path(root: str) -> Path:
    """Return the absolute audit-log path under <root>/build/."""
    return Path(root) / "build" / _AUDIT_FILENAME


def record_event(
    root: str,
    op: str,
    *,
    counts: dict[str, Any] | None = None,
    doc_ids: Iterable[str] | None = None,
    principal: str | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Append one event line to the audit log.

    Best-effort: any I/O failure is swallowed so a write-protected disk
    cannot break a destructive operation that has already succeeded in
    memory. The structured logger still captures the error elsewhere.
    """
    payload: dict[str, Any] = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
        "op": op,
        "schema_version": SCHEMA_VERSION,
    }
    if principal is not None:
        payload["principal"] = principal
    if doc_ids is not None:
        payload["doc_ids"] = sorted({d for d in doc_ids if d})
    if counts is not None:
        payload["counts"] = dict(counts)
    if extra:
        for key, value in extra.items():
            if key in payload:
                continue
            payload[key] = value

    path = _audit_log_path(root)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except OSError:
        # Audit logging is best-effort; do not propagate.
        pass


def read_recent(root: str, limit: int = 100) -> list:
    """Return the most recent `limit` audit events, oldest-first.

    Convenience for the Web UI's danger zone. Returns an empty list when
    the file does not exist yet (no destructive op has happened).
    """
    path = _audit_log_path(root)
    if not path.is_file():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    tail = lines[-max(0, int(limit)) :]
    out = []
    for line in tail:
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out

"""Structured logging for the doc-rag HTTP server.

The server is normally observed through `journalctl -u doc-rag-mcp`,
the file pointed to by `DOC_RAG_HTTP_LOG`, or by tailing stderr inside
Docker. All three want machine-readable lines once the deployment is
not just one developer reading the terminal.

This module configures the standard library `logging` system with:

- one of two formatters (`text` or `json`), selected by env var;
- a request-id contextvar that propagates an id into every log record
  emitted during a request;
- a level selectable by env var.

The HTTP middleware in `mcp_http.py` is responsible for *setting* the
request id at the start of each request. Anything logged from inside the
middleware-bounded call stack will pick it up automatically.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from contextvars import ContextVar

_request_id: ContextVar[str | None] = ContextVar("doc_rag_request_id", default=None)

_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

# A short list of fields we always pull onto a log record if present.
# Anything passed via `logger.info("...", extra={...})` is included as-is
# in JSON mode under the same keys.
_RESERVED = {
    "name",
    "msg",
    "args",
    "levelname",
    "levelno",
    "pathname",
    "filename",
    "module",
    "exc_info",
    "exc_text",
    "stack_info",
    "lineno",
    "funcName",
    "created",
    "msecs",
    "relativeCreated",
    "thread",
    "threadName",
    "processName",
    "process",
    "message",
    "asctime",
    "taskName",  # Python 3.12+ asyncio adds this; almost always None for us.
}


def new_request_id() -> str:
    """Generate a fresh request id (short uuid4 hex)."""
    return uuid.uuid4().hex[:16]


def set_request_id(rid: str | None) -> None:
    """Set the request id for the current logical execution context."""
    _request_id.set(rid)


def get_request_id() -> str | None:
    """Return the request id of the current context, or None."""
    return _request_id.get()


class _RequestIdFilter(logging.Filter):
    """Inject the current request id into every record under `request_id`."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id.get()
        return True


class _TextFormatter(logging.Formatter):
    """Human-readable single-line format, suitable for `journalctl` and tail.

    Format: `<ts> <level> <logger> [rid=<id>] <message>`
    """

    def format(self, record: logging.LogRecord) -> str:
        ts = time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(record.created))
        rid = getattr(record, "request_id", None)
        rid_part = f" [rid={rid}]" if rid else ""
        base = f"{ts} {record.levelname:<5} {record.name}{rid_part} {record.getMessage()}"
        if record.exc_info:
            base = base + "\n" + self.formatException(record.exc_info)
        return base


class _JsonFormatter(logging.Formatter):
    """One JSON object per log line, suitable for log shippers."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        rid = getattr(record, "request_id", None)
        if rid:
            payload["request_id"] = rid
        # Anything attached via `extra={...}` that isn't a stdlib field.
        for key, value in record.__dict__.items():
            if key in _RESERVED or key == "request_id":
                continue
            if key.startswith("_"):
                continue
            try:
                json.dumps(value)
            except (TypeError, ValueError):
                value = repr(value)
            payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


_CONFIGURED = False


def configure_logging() -> logging.Logger:
    """Configure the root doc-rag logger.

    Reads:
      DOC_RAG_LOG_LEVEL  — default INFO
      DOC_RAG_LOG_FORMAT — text|json, default text

    Idempotent — calling more than once does not stack handlers.
    Returns the `doc_rag` logger.
    """
    global _CONFIGURED

    level_name = (os.environ.get("DOC_RAG_LOG_LEVEL") or "INFO").upper()
    level = _LEVELS.get(level_name, logging.INFO)

    fmt = (os.environ.get("DOC_RAG_LOG_FORMAT") or "text").strip().lower()
    formatter: logging.Formatter
    if fmt == "json":
        formatter = _JsonFormatter()
    else:
        formatter = _TextFormatter()

    root_logger = logging.getLogger("doc_rag")
    root_logger.setLevel(level)

    if _CONFIGURED:
        # Re-apply level + formatter on existing handlers; do not add new.
        for handler in root_logger.handlers:
            handler.setLevel(level)
            handler.setFormatter(formatter)
        return root_logger

    handler = logging.StreamHandler()  # defaults to stderr
    handler.setLevel(level)
    handler.setFormatter(formatter)
    handler.addFilter(_RequestIdFilter())
    root_logger.addHandler(handler)
    root_logger.propagate = False
    _CONFIGURED = True
    return root_logger


def get_logger(name: str = "doc_rag") -> logging.Logger:
    """Return a configured logger; configure on first call."""
    if not _CONFIGURED:
        configure_logging()
    return logging.getLogger(name)

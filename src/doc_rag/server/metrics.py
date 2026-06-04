"""Prometheus metrics for the doc-rag HTTP server.

`prometheus_client` is an *optional* dependency installed via the
`[metrics]` extra. When it is not installed:
  - `metrics_available()` returns False;
  - `/metrics` returns 503 with a short hint;
  - the helper functions are no-ops so application code can call them
    unconditionally.

This keeps the default install footprint small while letting operators
opt in to Prometheus scraping with a single `pip install -e .[metrics]`.
"""

from __future__ import annotations

try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )

    _AVAILABLE = True
except ImportError:  # pragma: no cover — exercised in deployments without [metrics]
    _AVAILABLE = False
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"


_REGISTRY: CollectorRegistry | None = None
_MCP_REQUESTS: Counter | None = None
_MCP_DURATION: Histogram | None = None
_INGEST_DOCS_TOTAL: Counter | None = None
_INGEST_ERRORS_TOTAL: Counter | None = None
_FAISS_INDEX_SIZE: Gauge | None = None


def metrics_available() -> bool:
    """Return True if `prometheus_client` is importable."""
    return _AVAILABLE


def _ensure_registered() -> None:
    """Create the Prometheus metric objects on first use."""
    global _REGISTRY, _MCP_REQUESTS, _MCP_DURATION
    global _INGEST_DOCS_TOTAL, _INGEST_ERRORS_TOTAL, _FAISS_INDEX_SIZE

    if not _AVAILABLE or _REGISTRY is not None:
        return

    _REGISTRY = CollectorRegistry()
    _MCP_REQUESTS = Counter(
        "doc_rag_mcp_requests_total",
        "Total MCP tool invocations",
        ["tool", "status"],
        registry=_REGISTRY,
    )
    _MCP_DURATION = Histogram(
        "doc_rag_mcp_request_duration_seconds",
        "MCP request latency in seconds",
        ["tool"],
        registry=_REGISTRY,
        buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60),
    )
    _INGEST_DOCS_TOTAL = Counter(
        "doc_rag_ingest_documents_total",
        "Documents successfully ingested",
        registry=_REGISTRY,
    )
    _INGEST_ERRORS_TOTAL = Counter(
        "doc_rag_ingest_errors_total",
        "Ingest errors observed",
        registry=_REGISTRY,
    )
    _FAISS_INDEX_SIZE = Gauge(
        "doc_rag_faiss_index_size",
        "Number of vectors in the live FAISS index",
        registry=_REGISTRY,
    )


def record_mcp_request(tool: str, status: str, duration_seconds: float) -> None:
    """Record a finished MCP request. No-op if `[metrics]` is not installed."""
    if not _AVAILABLE:
        return
    _ensure_registered()
    assert _MCP_REQUESTS is not None and _MCP_DURATION is not None
    _MCP_REQUESTS.labels(tool=tool, status=status).inc()
    _MCP_DURATION.labels(tool=tool).observe(duration_seconds)


def record_ingest_result(documents: int, errors: int) -> None:
    """Bump ingest counters at the end of an ingest run."""
    if not _AVAILABLE:
        return
    _ensure_registered()
    assert _INGEST_DOCS_TOTAL is not None and _INGEST_ERRORS_TOTAL is not None
    if documents > 0:
        _INGEST_DOCS_TOTAL.inc(documents)
    if errors > 0:
        _INGEST_ERRORS_TOTAL.inc(errors)


def set_faiss_index_size(n_vectors: int) -> None:
    """Track current FAISS index size as a gauge."""
    if not _AVAILABLE:
        return
    _ensure_registered()
    assert _FAISS_INDEX_SIZE is not None
    _FAISS_INDEX_SIZE.set(int(n_vectors))


def render_text() -> bytes:
    """Return the current /metrics body in Prometheus text exposition format."""
    if not _AVAILABLE:
        return b"# doc-rag built without [metrics] extra; install prometheus-client to enable\n"
    _ensure_registered()
    assert _REGISTRY is not None
    return generate_latest(_REGISTRY)

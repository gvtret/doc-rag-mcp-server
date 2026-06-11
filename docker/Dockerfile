# syntax=docker/dockerfile:1

# --- UI builder stage --------------------------------------------------------
# v2.2+ ships a Svelte + Vite frontend under `ui/`. Build it in a Node
# stage so the runtime image stays Python-only. If `ui/` is missing the
# stage's COPY still succeeds (empty dir copy) and the build is a no-op.
FROM node:22-alpine AS ui-builder
WORKDIR /ui
COPY ui/package.json ui/package-lock.json* /ui/
RUN npm ci --no-audit --no-fund
COPY ui/ /ui/
RUN npm run build

# --- Python runtime stage ----------------------------------------------------
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_LINK_MODE=copy
ENV UV_COMPILE_BYTECODE=1
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:${PATH}"

WORKDIR /app

# uv as the installer (v2.1+: replaces pip path entirely).
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv

# System deps (minimal): antiword for legacy .doc; ca-certs for HTTPS pulls.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    antiword \
  && rm -rf /var/lib/apt/lists/*

# Copy lock + manifest first so dependency layer caches across code changes.
COPY pyproject.toml uv.lock /app/

# Reproducible install: --frozen refuses to update the lock.
# We install the server extra explicitly; faiss/embeddings/metrics stay
# opt-in for downstream images that need them. Docling is in base deps.
RUN uv sync --frozen --extra server --no-install-project

# Copy the rest of the source and install the project itself (editable
# is fine inside a container — the source tree is immutable at runtime).
COPY . /app

# Drop in the freshly-built UI bundle from the ui-builder stage so
# /opt/.../ui/dist is populated even though the source COPY above
# excluded it via .dockerignore on the host.
COPY --from=ui-builder /ui/dist /app/ui/dist

RUN uv sync --frozen --extra server

EXPOSE 3333

# Default: Streamable HTTP MCP endpoint on /mcp
CMD ["bash", "scripts/run_mcp_http.sh"]

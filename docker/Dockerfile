# syntax=docker/dockerfile:1.19
# ^ pinned: COPY --exclude (used below) requires BuildKit dockerfile
# syntax >= 1.19; `:1` would silently pick an older frontend that errors.

# --- UI builder stage --------------------------------------------------------
# The Svelte + Vite frontend under `ui/` is the only web UI doc-rag ships —
# there is no inline HTML fallback, so this stage is required, not optional.
# Built in a separate Node stage so the runtime image stays Python-only.
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

# 0 = model-free image (CI default); 1 = embed the offline embedding model.
ARG REQUIRE_MODEL=0
ENV HF_HOME=/app/build/hf-cache

# uv as the installer (v2.1+: replaces pip path entirely).
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv

# System deps (minimal): antiword for legacy .doc; ca-certs for HTTPS pulls.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    antiword \
  && rm -rf /var/lib/apt/lists/*

# User/group created early: the model layer below uses --chown=docrag:docrag,
# and a named --chown requires the user to already exist in the image.
RUN groupadd --gid 1000 docrag \
    && useradd --uid 1000 --gid docrag --no-create-home --shell /usr/sbin/nologin docrag

# Offline embedding model as its own early layer (before pyproject/code) so
# code changes never invalidate the ~2.3 GB layer. --chown at COPY time means
# the chown step at the bottom can skip models/ (see comment there).
COPY --chown=docrag:docrag models/ /app/models/

# Fail fast when the image must embed the model but models/bge-m3 is absent.
RUN test "$REQUIRE_MODEL" = "0" || test -f /app/models/bge-m3/config.json || (echo "[doc-rag] models/bge-m3 missing: run scripts/fetch_model.py" && exit 1)

# HF_HOME parent must exist and be writable even when the container runs
# without compose bind mounts (./build is not always mounted).
RUN mkdir -p /app/build/hf-cache && chown docrag:docrag /app/build /app/build/hf-cache

# Single source of truth for the uv extras set: REQUIRE_MODEL=1 additionally
# installs the faiss + embeddings extras (otherwise load_embedder returns
# None and a Docker deploy answers 503). Both uv sync RUNs below read this
# file so the two blocks cannot drift apart.
RUN if [ "$REQUIRE_MODEL" = "1" ]; then \
      printf -- "--extra server --extra faiss --extra embeddings" > /tmp/uv_extras; \
    else \
      printf -- "--extra server" > /tmp/uv_extras; \
    fi

# Copy lock + manifest first so dependency layer caches across code changes.
COPY pyproject.toml uv.lock /app/

# Reproducible install: --frozen refuses to update the lock.
RUN uv sync --frozen $(cat /tmp/uv_extras) --no-install-project

# Copy the rest of the source and install the project itself (editable
# is fine inside a container — the source tree is immutable at runtime).
# models/ is excluded: it is already in the image via the early layer above.
COPY --exclude=models . /app

# Drop in the freshly-built UI bundle from the ui-builder stage so
# /opt/.../ui/dist is populated even though the source COPY above
# excluded it via .dockerignore on the host.
COPY --from=ui-builder /ui/dist /app/ui/dist

RUN uv sync --frozen $(cat /tmp/uv_extras)

# When the image ships the offline model, its deps must really be installed.
RUN if [ "$REQUIRE_MODEL" = "1" ]; then python -c "import sentence_transformers"; fi

# Fix rapidocr 3.8.x missing arch_config.yaml (packaging bug).
# See: https://github.com/gvtret/doc-rag-mcp-server/issues/13
RUN if [ -f scripts/rapidocr/arch_config.yaml ]; then \
      DEST=$(find /app/.venv -path "*/rapidocr/inference_engine/pytorch/networks" -type d 2>/dev/null | head -1); \
      if [ -n "$DEST" ] && [ ! -f "$DEST/arch_config.yaml" ]; then \
        cp scripts/rapidocr/arch_config.yaml "$DEST/arch_config.yaml" && \
        echo "[doc-rag] patched rapidocr: installed arch_config.yaml"; \
      fi; \
    fi

# Drop root: bind-mounted ./build and ./sources must be writable by this
# UID on the host — see DEPLOY.txt for the pre-create/chown step.
# models/ is deliberately NOT re-owned here: it was copied with
# --chown=docrag:docrag in its own layer; chowning it now would copy up
# the whole 2.3 GB model layer. The non-recursive chown of /app itself
# touches only the /app inode, not its contents.
RUN chown docrag:docrag /app && find /app -mindepth 1 -maxdepth 1 -not -name models -exec chown -R docrag:docrag {} +
USER docrag

EXPOSE 3333

# Default: Streamable HTTP MCP endpoint on /mcp
CMD ["bash", "scripts/run_mcp_http.sh"]

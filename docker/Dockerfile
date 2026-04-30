# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System deps (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

# Copy project
COPY . /app

# Install only server runtime deps (CPU-safe)
# Note: sentence-transformers/torch are optional; without them, server will do keyword search.
RUN pip install --no-cache-dir -U pip \
 && pip install --no-cache-dir -e ".[server]" \
 && pip install --no-cache-dir -r requirements.txt

EXPOSE 3333

# Default: Streamable HTTP MCP endpoint on /mcp
CMD ["bash", "scripts/run_mcp_http.sh"]

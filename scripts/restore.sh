#!/usr/bin/env bash
# Restore a doc-rag backup produced by scripts/backup.sh.
#
# Refuses to overwrite an existing populated build/ directory unless
# --force is given. Always verifies the MANIFEST.sha256 file embedded
# inside the backup before touching anything on disk.
#
# Usage:
#   scripts/restore.sh BACKUP.tar.gz [--root /opt/doc-rag-mcp] [--force]
#
# Exit status:
#   0 — restore complete and verified
#   1 — bad arguments
#   2 — backup missing or unreadable
#   3 — sha256/tar tooling missing
#   4 — MANIFEST.sha256 verification failed
#   5 — target already populated and --force not given

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FORCE=0
BACKUP=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --root)
            ROOT="${2:?--root needs a directory}"
            shift 2
            ;;
        --force)
            FORCE=1
            shift
            ;;
        -h|--help)
            sed -n '/^# Usage:/,/^$/p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        -*)
            echo "unknown argument: $1" >&2
            exit 1
            ;;
        *)
            if [[ -n "$BACKUP" ]]; then
                echo "only one backup file accepted" >&2
                exit 1
            fi
            BACKUP="$1"
            shift
            ;;
    esac
done

if [[ -z "$BACKUP" ]]; then
    echo "usage: scripts/restore.sh BACKUP.tar.gz" >&2
    exit 1
fi
if [[ ! -f "$BACKUP" ]]; then
    echo "backup not found: $BACKUP" >&2
    exit 2
fi
for bin in tar sha256sum; do
    if ! command -v "$bin" >/dev/null 2>&1; then
        echo "$bin is required" >&2
        exit 3
    fi
done

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

tar -C "$WORK" -xzf "$BACKUP"

# Find the single top-level directory inside the staged backup.
STAGED="$(find "$WORK" -mindepth 1 -maxdepth 1 -type d | head -n 1)"
if [[ -z "$STAGED" ]] || [[ ! -f "$STAGED/MANIFEST.sha256" ]]; then
    echo "backup does not look like a doc-rag backup (no MANIFEST.sha256)" >&2
    exit 4
fi

# Verify checksums against the staged tree.
if ! (cd "$STAGED" && sha256sum --quiet -c MANIFEST.sha256); then
    echo "sha256 verification failed — backup is corrupt or tampered" >&2
    exit 4
fi

# Refuse to overwrite a populated build/.
if [[ -d "$ROOT/build" ]]; then
    if [[ -n "$(ls -A "$ROOT/build" 2>/dev/null)" ]] && [[ "$FORCE" -ne 1 ]]; then
        echo "$ROOT/build is not empty; pass --force to overwrite" >&2
        exit 5
    fi
fi

mkdir -p "$ROOT/build"

# Copy each entry from staged build/ over into $ROOT/build/.
if [[ -d "$STAGED/build" ]]; then
    cp -a "$STAGED/build/." "$ROOT/build/"
fi

# Optional sources/archived from --with-archived backups.
if [[ -d "$STAGED/sources/archived" ]]; then
    mkdir -p "$ROOT/sources/archived"
    cp -a "$STAGED/sources/archived/." "$ROOT/sources/archived/"
fi

echo "restored $BACKUP -> $ROOT"

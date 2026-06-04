#!/usr/bin/env bash
# Create a timestamped backup of the doc-rag corpus state.
#
# By default produces a .tar.gz containing:
#   build/manifest.json
#   build/chunks_jsonl/
#   build/embeddings/
#   build/index/
#   build/audit.log (if it exists)
#
# Pass --with-archived to also include sources/archived/ — useful when
# the original source documents are not stored elsewhere. The archive
# can be many hundreds of megabytes in that mode.
#
# A MANIFEST.sha256 file is written inside the tarball; scripts/restore.sh
# verifies it on extraction. Without that file you cannot reliably tell
# whether the backup is internally consistent.
#
# Usage:
#   scripts/backup.sh                              # writes ./doc-rag-backup-YYYYMMDD-HHMMSS.tar.gz
#   scripts/backup.sh --with-archived              # also include sources/archived/
#   scripts/backup.sh --output /path/to/dir        # choose output directory
#   scripts/backup.sh --root /opt/doc-rag-mcp      # back up a different install root
#
# Exit status:
#   0 — backup written and verified
#   1 — bad arguments
#   2 — repository state missing or unreadable
#   3 — sha256/tar tooling missing
#   4 — write failed

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$PWD"
WITH_ARCHIVED=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --with-archived)
            WITH_ARCHIVED=1
            shift
            ;;
        --output)
            OUT_DIR="${2:?--output needs a directory}"
            shift 2
            ;;
        --root)
            ROOT="${2:?--root needs a directory}"
            shift 2
            ;;
        -h|--help)
            sed -n '/^# Usage:/,/^$/p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

if ! command -v tar >/dev/null 2>&1; then
    echo "tar is required" >&2
    exit 3
fi
if ! command -v sha256sum >/dev/null 2>&1; then
    echo "sha256sum is required" >&2
    exit 3
fi

cd "$ROOT"
if [[ ! -d build ]]; then
    echo "no build/ under $ROOT — nothing to back up" >&2
    exit 2
fi

TS="$(date +%Y%m%d-%H%M%S)"
NAME="doc-rag-backup-${TS}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

STAGING="$WORK/$NAME"
mkdir -p "$STAGING/build"

# Always-included entries
for entry in manifest.json chunks_jsonl embeddings index audit.log; do
    src="build/$entry"
    if [[ -e "$src" ]]; then
        cp -a "$src" "$STAGING/build/"
    fi
done

if [[ "$WITH_ARCHIVED" -eq 1 ]] && [[ -d sources/archived ]]; then
    mkdir -p "$STAGING/sources"
    cp -a sources/archived "$STAGING/sources/"
fi

# Build MANIFEST.sha256 from the staged copy. Hashing the staged copy
# rather than the live tree avoids inconsistency if ingest runs while
# we tar.
(
    cd "$STAGING"
    find . -type f ! -name MANIFEST.sha256 -print0 \
        | LC_ALL=C sort -z \
        | xargs -0 sha256sum > MANIFEST.sha256
)

mkdir -p "$OUT_DIR"
OUTPUT="$OUT_DIR/$NAME.tar.gz"
tar -C "$WORK" -czf "$OUTPUT" "$NAME"

SIZE_MB="$(du -m "$OUTPUT" | awk '{print $1}')"
echo "wrote $OUTPUT (${SIZE_MB} MB)"

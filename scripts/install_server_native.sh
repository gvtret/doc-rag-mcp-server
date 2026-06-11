#!/usr/bin/env bash
set -euo pipefail

# Установка doc-rag на Linux-сервере без Docker:
# Python venv → bootstrap (неинтерактив) → systemd (автозапуск при загрузке).
#
# Запуск: от root после распаковки репозитория.
#   sudo bash scripts/install_server_native.sh [/path/to/doc-rag]
#
# Если путь не указан, считается корень репозитория по расположению этого скрипта.
#
# Опции: --minimal (без torch/embeddings; только HTTP/MCP, без семантики).
# По умолчанию ставится embeddings stack (torch CPU + sentence-transformers).
# Флаг указывайте перед путём INSTALL_ROOT при необходимости:
#   sudo bash scripts/install_server_native.sh --minimal /opt/doc-rag

if [[ "$(id -u)" -ne 0 ]]; then
  echo "ERROR: запускайте от root: sudo bash $0 \"$@\""
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_ROOT="/opt/doc-rag-mcp"

SKIP_EMBEDDINGS=

usage() {
  echo "Usage: sudo bash scripts/install_server_native.sh [INSTALL_ROOT]"
  echo "       INSTALL_ROOT по умолчанию: ${DEFAULT_ROOT}"
  echo "Flags:"
  echo "  --minimal   без torch/embeddings (только HTTP/MCP, без семантики)"
  exit "$1"
}

POS_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage 0 ;;
    --minimal) SKIP_EMBEDDINGS=1; shift ;;
    *) POS_ARGS+=("$1"); shift ;;
  esac
done

INSTALL_ROOT="${POS_ARGS[0]:-${INSTALL_ROOT:-$DEFAULT_ROOT}}"

if [[ ! -f "${SRC_ROOT}/pyproject.toml" ]] || [[ ! -f "${SRC_ROOT}/scripts/run_mcp_http.sh" ]]; then
  echo "ERROR: ${SRC_ROOT} не похож на корень doc-rag (нет pyproject.toml или scripts/run_mcp_http.sh)."
  exit 1
fi

mkdir -p "${INSTALL_ROOT}"
INSTALL_ROOT="$(cd "$INSTALL_ROOT" && pwd)"

SERVICE_USER=docrag

echo "[install] INSTALL_ROOT=${INSTALL_ROOT}"
if [[ -n "${SKIP_EMBEDDINGS}" ]]; then
  echo "[install] embeddings: SKIP (only HTTP/MCP)"
else
  echo "[install] embeddings: install (torch CPU + sentence-transformers)"
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq

echo "[install] apt: базовые пакеты + antiword (legacy .doc)…"
# С v2.0 OCR обрабатывается Docling (RapidOCR) внутри Python venv —
# системный tesseract больше не нужен.
apt-get install -y --no-install-recommends \
  ca-certificates \
  git \
  python3 \
  python3-venv \
  python3-dev \
  build-essential \
  rsync \
  antiword

if ! id "${SERVICE_USER}" &>/dev/null; then
  useradd --system \
    --home-dir /var/lib/docrag \
    --create-home \
    --shell /usr/sbin/nologin \
    "${SERVICE_USER}"
fi

echo "[install] sync project -> ${INSTALL_ROOT} ..."
# `build/` (manifest, chunks.jsonl, FAISS, blocks/) and `sources/`
# (incoming + archived documents) are data the operator owns — must
# survive a re-run of this script. We:
#   - `--exclude` both at the source side so they are never sent;
#   - `--filter 'protect …/'` so `--delete` cannot remove them on the
#     destination even if a future rsync version stops respecting
#     `--exclude` for deletion (belt + suspenders).
rsync -a --delete \
  --filter 'protect build/' \
  --filter 'protect sources/' \
  --exclude '.git' \
  --exclude '.venv' \
  --exclude 'build' \
  --exclude 'sources' \
  "${SRC_ROOT}/" "${INSTALL_ROOT}/"

mkdir -p "${INSTALL_ROOT}/build" "${INSTALL_ROOT}/sources/incoming" "${INSTALL_ROOT}/sources/archived"
chmod a+x "${INSTALL_ROOT}/scripts/run_mcp_http.sh" "${INSTALL_ROOT}/scripts/bootstrap.sh" 2>/dev/null || true

chown -R "${SERVICE_USER}:${SERVICE_USER}" "${INSTALL_ROOT}"

EMB_ENV="Y"
if [[ -n "${SKIP_EMBEDDINGS}" ]]; then
  EMB_ENV="N"
fi

echo "[install] bootstrap (venv через uv + базовые deps; Docling приходит как базовая зависимость)…"
sudo -u "${SERVICE_USER}" -H bash -c "
set -euo pipefail
cd '${INSTALL_ROOT}'
export HOME='/var/lib/docrag'
export DOC_RAG_BOOTSTRAP_NONINTERACTIVE='1'
export DOC_RAG_BOOTSTRAP_DEV='N'
export DOC_RAG_BOOTSTRAP_FAISS='Y'
export DOC_RAG_BOOTSTRAP_EMBEDDINGS='${EMB_ENV}'
export DOC_RAG_BOOTSTRAP_METRICS='N'
export DOC_RAG_BOOTSTRAP_INGEST='N'
bash scripts/bootstrap.sh
"
echo "[install] проверка Docling в venv:"
sudo -u "${SERVICE_USER}" -H env INSTALL_ROOT="${INSTALL_ROOT}" bash -c '
set -e
cd "$INSTALL_ROOT"
VPY="$INSTALL_ROOT/.venv/bin/python"
if [ -x "$VPY" ] && "$VPY" -c "import docling" 2>/dev/null; then
  echo "  OK: docling импортируется."
else
  echo "  WARN: docling не найден в venv. Запустите вручную: uv sync --frozen --extra server --extra faiss --extra embeddings"
fi
'

UNIT_SRC="${INSTALL_ROOT}/systemd/doc-rag-mcp.service.in"
DEFAULT_SRC="${INSTALL_ROOT}/deploy/etc-default-doc-rag.in"
if [[ ! -f "${UNIT_SRC}" ]]; then
  echo "ERROR: нет файла ${UNIT_SRC}"
  exit 1
fi

while IFS= read -r _line || [[ -n "${_line}" ]]; do
  printf '%s\n' "${_line//@INSTALL_ROOT@/${INSTALL_ROOT}}"
done < "${UNIT_SRC}" > /etc/systemd/system/doc-rag-mcp.service

if [[ ! -f /etc/default/doc-rag ]]; then
  if [[ ! -f "${DEFAULT_SRC}" ]]; then
    echo "[install] предупреждение: нет ${DEFAULT_SRC}, создаём минимальный /etc/default/doc-rag"
    printf '%s\n%s\n%s\n' 'DOC_RAG_HTTP_HOST=0.0.0.0' 'DOC_RAG_HTTP_PORT=3333' > /etc/default/doc-rag
  else
    while IFS= read -r _line || [[ -n "${_line}" ]]; do
      printf '%s\n' "${_line//@INSTALL_ROOT@/${INSTALL_ROOT}}"
    done < "${DEFAULT_SRC}" > /etc/default/doc-rag
  fi
  chmod 0644 /etc/default/doc-rag
else
  echo "[install] /etc/default/doc-rag уже есть — не перезаписываем."
fi

systemctl daemon-reload
systemctl enable doc-rag-mcp.service
systemctl restart doc-rag-mcp.service

echo ""
echo "Подсказка: индекс FAISS и manifest в build/ сохраняются при повторном запуске этого скрипта."
echo "  Новые поля manifest (OCR/coverage) появятся после ingest или rebuild."
echo "  Пересборка индекса нужна, если менялись чанки или модель эмбеддингов; только обновление кода — обычно нет."
echo ""
echo "OK: сервис doc-rag-mcp включён и перезапущен."
systemctl --no-pager status doc-rag-mcp.service || true

echo ""
echo "Проверка: curl -sS http://127.0.0.1:3333/health"
echo "Логи:     journalctl -u doc-rag-mcp -f"
echo "Правки env: nano /etc/default/doc-rag && systemctl restart doc-rag-mcp"

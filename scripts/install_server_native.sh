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
# Опции: --cpu (по умолчанию) | --gpu (нужны драйверы NVIDIA на хосте) | --minimal (без torch).
# Флаги указывайте перед путём INSTALL_ROOT при необходимости:
#   sudo bash scripts/install_server_native.sh --cpu /opt/doc-rag

if [[ "$(id -u)" -ne 0 ]]; then
  echo "ERROR: запускайте от root: sudo bash $0 \"$@\""
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_ROOT="/opt/doc-rag-mcp"

INSTALL_CPU_EMBEDDINGS=1
INSTALL_GPU_EMBEDDINGS=
SKIP_EMBEDDINGS=

usage() {
  echo "Usage: sudo bash scripts/install_server_native.sh [INSTALL_ROOT]"
  echo "       INSTALL_ROOT по умолчанию: ${DEFAULT_ROOT}"
  echo "Flags:"
  echo "  --cpu       torch CPU + embeddings (поведение по умолчанию)"
  echo "  --gpu       torch GPU + embeddings"
  echo "  --minimal   без torch/embeddings (только HTTP/MCP, без семантики)"
  exit "$1"
}

POS_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage 0 ;;
    --cpu) INSTALL_CPU_EMBEDDINGS=1; INSTALL_GPU_EMBEDDINGS=; SKIP_EMBEDDINGS=; shift ;;
    --gpu) INSTALL_GPU_EMBEDDINGS=1; INSTALL_CPU_EMBEDDINGS=; SKIP_EMBEDDINGS=; shift ;;
    --minimal) SKIP_EMBEDDINGS=1; INSTALL_CPU_EMBEDDINGS=; INSTALL_GPU_EMBEDDINGS=; shift ;;
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
TORCH_CHOICE=2
if [[ -n "${INSTALL_GPU_EMBEDDINGS:-}" ]]; then
  TORCH_CHOICE=1
fi
if [[ -n "${SKIP_EMBEDDINGS:-}" ]]; then
  TORCH_CHOICE=3
fi

echo "[install] INSTALL_ROOT=${INSTALL_ROOT}"
echo "[install] torch/bootstrap choice: ${TORCH_CHOICE} (1=GPU 2=CPU 3=skip)"

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y --no-install-recommends \
  ca-certificates \
  git \
  logrotate \
  python3 \
  python3-venv \
  python3-dev \
  build-essential \
  rsync

if ! id "${SERVICE_USER}" &>/dev/null; then
  useradd --system \
    --home-dir /var/lib/docrag \
    --create-home \
    --shell /usr/sbin/nologin \
    "${SERVICE_USER}"
fi

echo "[install] sync project -> ${INSTALL_ROOT} ..."
rsync -a --delete \
  --exclude '.git' \
  --exclude '.venv' \
  --exclude 'build' \
  "${SRC_ROOT}/" "${INSTALL_ROOT}/"

mkdir -p "${INSTALL_ROOT}/build" "${INSTALL_ROOT}/sources/incoming" "${INSTALL_ROOT}/sources/archived"
chmod a+x "${INSTALL_ROOT}/scripts/run_mcp_http.sh" "${INSTALL_ROOT}/scripts/bootstrap.sh" 2>/dev/null || true

chown -R "${SERVICE_USER}:${SERVICE_USER}" "${INSTALL_ROOT}"

echo "[install] bootstrap (venv + deps) ..."
sudo -u "${SERVICE_USER}" -H bash -c "
set -euo pipefail
cd '${INSTALL_ROOT}'
export HOME='/var/lib/docrag'
export DOC_RAG_BOOTSTRAP_NONINTERACTIVE='1'
export DOC_RAG_BOOTSTRAP_DEV='N'
export DOC_RAG_BOOTSTRAP_FAISS='Y'
export DOC_RAG_BOOTSTRAP_PDF='Y'
export DOC_RAG_BOOTSTRAP_SERVER='Y'
export DOC_RAG_BOOTSTRAP_INGEST='N'
export DOC_RAG_BOOTSTRAP_TORCH='${TORCH_CHOICE}'
bash scripts/bootstrap.sh
"

UNIT_SRC="${INSTALL_ROOT}/systemd/doc-rag-mcp.service.in"
DEFAULT_SRC="${INSTALL_ROOT}/deploy/etc-default-doc-rag.in"
LOGROT_SRC="${INSTALL_ROOT}/deploy/logrotate-doc-rag-mcp.in"
if [[ ! -f "${UNIT_SRC}" ]]; then
  echo "ERROR: нет файла ${UNIT_SRC}"
  exit 1
fi

while IFS= read -r _line || [[ -n "${_line}" ]]; do
  printf '%s\n' "${_line//@INSTALL_ROOT@/${INSTALL_ROOT}}"
done < "${UNIT_SRC}" > /etc/systemd/system/doc-rag-mcp.service

if [[ ! -f "${LOGROT_SRC}" ]]; then
  echo "ERROR: нет файла ${LOGROT_SRC}"
  exit 1
fi

while IFS= read -r _line || [[ -n "${_line}" ]]; do
  printf '%s\n' "${_line//@INSTALL_ROOT@/${INSTALL_ROOT}}"
done < "${LOGROT_SRC}" > /etc/logrotate.d/doc-rag-mcp
chmod 0644 /etc/logrotate.d/doc-rag-mcp

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
echo "OK: сервис doc-rag-mcp включён и перезапущен."
systemctl --no-pager status doc-rag-mcp.service || true

echo ""
echo "Проверка: curl -sS http://127.0.0.1:3333/health"
echo "Логи:     journalctl -u doc-rag-mcp -f"
echo "HTTP log: DOC_RAG_HTTP_LOG (по умолчанию ${INSTALL_ROOT}/build/http.log), ротация: /etc/logrotate.d/doc-rag-mcp (3 архива по 5M)"
echo "Правки env: nano /etc/default/doc-rag && systemctl restart doc-rag-mcp"

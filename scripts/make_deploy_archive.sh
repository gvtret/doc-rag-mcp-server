#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

WITH_DOCS=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-docs) WITH_DOCS=1; shift ;;
    -h|--help)
      echo "Usage: bash scripts/make_deploy_archive.sh [--with-docs]"
      echo "  default: архив без файлов из sources/archived и sources/incoming (только пустые каталоги)."
      echo "  --with-docs: включить все отслеживаемые в git документы (архив может быть очень большим)."
      exit 0
      ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

commit="$(git rev-parse --short HEAD 2>/dev/null || echo nogit)"
stamp="$(date +%Y%m%d)"
archive_basename="doc-rag-deploy-${stamp}-${commit}"
tmpdir="$(mktemp -d)"
trap 'rm -rf "${tmpdir}"' EXIT

# Верхний каталог в архиве всегда «doc-rag» — после распаковки: cd doc-rag
staging="${tmpdir}/doc-rag"
mkdir -p "${staging}"

git archive --format=tar HEAD | tar -x -C "${staging}"

mkdir -p "${staging}/build" "${staging}/sources/incoming" "${staging}/sources/archived"
touch "${staging}/build/.gitkeep" "${staging}/sources/incoming/.gitkeep" "${staging}/sources/archived/.gitkeep"

if [[ "${WITH_DOCS}" -eq 0 ]]; then
  find "${staging}/sources/archived" -mindepth 1 -delete 2>/dev/null || true
  find "${staging}/sources/incoming" -mindepth 1 -delete 2>/dev/null || true
  touch "${staging}/sources/incoming/.gitkeep" "${staging}/sources/archived/.gitkeep"
fi

cat > "${staging}/DEPLOY.txt" <<'EOF'
Развёртывание на сервере (Docker):

  1) tar xzf doc-rag-deploy-*.tar.gz && cd doc-rag
  2) cp .env.example .env
  3) Отредактируйте .env (порты, DOC_RAG_ALLOWED_ORIGINS, при необходимости API key).
  4) docker compose up -d --build

Проверка: curl -sS http://127.0.0.1:3333/health
Индексация в контейнере: docker compose exec doc-rag-mcp doc-rag ingest

---

Без Docker (venv + systemd, автозапуск при загрузке):

  sudo bash scripts/install_server_native.sh
  # или явный каталог и GPU: sudo bash scripts/install_server_native.sh --gpu /opt/doc-rag

Проверка: curl -sS http://127.0.0.1:3333/health
Индексация под пользователем docrag: sudo -u docrag -H bash -lc 'cd /путь/к/doc-rag && .venv/bin/doc-rag ingest'
(пути и cd замените на ваш INSTALL_ROOT; проще: см. README.)
EOF

out="${ROOT}/${archive_basename}.tar.gz"
tar -C "${tmpdir}" -czf "${out}" doc-rag

size="$(du -h "${out}" | cut -f1)"
echo "[doc-rag] Created: ${archive_basename}.tar.gz (${size})"

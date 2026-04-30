# TODO — после code review doc-rag

Список по убыванию серьёзности. Обновляйте галочки по мере работы.

---

## Критично / высокий риск

- [x] `cfg["_root"]` при загрузке конфига для поиска + `DOC_RAG_ROOT` / `project_root()` (`retrieval.py`; stdio MCP удалён).
- [x] Корень проекта: переменная окружения **`DOC_RAG_ROOT`** (см. README) для сценария вне editable install.
- [x] Endpoint `POST /search` в `http_server.py`: лимит **`top_k`** 1…50, общий путь с MCP через `doc_search`; аутентификация не делалась — остаётся явный debug-only для LAN.
- [x] `docker-compose`: **`DOC_RAG_ALLOWED_ORIGINS`** по умолчанию для localhost.

---

## Баги / расхождения поведения

- [x] Убран stdio; один путь ответа по HTTP MCP — текст в `search_tool` (расхождение stdio/HTTP снято вместе со stdio).
- [x] `http_server` и MCP tool используют **`doc_search`** из `retrieval.py`.
- [x] Обогащать результаты `source_file` из manifest по `doc_id` (чанки его не содержат).
- [x] Документация/скрипты: метод **`tools/list`**, см. `verify_mcp.sh`.

---

## Безопасность / устойчивость

- [x] Ограничения на размер и сложность PDF/DOCX при парсинге недоверенных файлов (max_file_mb / max_pdf_pages / max_docx_paragraphs).
- [x] `mcp_http.py`: middleware — явная переменная **`response`** до `try`/`finally`.

---

## Качество кода и репозиторий

- [x] Удалён `indexer.py.bak`.
- [x] Добавлен корневой `.gitignore` (игнорируем `build/`, `.venv/`, логи, кеши).
- [x] Заменён `datetime.utcnow()` в `pipeline.py` на timezone-aware UTC.
- [x] Убрана неиспользуемая переменная `path_lower` в `parsers.py`.

---

## Тесты и CI

- [x] Добавлены минимальные `tests/` (без запуска сервера; smoke на `initialize`/`tools/list`, плюс unit на lexical search). `pytest` ставится через `.[dev]`.

---

## Низкий приоритет / продукт

- [x] Удалены legacy `server/server.py` и `tools/*` (дубликаты основного пакета).

# RAG-аудит: план исправлений и улучшений (v2, после ревью плана)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Закрыть баги целостности данных (A1-A12), поднять качество ретривала
(B1-B6) и латентность (C1-C8) по итогам LLM/RAG-аудита; eval-харнесс —
предусловие качественных изменений.

**Architecture:** Фаза 1 — cleanup и integrity. Фаза 2 — eval-харнесс (baseline
снимается ДО алгоритмических изменений). Фаза 3 — алгоритмика (токенизатор,
таблицы, префиксы, BM25). Фаза 4 — переиндексация `--force` с замером до/после.

**Tech Stack:** Python 3.11+, FAISS (IndexFlatIP не меняется), tokenizers 0.22,
pytest.

**Spec:** аудит-отчёт этой сессии; ключевые утверждения проверены ревьюером
плана по коду и замерами (lexical 6.3-8.6 с на запрос; 24.5% чанков > 512
реальных токенов; max чанк 14233 токена — таблица).

## Global Constraints

- Тексты: ASCII + кириллица; без эмодзи; коммит-субъект <= 72 симв.
- Тесты: `.venv/bin/python -m pytest`; ruff check + format --check на
  тронутых файлах (`/home/trgv/.local/bin/ruff`).
- Формат `faiss.index` не меняется; новые поля IndexMeta — в конец dataclass
  с дефолтами; старые JSON читаются.
- Никаких обращений к сети в пути запроса.
- `tests/test_index_fingerprint.py` и `tests/test_degraded_mode.py` уже
  существуют — расширять, не создавать дубли.
- Versioning constants (вводятся Task 6, инкрементятся в T11/T12/T13):
  `CHUNKING_SCHEMA_VERSION` (модульный constants в
  `doc_rag/raglib/chunking_meta.py`), `EMBEDDING_TEXT_VERSION`,
  fingerprint включает sha256 `tokenizer.json` актуальной модели.

## Review Focus

1. Рассинхрон векторов и текстов при смене конфига чанкинга/токенизатора/
   текста эмбеддинга — молча. (Task 6 + bump-инвариант: T11/T12/T13 обязаны
   инкрементить константы; тест в Task 6.)
2. Обрыв посреди rebuild -> нет индекса или рассогласованные файлы.
   (Task 2: tmp+rename; Task 5: ntotal == len(chunk_ids) fail-closed.)
3. Фильтр doc_id возвращает чужие документы в semantic, lexical И hybrid.
   (Task 7: тест на всех трёх режимах.)
4. Битая мета -> fail-closed, а не позиционный legacy-fallback.
   (Task 5: тест «corrupt meta -> semantic None -> lexical, НЕ legacy».)
5. Русский запрос к английскому документу через lexical: словоформы.
   (Task 14: тест «прибора/приборы -> прибор» + негативный тест «слово,
   которое не должно склеиться».)

---

## Фаза 1 — Cleanup и integrity

### Task 1: Удалить мёртвый raglib/indexer.py

**Files:**
- Delete: `src/doc_rag/raglib/indexer.py`
- Modify: докстринги-упоминания (`vector_index.py:3`, `model_loader.py:3`,
  `faiss_store.py:3`)

- [ ] **Step 1:** `grep -rn "raglib.indexer\|from doc_rag.raglib import indexer\|raglib import model_loader, model_paths" src tests scripts` —
  подтвердить нулевые импорты (кроме самого файла).
- [ ] **Step 2:** удалить файл, поправить докстринги.
- [ ] **Step 3:** полный pytest PASS (раньше делали это в конце Фазы 1 —
  теперь первым, чтобы Task 6 правил один IndexMeta, а не два).
- [ ] **Step 4: Commit** — `chore: удалить мёртвый indexer.py`.

### Task 2: Атомарная запись

**Files:**
- Create: `src/doc_rag/raglib/atomic.py`
- Modify: `faiss_store.py:save (201-211)`, `pipeline.py:_write_chunks
  (828-831)`, `pipeline.py:_append_chunks (834-839)`, `pipeline.py` manifest
  writes (`:1035-1038`, `:1104-1107`)
- Test: `tests/test_atomic_writes.py`

**Interfaces:**
- Produces:
  - `atomic_write_bytes(path: str, data: bytes) -> None` — tmp в том же
    каталоге, fsync, os.replace.
  - `atomic_write_text(path: str, text: str) -> None`.
  - Для faiss: НЕ bytes — паттерн `write_index(tmp_path)` + `os.replace` в
    save() (см. Step 3). Последовательность: index.tmp -> replace, затем
    meta (atomic_write_text), затем fsync каталога.
- `_append_chunks` НЕ атомарен (это механизм докатки после обрыва;
  replace на каждый вызов даст O(n^2)) — только `f.flush()` + `os.fsync(f.fileno())`.

- [ ] **Step 1: Падающие тесты** — (а) monkeypatch faiss.write_index ->
  raise посреди save(): старые faiss.index И index_meta.json на месте и
  согласованы; (б) atomic_write_bytes оставляет файл на месте при падении
  json.dump; (в) _append_chunks после падения процесса (симуляция: неполная
  строка) не теряет ранее записанные.
- [ ] **Step 2: FAIL.**
- [ ] **Step 3: Реализовать** — по интерфейсам выше.
- [ ] **Step 4: PASS + полный набор.**
- [ ] **Step 5: Commit** — `fix: атомарная запись индекса, манифеста, чанков`.

### Task 3: Единый VectorIndex-кэш с mtime-валидацией

**Files:**
- Modify: `src/doc_rag/server/retrieval.py:21,536-546`
- Test: `tests/test_retrieval.py`

**Interfaces:**
- Produces: `_VECTOR_INDEX_CACHE: dict[str, dict]` — ключ `f"{root}:{namespace}"`,
  значение `{"index": VectorIndex | None, "index_mtime": float | None,
  "meta_mtime": float | None}`; функция `invalidate_index_cache() -> None`.
- Дизайн один для T3 и T15 (не два кэша): namespace из MCP-аргументов
  недоверенный — кэш ограничен 8 записями LRU-вытеснением.
- Экспорт `invalidate_index_cache()` и вызов из конца
  `_run_index_job_background` (mcp_http.py) — belt-and-suspenders к mtime.

- [ ] **Step 1: Тест** — индекс в tmp, запрос (кэш), перезаписать
  faiss.index+meta другим набором, запрос -> новые id. FAIL.
- [ ] **Step 2: FAIL.**
- [ ] **Step 3: Реализовать** — перед использованием стора сверять
  `st_mtime` обоих файлов с кэшем; расхождение -> пересоздать VectorIndex.
  Статический mtime (Linux наносекунды); size не использовать.
- [ ] **Step 4: PASS.**
- [ ] **Step 5: Commit** — `fix: поиск видит свежий индекс после rebuild`.

### Task 4: Off-by-i в faiss_store.add()

**Files:**
- Modify: `faiss_store.py:126-137`
- Test: `tests/test_vectorstore.py` (расширить существующий test_reconstruct)

- [ ] **Step 1: Тест** — add(4 вектора, ids a-d) -> reconstruct каждого id
  байт-в-байт равен входному вектору (стиль test_faiss_reconstruct). FAIL.
- [ ] **Step 2: FAIL.**
- [ ] **Step 3:** `base = len(self._meta.chunk_ids)` ДО цикла.
- [ ] **Step 4: PASS + test_vectorstore.**
- [ ] **Step 5: Commit** — `fix: позиции id_to_pos при инкрементальном add`.

### Task 5: Fail-closed целостности + блокировка legacy

**Files:**
- Modify: `faiss_store.py:_ensure_loaded (71-101)`,
  `vector_index.py:_load_existing_ids (64-75)`,
  `server/retrieval.py:semantic_search (524-635)` legacy-гейт
- Test: `tests/test_vectorstore.py`, `tests/test_retrieval.py`

**Interfaces:**
- Produces: при `int(index.ntotal) != len(meta.chunk_ids)` -> лог ERROR,
  `_index = None`, `_meta = None` (degraded); `is_available() -> False`.
- Legacy-ветка semantic_search: перед использованием проверять тот же
  инвариант (ntotal == len(chunks.jsonl)-совместимость недоступна legacy —
  вместо этого: читать index_meta.json если есть; если meta есть и
  рассогласована с ntotal -> НЕ входить в legacy, вернуть None). Если meta
  отсутствует (совсем старая установка) — legacy разрешён (это его
  единственный сценарий).

- [ ] **Step 1: Тесты** — (а) meta len=3, index ntotal=2 -> `_index is
  None`, лог ERROR; (б) corrupt meta -> semantic_search вернул None и
  результат пришёл из lexical, а НЕ из legacy (проверить отсутствие
  позиционного маппинга: чанки в другом порядке, чем позиции индекса).
  FAIL.
- [ ] **Step 2: FAIL.**
- [ ] **Step 3: Реализовать** по интерфейсам; `_load_existing_ids` —
  глотание исключения заменить: ошибка -> лог ERROR + считать существующие
  id пустыми ТОЛЬКО если meta отсутствует, иначе fail-closed (полный
  rebuild через return False из _store_matches_config пути — см. уже
  существующий механизм).
- [ ] **Step 4: PASS + полный набор.**
- [ ] **Step 5: Commit** — `fix: fail-closed при рассинхроне, legacy-гейт`.

### Task 6: Fingerprint конфигурации чанкинга и текста эмбеддинга

**Files:**
- Create: `src/doc_rag/raglib/chunking_meta.py`
- Modify: `faiss_store.py` (IndexMeta: поле `config_fingerprint: str = ""`,
  to_dict/from_dict, `init(..., config_fingerprint: str = "")`),
  `vector_index.py` (_store_matches_config, build: передавать), 
  `model_paths.py` (или chunking_meta — хелпер расчёта)
- Test: `tests/test_index_fingerprint.py` (расширить)

**Interfaces:**
- Produces: `chunking_meta.config_fingerprint(cfg, tokenizer_path) -> str` —
  sha256 от {strategy, target_tokens, overlap_tokens,
  CHUNKING_SCHEMA_VERSION, EMBEDDING_TEXT_VERSION, sha256(tokenizer.json)}.
- Константы: `CHUNKING_SCHEMA_VERSION = "2"` (1 = текущее поведение
  len//4), `EMBEDDING_TEXT_VERSION = "1"` (без префиксов). T11 инкрементит
  первую, T13 — вторую.
- Сравнение в _store_matches_config: аналогично weights_sha256 (пустой have
  -> False + WARN «index predates config fingerprint»).

- [ ] **Step 1: Тесты** — (а) индекс с fp="abc", конфиг даёт "def" ->
  build(force=False) полный rebuild; (б) легаси-мета без поля -> полный
  rebuild один раз + WARN; (в) смена tokenizer.json меняет fp при прочих
  равных. FAIL.
- [ ] **Step 2: FAIL.**
- [ ] **Step 3: Реализовать** — по интерфейсам; вызов `store.init` в build()
  расширяется.
- [ ] **Step 4: PASS.**
- [ ] **Step 5: Commit** — `fix: config_fingerprint форсирует переиндексацию
  при смене чанкинга/токенизатора/текста`.

**ВАЖНО (операционно):** текущий прод-индекс — легаси (без weights_sha256 и
config_fingerprint) -> первый же ingest/rebuild после деплоя даст полный
rebuild. Это ПОЛЕЗНО (нужна переиндексация T16), но совмещается с ней явно:
мержить Task 6 в master одновременно с готовностью к T16, не между.

### Task 7: Фильтры на FAISS + lexical + hybrid

**Files:**
- Modify: `vector_index.py:294-308` (search_with_filter),
  `faiss_store.py` (новый метод), `server/retrieval.py` (передача фильтров
  в lexical + hybrid), `pipeline.py` — нет
- Test: `tests/test_filters_faiss.py`

**Interfaces:**
- Реализация на слое VectorIndex (не FaissVectorStore — у стора нет
  is_table/section_path; они есть в chunks у retrieval):
  `VectorIndex.search_with_filter(query, top_k, filters, chunks) -> tuple[
  ndarray, list[str], list[dict]]` — 3-tuple (существующий контракт,
  `retrieval.py:551` распаковывает три значения).
- Поддерживаемые ключи: `doc_id` (str), `doc_ids` (list), `section_path`
  (prefix match), `is_table` (bool; MCP-ключ `tables_only` маппится в
  `is_table=True` в search_tool.py:80-88 — там уже так).
- Алгоритм: metadata pre-filter по chunks (doc_id из chunk_id split(":")[0],
  section_path, is_table) -> разрешённые chunk_ids -> их позиции через
  `_id_to_pos` -> если позиций <= 5000: собрать подматрицу через
  `index.reconstruct_batch`-эквивалент (numpy по `index.reconstruct(pos)`)
  и точный dot-скан; иначе oversample `index.search(vec, top_k*20)` +
  пост-фильтрация. Возврат ровно top_k лучших разрешённых.
- lexical_search: параметр `filters` + pre-filter по тем же метаданным;
  hybrid_search передаёт filters в ОБОИХ канала; doc_search lexical-fallback
  (`:712`) передаёт filters.

- [ ] **Step 1: Тесты** — (а) semantic: 3 дока, filter doc_id="doc-b" ->
  только doc-b; (б) tables_only -> только is_table; (в) section_path
  prefix; (г) несуществующий doc_id -> пустой список; (д) lexical и hybrid
  с тем же фильтром; (е) фильтр + top_k больше числа разрешённых -> все
  разрешённые. FAIL.
- [ ] **Step 2: FAIL.**
- [ ] **Step 3: Реализовать** по интерфейсам.
- [ ] **Step 4: PASS.**
- [ ] **Step 5: Commit** — `fix: фильтры работают во всех режимах ретривала`.

### Task 8: Постпроцессинг единый (A7/A8)

**Files:**
- Modify: `pipeline.py:533-539` (regex), `:486` (уже есть), добавить вызовы
  в rebuild-путь и пересмотр rebuild_docs
- Test: `tests/test_postprocess_consistency.py`

- [ ] **Step 1: Тесты** — (а) одинаковый документ через ingest и rebuild ->
  идентичный текст чанка; (б) markdown-таблица в чанке сохраняет `\n`;
  (в) `re.sub(r"\n{3,}", "\n\n")` фактически работает (пустых assert-ов
  нет). FAIL.
- [ ] **Step 2: FAIL.**
- [ ] **Step 3: Реализовать** — A8: `cleaned = re.sub(r"[ \t]+", " ",
  cleaned)` затем `re.sub(r"\n{3,}", "\n\n", cleaned)` (одиночные \n
  сохраняются). A7: в rebuild вызывать `_postprocess_chunks` на полном
  объединённом наборе до `_dedup_chunks` (дубли шапок считаются по всему
  корпусу). **rebuild_docs: НЕ вызывать постпроцессинг** (он изменил бы
  чанки чужих документов при инкрементальном обновлении индекса только по
  целевым doc_id) — вместо этого пометить в докстринге rebuild_docs:
  полный ребилд требуется для консистентного постпроцессинга. Дырки в
  chunk_idx после фильтрации — приемлемы, не «чинить».
- [ ] **Step 4: PASS + полный набор.**
- [ ] **Step 5: Commit** — `fix: единый постпроцессинг, сохранить структуру`.

### Task 9: Мелкие integrity-фиксы (A6-hост, A12, C3)

**Files:**
- Modify: `vector_index.py:246` (set out of loop), 
  `mcp_http.py:1341-1415` (checks["faiss_index"]=False в semantic-режиме ->
  ready=False, reason `index_missing`), `_load_existing_ids` — уже покрыт
  Task 5
- Test: `tests/test_startup_model.py` (расширить)

- [ ] **Step 1: Тесты** — semantic-режим без faiss.index -> ready=False,
  "index_missing" в reasons; lexical-режим -> не гейтится. FAIL.
- [ ] **Step 2: FAIL.**
- [ ] **Step 3: Реализовать** оба фикса.
- [ ] **Step 4: PASS.**
- [ ] **Step 5: Commit** — `fix: readiness без индекса, мелкий перф`.
  (C9 re-embed-offset — сознательно отложена; RF-пункт из v1 плана снят.)

---

## Фаза 2 — Eval-харнесс

### Task 10: Eval-харнесс ретривала

**Files:**
- Create: `scripts/eval_retrieval.py`, `eval/queries_golden.yaml`
- Test: `tests/test_eval_retrieval.py`

**Interfaces:**
- CLI: `python scripts/eval_retrieval.py [--root <DOC_RAG_ROOT>]
  [--queries eval/queries_golden.yaml] [--modes semantic,hybrid,lexical]`
  — root передаётся через env DOC_RAG_ROOT до вызова retrieval-функций
  (doc_search() сам грузит конфиг без аргументов; для --modes вызывать
  hybrid_search/semantic_search/lexical_search напрямую, не doc_search).
- Queries YAML: `{query, relevant_doc_ids: [...], relevant_chunk_ids: (опц.)}`.
- Метрики: recall@10, MRR@10, nDCG@10 по doc_id и chunk_id; вывод таблица +
  JSON в `build/eval/<ts>.json`. Заготовка queries: 10 запросов по реальному
  манифесту (30 русских кросс-языковых + 30 английских + 20 кодовых
  разметит оператор — в плане только структура).

- [ ] **Step 1: Тест** — синтетический корпус (3 дока, DOC_RAG_ROOT в tmp)
  + золотой файл -> recall@10 == 1.0; nDCG сверить с ручным расчётом. FAIL.
- [ ] **Step 2: FAIL.**
- [ ] **Step 3: Реализовать.**
- [ ] **Step 4: PASS + baseline на реальном корпусе** (текущий чанкер,
  текущие векторы) — JSON в build/eval/baseline-*.json.
- [ ] **Step 5: Commit** — `feat: eval-харнесс ретривала (recall/MRR/nDCG)`.

---

## Фаза 3 — Алгоритмические улучшения

### Task 11: Настоящий токенизатор (и таблицы — см. T12)

**Files:**
- Create: `src/doc_rag/raglib/chunking_meta.py` уже создан (T6) — сюда
  `count_tokens`
- Modify: `chunker_recursive.py` (_estimate_tokens, _chunk_node:231-250,
  _chunk_node flush), `pipeline.py:_chunk_text (100-119)`
- Test: `tests/test_chunker_tokens.py`

**Interfaces:**
- Produces: `count_tokens(text: str, tokenizer_path: str | None = None) ->
  int` — tokenizers.Tokenizer.from_file, ленивый синглтон под threading.Lock;
  путь через `model_paths.resolve_model_path(cfg)` + `/tokenizer.json`
  (учитывает model_path/build/models/models — НЕ хардкод models/bge-m3);
  файла нет -> fallback `max(1, len(text)//4)` + WARN один раз.
- Жёсткий cap: 504 токена (512 минус спец-токены и запас) — applies to ВСЕХ
  чанков включая табличные (T12 делит таблицы по строкам).

- [ ] **Step 1: Тесты** — (а) фикстура: известный текст (зафиксировать
  литерально в тесте, например 2048 символов кириллической технической
  фразы) -> count_tokens == конкретное число (вычислить в тесте через тот
  же токенизатор, не хардкодить 600 — контракт: count_tokens совпадает с
  прямым вызовом tokenizer.encode и отличается от len//4); (б) чанкер не
  выдаёт чанков > 504 реальных токенов на корпусе из таблицы 14233 токена;
  (в) fallback при отсутствии tokenizer.json. FAIL.
- [ ] **Step 2: FAIL.**
- [ ] **Step 3: Реализовать.**
- [ ] **Step 4: PASS + статистика на реальных чанках** (>512 упало с 24.5%
  до ~0). Инкремент `CHUNKING_SCHEMA_VERSION = "3"`.
- [ ] **Step 5: Commit** — `fix: чанкер считает настоящие токены bge-m3`.

### Task 12: Таблицы — без summary, с делением по строкам

**Files:**
- Modify: `chunker_recursive.py:92-100,161-182` + деление таблиц
- Test: `tests/test_chunker_tables.py`

**Interfaces:**
- Таблица > 504 токенов -> несколько чанков, каждый начинается с
  `section_path` + header-строки таблицы (первая строка markdown-таблицы),
  затем строки таблицы; границы по строкам `\n`, не посреди строки.
- Summary-чанки (`is_table_summary`) удаляются entirely (генерация
  `:92-100` и вставка `:174-182`).
- Инкремент `CHUNKING_SCHEMA_VERSION = "4"` (вместе с T11 или после —
  оба в одной переиндексации).

- [ ] **Step 1: Тесты** — (а) таблица <= 504 -> ОДИН чанк с header +
  section_path в тексте, без summary; (б) таблица 14233 токена -> несколько
  чанков, каждый <= 504 токенов, каждый начинается с header; (в) ни одного
  is_table_summary в выдаче. FAIL.
- [ ] **Step 2: FAIL.**
- [ ] **Step 3: Реализовать.**
- [ ] **Step 4: PASS.**
- [ ] **Step 5: Commit** — `fix: таблицы делятся по строкам, summary убраны`.

### Task 13: Контекстные префиксы заголовков

**Files:**
- Create: helper `doc_rag/raglib/embedding_text.py`
- Modify: `vector_index.py` (build: текст = embedding_text(chunk)), 
  `server/retrieval.py` (выдача — чистый text без изменений), 
  `search_tool.py` — нет
- Test: `tests/test_context_prefix.py`

**Interfaces:**
- Produces: `embedding_text.build(chunk: dict, title_hint: str | None) ->
  str` — `"{title_hint} > {section_path}\n\n{text}"`; без section_path —
  plain text; бюджет: префикс (title_hint + section_path) обрезается до 40
  count_tokens (Task 11), затем EMBEDDING_TEXT_VERSION.
- Применяется ТОЛЬКО на стороне build() (эмбеддинг-текст документов
  корпуса). search() эмбеддит запрос как есть (префикс запросу не нужен).
- title_hint: карта doc_id -> title из манифеста; владелец — retrieval.py
  (кэш по mtime манифеста рядом с _MANIFEST_CACHE), передаётся в build()
  параметром `title_hints: dict[str, str] | None = None` (build получает
  его из pipeline, pipeline — из `_manifest_shell`/manifest). Очистка:
  title_hint strip + обрезка 80 символов; мусорные имена файлов
  (`rfc1928.errata`) — как есть (операторская разметка потом улучшит).
- Инкремент `EMBEDDING_TEXT_VERSION = "2"`.
- Выдача результатов (doc_search/semantic_search) НЕ меняется: LLM получает
  чистый text.

- [ ] **Step 1: Тесты** — (а) чанк с section_path -> эмбеддинг-текст с
  префиксом (проверка через подмену embedder: FakeST получает текст с
  префиксом); (б) без section_path -> как раньше; (в) префикс длиннее 40
  токенов обрезан; (г) выдача результатов содержит чистый text. FAIL.
- [ ] **Step 2: FAIL.**
- [ ] **Step 3: Реализовать.**
- [ ] **Step 4: PASS.**
- [ ] **Step 5: Commit** — `feat: префиксы заголовков в эмбеддинге чанков`.

### Task 14: BM25-индекс + RRF-веса

**Files:**
- Create: `src/doc_rag/server/lexical_index.py`
- Modify: `retrieval.py:468-521` (lexical_search через индекс),
  `:638-691` (RRF weights), `doc_search` fallback `:712` (filters)
- Test: `tests/test_lexical_index.py`

**Interfaces:**
- Produces: `LexicalIndex.build(chunks) -> LexicalIndex`: postings term ->
  [(chunk_idx, tf)], doc_len[], avgdl, df; BM25 k1=1.2, b=0.75; синглтон
  с mtime-инвалидацией chunks.jsonl.
- Токенизация: `_tokenize` текущий regex + стемминг-нормализатор:
  ЯВНЫЙ список суффиксов RU (порядок применения, только слова >= 5 букв):
  ["ования", "ироваться", "ированный", "ировала", "ирует", "ировал",
  "ация", "ями", "ами", "ого", "его", "ому", "ему", "ых", "ий", "ая",
  "ое", "ые", "у", "ю", "а", "я", "о", "е", "ы", "и"] — снимать ОДИН
  суффикс за проход, самый длинный совпавший. EN: без стемминга (s/es
  снятие для слов >= 4 букв: "ies"->"y", "es", "s"). Числовые/кодовые
  токены (`\d+[.-]\d+` и `\d+-\d+-\d+`) — без стемминга, бонус x1.5.
  Phrase-бонус сохранён, нормализован: `min(5.0, 2.0 + 3.0*count)`.
- search(query, top_k, filters=None) — фильтры из Task 7 (pre-filter по
  metadata map, передаваемой при build).
- RRF: `_rrf_fusion(*lists, k=60, weights: list[float] | None = None)` —
  `weights[i] / (k + rank + 1)`; hybrid_search передаёт
  `[semantic_weight, lexical_weight]` (порядок списков уже semantic-first).
- Зависимости: НИКАКИХ новых пакетов (snowball/nltk/pymorphy в .venv нет и
  не добавляем — зависимость-заморозка). Нормализатор — собственный список
  суффиксов.

- [ ] **Step 1: Тесты** — (а) BM25-свойство: длинный документ (x5 повторов
  текста) не выигрывает у короткого релевантного только за счёт длины;
  (б) «прибора»/«приборы» -> документ со словом «прибор»; (в) НЕГАТИВНЫЙ:
  «прибор» не матчит «прибadesh»-подобное слово, не имеющее общего корня
  (например «прибой» не должен матчить «прибор»); (г) кодовый токен
  «62056-5-3» ищется целиком; (д) кэш по mtime; (е) filters работают;
  (ж) RRF с weights=(2.0, 1.0) ставит semantic-совпадение выше. FAIL.
- [ ] **Step 2: FAIL.**
- [ ] **Step 3: Реализовать.**
- [ ] **Step 4: PASS + eval** (Task 10): lexical recall@10 не упал, hybrid
  латентность < 200 мс на 20k (замер в отчёте, не строгий тест).
  Инкремент ничего из chunking-констант (лексика не меняет чанки).
- [ ] **Step 5: Commit** — `feat: предвычисленный BM25-индекс (RU/EN) + RRF-веса`.

### Task 15: Serving-кэши и doc_download

**Files:**
- Modify: `retrieval.py` (id_to_chunk кэш; indexed_catalog манифест —
  расширить _MANIFEST_CACHE: хранить полный parsed manifest по mtime),
  `mcp_http.py:_doc_download_tool (892-937)`
- Test: `tests/test_retrieval.py`, `tests/test_mcp_tools.py` (если есть;
  иначе в test_retrieval)

**Interfaces:**
- id_to_chunk: кэш рядом с _CHUNKS_CACHE (тот же mtime-ключ) — dict
  строится один раз.
- indexed_catalog: _MANIFEST_CACHE расширяется `{"root","mtime","map",
  "manifest"}` — полный parsed JSON по mtime; пустые/битые -> не кэшировать.
- doc_download: лимит 25 МБ на ФАЙЛ (read_bytes до b64; payload ~33 МБ base64
  — известное следствие); резолв: сперва манифест source_file (точный путь),
  fallback — rglob по sources/ (файлы вне манифеста остаются доступными);
  ошибка превышения: `"Error: file exceeds 25 MB limit"`.

- [ ] **Step 1: Тесты** — (а) второй вызов indexed_catalog не перечитывает
  манифест (monkeypatch open -> счётчик); (б) doc_download 26 МБ ->
  понятная ошибка; (в) файл вне манифеста, но в sources/ — скачивается
  (fallback). FAIL.
- [ ] **Step 2: FAIL.**
- [ ] **Step 3: Реализовать.**
- [ ] **Step 4: PASS.**
- [ ] **Step 5: Commit** — `perf: кэши горячего пути; лимит и резолв doc_download`.

---

## Фаза 4 — Переиндексация и замер

### Task 16: Переиндексация --force и eval до/после

**Files:**
- Modify: `docs/bench-results.md` (результаты), опционально
  `pipeline.py` (staging-режим rebuild — см. Step 0)
- Test: ручная операция + eval JSON

- [ ] **Step 0: Staging-rebuild (код, если время позволяет):** rebuild
  --force сейчас сносит `docs_md/` и `chunks_jsonl/` ДО парсинга
  (`pipeline.py:893-896`) — сервис на часы без выдачи. Минимальная защита:
  парсить в `build/staging/{docs_md,chunks_jsonl}` и swap в конце
  (`os.replace` каталогов), старые файлы не удалять до успешной сборки.
  Если это расширение среза — задокументировать окно простоя и выполнять
  в maintenance-окно; выбор фиксируется в отчёте.
- [ ] **Step 1: baseline уже снят** (Task 10).
- [ ] **Step 2:** `doc-rag rebuild --force` (обязательно force: без него
  манифест пропускает все файлы и чанки останутся старыми). Реальное
  время: перепарс 195 PDF через Docling (часы) + эмбеддинг (~40-45 мин при
  7.9 чанк/с). Выполнять в maintenance-окно.
- [ ] **Step 3:** eval после (те же queries) -> сравнение до/после:
  recall/MRR/nDCG таблица в docs/bench-results.md.
- [ ] **Step 4:** рестарт сервиса, /health/ready 200, выборочная проверка
  оператором; `weights_sha256` и `config_fingerprint` в новой мете.
- [ ] **Step 5: Commit** — `docs: результаты eval после переиндексации`.

---

## Самопроверка плана

- Аудит-находки: A1(T3), A2(T2), A3(T6+T11+T12+T13 через константы),
  A4(T4), A5(T7), A6(T5), A7(T8), A8(T8), A9(T2), A10(T2), A11(T4 покрывает),
  A12(T9), A13(T16-решение/legacy гейт T5), B1(T11), B2(T13), B3(T12),
  B4(частично: B1 решает окно; recursive overlap сознательно не введён —
  префикс+токенизация дают больший эффект на ту же переиндексацию),
  B5(T14), B6(T14), C1(T14), C2(отложена: MinHash — отдельная задача),
  C3(T9), C4(не делать: flat IP при 20k — 9 мс), C5(T15), C6(отложена),
  C7(T15), C8(T3/T15 единый дизайн), C9(отложена, RF снят), D1(T1),
  D2(T3+T15), D4(T10), D6(T15), D7(T5/T16). B7/B8/B9/D3/D5 — после eval /
  отдельные задачи.
- Review Focus 1(T6+константы), 2(T2+T5), 3(T7), 4(T5), 5(T14) — каждый
  pinned к тесту.
- Порядок: T1 (один IndexMeta) -> ... -> T10 baseline ДО T11-T14 ->
  T16 --force.

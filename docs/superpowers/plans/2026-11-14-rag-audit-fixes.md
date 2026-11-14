# RAG-аудит: план исправлений и улучшений

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Закрыть баги целостности данных (A1-A13), поднять качество ретривала
(B1-B4, B6) и латентность (C1-C8) по итогам LLM/RAG-аудита; добавить
eval-харнесс как предусловие всех качественных изменений.

**Architecture:** Фаза 1 — integrity/quick-wins (без смены формата данных,
кроме chunking_fingerprint). Фаза 2 — eval-харнесс (золотые запросы + метрики).
Фаза 3 — алгоритмические улучшения: настоящий токенизатор в чанкере,
контекстные префиксы заголовков, предвычисленный BM25-индекс. Финал —
переиндексация и замер до/после на eval.

**Tech Stack:** Python 3.11+, FAISS, sentence-transformers, tokenizers 0.22
(быстрый токенизатор bge-m3 уже в образе), pytest.

**Spec:** аудит-отчёт этой сессии (полный текст — в конце плана, раздел
«Исходный аудит»). Спец-документ не создавался; решения приняты в чате.

## Global Constraints

- Тексты: ASCII + кириллица; без эмодзи; коммит-субъект <= 72 симв.
- Тесты: `.venv/bin/python -m pytest`; ruff check + format --check на
  тронутых файлах (`/home/trgv/.local/bin/ruff`).
- Формат `faiss.index` не менять (IndexFlatIP остаётся); меты-поля добавлять
  с дефолтами, старые JSON читаются.
- Все новые поля IndexMeta — в конец dataclass со значениями по умолчанию.
- Legacy FAISS fallback в semantic_search не трогаем (удаляется отдельной
  задачей 15 после проверки, что он никому не нужен).
- Никаких обращений к сети в пути запроса.
- Порядок фаз строго: 1 -> 2 -> 3 -> 4 (переиндексация) — каждая фаза
  коммитится зелёным набором тестов.

## Review Focus

1. Рассинхрон векторов и текстов при смене конфига чанкинга — молча.
   (Task 5: тест на chunking_fingerprint mismatch -> full rebuild.)
2. Обрыв процесса посреди rebuild оставляет систему без индекса или с
   рассогласованными faiss.index/index_meta.json. (Tasks 2-4: tmp+rename,
   проверка ntotal == len(chunk_ids).)
3. Фильтр doc_id возвращает чужие документы. (Task 7: тест «фильтр по одному
   из трёх документов -> только чанки этого документа».)
4. Ре-эмбеддинг после обрыва начинает всё заново и теряет посчитанные
   батчи. (Task 8: тест продолжения с offset.)
5. Русский запрос к английскому документу через lexical-канал. (Task 12:
   стемминг/нормализация, тест на словоформы «прибора/приборы».)

---

## Фаза 1 — Integrity и quick wins

### Task 1: Атомарная запись save()/манифеста/чанков

**Files:**
- Modify: `src/doc_rag/raglib/vectorstore/faiss_store.py:201-211` (save)
- Modify: `src/doc_rag/raglib/pipeline.py:828-839` (_write_chunks/_append_chunks), `:1035-1038`, `:1104-1107` (manifest writes)
- Test: `tests/test_atomic_writes.py`

**Interfaces:**
- Produces: helper `atomic_write_bytes(path: str, data: bytes) -> None` в
  `doc_rag/raglib/atomic.py` (tmp в том же каталоге + os.replace + fsync);
  используется везде.

- [ ] **Step 1: Падающие тесты** — save() не оставляет целевой файл
  повреждённым при падении faiss.write_index (monkeypatch write_index ->
  raise, проверить, что старый faiss.index на месте); манифест: то же.
- [ ] **Step 2: Запустить** — FAIL.
- [ ] **Step 3: Реализовать** — tmp+os.replace в save() (сначала index tmp,
  потом meta tmp, оба replace); манифест и чанки через helper.
- [ ] **Step 4: Тесты PASS + полный набор.**
- [ ] **Step 5: Commit** — `fix: атомарная запись индекса, манифеста и чанков`.

### Task 2: Инвалидация _VECTOR_INDEX_CACHE по mtime

**Files:**
- Modify: `src/doc_rag/server/retrieval.py:536-546`
- Test: `tests/test_retrieval.py`

**Interfaces:**
- Produces: `retrieval.invalidate_index_cache() -> None` (экспорт для
  mcp_http).

- [ ] **Step 1: Тест** — построить индекс в tmp, сделать запрос (кэш
  заполнился), перезаписать faiss.index другим набором векторов, запрос
  должен вернуть НОВЫЕ id. Сейчас FAIL.
- [ ] **Step 2: FAIL подтверждён.**
- [ ] **Step 3: Реализовать** — в кэше хранить `(mtime, size)` пары обоих
  файлов; расхождение -> пересоздать VectorIndex. Экспортировать
  invalidate_index_cache().
- [ ] **Step 4: PASS.**
- [ ] **Step 5: Commit** — `fix: поиск видит свежий индекс после rebuild`.

### Task 3: Off-by-i в faiss_store.add()

**Files:**
- Modify: `src/doc_rag/raglib/vectorstore/faiss_store.py:126-137`
- Test: `tests/test_vectorstore.py` (усилить test_reconstruct)

- [ ] **Step 1: Тест** — добавить 4 вектора, reconstruct каждого id и
  сравнить байт-в-байт с входными векторами (зеркалит стиль
  test_faiss_reconstruct). FAIL.
- [ ] **Step 2: FAIL.**
- [ ] **Step 3:** `base = len(self._meta.chunk_ids)` до цикла.
- [ ] **Step 4: PASS + весь test_vectorstore.**
- [ ] **Step 5: Commit** — `fix: позиции id_to_pos при инкрементальном add`.

### Task 4: Проверка целостности при загрузке индекса

**Files:**
- Modify: `src/doc_rag/raglib/vectorstore/faiss_store.py:_ensure_loaded`
- Test: `tests/test_vectorstore.py`

- [ ] **Step 1: Тест** — meta с chunk_ids len=3, index ntotal=2 ->
  _ensure_loaded либо fail-closed (index=None + WARN), либо лог ERROR; НЕ
  молча. FAIL.
- [ ] **Step 2: FAIL.**
- [ ] **Step 3:** после read_index сверить `int(index.ntotal) ==
  len(meta.chunk_ids)`; несовпадение -> лог ERROR, `_index = None`
  (degraded, semantic вернёт fallback).
- [ ] **Step 4: PASS.**
- [ ] **Step 5: Commit** — `fix: fail-closed при рассинхроне индекса и меты`.

### Task 5: chunking_fingerprint в IndexMeta

**Files:**
- Modify: `src/doc_rag/raglib/vectorstore/faiss_store.py` (IndexMeta + init), `src/doc_rag/raglib/vector_index.py` (_store_matches_config, build)
- Test: `tests/test_index_fingerprint.py`

**Interfaces:**
- Produces: `model_paths.chunking_fingerprint(cfg) -> str` — sha256 от
  {strategy, target_tokens, overlap_tokens, postprocess_version}.

- [ ] **Step 1: Тест** — индекс с chunking_fingerprint="abc", конфиг с "def"
  -> build(force=False) делает полный rebuild; старый JSON без поля ->
  полный rebuild один раз (WARN). FAIL.
- [ ] **Step 2: FAIL.**
- [ ] **Step 3: Реализовать** — поле в оба IndexMeta (faiss_store + при
  удалении indexer.py в Task 14 остаётся один), расчёт в model_paths,
  сравнение в _store_matches_config (аналогично weights_sha256: пустой have
  -> False + WARN).
- [ ] **Step 4: PASS.**
- [ ] **Step 5: Commit** — `fix: смена чанкинга форсирует переиндексацию`.

### Task 6: Фильтры на FAISS (post-filter)

**Files:**
- Modify: `src/doc_rag/raglib/vectorstore/faiss_store.py` (add
  search_with_filter), `src/doc_rag/raglib/vector_index.py:294-308`
- Test: `tests/test_filters_faiss.py`

**Interfaces:**
- Produces: `FaissVectorStore.search_with_filter(query_vec, top_k, filters)
  -> tuple[ndarray, list[str]]`; поддерживаемые ключи фильтра: `doc_id`
  (str), `doc_ids` (list[str]), `tables_only` (bool -> is_table).

- [ ] **Step 1: Тест** — 3 документа в индексе, search_with_filter
  (doc_id="doc-b") -> только чанки doc-b. FAIL (сейчас игнор).
- [ ] **Step 2: FAIL.**
- [ ] **Step 3: Реализовать** — oversample: top_k*20 пост-фильтрация по
  `_id_to_pos`/`chunk_ids` + метаданным `doc_id` из chunk_id (split(":")[0])
  и is_table из chunks.jsonl (нужен доступ к чанкам — передавать
  `doc_id -> [chunk_ids]` или читать из meta chunk_ids префиксов); при
  недостатке результатов после фильтра — честный пост-скан (np dot по
  подмножеству позиций через reconstruct-free путь: держать вектора? нет —
  использовать `index.search(vec, k=n)` полный скан один раз, дешево при
  20k). Топ-6 после фильтрации.
- [ ] **Step 4: PASS** — включая «фильтр по несуществующему doc -> пустой
  список, не топ всего корпуса».
- [ ] **Step 5: Commit** — `fix: doc_id/tables_only фильтры работают на FAISS`.

### Task 7: Мелкие integrity-фиксы

**Files:**
- Modify: `pipeline.py` (_postprocess: regex fix A8 + вызов в rebuild/
  rebuild_docs A7), `vector_index.py:246` (set out of loop C3),
  `vector_index.py:_load_existing_ids` (fail-closed A6),
  `mcp_http.py:1341-1415` (faiss_index в reasons для semantic A12)
- Test: `tests/test_postprocess_consistency.py`

- [ ] **Step 1: Тесты** — одинаковый чанк через ingest и rebuild даёт
  идентичный текст; битая meta при инкременте -> полный rebuild.
- [ ] **Step 2: FAIL.**
- [ ] **Step 3: Реализовать все 4 фикса** (каждый мелкий).
- [ ] **Step 4: PASS + полный набор.**
- [ ] **Step 5: Commit** — `fix: постпроцессинг единый, fail-closed, мелкие`.

---

## Фаза 2 — Eval-харнесс

### Task 8: Eval-харнесс ретривала

**Files:**
- Create: `scripts/eval_retrieval.py`
- Create: `eval/queries_golden.yaml` (заготовка: 10 запросов, размеченных
  по doc_id из манифеста — полная разметка 80+ вручную оператором)
- Test: `tests/test_eval_retrieval.py`

**Interfaces:**
- Produces: CLI `python scripts/eval_retrieval.py [--config config/config.yaml]
  [--queries eval/queries_golden.yaml] [--modes semantic,hybrid,lexical]`
  -> таблица recall@10 / MRR@10 / nDCG@10 по doc_id и chunk_id, JSON в
  `build/eval/<ts>.json`. Формат queries YAML: список
  `{query, relevant_doc_ids: [...], relevant_chunk_ids: [...] (опц.)}`.

- [ ] **Step 1: Тест** — на синтетическом корпусе (3 дока) золотой файл с
  известной релевантностью -> recall@10 == 1.0 для semantic; проверка
  расчёта nDCG на руках. FAIL.
- [ ] **Step 2: FAIL.**
- [ ] **Step 3: Реализовать** — загрузка конфига+чанков+индекса (без сети),
  запуск doc_search по каждому запросу, метрики doc-level и chunk-level,
  вывод таблицы + JSON.
- [ ] **Step 4: PASS + прогон на реальном корпусе** (модель уже локальна)
  — сохранить baseline в build/eval/.
- [ ] **Step 5: Commit** — `feat: eval-харнесс ретривала (recall/MRR/nDCG)`.

---

## Фаза 3 — Алгоритмические улучшения

### Task 9: Настоящий токенизатор в чанкере

**Files:**
- Modify: `src/doc_rag/raglib/chunker_recursive.py` (_estimate_tokens),
  `src/doc_rag/raglib/pipeline.py:100-119` (_chunk_text)
- Test: `tests/test_chunker_tokens.py`

**Interfaces:**
- Produces: `chunker_recursive.count_tokens(text: str) -> int` — быстрый
  токенизатор из `models/bge-m3/tokenizer.json` (tokenizers.Tokenizer,
  ленивый синглтон, fallback на len//4 если файла нет); жёсткий cap:
  чанк > 512 токенов (min 480 после спец-токенов) разбивается.

- [ ] **Step 1: Тест** — кириллический текст 2048 симв. -> count_tokens
  возвращает ~600 (не 512); чанкер не выдаёт чанков длиннее 512 реальных
  токенов. FAIL (сейчас ~608 проходит).
- [ ] **Step 2: FAIL.**
- [ ] **Step 3: Реализовать** — tokenizers.Tokenizer.from_file (lazy, lock),
  интеграция в _chunk_node и _chunk_text (fallback len//4).
- [ ] **Step 4: PASS + перегон статистики** на реальных чанках (доля
  >512 упала с 24.8% до ~0).
- [ ] **Step 5: Commit** — `fix: чанкер считает настоящие токены bge-m3`.

### Task 10: Контекстные префиксы заголовков

**Files:**
- Modify: `src/doc_rag/raglib/vector_index.py` (_encode → текст с
  префиксом), `src/doc_rag/server/retrieval.py` (выдача без префикса —
  чистый text)
- Test: `tests/test_context_prefix.py`

**Interfaces:**
- Produces: `vector_index._embedding_text(chunk: dict, title_hint: str) ->
  str` — `"{title_hint} > {section_path}\n\n{text}"`, если section_path
  есть; budget: префикс <= 40 токенов (обрезка с конца префикса). Поле
  `embedding_text` НЕ пишется в chunks.jsonl — префикс собирается на лету
  из section_path + doc title (манифест).

- [ ] **Step 1: Тест** — чанк с section_path «5.3 Requirements» получает
  эмбеддинг-текст с префиксом; без section_path — как раньше; префикс
  длиннее 40 токенов обрезается. FAIL.
- [ ] **Step 2: FAIL.**
- [ ] **Step 3: Реализовать** — в build() и search() единая сборка текста;
  title_hint из манифеста (doc_id -> title map, кэш).
- [ ] **Step 4: PASS.**
- [ ] **Step 5: Commit** — `feat: контекстные префиксы разделов в эмбеддинге`.

### Task 11: BM25-индекс вместо пересчёта на запрос

**Files:**
- Create: `src/doc_rag/server/lexical_index.py`
- Modify: `src/doc_rag/server/retrieval.py:468-521` (lexical_search через
  индекс), `:638-691` (RRF weights)
- Test: `tests/test_lexical_index.py`

**Interfaces:**
- Produces: `LexicalIndex.build(chunks) -> LexicalIndex` (postings: term ->
  [(chunk_idx, tf)], doc_len[], avgdl, df; k1=1.2, b=0.75; RU Snowball +
  EN Porter стемминг опционально — если snowball недоступен, лёгкий
  внутренний нормализатор окончаний для слов >5 букв), `search(query,
  top_k, filters=None)`; кэш синглтона по mtime chunks.jsonl; phrase-бонус
  сохранён, нормализован (max 5.0); числовые/кодовые токены (regex
  `\d+[.-]\d+`) без стемминга и с бонусом x1.5.
- RRF: веса lexical_weight/semantic_weight передаются в _rrf_fusion
  (w_i/(k+rank)).

- [ ] **Step 1: Тесты** — BM25-свойства: длинный документ не выигрывает
  только за счёт длины (b=0.75 нормализация); словоформы «прибора»/
  «приборы» матчат «прибор»; build по mtime кэшируется; поиск < 100 мс на
  20k чанках (маркер, не строгий тест). FAIL.
- [ ] **Step 2: FAIL.**
- [ ] **Step 3: Реализовать.**
- [ ] **Step 4: PASS + eval до/после** (Task 8 харнесс): lexical-режим не
  стал хуже по recall, hybrid быстрее.
- [ ] **Step 5: Commit** — `feat: предвычисленный BM25-индекс (RU/EN)`.

### Task 12: Убрать table-summary чанки + header-injection

**Files:**
- Modify: `src/doc_rag/raglib/chunker_recursive.py:92-100,161-182`
- Test: `tests/test_chunker_tables.py`

- [ ] **Step 1: Тест** — таблица -> один чанк (не два), текст таблицы
  содержит header первой строки + section_path. FAIL.
- [ ] **Step 2: FAIL.**
- [ ] **Step 3: Реализовать** — удалить генерацию summary; добавить
  section_path первой строкой в текст табличного чанка.
- [ ] **Step 4: PASS.**
- [ ] **Step 5: Commit** — `fix: табличные summary-чанки убраны`.

### Task 13: Serving-фиксы (C5-C8)

**Files:**
- Modify: `src/doc_rag/server/retrieval.py` (кэш id_to_chunk C5; кэш
  manifest в indexed_catalog C7; словарь VectorIndex-кэшей C8),
  `src/doc_rag/server/mcp_http.py` (doc_download лимит размера D6)
- Test: `tests/test_retrieval.py`

- [ ] **Step 1: Тесты** — второй вызов indexed_catalog не перечитывает
  манифест (mtime-кэш); doc_download > лимит -> понятная ошибка.
- [ ] **Step 2: FAIL.**
- [ ] **Step 3: Реализовать** — id_to_chunk кэш рядом с _CHUNKS_CACHE;
  manifest по mtime; VectorIndex кэш dict[(root,namespace)]; doc_download:
  лимит 25 МБ, резолв через манифест source_file.
- [ ] **Step 4: PASS.**
- [ ] **Step 5: Commit** — `perf: кэши горячего пути, лимит doc_download`.

---

## Фаза 4 — Очистка и переиндексация

### Task 14: Удалить raglib/indexer.py

**Files:**
- Delete: `src/doc_rag/raglib/indexer.py`
- Modify: докстринги-упоминания (vector_index.py:3, model_loader.py:3,
  faiss_store.py:3)

- [ ] **Step 1: Подтвердить нулевые импорты** (grep снова).
- [ ] **Step 2: Удалить + поправить докстринги.**
- [ ] **Step 3: Полный pytest PASS.**
- [ ] **Step 4: Commit** — `chore: удалить мёртвый indexer.py`.

### Task 15: Legacy fallback — решение

- [ ] **Step 1:** grep по деплой-докам: есть ли установки без index_meta.json.
- [ ] **Step 2:** если нет упоминаний — удалить legacy-ветку semantic_search
  (строки 569-635) + _SEMANTIC_CACHE, упростить до одного пути; покрыть
  тестами. Если упоминания есть — оставить, пометить deprecated в доке.
- [ ] **Step 3: PASS. Commit.**

### Task 16: Переиндексация и замер

- [ ] **Step 1: baseline уже в build/eval/** (Task 8).
- [ ] **Step 2:** `doc-rag rebuild` (фоновая джоба через UI/CLI; ~40 мин).
- [ ] **Step 3:** eval после — сравнить recall/MRR/nDCG до/после; таблицу
  результатов приложить в docs/bench-results.md.
- [ ] **Step 4:** перезапуск сервиса, /health/ready 200, выборочная проверка
  поиска оператором.
- [ ] **Step 5: Commit** — `docs: результаты eval после переиндексации`.

---

## Самопроверка плана

- Аудит-находки: A1(T2), A2(T1), A3(T5), A4(T3), A5(T6), A6(T7), A7(T7),
  A8(T7), A9(T1), A10(T1), A11(T3 покрывает), A12(T7), A13(T15), B1(T9),
  B2(T10), B3(T12), B4(T9 частично: recursive overlap сознательно отложен —
  префикс+B1 дают больший эффект; отметить в отчёте), B5(T11), B6(T11),
  C1(T11), C2(сознательно отложена: MinHash — отдельная задача, дедуп
  работает, просто медленный), C3(T7), C5(T13), C6(сознательно отложена:
  отдельный executor — риск-изменение конкурентности, требует наблюдения),
  C7(T13), C8(T13), C9(отложена: низкий приоритет), D1(T14), D2(T2+T13
  частично), D6(T13), D4(T8), D7(T15). B7/B8/B9/D3/D5/C4 — осознанно не
  включены (после eval / не нужны / отдельные задачи).
- Review Focus 1-5 покрыты тестами Tasks 5, 1/4, 6, 8(нет — Task 9
  continuation... уточнение: C9 отложена, Review Focus 4 = Task 7 fail-closed
  _load_existing_ids), 11.
- Имена согласованы: count_tokens, chunking_fingerprint,
  LexicalIndex.build/search, invalidate_index_cache.

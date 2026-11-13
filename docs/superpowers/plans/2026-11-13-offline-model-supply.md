# Офлайн-поставка модели BGE-m3: план реализации

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Модель `BAAI/bge-m3` поставляется вместе с образом, грузится при
старте сервера без обращений к HF в пути запроса, а ревизия весов сверяется
с HF по хешам раз в N дней с предупреждением оператору.

**Architecture:** Новый слой `model_paths` (поиск локальной копии и
фингерпринт весов) → `model_loader` (единый thread-safe загрузчик с
`local_files_only=True`) → все три прежних места загрузки делегируют ему.
Фингерпринт записывается в `index_meta.json`, несовпадение помечает индекс
устаревшим и блокирует тихое смешение ревизий. `model_watcher` только
сравнивает хеши весов с HF и поднимает флаг обновления; скачивания и
подмены в рантайме нет.

**Tech Stack:** Python 3.11+ (образ), FastAPI/asyncio, sentence-transformers
5.x, huggingface_hub 1.x (`HfApi.model_info(files_metadata=True)`), Docker
BuildKit, pytest.

**Spec:** `docs/superpowers/specs/2026-11-12-offline-model-supply-design.md`

## Global Constraints

- Текстовые файлы: только ASCII и кириллица; без эмодзи и спецсимволов.
- Путь обслуживания запросов не обращается к HF: загрузчик модели всегда
  `local_files_only=True`.
- Любой сетевой вызов watcher-а: `timeout=10` секунд.
- Целевой Python — 3.11 (образ); локально 3.13 — код обязан работать на
  обоих, без синтаксиса новее 3.11.
- Тесты: `.venv/bin/python -m pytest` (pytest 9.1.1), каталог `tests/`.
- Формат коммита: одна строка темы до 72 символов, пустая строка, тело не
  более двух предложений.
- `models/` не попадает в git и **не** добавляется в `.dockerignore`.
- Опциональные зависимости (torch, sentence-transformers, huggingface_hub)
  импортируются лениво или через guarded-импорт — сервер обязан
  стартовать при `--minimal` без embeddings-extra.

## Review Focus

Пять классов входа/состояния, которые спека подразумевает, но которые легко
сломать незаметно. Каждый привязан к тесту в своей задаче.

1. Каталога модели нет вообще (свежий checkout без fetch) — сервер стартует,
   семантика отдаёт явную ошибку модели, а не 504 и не падение. (Task 5)
2. HF недоступен в момент сверки — поиск и старт не блокируются, в лог
   INFO, повтор на следующем цикле. (Task 6)
3. Индекс построен другими весами (включая старый индекс без фингерпринта) —
   никакого молчаливого смешения векторов; устаревание видно в каталоге и
   `/health/ready`. (Task 4)
4. `update_check_interval_hours: 0`, отрицательное или мусорное значение —
   сверка выключается, процесс не падает. (Task 6)
5. `embeddings.model_path` указывает на не-модель (пустой или битый каталог) —
   понятная ошибка загрузчика, degraded-старт, watcher выключен. (Task 3)

---

### Task 1: Поиск локальной модели и фингерпринт весов

**Files:**
- Create: `src/doc_rag/raglib/model_paths.py`
- Test: `tests/test_model_paths.py`

**Interfaces:**
- Produces:
  - `resolve_model_path(cfg: dict[str, Any], root: str | None = None) -> Path | None` —
    порядок: `embeddings.model_path` → `<root>/build/models/bge-m3` →
    `<root>/models/bge-m3`. Каталог считается моделью, только если в нём
    есть `config.json` и `modules.json`; иначе `None`. Без `root`:
    `DOC_RAG_ROOT`, иначе `Path(__file__).resolve().parents[3]` (корень
    репозитория); если там нет `config/config.yaml`, корень неизвестен.
  - `weight_files(model_dir: Path) -> list[Path]` — `pytorch_model*.bin`,
    `*.safetensors`, `*.pt`, исключая подкаталоги `onnx/` и `imgs/`;
    отсортированы по относительному пути.
  - `fingerprint_from_pairs(pairs: Iterable[tuple[str, str]]) -> str` —
    нормализация и агрегат: строки `<sha256>  <relpath>` сортируются,
    `sha256` от их конкатенации с `\n`. Единая функция для локальной и
    удалённой стороны (Task 6 использует её же).
  - `write_weights_sha256(model_dir: Path) -> Path` — пишет
    `WEIGHTS.sha256` в формате `sha256sum` (отсортировано).
  - `model_fingerprint(model_dir: Path) -> str | None` — если
    `WEIGHTS.sha256` есть — агрегат от его строк; если нет — агрегат,
    вычисленный по фактическим весам (`weight_files`) и записанный в
    `WEIGHTS.sha256` (миграция для `model_path`-деплоев). `None`, только
    если весовых файлов нет вовсе.
  - `fingerprint_for_cfg(cfg: dict[str, Any]) -> str | None` — резолв пути
    + `model_fingerprint`.

- [ ] **Step 1: Написать падающие тесты**

```python
from doc_rag.raglib.model_paths import (
    fingerprint_from_pairs, model_fingerprint, resolve_model_path,
    write_weights_sha256,
)

def make_model_dir(tmp_path):
    d = tmp_path / "models" / "bge-m3"
    d.mkdir(parents=True)
    (d / "config.json").write_text("{}")
    (d / "modules.json").write_text("[]")
    (d / "pytorch_model.bin").write_bytes(b"w")
    return d

def test_resolve_prefers_model_path(tmp_path):
    custom = make_model_dir(tmp_path)
    (tmp_path / "models" / "bge-m3").rename(custom)
    cfg = {"_root": str(tmp_path), "embeddings": {"model_path": str(custom)}}
    assert resolve_model_path(cfg) == custom

def test_resolve_falls_back_to_build_then_repo(tmp_path):
    repo = make_model_dir(tmp_path)
    build = tmp_path / "build" / "models" / "bge-m3"
    build.parent.mkdir(parents=True)
    repo.rename(build)
    cfg = {"_root": str(tmp_path), "embeddings": {}}
    assert resolve_model_path(cfg) == build

def test_resolve_requires_model_files(tmp_path):
    d = tmp_path / "models" / "bge-m3"
    d.mkdir(parents=True)
    cfg = {"_root": str(tmp_path), "embeddings": {}}
    assert resolve_model_path(cfg) is None

def test_fingerprint_changes_with_weights(tmp_path):
    d = make_model_dir(tmp_path)
    (d / "onnx").mkdir()
    (d / "onnx" / "model.onnx_data").write_bytes(b"ignored")
    fp1 = model_fingerprint(d)
    (d / "pytorch_model.bin").write_bytes(b"changed")
    fp2 = model_fingerprint(d)
    assert fp1 != fp2
    assert "onnx" not in (d / "WEIGHTS.sha256").read_text()

def test_model_fingerprint_migrates_missing_file(tmp_path):
    d = make_model_dir(tmp_path)
    assert model_fingerprint(d) is not None
    assert (d / "WEIGHTS.sha256").is_file()

def test_fingerprint_from_pairs_is_order_insensitive():
    pairs = [("b" * 64, "b.bin"), ("a" * 64, "a.bin")]
    assert fingerprint_from_pairs(pairs) == fingerprint_from_pairs(reversed(pairs))
```

- [ ] **Step 2: Запустить тесты — убедиться, что падают**

Run: `.venv/bin/python -m pytest tests/test_model_paths.py -v`
Expected: FAIL, `ModuleNotFoundError: doc_rag.raglib.model_paths`.

- [ ] **Step 3: Реализовать `model_paths.py`**

Модуль без импорта torch/sentence-transformers — его используют и сервер,
и скрипт скачивания. Использует только stdlib (`hashlib`, `pathlib`, `os`).

- [ ] **Step 4: Запустить тесты — убедиться, что проходят**

Run: `.venv/bin/python -m pytest tests/test_model_paths.py -v`
Expected: PASS.

- [ ] **Step 5: Коммит**

```bash
git add src/doc_rag/raglib/model_paths.py tests/test_model_paths.py
git commit -m "feat(raglib): поиск локальной модели и фингерпринт весов"
```

---

### Task 2: Скрипт скачивания модели

**Files:**
- Create: `scripts/fetch_model.py`
- Modify: `.gitignore` (секция «Generated artifacts», строка 11)
- Test: `tests/test_fetch_model.py`

**Interfaces:**
- Consumes: `model_paths.write_weights_sha256`, `model_paths.model_fingerprint`.
- Produces:
  - `download_model(model: str, dest: Path, *, force: bool = False) -> dict[str, Any]` —
    качает snapshot в `<dest>.staging`, валидирует `config.json` +
    `modules.json`, пишет `WEIGHTS.sha256`, удаляет служебный
    `.cache/huggingface` внутри snapshot-а, затем: если `<dest>` существует —
    переименовывает в `<dest>.old`, `os.replace(staging, dest)`, удаляет
    `.old`. Возвращает `{"dest": str, "fingerprint": str, "weight_files": int}`.
    При любой ошибке staging удаляется в `finally`.
  - CLI `python scripts/fetch_model.py [--model BAAI/bge-m3] [--dest models/bge-m3] [--force]`,
    код 0/1. Перед скачиванием:
    `os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")` (таймаут чтения
    из спеки; общий бюджет не ограничен).

- [ ] **Step 1: Написать падающие тесты**

Тесты загружают скрипт через
`importlib.util.spec_from_file_location("fetch_model", ROOT/"scripts/fetch_model.py")`
(каталог `scripts/` не пакет, `sys.path` не правится).

```python
def load_fetch_model():
    spec = importlib.util.spec_from_file_location(
        "fetch_model", Path(__file__).resolve().parents[1] / "scripts" / "fetch_model.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def test_download_uses_staging_and_writes_fingerprint(tmp_path, monkeypatch):
    mod = load_fetch_model()
    def fake_snapshot_download(repo_id, local_dir, ignore_patterns=None, **kw):
        p = Path(local_dir)
        p.mkdir(parents=True)
        (p / "config.json").write_text("{}")
        (p / "modules.json").write_text("[]")
        (p / "pytorch_model.bin").write_bytes(b"w")
        return str(p)
    monkeypatch.setattr(mod, "snapshot_download", fake_snapshot_download)
    info = mod.download_model("BAAI/bge-m3", tmp_path / "m")
    assert (tmp_path / "m" / "WEIGHTS.sha256").is_file()
    assert not (tmp_path / "m.staging").exists()
    assert info["weight_files"] == 1

def test_download_replaces_existing_model(tmp_path, monkeypatch):
    # второй download поверх существующей модели не падает с ENOTEMPTY
    ...

def test_download_rejects_incomplete_snapshot(tmp_path, monkeypatch):
    mod = load_fetch_model()
    def broken(repo_id, local_dir, **kw):
        Path(local_dir).mkdir(parents=True)
        return str(local_dir)
    monkeypatch.setattr(mod, "snapshot_download", broken)
    with pytest.raises(RuntimeError):
        mod.download_model("BAAI/bge-m3", tmp_path / "m")
    assert not (tmp_path / "m").exists()
    assert not (tmp_path / "m.staging").exists()
```

Отдельный тест: `ignore_patterns` из `mock.call_args` содержит `"onnx/*"` и
`"imgs/*"`.

- [ ] **Step 2: Запустить тесты — убедиться, что падают**

Run: `.venv/bin/python -m pytest tests/test_fetch_model.py -v`
Expected: FAIL, файл отсутствует.

- [ ] **Step 3: Реализовать скрипт**

`from huggingface_hub import snapshot_download` на уровне модуля (тесты
патчат `mod.snapshot_download`). `--force` удаляет staging перед
скачиванием. В `.gitignore` (секция «Generated artifacts») добавить:

```
# Model artefacts (fetch: python scripts/fetch_model.py)
models/*
!models/.gitkeep
```

и создать `models/.gitkeep`.

- [ ] **Step 4: Запустить тесты и проверить CLI**

Run: `.venv/bin/python -m pytest tests/test_fetch_model.py -v`
Expected: PASS.
Run: `.venv/bin/python scripts/fetch_model.py --help`
Expected: usage с `--model`, `--dest`, `--force`, код 0.

- [ ] **Step 5: Коммит**

```bash
git add scripts/fetch_model.py tests/test_fetch_model.py .gitignore models/.gitkeep
git commit -m "feat: скрипт скачивания модели bge-m3 в models/"
```

---

### Task 3: Единый загрузчик модели вместо трёх копий

**Files:**
- Create: `src/doc_rag/raglib/model_loader.py`
- Modify: `src/doc_rag/raglib/vector_index.py:21-49` (`_load_embedder` делегирует)
- Modify: `src/doc_rag/raglib/indexer.py:96-119` (`_load_embedder` делегирует)
- Modify: `src/doc_rag/server/retrieval.py:497-559` (legacy-путь; условие кэша
  `529-535` больше не проверяет `embedder`)
- Test: `tests/test_model_loader.py`

**Interfaces:**
- Consumes: `model_paths.resolve_model_path`, `model_paths.fingerprint_for_cfg`.
- Produces:
  - Модульный guarded-импорт:
    `try: from sentence_transformers import SentenceTransformer
    except ImportError: SentenceTransformer = None` (тесты патчат
    `doc_rag.raglib.model_loader.SentenceTransformer`).
  - `load_embedder(cfg: dict[str, Any]) -> Any | None` — единственная точка
    создания модели. `threading.Lock` вокруг проверки и загрузки
    (single-flight). Ключ кэша — `(str(model_dir) или model_name, device)`.
    Локальный каталог: `SentenceTransformer(str(path), device=device,
    local_files_only=True)`; `max_seq_length` из конфига; ошибка на
    выбранном устройстве — повтор на `cpu`; ошибка и там — `None` + `error`.
    Fallback для существующих установок: локального каталога нет, но
    `model_name` есть в HF-кэше — `SentenceTransformer(model_name,
    local_files_only=True)`; `path=None`, статус `"source": "hf_cache"`.
  - `current_embedder() -> Any | None`, `status() -> dict[str, Any]`
    (`{"ready", "path", "fingerprint", "error", "source"}`),
    `reset_for_tests() -> None`.

- [ ] **Step 1: Написать падающие тесты**

```python
from doc_rag.raglib import model_loader
from doc_rag.raglib.model_loader import load_embedder, status, reset_for_tests

def test_load_uses_local_files_only(tmp_path, monkeypatch):
    d = make_model_dir(tmp_path)  # хелпер из tests/test_model_paths.py
    calls = []
    class FakeST:
        def __init__(self, path, device=None, local_files_only=False, **kw):
            calls.append({"path": path, "local_files_only": local_files_only})
            self.max_seq_length = 0
    monkeypatch.setattr(model_loader, "SentenceTransformer", FakeST)
    cfg = {"_root": str(tmp_path), "embeddings": {}}
    assert load_embedder(cfg) is not None
    assert calls == [{"path": str(d), "local_files_only": True}]

def test_single_flight_loads_once(tmp_path, monkeypatch):
    # 8 потоков вызывают load_embedder; конструктор вызван ровно 1 раз
    ...

def test_missing_model_returns_none_with_error(tmp_path, monkeypatch):
    monkeypatch.setattr(model_loader, "SentenceTransformer", None)
    cfg = {"_root": str(tmp_path), "embeddings": {}}
    assert load_embedder(cfg) is None
    assert status()["ready"] is False
    assert status()["error"]

def test_bad_model_path_degrades(tmp_path, monkeypatch):
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "config.json").write_text("{}")  # нет modules.json -> не модель
    cfg = {"_root": str(tmp_path), "embeddings": {"model_path": str(bad)}}
    assert load_embedder(cfg) is None
    assert status()["error"]
```

- [ ] **Step 2: Запустить тесты — убедиться, что падают**

Run: `.venv/bin/python -m pytest tests/test_model_loader.py -v`
Expected: FAIL, модуль отсутствует.

- [ ] **Step 3: Реализовать загрузчик и переключить три места**

`vector_index._load_embedder` и `indexer._load_embedder` сохраняют имена и
сигнатуры (17 тестов патчат их по строковому пути) и делегируют в
`model_loader.load_embedder(cfg)`. В `retrieval.semantic_search` legacy-ветка
заменяет собственный `SentenceTransformer` на `model_loader.load_embedder(cfg)`;
из условия кэша на строках 529-535 убирается проверка
`_SEMANTIC_CACHE.get("embedder") is not None` — иначе `faiss.read_index`
будет вызываться на каждый запрос. `migrate_vectors.py` работает через
`vector_index._load_embedder` без изменений.

- [ ] **Step 4: Запустить тесты**

Run: `.venv/bin/python -m pytest tests/test_model_loader.py tests/test_vector_index.py tests/test_dedup.py -v`
Expected: PASS, включая все существующие monkeypatch-тесты.

- [ ] **Step 5: Коммит**

```bash
git add src/doc_rag/raglib/model_loader.py src/doc_rag/raglib/vector_index.py \
  src/doc_rag/raglib/indexer.py src/doc_rag/server/retrieval.py tests/test_model_loader.py
git commit -m "refactor: единый загрузчик модели с local_files_only"
```

---

### Task 4: Фингерпринт в мете индекса и защита от смешения ревизий

**Files:**
- Modify: `src/doc_rag/raglib/indexer.py:20-50` (`IndexMeta`)
- Modify: `src/doc_rag/raglib/vectorstore/faiss_store.py:21-49` (`IndexMeta`),
  `:100-116` (`init`)
- Modify: `src/doc_rag/raglib/vectorstore/pgvector_store.py:67`, `qdrant_store.py:98`
  (`init` — принять и сохранить параметр)
- Modify: `src/doc_rag/raglib/vector_index.py:98-124` (`_store_matches_config`),
  `:256` (вызов `store.init`)
- Modify: `src/doc_rag/server/retrieval.py:194-284` (`indexed_catalog`)
- Test: `tests/test_index_fingerprint.py`

**Interfaces:**
- Consumes: `model_paths.fingerprint_for_cfg`, `model_loader.status`.
- Produces:
  - Поле `weights_sha256: str = ""` в обеих `IndexMeta`, включено в
    `to_dict`/`from_dict` (старый JSON без поля — пустая строка).
  - Механизм передачи: `init(self, chunks, dims, *, weights_sha256: str = "", ...)`
    у всех трёх бэкендов; `VectorIndex.build` вызывает
    `store.init(..., weights_sha256=model_paths.fingerprint_for_cfg(self._cfg) or "")`.
    `IndexMeta` бэкенда сохраняет это поле в `index_meta.json`.
  - `_store_matches_config`: непустые `want` и `have` и их неравенство →
    `False`; пустой `have` (старый индекс) → тоже `False` + WARN «index has
    no weights fingerprint: full rebuild required», т.е. первый ingest после
    апгрейда один раз переиндексирует корпус, а не смешает ревизии.
  - `indexed_catalog()` добавляет: `model_ready` (bool), `weights_sha256`
    (str | None), `index_stale` (bool), `stale_reason` (str | None),
    `model_update_available` (bool; до Task 6 — `False`).
    `index_stale` = оба фингерпринта известны и различаются;
    `stale_reason = "weights fingerprint mismatch"`.

- [ ] **Step 1: Написать падающие тесты**

```python
from doc_rag.raglib.vectorstore.faiss_store import IndexMeta

def test_meta_roundtrip_keeps_fingerprint():
    meta = IndexMeta(protocol_version="1", model_name="BAAI/bge-m3",
                     metric="ip", normalize=True, dim=1024,
                     chunk_ids=[], weights_sha256="abc")
    assert IndexMeta.from_dict(meta.to_dict()).weights_sha256 == "abc"

def test_old_meta_without_field_loads_empty():
    data = {"protocol_version": "1", "model_name": "BAAI/bge-m3",
            "metric": "ip", "normalize": True, "dim": 1024, "chunk_ids": []}
    assert IndexMeta.from_dict(data).weights_sha256 == ""

def test_incremental_refused_on_fingerprint_mismatch(tmp_path, monkeypatch):
    # индекс построен с weights_sha256="old", фингерпринт конфига "new"
    # build(force=False) переиндексирует всё, а не дополняет
    ...

def test_incremental_refused_on_legacy_meta(tmp_path, monkeypatch):
    # индекс без фингерпринта -> полный rebuild, WARN в лог
    ...

def test_catalog_marks_stale_on_fingerprint_mismatch(tmp_path, monkeypatch):
    cat = indexed_catalog()
    assert cat["index_stale"] is True
    assert cat["stale_reason"] == "weights fingerprint mismatch"
    assert cat["model_ready"] is True
```

- [ ] **Step 2: Запустить тесты — убедиться, что падают**

Run: `.venv/bin/python -m pytest tests/test_index_fingerprint.py -v`
Expected: FAIL (`weights_sha256` отсутствует, `index_stale` в каталоге нет).

- [ ] **Step 3: Реализовать**

Поле — в конец датаклассов со значением по умолчанию `""` (позиционные
вызовы не ломаются). Бэкенды принимают `weights_sha256` в `init` и пишут его
в мету. `indexed_catalog()` читает `index_meta.json` напрямую (не через
дорогой разбор манифеста).

- [ ] **Step 4: Запустить тесты**

Run: `.venv/bin/python -m pytest tests/test_index_fingerprint.py tests/test_vector_index.py tests/test_ingest.py -v`
Expected: PASS.

- [ ] **Step 5: Коммит**

```bash
git add src/doc_rag/raglib/indexer.py src/doc_rag/raglib/vectorstore/faiss_store.py \
  src/doc_rag/raglib/vectorstore/pgvector_store.py src/doc_rag/raglib/vectorstore/qdrant_store.py \
  src/doc_rag/raglib/vector_index.py src/doc_rag/server/retrieval.py tests/test_index_fingerprint.py
git commit -m "fix(index): фингерпринт весов защищает индекс от смешения ревизий"
```

---

### Task 5: Загрузка модели при старте, readiness и диагностика

**Files:**
- Modify: `src/doc_rag/server/mcp_http.py:1171` (создание `app`),
  `1262-1302` (`_readiness_state`), `1485-1503` (`_FIELD_CONSTRAINTS`)
- Modify: `src/doc_rag/server/search_tool.py:9-14` (тексты и выбор)
- Modify: `tests/test_health_metrics.py:39-52`, `tests/test_degraded_mode.py:73-78`
  (ожидания меняются вместе с поведением)
- Modify: `tests/conftest.py` — autouse-фикстура сброса глобального состояния
- Test: `tests/test_startup_model.py`

**Interfaces:**
- Consumes: `model_loader.load_embedder`, `model_loader.status`,
  `retrieval.indexed_catalog`.
- Produces:
  - `app = FastAPI(..., lifespan=_lifespan)`; `_lifespan` вызывает
    `await asyncio.to_thread(load_embedder, load_config())` до `yield`;
    при `None` — ERROR с инструкцией, работа продолжается.
  - `_readiness_state()`: `checks["model_ready"]`; `ready` дополнительно
    требует `model_ready`, если `mcp.retrieval_mode in ("semantic", "hybrid")`;
    `reasons` пополняется `model_missing` и `index_stale`. `index_stale`
    определяется чтением `build/index/index_meta.json` напрямую (без
    `indexed_catalog()`, чтобы probe оставался дешёвым). Порог
    `checks["disk_space"]` остаётся 1 GiB — рантайм ничего не качает.
  - `search_tool`: три текста — `_FALLBACK_NOTICE` (индекса нет),
    `_MODEL_NOTICE` (модель не найдена, команда `scripts/fetch_model.py`),
    `_STALE_NOTICE` (веса изменились: `scripts/fetch_model.py`, рестарт,
    `doc-rag rebuild`). Выбор по `cat["model_ready"]` / `cat["index_stale"]`
    / наличию индекса.
  - `_FIELD_CONSTRAINTS`: только `embeddings.update_check_interval_hours`
    (`{"type": "int", "min": 0}`). `embeddings.model_path` в UI-форму
    **не выводится**: `ui_config_patch` отвергает отсутствующие в конфиге
    ключи, а ключ закомментирован.

- [ ] **Step 1: Написать падающие тесты**

`_client()` в существующих тестах создаёт `TestClient(app)` без `with`,
поэтому lifespan не исполняется; новые тесты используют
`with TestClient(app) as client:` либо патчат `model_loader.status`.

```python
def test_readiness_fails_without_model(tmp_corpus_root, monkeypatch):
    # manifest и faiss.index есть, model_loader.status -> ready=False
    state = _readiness_state()
    assert state["ready"] is False
    assert "model_missing" in state["reasons"]

def test_readiness_ok_with_model(tmp_corpus_root, monkeypatch):
    monkeypatch.setattr(model_loader, "status",
                        lambda: {"ready": True, "error": None, "path": "/m",
                                 "fingerprint": "fp", "source": "local"})
    assert _readiness_state()["ready"] is True

def test_readiness_ok_without_model_in_lexical_mode(tmp_corpus_root, monkeypatch):
    # mcp.retrieval_mode == "bm25" -> модель не требуется
    ...

def test_stale_notice_mentions_rebuild(...):
    assert "rebuild" in _STALE_NOTICE and "fetch_model.py" in _STALE_NOTICE
```

Существующие тесты правятся:
`test_health_metrics.py:39-52` — патч `model_loader.status` на готовую
модель (иначе гейт даст 503); `test_degraded_mode.py:73-78` — ожидаемый
текст `_MODEL_NOTICE` вместо `_FALLBACK_NOTICE`.

- [ ] **Step 2: Запустить тесты — убедиться, что падают**

Run: `.venv/bin/python -m pytest tests/test_startup_model.py -v`
Expected: FAIL.

- [ ] **Step 3: Реализовать lifespan, readiness и тексты**

lifespan — `@contextlib.asynccontextmanager` рядом с созданием `app`; внутри
только загрузчик (сеть не используется, зависнуть на HF невозможно).

- [ ] **Step 4: Запустить тесты**

Run: `.venv/bin/python -m pytest tests/test_startup_model.py tests/test_health_metrics.py tests/test_degraded_mode.py -v`
Expected: PASS.

- [ ] **Step 5: Коммит**

```bash
git add src/doc_rag/server/mcp_http.py src/doc_rag/server/search_tool.py \
  tests/test_startup_model.py tests/test_health_metrics.py tests/test_degraded_mode.py \
  tests/conftest.py
git commit -m "feat(server): загрузка модели при старте и честный readiness"
```

---

### Task 6: Фоновая сверка весов с HF Hub

**Files:**
- Create: `src/doc_rag/server/model_watcher.py`
- Modify: `src/doc_rag/server/mcp_http.py` (запуск/остановка задачи)
- Modify: `src/doc_rag/server/retrieval.py` (`model_update_available`)
- Modify: `config/config.yaml:52-61` (секция `embeddings`)
- Test: `tests/test_model_watcher.py`

**Interfaces:**
- Consumes: `model_paths.fingerprint_from_pairs`, `model_fingerprint`,
  `resolve_model_path`, `model_loader.status`.
- Produces:
  - `async def check_once(cfg, *, api=None) -> str | None` — `None`, если
    изменений нет, сверка невозможна или ошибка сети. `api` — инъекция;
    по умолчанию `HfApi()` импортируется **лениво** внутри функции
    (`--minimal` без huggingface_hub не должен ломать импорт сервера).
    `api.model_info(repo_id, files_metadata=True, timeout=10)`; из
    `siblings` берутся только записи с `lfs.sha256` (у не-LFS файлов,
    например `*.pt`, хеш содержимого без скачивания недоступен — они не
    участвуют, чтобы не давать вечный ложный «update available»);
    удалённый агрегат — `fingerprint_from_pairs`.
  - `def start(cfg) -> asyncio.Task | None` — `None`, если интервал `0`,
    задан `embedding_path`, модель не резолвится или `huggingface_hub`
    недоступен. Иначе `asyncio.create_task`: проверка сразу, затем
    `asyncio.sleep(interval)`. Задача начинает исполняться после первого
    `await` event loop, т.е. после завершения startup.
  - `def stop() -> None` — отмена задачи.
  - `def state() -> dict[str, Any]` — `{"update_available", "checked_at", "error"}`.
  - Конфиг: `update_check_interval_hours: 72`; `model_path` остаётся
    закомментированным (в UI не выводится).
  - `indexed_catalog()`: `"model_update_available": model_watcher.state()["update_available"]`.

- [ ] **Step 1: Написать падающие тесты**

```python
import asyncio
from doc_rag.server import model_watcher

def fake_api(hash_by_name):
    class Sib:
        def __init__(self, name, lfs): self.rfilename, self.lfs = name, lfs
    class Info:
        def __init__(self): self.siblings = [...]
    class Api:
        def model_info(self, *a, **kw): return Info()
    return Api()

def test_check_reports_update_when_weight_hash_differs(...):
    assert asyncio.run(model_watcher.check_once(cfg, api=fake_api(...))) == "newhash"

def test_check_none_when_hashes_match(...): ...
def test_hf_error_is_swallowed(...):
    assert asyncio.run(model_watcher.check_once(cfg, api=raising_api())) is None
    assert model_watcher.state()["error"]
def test_interval_zero_disables(...):
    assert model_watcher.start(cfg0) is None
def test_model_path_disables_watcher(...): ...
def test_interval_parsing_tolerates_garbage(...):
    # "abc" и -5 -> выключено, без исключения
```

- [ ] **Step 2: Запустить тесты — убедиться, что падают**

Run: `.venv/bin/python -m pytest tests/test_model_watcher.py -v`
Expected: FAIL.

- [ ] **Step 3: Реализовать модуль и подключить в lifespan**

`start(cfg)` — сразу после `load_embedder`; `stop()` — после `yield`.
Логи: INFO при отсутствии изменений/сети, WARN при найденном обновлении.

- [ ] **Step 4: Запустить тесты**

Run: `.venv/bin/python -m pytest tests/test_model_watcher.py tests/test_startup_model.py -v`
Expected: PASS.

- [ ] **Step 5: Коммит**

```bash
git add src/doc_rag/server/model_watcher.py src/doc_rag/server/mcp_http.py \
  src/doc_rag/server/retrieval.py config/config.yaml tests/test_model_watcher.py
git commit -m "feat(server): фоновая сверка хешей весов с HF Hub"
```

---

### Task 7a: Образ и CI

**Files:**
- Modify: `docker/Dockerfile` (после `ENV PATH`; `COPY . /app` строка 45;
  `uv sync` строки 41, 52)
- Modify: `docker-compose.yml:3-5` (`build.args`)
- Modify: `.github/workflows/build.yml:24-39`
- Test: ручная проверка сборки

**Interfaces:**
- Produces:
  - Dockerfile: `ARG REQUIRE_MODEL=0`; `ENV HF_HOME=/app/build/hf-cache`;
    `COPY --chown=docrag:docrag models/ /app/models/` ранним слоем (до
    `COPY pyproject.toml`); после `uv sync` — `RUN python -c "import
    sentence_transformers"` при `REQUIRE_MODEL=1` и два extras:
    `uv sync --frozen --extra server --extra faiss --extra embeddings`
    (иначе `load_embedder` вернёт `None` и Docker-деплой даст 503);
    проверка `RUN test "$REQUIRE_MODEL" = "0" || test -f /app/models/bge-m3/config.json || (echo "[doc-rag] models/bge-m3 missing: run scripts/fetch_model.py" && exit 1)`;
    финальный `COPY --exclude=models . /app` (BuildKit: `dockerfile:1` +
    `setup-buildx-action@v3`).
  - `docker-compose.yml`: `build.args.REQUIRE_MODEL: "1"`.
  - CI: `build-args: REQUIRE_MODEL=0`.

- [ ] **Step 1: Внести правки Dockerfile, compose и CI**

Порядок слоёв: модель отдельным слоем до кода — при правках кода 2.3 ГБ
слой не пересобирается. `--chown` на COPY модели, чтобы `RUN chown -R /app`
не создавал вторую копию через copy-up.

- [ ] **Step 2: Проверить сборку без модели (CI-сценарий)**

Run: `docker build --build-arg REQUIRE_MODEL=0 -f docker/Dockerfile -t doc-rag:plan-test .`
Expected: успешная сборка.

- [ ] **Step 3: Проверить fail-fast с REQUIRE_MODEL=1**

Run: `docker build --build-arg REQUIRE_MODEL=1 -f docker/Dockerfile -t doc-rag:plan-test .`
Expected: сборка падает с `[doc-rag] models/bge-m3 missing`.

- [ ] **Step 4: Проверить сборку с моделью**

Run: `python scripts/fetch_model.py --dest models/bge-m3` затем
`docker build --build-arg REQUIRE_MODEL=1 -f docker/Dockerfile -t doc-rag:plan-test .`
Expected: успешная сборка; `docker run --rm doc-rag:plan-test python -c "import sentence_transformers"` — код 0.

- [ ] **Step 5: Коммит**

```bash
git add docker/Dockerfile docker-compose.yml .github/workflows/build.yml
git commit -m "build: модель в образе, extras embeddings, CI-сборка без модели"
```

---

### Task 7b: Каналы доставки модели

**Files:**
- Modify: `scripts/make_deploy_archive.sh:27-34, 42-66`
- Modify: `scripts/install_server_native.sh:96-103` (rsync)
- Modify: `scripts/bootstrap.sh`
- Test: ручная проверка архива и rsync

**Interfaces:**
- Produces:
  - `make_deploy_archive.sh`: после `git archive` копировать модель в
    staging (`cp -a "$ROOT/models/." "$staging/models/"` при наличии); при
    отсутствии — предупреждение, архив всё равно собирается; в heredoc
    `DEPLOY.txt` — строка про проверку модели.
  - `install_server_native.sh`: rsync с `--filter 'protect models/'` и
    `--exclude 'models'` (gitignored каталог иначе снесётся `--delete`);
    `models/` копируется из источника отдельно при наличии. `fetch_model.py`
    — только явный fallback с предупреждением, не автоматически.
  - `bootstrap.sh`: при отсутствии `models/bge-m3/config.json` — вызвать
    `python scripts/fetch_model.py` с сообщением.

- [ ] **Step 1: Внести правки скриптов**

- [ ] **Step 2: Проверить архив**

Run: `bash scripts/make_deploy_archive.sh` затем
`tar -tf <staging>.tar 2>/dev/null | grep -c 'models/bge-m3/'` — если
проверка через несжатый промежуточный tar недоступна, использовать
`tar -tzf` с `--occurrence`/head и не ждать полного потока.
Expected: число больше 0 при наличии модели; архив собирается и без неё.

- [ ] **Step 3: Проверить rsync-защиту**

Run: тест на временном каталоге — `rsync -a --delete --filter 'protect models/' --exclude 'models' src/ dst/`,
где в `dst/models/bge-m3` лежит файл.
Expected: файл пережил синхронизацию.

- [ ] **Step 4: Коммит**

```bash
git add scripts/make_deploy_archive.sh scripts/install_server_native.sh scripts/bootstrap.sh
git commit -m "build: доставка модели архивом и rsync без сноса models/"
```

---

### Task 8: UI и документация

**Files:**
- Modify: `ui/src/lib/types.ts` (тип каталога индекса)
- Modify: `ui/src/lib/configSchema.ts` (ключ `update_check_interval_hours`)
- Modify: `ui/src/pages/Documents.svelte:210-211` (статус индекса, баннеры)
- Modify: `docs/install.md`, `docs/deploy.md`, `docs/troubleshooting.md`,
  а также файл, где описаны ключи `embeddings` (`grep -rn "embeddings" docs/`)
- Test: `cd ui && npm run build`

**Interfaces:**
- Consumes: `model_ready`, `index_stale`, `stale_reason`,
  `model_update_available`, `weights_sha256`.
- Produces: три баннера: «модель не найдена, семантика недоступна»
  (ERROR), «индекс устарел: веса модели изменились, нужен rebuild» (WARN),
  «на HF доступна новая ревизия весов» (INFO). Ключ
  `update_check_interval_hours` — в форме настроек; `model_path` — нет.

- [ ] **Step 1: Добавить поля в типы и схему конфига**

- [ ] **Step 2: Добавить баннеры в `Documents.svelte`**

Расширить существующий блок статуса индекса, не создавая второй источник
истины.

- [ ] **Step 3: Обновить документацию**

`docs/install.md` — `scripts/fetch_model.py`; `docs/deploy.md` — доставка
модели по каналам; `docs/troubleshooting.md` — 504/модель не найдена/индекс
устарел; ключи `embeddings.*` — в соответствующем файле. ASCII + кириллица.

- [ ] **Step 4: Проверить сборку UI и тесты Python**

Run: `cd ui && npm run build`
Expected: успешная сборка, ошибок типов нет.
Run: `.venv/bin/python -m pytest -q`
Expected: весь набор PASS.

- [ ] **Step 5: Коммит**

```bash
git add ui/src/lib/types.ts ui/src/lib/configSchema.ts ui/src/pages/Documents.svelte docs/
git commit -m "feat(ui): статус модели и индекса, ключ сверки в настройках"
```

---

## Самопроверка плана

- Покрытие спеки: офлайн-поставка (Task 2, 7a, 7b), загрузка при старте
  (Task 3, 5), сверка без подмены (Task 6), параметр интервала (Task 5, 6),
  фингерпринт и BUG-006 (Task 4), degraded-режим и диагностика (Task 5, 8),
  каналы доставки (Task 7a, 7b), таймаут скачивания (Task 2), защита
  rsync (Task 7b).
- Устранены находки ревью плана: C1 (replace через `.old`), C2 (параметр
  `weights_sha256` в `init` трёх бэкендов), C3 (правка существующих тестов
  readiness/fallback), C4 (extras `faiss`/`embeddings`, `--chown`, compose
  build-args), C5 (rsync protect), I1-I14 и M1-M12.
- Консистентность имён: `resolve_model_path`, `fingerprint_from_pairs`,
  `write_weights_sha256`, `model_fingerprint`, `fingerprint_for_cfg`,
  `load_embedder`, `current_embedder`, `status`, `reset_for_tests`,
  `check_once`, `start`, `stop`, `state` — одинаково во всех задачах.
- Review Focus: каждый из пяти пунктов привязан к тесту в указанной задаче.

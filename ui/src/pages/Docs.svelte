<script lang="ts">
  import { appState } from "../lib/state.svelte";

  type Section = { id: string; title: string };
  const sections: Section[] = [
    { id: "overview", title: "Обзор сервиса" },
    { id: "architecture", title: "Архитектура" },
    { id: "config", title: "Параметры конфигурации" },
    { id: "env", title: "Переменные окружения" },
    { id: "api", title: "API (REST)" },
    { id: "mcp", title: "MCP-протокол" },
    { id: "search", title: "Режимы поиска" },
    { id: "quality", title: "Качество документов" },
  ];

  let activeSection = $state("overview");

  function scrollTo(id: string) {
    activeSection = id;
    document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" });
  }
</script>

<div class="docs-page">
<aside class="doc-nav">
  <nav>
    {#each sections as s (s.id)}
      <button
        class="doc-nav-btn"
        class:active={activeSection === s.id}
        onclick={() => scrollTo(s.id)}
      >{s.title}</button>
    {/each}
  </nav>
  <div class="doc-nav-links">
    <a href="/docs" target="_blank" rel="noopener">Swagger UI</a>
    <a href="/redoc" target="_blank" rel="noopener">ReDoc</a>
    <a href="/openapi.json" target="_blank" rel="noopener">OpenAPI JSON</a>
  </div>
</aside>

<article class="doc-body">
  <!-- ===== Обзор ===== -->
  <section id="overview">
    <h2>Обзор сервиса</h2>
    <p>
      <strong>doc-rag</strong> — локальный RAG-сервер (Retrieval-Augmented Generation)
      для инженерных документов. Индексирует PDF, DOCX, MD, TXT в поисковый индекс,
      предоставляет семантический/lexical/гибридный поиск и генерацию ответов
      с цитированием источников через MCP-протокол (для Cursor, VSCode) и REST API.
    </p>
    <p>Ключевые возможности:</p>
    <ul>
      <li><strong>Парсинг</strong> — Docling ( основной ), Unstructured ( fallback при cascade ), python-docx</li>
      <li><strong>Чанкинг</strong> — фиксированный или рекурсивный (по заголовкам и блокам)</li>
      <li><strong>Эмбеддинги</strong> — SentenceTransformer (BAAI/bge-large-en-v1.5), CPU/CUDA</li>
      <li><strong>Векторный индекс</strong> — FAISS (по умолчанию), Qdrant, pgvector</li>
      <li><strong>Гибридный поиск</strong> — RRF (Reciprocal Rank Fusion) семантика + лексика</li>
      <li><strong>RAG-генерация</strong> — OpenAI-compatible API (Ollama, llama.cpp, vLLM, OpenAI)</li>
      <li><strong>Качество</strong> — автоматические проверки документов, бейджи в UI</li>
      <li><strong>Web UI</strong> — Svelte-интерфейс с управлением, логами, конфигурацией</li>
    </ul>
  </section>

  <!-- ===== Архитектура ===== -->
  <section id="architecture">
    <h2>Архитектура</h2>
    <pre class="arch-diagram">
┌─────────────┐   ┌──────────────┐   ┌─────────────────┐
│  Источники   │──▶│   Парсеры    │──▶│  Блоки (JSONL)  │
│  (PDF/DOCX)  │   │ Docling/Unst │   │  headings/tables│
└─────────────┘   └──────────────┘   └────────┬────────┘
                                              │
                   ┌──────────────┐   ┌────────▼────────┐
                   │  Эмбеддинги  │◀──│    Чанкер       │
                   │ SentenceTr.  │   │ fixed/recursive │
                   └──────┬───────┘   └─────────────────┘
                          │
              ┌───────────▼───────────┐
              │    Векторный индекс    │
              │  FAISS / Qdrant / PG  │
              └───────────┬───────────┘
                          │
         ┌────────────────┼────────────────┐
         ▼                ▼                ▼
   ┌──────────┐   ┌────────────┐   ┌────────────┐
   │doc_search│   │doc_generate│   │  Web UI    │
   │  MCP tool│   │  MCP tool  │   │  /ui-next  │
   └──────────┘   └────────────┘   └────────────┘</pre>
    <p>Поток данных: <code>Источник → Парсер → Блоки → Чанкер → Эмбеддинги → Индекс</code>.
       При запросе: <code>Запрос → Поиск → Контекст → LLM → Ответ с цитатами</code>.</p>
  </section>

  <!-- ===== Конфигурация ===== -->
  <section id="config">
    <h2>Параметры конфигурации</h2>
    <p>Файл: <code>config/config.yaml</code>. Редактируется через
      <button class="link-btn" onclick={() => { appState.page = "config"; }}>Конфигурация</button>
      (вкладка «Параметры сервера») или напрямую.
    </p>

    <h3>Парсинг (<code>parsing</code>)</h3>
    <table class="doc-table">
      <thead><tr><th>Ключ</th><th>Тип</th><th>По умолч.</th><th>Описание</th></tr></thead>
      <tbody>
        <tr><td><code>pdf_backend</code></td><td>select</td><td><code>docling</code></td>
          <td>Backend для PDF. <code>docling</code> — основной; <code>cascade</code> — Docling с fallback на Unstructured; <code>auto</code> — совместимость.</td></tr>
        <tr><td><code>docx_backend</code></td><td>select</td><td><code>python-docx</code></td>
          <td>Backend для DOCX. <code>python-docx</code> — быстрый; <code>docling</code> — структурный, дороже.</td></tr>
        <tr><td><code>normalize_whitespace</code></td><td>bool</td><td><code>true</code></td>
          <td>Нормализация пробелов при парсинге.</td></tr>
        <tr><td><code>min_chars_per_page</code></td><td>int</td><td><code>20</code></td>
          <td>Порог «пустой страницы» для метрик покрытия.</td></tr>
      </tbody>
    </table>

    <h3>Заголовки (<code>sectioning</code>)</h3>
    <table class="doc-table">
      <thead><tr><th>Ключ</th><th>Тип</th><th>По умолч.</th><th>Описание</th></tr></thead>
      <tbody>
        <tr><td><code>enable_numbered_headings</code></td><td>bool</td><td><code>true</code></td>
          <td>Распознавание нумерованных заголовков (1.2.3 ...).</td></tr>
        <tr><td><code>enable_allcaps_headings</code></td><td>bool</td><td><code>true</code></td>
          <td>Распознавание ЗАГОЛОВКОВ КАПСОМ.</td></tr>
        <tr><td><code>min_heading_len</code></td><td>int</td><td><code>4</code></td>
          <td>Минимальная длина заголовка (символы).</td></tr>
        <tr><td><code>max_heading_len</code></td><td>int</td><td><code>120</code></td>
          <td>Максимальная длина заголовка.</td></tr>
      </tbody>
    </table>

    <h3>Чанкинг (<code>chunking</code>)</h3>
    <table class="doc-table">
      <thead><tr><th>Ключ</th><th>Тип</th><th>По умолч.</th><th>Описание</th></tr></thead>
      <tbody>
        <tr><td><code>strategy</code></td><td>select</td><td><code>fixed</code></td>
          <td><code>fixed</code> — фиксированный размер по символам; <code>recursive</code> — структурный (по заголовкам и блокам).</td></tr>
        <tr><td><code>target_tokens</code></td><td>int</td><td><code>512</code></td>
          <td>Целевой размер чанка в токенах.</td></tr>
        <tr><td><code>overlap_tokens</code></td><td>int</td><td><code>64</code></td>
          <td>Перекрытие между чанками.</td></tr>
        <tr><td><code>dedup_similarity_threshold</code></td><td>float [0..1]</td><td><code>0.85</code></td>
          <td>Порог дедупликации (word-bigram Jaccard). 0 = выключено.</td></tr>
      </tbody>
    </table>

    <h3>Эмбеддинги (<code>embeddings</code>)</h3>
    <table class="doc-table">
      <thead><tr><th>Ключ</th><th>Тип</th><th>По умолч.</th><th>Описание</th></tr></thead>
      <tbody>
        <tr><td><code>model_name</code></td><td>string</td><td><code>BAAI/bge-large-en-v1.5</code></td>
          <td>Имя модели sentence-transformers.</td></tr>
        <tr><td><code>device</code></td><td>select</td><td><code>cpu</code></td>
          <td><code>cpu</code> — без GPU; <code>cuda</code> — NVIDIA GPU.</td></tr>
        <tr><td><code>batch_size</code></td><td>int</td><td><code>32</code></td>
          <td>Размер батча для эмбеддингов.</td></tr>
        <tr><td><code>normalize</code></td><td>bool</td><td><code>true</code></td>
          <td>Нормализация векторов (рекомендуется для inner product).</td></tr>
      </tbody>
    </table>

    <h3>Индекс (<code>index</code>)</h3>
    <table class="doc-table">
      <thead><tr><th>Ключ</th><th>Тип</th><th>По умолч.</th><th>Описание</th></tr></thead>
      <tbody>
        <tr><td><code>backend</code></td><td>select</td><td><code>faiss</code></td>
          <td><code>faiss</code> — локальный; <code>qdrant</code> — HTTP-сервер; <code>pgvector</code> — PostgreSQL.</td></tr>
        <tr><td><code>metric</code></td><td>select</td><td><code>ip</code></td>
          <td><code>ip</code> — inner product (для нормализованных ≈ косинус); <code>l2</code> — евклидово.</td></tr>
        <tr><td><code>top_k</code></td><td>int</td><td><code>6</code></td>
          <td>Количество результатов по умолчанию.</td></tr>
      </tbody>
    </table>

    <h3>MCP / Поиск (<code>mcp</code>)</h3>
    <table class="doc-table">
      <thead><tr><th>Ключ</th><th>Тип</th><th>По умолч.</th><th>Описание</th></tr></thead>
      <tbody>
        <tr><td><code>retrieval_mode</code></td><td>select</td><td><code>semantic</code></td>
          <td><code>semantic</code> — векторный; <code>lexical</code> — TF-IDF; <code>hybrid</code> — комбинация (RRF).</td></tr>
        <tr><td><code>rag_generate.base_url</code></td><td>string</td><td>—</td>
          <td>URL LLM API (OpenAI-compatible). Альтернатива: <code>DOC_RAG_LLM_BASE_URL</code>.</td></tr>
        <tr><td><code>rag_generate.model</code></td><td>string</td><td>—</td>
          <td>Имя модели LLM. Альтернатива: <code>DOC_RAG_LLM_MODEL</code>.</td></tr>
        <tr><td><code>rag_generate.max_tokens</code></td><td>int</td><td><code>1024</code></td>
          <td>Максимум токенов в ответе LLM.</td></tr>
        <tr><td><code>rag_generate.temperature</code></td><td>float</td><td><code>0.3</code></td>
          <td>Temperature генерации.</td></tr>
      </tbody>
    </table>

    <h3>Качество (<code>quality</code>)</h3>
    <table class="doc-table">
      <thead><tr><th>Ключ</th><th>Тип</th><th>По умолч.</th><th>Описание</th></tr></thead>
      <tbody>
        <tr><td><code>fail_on_severity</code></td><td>select</td><td><code>never</code></td>
          <td><code>never</code> — не блокировать; <code>warn</code> — блокировать при warnings; <code>error</code> — только при ошибках.</td></tr>
      </tbody>
    </table>

    <h3>Qdrant (<code>index.qdrant</code>)</h3>
    <table class="doc-table">
      <thead><tr><th>Ключ</th><th>Тип</th><th>Описание</th></tr></thead>
      <tbody>
        <tr><td><code>url</code></td><td>string</td><td>URL Qdrant-сервера (напр. <code>http://localhost:6333</code>).</td></tr>
        <tr><td><code>collection</code></td><td>string</td><td>Имя коллекции (по умолч. <code>default</code>).</td></tr>
        <tr><td><code>api_key</code></td><td>string</td><td>API-ключ (опционально).</td></tr>
      </tbody>
    </table>

    <h3>pgvector (<code>index.pgvector</code>)</h3>
    <table class="doc-table">
      <thead><tr><th>Ключ</th><th>Тип</th><th>Описание</th></tr></thead>
      <tbody>
        <tr><td><code>dsn</code></td><td>string</td><td>PostgreSQL DSN (напр. <code>postgresql://user:pass@host/db</code>).</td></tr>
        <tr><td><code>table</code></td><td>string</td><td>Имя таблицы для векторов.</td></tr>
      </tbody>
    </table>
  </section>

  <!-- ===== Переменные окружения ===== -->
  <section id="env">
    <h2>Переменные окружения</h2>
    <p>Редактируются через
      <button class="link-btn" onclick={() => { appState.page = "config"; }}>Конфигурация</button>
      → вкладка «Окружение сервиса». Также доступны в <code>/etc/default/doc-rag</code> или <code>&lt;root&gt;/.env</code>.
    </p>
    <table class="doc-table">
      <thead><tr><th>Переменная</th><th>Тип</th><th>Описание</th></tr></thead>
      <tbody>
        <tr><td><code>DOC_RAG_HTTP_HOST</code></td><td>string</td><td>Адрес привязки. <code>0.0.0.0</code> — вся сеть.</td></tr>
        <tr><td><code>DOC_RAG_HTTP_PORT</code></td><td>int</td><td>TCP-порт (по умолч. 3333).</td></tr>
        <tr><td><code>DOC_RAG_ALLOWED_ORIGINS</code></td><td>string</td><td>CORS allow-list через запятую.</td></tr>
        <tr><td><code>DOC_RAG_HTTP_LOG</code></td><td>string</td><td>Путь к файлу HTTP-лога.</td></tr>
        <tr><td><code>DOC_RAG_UI_RESTART_ENABLED</code></td><td>bool</td><td>Разрешить рестарт из UI.</td></tr>
        <tr><td><code>DOC_RAG_UI_RESTART_CMD</code></td><td>string</td><td>Команда рестарта (напр. <code>sudo systemctl restart doc-rag-mcp</code>).</td></tr>
        <tr><td><code>DOC_RAG_UI_MAX_UPLOAD_MB</code></td><td>int</td><td>Лимит размера загружаемого файла.</td></tr>
        <tr><td><code>DOC_RAG_MAX_CONCURRENCY</code></td><td>int</td><td>Макс. одновременных MCP-инструментов.</td></tr>
        <tr><td><code>DOC_RAG_RATE_LIMIT_RPS</code></td><td>int</td><td>Rate limit (запросов/сек). 0 = выкл.</td></tr>
        <tr><td><code>DOC_RAG_RATE_LIMIT_BURST</code></td><td>int</td><td>Ёмкость burst-бакета.</td></tr>
        <tr><td><code>DOC_RAG_LOG_LEVEL</code></td><td>string</td><td>Уровень логов (DEBUG/INFO/WARNING/ERROR).</td></tr>
        <tr><td><code>DOC_RAG_LOG_FORMAT</code></td><td>select</td><td><code>text</code> — человекочитаемый; <code>json</code> — для лог-шипперов.</td></tr>
        <tr><td><code>DOC_RAG_API_KEY</code></td><td>secret</td><td>API-ключ для доступа к MCP и UI.</td></tr>
        <tr><td><code>DOC_RAG_LLM_BASE_URL</code></td><td>string</td><td>URL LLM API для RAG-генерации.</td></tr>
        <tr><td><code>DOC_RAG_LLM_MODEL</code></td><td>string</td><td>Имя модели LLM.</td></tr>
        <tr><td><code>DOC_RAG_LLM_API_KEY</code></td><td>secret</td><td>API-ключ LLM (если нужен).</td></tr>
        <tr><td><code>DOC_RAG_EXPOSE_SOURCE_PATHS</code></td><td>bool</td><td>Показывать полные пути к файлам в результатах.</td></tr>
      </tbody>
    </table>
  </section>

  <!-- ===== API ===== -->
  <section id="api">
    <h2>REST API</h2>
    <p>Полная спецификация: <a href="/docs" target="_blank">Swagger UI</a> |
       <a href="/redoc" target="_blank">ReDoc</a> |
       <a href="/openapi.json" target="_blank">OpenAPI JSON</a>.</p>

    <h3>Здоровье и статус</h3>
    <table class="doc-table">
      <thead><tr><th>Метод</th><th>Путь</th><th>Описание</th></tr></thead>
      <tbody>
        <tr><td><code>GET</code></td><td><code>/health</code></td><td>Комбинированный health-check (legacy).</td></tr>
        <tr><td><code>GET</code></td><td><code>/health/live</code></td><td>Liveness — процесс жив. Всегда 200.</td></tr>
        <tr><td><code>GET</code></td><td><code>/health/ready</code></td><td>Readiness — 200 если можно отвечать, 503 иначе.</td></tr>
        <tr><td><code>GET</code></td><td><code>/metrics</code></td><td>Prometheus метрики (требует <code>[metrics]</code> extra).</td></tr>
      </tbody>
    </table>

    <h3>UI</h3>
    <table class="doc-table">
      <thead><tr><th>Метод</th><th>Путь</th><th>Описание</th></tr></thead>
      <tbody>
        <tr><td><code>GET</code></td><td><code>/ui/status</code></td><td>Статус сервиса (job, indexed, logs).</td></tr>
        <tr><td><code>GET</code></td><td><code>/ui/quality</code></td><td>Сводка качества документов.</td></tr>
        <tr><td><code>GET</code></td><td><code>/ui/config/raw</code></td><td>Сырой YAML конфига.</td></tr>
        <tr><td><code>POST</code></td><td><code>/ui/config/save</code></td><td>Сохранить конфиг.</td></tr>
        <tr><td><code>GET</code></td><td><code>/ui/config/parsed</code></td><td>Парсенный конфиг (для формы).</td></tr>
        <tr><td><code>POST</code></td><td><code>/ui/config/patch</code></td><td>Патч полей конфига (с комментариями).</td></tr>
        <tr><td><code>POST</code></td><td><code>/ui/config/validate</code></td><td>Валидация без записи.</td></tr>
        <tr><td><code>GET</code></td><td><code>/ui/env</code></td><td>Env-переменные (секреты замаскированы).</td></tr>
        <tr><td><code>POST</code></td><td><code>/ui/env/save</code></td><td>Сохранить env в <code>&lt;root&gt;/.env</code>.</td></tr>
        <tr><td><code>POST</code></td><td><code>/ui/upload</code></td><td>Загрузка документов.</td></tr>
        <tr><td><code>POST</code></td><td><code>/ui/delete</code></td><td>Удаление документа по doc_id.</td></tr>
        <tr><td><code>POST</code></td><td><code>/ui/rebuild</code></td><td>Пересборка индекса.</td></tr>
        <tr><td><code>POST</code></td><td><code>/ui/wipe</code></td><td>Полная очистка индекса.</td></tr>
        <tr><td><code>POST</code></td><td><code>/ui/clean-orphans</code></td><td>Удаление осиротевших записей.</td></tr>
        <tr><td><code>POST</code></td><td><code>/ui/clear-incoming</code></td><td>Очистка папки incoming.</td></tr>
        <tr><td><code>POST</code></td><td><code>/ui/restart</code></td><td>Рестарт сервиса (требует <code>DOC_RAG_UI_RESTART_ENABLED=1</code>).</td></tr>
        <tr><td><code>GET</code></td><td><code>/ui/document-preview</code></td><td>Превью документа (title + preview).</td></tr>
      </tbody>
    </table>

    <h3>RAG</h3>
    <table class="doc-table">
      <thead><tr><th>Метод</th><th>Путь</th><th>Описание</th></tr></thead>
      <tbody>
        <tr><td><code>POST</code></td><td><code>/api/v1/generate</code></td>
          <td>RAG-генерация: retrieve + LLM. Параметры: <code>query</code>, <code>top_k</code>, <code>namespace</code>.</td></tr>
        <tr><td><code>GET</code></td><td><code>/api/v1/manifest</code></td><td>Manifest в JSON (для CI).</td></tr>
      </tbody>
    </table>

    <h3>MCP</h3>
    <table class="doc-table">
      <thead><tr><th>Метод</th><th>Путь</th><th>Описание</th></tr></thead>
      <tbody>
        <tr><td><code>GET</code></td><td><code>/mcp</code></td><td>SSE-уведомления (long-polling).</td></tr>
        <tr><td><code>POST</code></td><td><code>/mcp</code></td><td>JSON-RPC: <code>initialize</code>, <code>tools/list</code>, <code>tools/call</code>.</td></tr>
        <tr><td><code>GET</code></td><td><code>/ui/mcp/cursor.json</code></td><td>Конфиг MCP для Cursor.</td></tr>
        <tr><td><code>GET</code></td><td><code>/ui/mcp/vscode.json</code></td><td>Конфиг MCP для VSCode.</td></tr>
      </tbody>
    </table>
  </section>

  <!-- ===== MCP ===== -->
  <section id="mcp">
    <h2>MCP-инструменты</h2>
    <p>Доступны через <code>POST /mcp</code> (JSON-RPC) и интегрированы в Cursor, VSCode и другие MCP-клиенты.</p>

    <h3><code>doc_search</code></h3>
    <p>Поиск по базе документов. Возвращает чанки с оценками и структурированными цитатами.</p>
    <table class="doc-table">
      <thead><tr><th>Параметр</th><th>Тип</th><th>Обяз.</th><th>Описание</th></tr></thead>
      <tbody>
        <tr><td><code>query</code></td><td>string</td><td>да</td><td>Текст запроса.</td></tr>
        <tr><td><code>top_k</code></td><td>integer</td><td>нет</td><td>Количество результатов (по умолч. 6).</td></tr>
        <tr><td><code>namespace</code></td><td>string</td><td>нет</td><td>Пространство имён коллекции.</td></tr>
        <tr><td><code>doc_id</code></td><td>string</td><td>нет</td><td>Фильтр по ID документа.</td></tr>
        <tr><td><code>section_path</code></td><td>string</td><td>нет</td><td>Фильтр по префиксу секции.</td></tr>
        <tr><td><code>tables_only</code></td><td>boolean</td><td>нет</td><td>Только табличные чанки.</td></tr>
      </tbody>
    </table>

    <h3><code>doc_generate</code></h3>
    <p>RAG-генерация: поиск + LLM с цитированием.</p>
    <table class="doc-table">
      <thead><tr><th>Параметр</th><th>Тип</th><th>Обяз.</th><th>Описание</th></tr></thead>
      <tbody>
        <tr><td><code>query</code></td><td>string</td><td>да</td><td>Вопрос для генерации ответа.</td></tr>
        <tr><td><code>top_k</code></td><td>integer</td><td>нет</td><td>Количество контекстных чанков (по умолч. 5).</td></tr>
        <tr><td><code>namespace</code></td><td>string</td><td>нет</td><td>Пространство имён коллекции.</td></tr>
        <tr><td><code>max_tokens</code></td><td>integer</td><td>нет</td><td>Лимит токенов ответа LLM.</td></tr>
      </tbody>
    </table>
  </section>

  <!-- ===== Режимы поиска ===== -->
  <section id="search">
    <h2>Режимы поиска</h2>
    <table class="doc-table">
      <thead><tr><th>Режим</th><th>Описание</th><th>Когда использовать</th></tr></thead>
      <tbody>
        <tr><td><code>semantic</code></td><td>Векторный поиск (SentenceTransformer эмбеддинги + FAISS/Qdrant/pgvector).</td>
          <td>По умолчанию. Лучший для семантических запросов.</td></tr>
        <tr><td><code>lexical</code></td><td>TF-IDF с фразовым матчингом, IDF-взвешиванием, бонусом за покрытие.</td>
          <td>Точные термины, названия, коды.</td></tr>
        <tr><td><code>hybrid</code></td><td>Комбинация: lexical + semantic параллельно, слияние через RRF (k=60).</td>
          <td>Универсальный режим. Лучшее из двух миров.</td></tr>
      </tbody>
    </table>
    <p>Настройка: <code>mcp.retrieval_mode</code> в <code>config.yaml</code> или через
      <button class="link-btn" onclick={() => { appState.page = "config"; }}>Конфигурация</button>
      → «Индекс» → «Режим поиска».
    </p>
  </section>

  <!-- ===== Качество ===== -->
  <section id="quality">
    <h2>Качество документов</h2>
    <p>Сервис автоматически проверяет каждый документ при индексации. Результаты —
      <button class="link-btn" onclick={() => { appState.page = "config"; }}>Конфигурация</button>
      (бейджи в таблице документов) и <code>/ui/quality</code>.</p>
    <h3>Типы проверок</h3>
    <table class="doc-table">
      <thead><tr><th>Код</th><th>Описание</th><th>Severity</th></tr></thead>
      <tbody>
        <tr><td><code>empty_pages</code></td><td>Пустые страницы.</td><td>warn</td></tr>
        <tr><td><code>low_text_density</code></td><td>Страницы с подозрительно низкой плотностью текста.</td><td>info</td></tr>
        <tr><td><code>broken_table</code></td><td>Битые таблицы (пустые ячейки, несовпадение столбцов).</td><td>warn</td></tr>
        <tr><td><code>formula_garbage</code></td><td>Мусор в формулах (неизвестные Unicode-блоки).</td><td>warn</td></tr>
        <tr><td><code>unreadable_chars</code></td><td>Высокий % нечитаемых символов.</td><td>warn</td></tr>
        <tr><td><code>duplicate_headers</code></td><td>Дублирующиеся заголовки/колонтитулы.</td><td>info</td></tr>
        <tr><td><code>duplicate_footers</code></td><td>Дублирующиеся колонтитулы.</td><td>info</td></tr>
      </tbody>
    </table>
  </section>
</article>
</div>

<style>
  .docs-page {
    display: flex;
    height: 100%;
  }
  .doc-nav {
    width: 200px;
    min-width: 200px;
    border-right: 1px solid var(--border-subtle);
    padding: 16px 0;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    overflow-y: auto;
  }
  .doc-nav nav {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }
  .doc-nav-btn {
    display: block;
    width: 100%;
    text-align: left;
    padding: 6px 16px;
    background: none;
    border: none;
    color: var(--text-secondary);
    font-size: 0.85rem;
    cursor: pointer;
    font-family: ui-monospace, "JetBrains Mono", "Fira Code", SFMono-Regular,
      Menlo, Consolas, monospace;
  }
  .doc-nav-btn:hover {
    color: var(--text-primary);
    background: var(--bg-elevated);
  }
  .doc-nav-btn.active {
    color: var(--accent-ok);
  }
  .doc-nav-links {
    padding: 12px 16px;
    border-top: 1px solid var(--border-subtle);
    display: flex;
    flex-direction: column;
    gap: 6px;
    font-size: 0.8rem;
  }
  .doc-body {
    flex: 1;
    overflow-y: auto;
    padding: 24px 32px 48px;
    max-width: 900px;
  }
  .doc-body section {
    margin-bottom: 36px;
  }
  .doc-body h2 {
    font-size: 1.15rem;
    color: var(--accent-info);
    border-bottom: 1px solid var(--border-subtle);
    padding-bottom: 6px;
    margin: 0 0 16px;
    font-family: ui-monospace, "JetBrains Mono", "Fira Code", SFMono-Regular,
      Menlo, Consolas, monospace;
  }
  .doc-body h3 {
    font-size: 0.95rem;
    color: var(--text-primary);
    margin: 20px 0 8px;
    font-family: ui-monospace, "JetBrains Mono", "Fira Code", SFMono-Regular,
      Menlo, Consolas, monospace;
  }
  .doc-body p, .doc-body li {
    color: var(--text-secondary);
    font-size: 0.9rem;
    line-height: 1.6;
  }
  .doc-body ul {
    padding-left: 20px;
    margin: 8px 0;
  }
  .doc-body li {
    margin-bottom: 4px;
  }
  .doc-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
    margin: 8px 0 16px;
  }
  .doc-table th {
    text-align: left;
    padding: 6px 10px;
    border-bottom: 2px solid var(--border-strong);
    color: var(--text-muted);
    font-weight: 500;
    font-family: ui-monospace, "JetBrains Mono", "Fira Code", SFMono-Regular,
      Menlo, Consolas, monospace;
    font-size: 0.8rem;
  }
  .doc-table td {
    padding: 5px 10px;
    border-bottom: 1px solid var(--border-subtle);
    color: var(--text-secondary);
    vertical-align: top;
  }
  .doc-table td:first-child {
    white-space: nowrap;
  }
  .doc-table tr:hover td {
    background: var(--bg-elevated);
  }
  .arch-diagram {
    background: var(--bg-elevated);
    border: 1px solid var(--border-subtle);
    border-radius: 4px;
    padding: 16px;
    font-size: 0.8rem;
    overflow-x: auto;
    color: var(--text-secondary);
    line-height: 1.4;
    margin: 8px 0 16px;
  }
  .link-btn {
    background: none;
    border: none;
    color: var(--accent-info);
    cursor: pointer;
    font-size: inherit;
    font-family: inherit;
    padding: 0;
    text-decoration: underline;
  }
  .link-btn:hover {
    color: var(--accent-ok);
  }
</style>

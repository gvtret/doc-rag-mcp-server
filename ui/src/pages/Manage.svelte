<script lang="ts">
  import { appState } from "../lib/state.svelte";
  import {
    uploadFiles,
    rebuildIndex,
    wipeIndex,
    cleanOrphans,
    clearIncoming,
    restartService,
    ingest,
  } from "../lib/api";

  type Banner = { tone: "ok" | "err" | "info" | "warn"; text: string };
  type UploadResult = { saved: number; dups: string[]; errors: string[] };

  let banner: Banner | null = $state(null);
  let uploadBanners: Banner[] = $state([]);
  let busy = $state(false);
  let dragging = $state(false);
  let fileInput: HTMLInputElement;

  const running = $derived(appState.status?.running ?? false);
  const job = $derived(appState.status?.job ?? null);
  const rebuildInterrupted = $derived(appState.status?.rebuild_interrupted ?? false);

  // ---- Upload ----
  function onDragOver(e: DragEvent) {
    e.preventDefault();
    dragging = true;
  }
  function onDragLeave() {
    dragging = false;
  }
  function onDrop(e: DragEvent) {
    e.preventDefault();
    dragging = false;
    const files = e.dataTransfer?.files;
    if (files && files.length > 0) void doUpload(Array.from(files));
  }
  function onFileSelect() {
    const files = fileInput?.files;
    if (files && files.length > 0) void doUpload(Array.from(files));
  }

  async function doUpload(files: File[]) {
    if (busy) return;
    busy = true;
    uploadBanners = [];
    try {
      const r = await uploadFiles(files);
      if (r.result) {
        const res = r.result as UploadResult;
        const list: Banner[] = [];
        if (res.saved > 0) {
          list.push({ tone: "ok", text: `Загружено: ${res.saved} файл(ов)` });
        }
        for (const msg of res.dups) {
          list.push({ tone: "warn", text: msg });
        }
        for (const msg of res.errors) {
          list.push({ tone: "err", text: msg });
        }
        if (list.length === 0) {
          list.push({ tone: "err", text: "Файлы не сохранены" });
        }
        uploadBanners = list;
      } else {
        uploadBanners = [{ tone: "err", text: r.error ?? "Ошибка загрузки" }];
      }
    } catch (e) {
      uploadBanners = [{ tone: "err", text: e instanceof Error ? e.message : String(e) }];
    } finally {
      busy = false;
      if (fileInput) fileInput.value = "";
    }
  }

  // ---- Operations ----
  async function doRebuild() {
    if (busy || running) return;
    if (!confirm("Пересоздать индекс? Текущий индекс будет перезаписан.")) return;
    busy = true;
    banner = null;
    try {
      const r = await rebuildIndex();
      if (r.ok) {
        banner = { tone: "info", text: "Пересборка запущена. Следите за прогрессом в статусе." };
      } else {
        banner = { tone: "err", text: r.error ?? "Ошибка" };
      }
    } catch (e) {
      banner = { tone: "err", text: e instanceof Error ? e.message : String(e) };
    } finally {
      busy = false;
    }
  }

  async function doRebuildIndex() {
    if (busy || running) return;
    if (!confirm("Пересобрать только индекс из chunks.jsonl?")) return;
    busy = true;
    banner = null;
    try {
      const fd = new FormData();
      const r = await fetch("/ui/rebuild-index", { method: "POST", body: fd, credentials: "same-origin" });
      if (r.ok) {
        banner = { tone: "info", text: "Пересборка индекса запущена." };
      } else {
        banner = { tone: "err", text: `HTTP ${r.status}` };
      }
    } catch (e) {
      banner = { tone: "err", text: e instanceof Error ? e.message : String(e) };
    } finally {
      busy = false;
    }
  }

  async function doRebuildSkipIndex() {
    if (busy || running) return;
    if (!confirm("Полная пересборка БЕЗ индекса? (для локальной обработки на GPU)")) return;
    busy = true;
    banner = null;
    try {
      const fd = new FormData();
      const r = await fetch("/ui/rebuild-skip-index", { method: "POST", body: fd, credentials: "same-origin" });
      if (r.ok) {
        banner = { tone: "info", text: "Пересборка запущена (без индекса)." };
      } else {
        banner = { tone: "err", text: `HTTP ${r.status}` };
      }
    } catch (e) {
      banner = { tone: "err", text: e instanceof Error ? e.message : String(e) };
    } finally {
      busy = false;
    }
  }

  async function doIngest() {
    if (busy || running) return;
    if (!confirm("Обработать файлы из папки загрузок?")) return;
    busy = true;
    banner = null;
    try {
      const r = await ingest();
      if (r.ok) {
        banner = { tone: "info", text: "Обработка запущена. Следите за прогрессом в статусе." };
      } else {
        banner = { tone: "err", text: r.error ?? "Ошибка" };
      }
    } catch (e) {
      banner = { tone: "err", text: e instanceof Error ? e.message : String(e) };
    } finally {
      busy = false;
    }
  }

  async function doWipe() {
    if (busy || running) return;
    const confirm1 = confirm("Удалить ВСЕ документы и индекс? Это действие необратимо.");
    if (!confirm1) return;
    const word = prompt("Введите DELETE для подтверждения:");
    if (word !== "DELETE") {
      banner = { tone: "err", text: "Подтверждение не введено. Операция отменена." };
      return;
    }
    busy = true;
    banner = null;
    try {
      const r = await wipeIndex("DELETE");
      if (r.ok) {
        banner = { tone: "ok", text: "Индекс и документы удалены." };
      } else {
        banner = { tone: "err", text: r.error ?? "Ошибка" };
      }
    } catch (e) {
      banner = { tone: "err", text: e instanceof Error ? e.message : String(e) };
    } finally {
      busy = false;
    }
  }

  async function doCleanOrphans() {
    if (busy || running) return;
    busy = true;
    banner = null;
    try {
      const r = await cleanOrphans();
      if (r.ok) {
        banner = { tone: "ok", text: r.message ?? "Orphans удалены." };
      } else {
        banner = { tone: "err", text: r.error ?? "Ошибка" };
      }
    } catch (e) {
      banner = { tone: "err", text: e instanceof Error ? e.message : String(e) };
    } finally {
      busy = false;
    }
  }

  async function doClearIncoming() {
    if (busy || running) return;
    if (!confirm("Удалить все файлы из папки incoming?")) return;
    busy = true;
    banner = null;
    try {
      const r = await clearIncoming();
      if (r.ok) {
        banner = { tone: "ok", text: r.message ?? "Папка incoming очищена." };
      } else {
        banner = { tone: "err", text: r.error ?? "Ошибка" };
      }
    } catch (e) {
      banner = { tone: "err", text: e instanceof Error ? e.message : String(e) };
    } finally {
      busy = false;
    }
  }

  async function doRestart() {
    if (busy) return;
    if (!confirm("Перезапустить сервис doc-rag-mcp?")) return;
    busy = true;
    banner = null;
    try {
      const r = await restartService();
      if (r.ok) {
        banner = { tone: "ok", text: r.message ?? "Перезапуск отправлен." };
      } else {
        banner = {
          tone: "err",
          text: r.error ?? "Перезапуск недоступен. Включите DOC_RAG_UI_RESTART_ENABLED=1.",
        };
      }
    } catch (e) {
      banner = { tone: "err", text: e instanceof Error ? e.message : String(e) };
    } finally {
      busy = false;
    }
  }
</script>

<section>
  {#if banner}
    <div class="banner {banner.tone}">{banner.text}</div>
  {/if}

  {#if running}
    <div class="banner info">
      Выполняется: <strong>{job ?? "…"}</strong>. Дождитесь завершения перед новыми операциями.
    </div>
  {:else if rebuildInterrupted}
    <div class="banner warn">
      Предыдущий rebuild был прерван. Нажмите «Продолжить» для возобновления.
      <button
        type="button"
        class="btn inline"
        disabled={busy}
        onclick={() => void doRebuild()}
      >
        Продолжить
      </button>
    </div>
  {/if}

  <fieldset class="group">
    <legend>Загрузка файлов</legend>
    <div
      class="dropzone"
      class:dragging
      ondragover={onDragOver}
      ondragleave={onDragLeave}
      ondrop={onDrop}
      role="button"
      tabindex="0"
      title="Поддерживаемые форматы: PDF, DOCX, DOC, TXT, MD. Максимум 100 МБ на файл."
      onclick={() => fileInput?.click()}
      onkeydown={(e) => e.key === "Enter" && fileInput?.click()}
    >
      <span class="drop-label">
        {#if dragging}
          Отпустите файлы
        {:else}
          Перетащите файлы сюда или нажмите для выбора
        {/if}
      </span>
      <input
        bind:this={fileInput}
        type="file"
        multiple
        accept=".pdf,.docx,.doc,.txt,.md"
        class="hidden-input"
        onchange={onFileSelect}
      />
    </div>
  </fieldset>

  {#if uploadBanners.length > 0}
    <div class="upload-results">
      {#each uploadBanners as b}
        <div class="banner {b.tone}">{b.text}</div>
      {/each}
    </div>
  {/if}

  <fieldset class="group">
    <legend>Операции</legend>
    <div class="op-grid">
      <button
        type="button"
        class="btn"
        disabled={busy || running}
        title="Полная пересборка: парсинг + чанкинг + индекс. Все документы обрабатываются заново."
        onclick={() => void doRebuild()}
      >
        Полная пересборка
      </button>
      <button
        type="button"
        class="btn"
        disabled={busy || running}
        title="Пересобрать только FAISS-индекс из существующих чанков. Без повторного парсинга документов."
        onclick={() => void doRebuildIndex()}
      >
        Только индекс
      </button>
      <button
        type="button"
        class="btn"
        disabled={busy || running}
        title="Полный парсинг и чанкинг без построения индекса. Для обработки на GPU локально, затем загрузки индекса на сервер."
        onclick={() => void doRebuildSkipIndex()}
      >
        Только парсинг/чанкинг
      </button>
      <button
        type="button"
        class="btn"
        disabled={busy || running}
        title="Обработать файлы из папки загрузок: парсинг + чанкинг + индексация + архивация."
        onclick={() => void doIngest()}
      >
        Обработать загрузки
      </button>
      <button
        type="button"
        class="btn danger"
        disabled={busy || running}
        title="Удалить ВСЕ документы и индекс. Необратимо. Требует ввода DELETE для подтверждения."
        onclick={() => void doWipe()}
      >
        Очистить индекс
      </button>
      <button
        type="button"
        class="btn"
        disabled={busy || running}
        title="Найти и удалить документы, которые есть в файловой системе, но отсутствуют в манифесте."
        onclick={() => void doCleanOrphans()}
      >
        Удалить потерянные
      </button>
      <button
        type="button"
        class="btn danger"
        disabled={busy || running}
        title="Удалить все файлы из папки загрузок (sources/incoming)."
        onclick={() => void doClearIncoming()}
      >
        Очистить загрузки
      </button>
      <button
        type="button"
        class="btn warn"
        disabled={busy}
        title="Перезапустить сервис doc-rag-mcp. Прервёт текущие фоновые задачи."
        onclick={() => void doRestart()}
      >
        Перезапустить сервис
      </button>
    </div>
    <p class="hint">
      Очистка и Удаление требуют подтверждения. Пересборка и операции с данными блокируются во время загрузки/пересборки.
    </p>
  </fieldset>
</section>

<style>
  section {
    display: flex;
    flex-direction: column;
    gap: 12px;
    padding: 16px 24px;
    color: var(--text-secondary);
    height: 100%;
    box-sizing: border-box;
  }
  .banner {
    padding: 6px 10px;
    border-radius: 3px;
    font-size: 0.85rem;
    border-left: 3px solid;
  }
  .banner.ok {
    background: rgba(34, 197, 94, 0.1);
    border-color: var(--accent-ok);
    color: var(--text-primary);
  }
  .banner.err {
    background: rgba(239, 68, 68, 0.1);
    border-color: var(--accent-error);
    color: var(--text-primary);
  }
  .banner.info {
    background: rgba(59, 130, 246, 0.08);
    border-color: var(--accent-info);
    color: var(--text-secondary);
  }
  .banner.warn {
    background: rgba(245, 158, 11, 0.08);
    border-color: var(--accent-warn);
    color: var(--text-primary);
    display: flex;
    align-items: center;
    gap: 10px;
  }
  .banner .inline {
    margin-left: auto;
    padding: 3px 10px;
    font-size: 0.8rem;
    flex-shrink: 0;
  }
  .group {
    border: 1px solid var(--border-strong);
    border-radius: 3px;
    padding: 10px 14px 14px;
    margin: 0;
    display: flex;
    flex-direction: column;
    gap: 10px;
  }
  .group legend {
    color: var(--text-primary);
    font-size: 0.85rem;
    font-weight: 600;
    padding: 0 6px;
  }
  .dropzone {
    border: 2px dashed var(--border-strong);
    border-radius: 4px;
    padding: 28px 16px;
    text-align: center;
    cursor: pointer;
    transition: border-color 0.15s, background 0.15s;
  }
  .dropzone:hover,
  .dropzone.dragging {
    border-color: var(--accent-info);
    background: rgba(59, 130, 246, 0.06);
  }
  .drop-label {
    color: var(--text-muted);
    font-size: 0.85rem;
  }
  .hidden-input {
    display: none;
  }
  .op-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
  }
  .btn {
    background: var(--bg-elevated);
    color: var(--text-primary);
    border: 1px solid var(--border-strong);
    border-radius: 3px;
    padding: 6px 12px;
    font-size: 0.85rem;
    font-family: ui-monospace, "JetBrains Mono", "Fira Code", SFMono-Regular, Menlo, Consolas, monospace;
    cursor: pointer;
  }
  .btn:hover:not(:disabled) {
    border-color: var(--text-muted);
  }
  .btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
  .btn.danger {
    border-color: var(--accent-error);
    color: var(--accent-error);
  }
  .btn.danger:hover:not(:disabled) {
    background: rgba(239, 68, 68, 0.1);
  }
  .btn.warn {
    border-color: var(--accent-warn);
    color: var(--accent-warn);
  }
  .btn.warn:hover:not(:disabled) {
    background: rgba(245, 158, 11, 0.1);
  }
  .hint {
    margin: 0;
    font-size: 0.78rem;
    color: var(--text-muted);
  }
</style>

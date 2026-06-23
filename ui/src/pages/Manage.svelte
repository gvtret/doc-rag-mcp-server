<script lang="ts">
  import { appState } from "../lib/state.svelte";
  import {
    uploadFiles,
    rebuildIndex,
    wipeIndex,
    cleanOrphans,
    clearIncoming,
    restartService,
  } from "../lib/api";

  type Banner = { tone: "ok" | "err" | "info"; text: string } | null;

  let banner: Banner = $state(null);
  let busy = $state(false);
  let dragging = $state(false);
  let fileInput: HTMLInputElement;

  const running = $derived(appState.status?.running ?? false);
  const job = $derived(appState.status?.job ?? null);

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
    banner = null;
    try {
      const r = await uploadFiles(files);
      if (r.ok) {
        banner = { tone: "ok", text: r.message ?? `Загружено файлов: ${files.length}` };
      } else {
        banner = { tone: "err", text: r.error ?? "Ошибка загрузки" };
      }
    } catch (e) {
      banner = { tone: "err", text: e instanceof Error ? e.message : String(e) };
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
        banner = { tone: "info", text: "Rebuild запущен. Следите за прогрессом в статусе." };
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
        accept=".pdf,.docx,.txt,.md,.html,.htm"
        class="hidden-input"
        onchange={onFileSelect}
      />
    </div>
  </fieldset>

  <fieldset class="group">
    <legend>Операции</legend>
    <div class="op-grid">
      <button
        type="button"
        class="btn"
        disabled={busy || running}
        onclick={() => void doRebuild()}
      >
        Rebuild index
      </button>
      <button
        type="button"
        class="btn danger"
        disabled={busy || running}
        onclick={() => void doWipe()}
      >
        Wipe index
      </button>
      <button
        type="button"
        class="btn"
        disabled={busy || running}
        onclick={() => void doCleanOrphans()}
      >
        Clean orphans
      </button>
      <button
        type="button"
        class="btn danger"
        disabled={busy || running}
        onclick={() => void doClearIncoming()}
      >
        Clear incoming
      </button>
      <button
        type="button"
        class="btn warn"
        disabled={busy}
        onclick={() => void doRestart()}
      >
        Restart service
      </button>
    </div>
    <p class="hint">
      Wipe и Clear требуют подтверждения. Rebuild и операции с данными блокируются во время ingest/rebuild.
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

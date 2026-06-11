<script lang="ts">
  import { fetchConfigRaw, restartService, saveConfig } from "../lib/api";

  // Plain-text editor for now (no YAML highlighting); the server
  // validates on save and returns a readable error. A richer editor
  // (CodeMirror + YAML grammar) lands in a follow-up if needed.

  type LoadState =
    | { kind: "loading" }
    | { kind: "loaded"; path: string; loadedYaml: string }
    | { kind: "error"; message: string };

  // NOTE: avoid a local variable literally named `state`. Svelte 4
  // store-detection (`$state`) shadows the Svelte 5 `$state(...)` rune
  // in svelte-check when the names collide, and the error message
  // (`Cannot use 'state' as a store`) is misleading.
  let loadState = $state<LoadState>({ kind: "loading" });
  let text = $state("");
  let busy = $state(false);

  type Banner = { tone: "ok" | "err"; text: string } | null;
  let saveMsg: Banner = $state(null);
  let restartMsg: Banner = $state(null);

  const dirty = $derived.by(() => {
    if (loadState.kind !== "loaded") return false;
    return text !== loadState.loadedYaml;
  });

  async function load() {
    loadState = { kind: "loading" };
    try {
      const r = await fetchConfigRaw();
      if (r.ok) {
        loadState = { kind: "loaded", path: r.path, loadedYaml: r.yaml };
        text = r.yaml;
        saveMsg = null;
      } else {
        loadState = { kind: "error", message: r.error };
      }
    } catch (e) {
      loadState = {
        kind: "error",
        message: e instanceof Error ? e.message : String(e),
      };
    }
  }

  async function onSave() {
    if (loadState.kind !== "loaded" || busy) return;
    if (!confirm("Сохранить config.yaml? Сервис может потребовать перезапуска для применения некоторых изменений.")) {
      return;
    }
    busy = true;
    saveMsg = null;
    try {
      const r = await saveConfig(text);
      if (r.ok) {
        // Treat the just-saved buffer as the new on-disk baseline so
        // the "dirty" indicator clears without re-fetching.
        loadState = { ...loadState, loadedYaml: text };
        saveMsg = {
          tone: "ok",
          text: `Сохранено: ${r.path ?? "(путь не возвращён)"}`,
        };
      } else {
        saveMsg = { tone: "err", text: r.error ?? "Неизвестная ошибка." };
      }
    } catch (e) {
      saveMsg = {
        tone: "err",
        text: e instanceof Error ? e.message : String(e),
      };
    } finally {
      busy = false;
    }
  }

  async function onReload() {
    if (busy) return;
    if (
      loadState.kind === "loaded" &&
      dirty &&
      !confirm("Несохранённые изменения будут потеряны. Перезагрузить с диска?")
    ) {
      return;
    }
    await load();
  }

  async function onRestart() {
    if (busy) return;
    if (
      !confirm(
        "Перезапустить сервис doc-rag-mcp? Текущие фоновые задачи (ingest/rebuild) будут прерваны.",
      )
    ) {
      return;
    }
    busy = true;
    restartMsg = null;
    try {
      const r = await restartService();
      if (r.ok) {
        restartMsg = {
          tone: "ok",
          text: r.message ?? "Команда перезапуска отправлена.",
        };
      } else {
        restartMsg = {
          tone: "err",
          text:
            r.error ??
            "Перезапуск недоступен. Включите DOC_RAG_UI_RESTART_ENABLED=1 на сервере.",
        };
      }
    } catch (e) {
      restartMsg = {
        tone: "err",
        text: e instanceof Error ? e.message : String(e),
      };
    } finally {
      busy = false;
    }
  }

  load();
</script>

<section>
  {#if loadState.kind === "loading"}
    <p class="muted">Загружаем <code>config/config.yaml</code>…</p>
  {:else if loadState.kind === "error"}
    <p class="error">
      Не удалось загрузить конфиг: <code>{loadState.message}</code>
    </p>
    <button type="button" class="btn" onclick={() => void load()}>
      Попробовать ещё раз
    </button>
  {:else}
    <div class="meta mono">
      <span class="muted">file:</span>
      <code>{loadState.path}</code>
      {#if dirty}
        <span class="dot dirty" title="несохранённые изменения"></span>
        <span class="muted">несохранённые изменения</span>
      {/if}
    </div>

    {#if saveMsg}
      <div class="banner {saveMsg.tone}">{saveMsg.text}</div>
    {/if}
    {#if restartMsg}
      <div class="banner {restartMsg.tone}">{restartMsg.text}</div>
    {/if}

    <div class="editor-wrap">
      <textarea
        class="editor mono"
        bind:value={text}
        spellcheck="false"
        autocomplete="off"
        autocapitalize="off"
      ></textarea>
    </div>

    <div class="actions">
      <button
        type="button"
        class="btn primary"
        disabled={!dirty || busy}
        onclick={() => void onSave()}
      >
        Сохранить
      </button>
      <button
        type="button"
        class="btn"
        disabled={busy}
        onclick={() => void onReload()}
      >
        Перезагрузить с диска
      </button>
      <button
        type="button"
        class="btn warn"
        disabled={busy}
        onclick={() => void onRestart()}
      >
        Перезапустить сервис
      </button>
      <span class="muted spacer">
        Изменения config'а вступят в силу после рестарта сервиса.
      </span>
    </div>
  {/if}
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
  .muted {
    color: var(--text-muted);
  }
  .error {
    color: var(--accent-error);
  }
  .meta {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.85rem;
  }
  .meta .dot.dirty {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--accent-warn);
    box-shadow: 0 0 6px rgba(245, 158, 11, 0.5);
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
  .editor-wrap {
    flex: 1;
    min-height: 280px;
    display: flex;
  }
  .editor {
    flex: 1;
    width: 100%;
    background: var(--bg-elevated);
    color: var(--text-primary);
    border: 1px solid var(--border-strong);
    padding: 10px 12px;
    font-size: 0.85rem;
    line-height: 1.45;
    resize: vertical;
    tab-size: 2;
  }
  .editor:focus {
    outline: none;
    border-color: var(--accent-info);
  }
  .actions {
    display: flex;
    gap: 8px;
    align-items: center;
    flex-wrap: wrap;
  }
  .spacer {
    margin-left: auto;
    font-size: 0.8rem;
  }
  .btn {
    background: var(--bg-elevated);
    color: var(--text-primary);
    border: 1px solid var(--border-strong);
    border-radius: 3px;
    padding: 6px 12px;
    font-size: 0.85rem;
    font-family: ui-monospace, "JetBrains Mono", "Fira Code", SFMono-Regular,
      Menlo, Consolas, monospace;
    cursor: pointer;
  }
  .btn:hover:not(:disabled) {
    border-color: var(--text-muted);
  }
  .btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
  .btn.primary {
    background: var(--accent-ok);
    color: var(--bg-base);
    border-color: var(--accent-ok);
    font-weight: 600;
  }
  .btn.primary:hover:not(:disabled) {
    background: #16a34a;
  }
  .btn.warn {
    border-color: var(--accent-warn);
    color: var(--accent-warn);
  }
  .btn.warn:hover:not(:disabled) {
    background: rgba(245, 158, 11, 0.1);
  }
</style>

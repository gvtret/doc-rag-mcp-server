<script lang="ts">
  import {
    fetchConfigParsed,
    fetchConfigRaw,
    fetchEnv,
    patchConfig,
    restartService,
    saveConfig,
    saveEnv,
  } from "../lib/api";
  import { CONFIG_SCHEMA, type Field } from "../lib/configSchema";
  import { getByPath } from "../lib/dotpath";
  import { ENV_META } from "../lib/envSchema";
  import type { EnvField, EnvSecret } from "../lib/types";

  // Two editors over the same config/config.yaml:
  //   - "form": structured fields (schema-driven), comment-preserving
  //     field-level save via /ui/config/patch.
  //   - "raw":  the original plain-text YAML editor (Advanced).
  // The raw textarea still validates server-side on save.

  type Tab = "form" | "raw" | "env";
  let tab = $state<Tab>("form");

  type Banner = { tone: "ok" | "err"; text: string } | null;
  let busy = $state(false);
  let restartMsg: Banner = $state(null);

  // ---- form tab ---------------------------------------------------------
  type FormVal = string | number | boolean;
  type FormState =
    | { kind: "loading" }
    | { kind: "loaded"; path: string }
    | { kind: "error"; message: string };

  let formState = $state<FormState>({ kind: "loading" });
  // Bound to inputs of mixed types (text/number/checkbox/select), so the
  // value type is intentionally loose; coerceIn normalizes on load.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  let formValues = $state<Record<string, any>>({});
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  let formBaseline = $state<Record<string, any>>({});
  let formMsg: Banner = $state(null);
  let formLoaded = false;

  const allFields: Field[] = CONFIG_SCHEMA.flatMap((s) => s.fields);

  function coerceIn(field: Field, raw: unknown): FormVal {
    if (field.type === "bool") return raw === true;
    if (field.type === "int" || field.type === "float") {
      return typeof raw === "number" ? raw : Number(raw ?? 0);
    }
    return raw == null ? "" : String(raw);
  }

  const changedPaths = $derived.by(() =>
    allFields
      .map((f) => f.path)
      .filter((p) => formValues[p] !== formBaseline[p]),
  );
  const formDirty = $derived(changedPaths.length > 0);

  async function loadForm() {
    formState = { kind: "loading" };
    formMsg = null;
    try {
      const r = await fetchConfigParsed();
      if (r.ok) {
        const vals: Record<string, FormVal> = {};
        for (const f of allFields) vals[f.path] = coerceIn(f, getByPath(r.config, f.path));
        formValues = vals;
        formBaseline = { ...vals };
        formState = { kind: "loaded", path: r.path };
        formLoaded = true;
      } else {
        formState = { kind: "error", message: r.error };
      }
    } catch (e) {
      formState = { kind: "error", message: e instanceof Error ? e.message : String(e) };
    }
  }

  async function onFormSave() {
    if (formState.kind !== "loaded" || busy || !formDirty) return;
    busy = true;
    formMsg = null;
    try {
      const updates: Record<string, unknown> = {};
      for (const p of changedPaths) updates[p] = formValues[p];
      const r = await patchConfig(updates);
      if (r.ok) {
        formBaseline = { ...formValues };
        formMsg = { tone: "ok", text: `Сохранено: ${r.path ?? "(путь не возвращён)"}` };
      } else {
        formMsg = { tone: "err", text: r.error ?? "Неизвестная ошибка." };
      }
    } catch (e) {
      formMsg = { tone: "err", text: e instanceof Error ? e.message : String(e) };
    } finally {
      busy = false;
    }
  }

  function onFormReset() {
    if (busy) return;
    formValues = { ...formBaseline };
    formMsg = null;
  }

  // ---- raw tab ----------------------------------------------------------
  type RawState =
    | { kind: "loading" }
    | { kind: "loaded"; path: string; loadedYaml: string }
    | { kind: "error"; message: string };

  let rawState = $state<RawState>({ kind: "loading" });
  let text = $state("");
  let rawMsg: Banner = $state(null);
  let rawLoaded = false;

  const rawDirty = $derived.by(() => {
    if (rawState.kind !== "loaded") return false;
    return text !== rawState.loadedYaml;
  });

  async function loadRaw() {
    rawState = { kind: "loading" };
    try {
      const r = await fetchConfigRaw();
      if (r.ok) {
        rawState = { kind: "loaded", path: r.path, loadedYaml: r.yaml };
        text = r.yaml;
        rawMsg = null;
        rawLoaded = true;
      } else {
        rawState = { kind: "error", message: r.error };
      }
    } catch (e) {
      rawState = { kind: "error", message: e instanceof Error ? e.message : String(e) };
    }
  }

  async function onRawSave() {
    if (rawState.kind !== "loaded" || busy) return;
    if (!confirm("Сохранить config.yaml? Сервис может потребовать перезапуска для применения некоторых изменений.")) {
      return;
    }
    busy = true;
    rawMsg = null;
    try {
      const r = await saveConfig(text);
      if (r.ok) {
        rawState = { ...rawState, loadedYaml: text };
        rawMsg = { tone: "ok", text: `Сохранено: ${r.path ?? "(путь не возвращён)"}` };
      } else {
        rawMsg = { tone: "err", text: r.error ?? "Неизвестная ошибка." };
      }
    } catch (e) {
      rawMsg = { tone: "err", text: e instanceof Error ? e.message : String(e) };
    } finally {
      busy = false;
    }
  }

  async function onRawReload() {
    if (busy) return;
    if (rawState.kind === "loaded" && rawDirty && !confirm("Несохранённые изменения будут потеряны. Перезагрузить с диска?")) {
      return;
    }
    await loadRaw();
  }

  // ---- service env tab --------------------------------------------------
  type EnvState =
    | { kind: "loading" }
    | { kind: "loaded"; path: string }
    | { kind: "error"; message: string };

  let envState = $state<EnvState>({ kind: "loading" });
  let envFields = $state<EnvField[]>([]);
  let envSecrets = $state<EnvSecret[]>([]);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  let envValues = $state<Record<string, any>>({});
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  let envBaseline = $state<Record<string, any>>({});
  let envMsg: Banner = $state(null);
  let envLoaded = false;

  function envIn(f: EnvField): string | boolean {
    if (f.type === "bool") return ["1", "true", "yes", "on"].includes(f.value.toLowerCase());
    return f.value;
  }

  const envChangedKeys = $derived.by(() =>
    envFields.map((f) => f.key).filter((k) => envValues[k] !== envBaseline[k]),
  );
  const envDirty = $derived(envChangedKeys.length > 0);

  function envMeta(key: string): { label: string; hint?: string } {
    return ENV_META[key] ?? { label: key };
  }

  async function loadEnv() {
    envState = { kind: "loading" };
    envMsg = null;
    try {
      const r = await fetchEnv();
      if (r.ok) {
        envFields = r.fields;
        envSecrets = r.secrets;
        const vals: Record<string, string | boolean> = {};
        for (const f of r.fields) vals[f.key] = envIn(f);
        envValues = vals;
        envBaseline = { ...vals };
        envState = { kind: "loaded", path: r.path };
        envLoaded = true;
      } else {
        envState = { kind: "error", message: r.error };
      }
    } catch (e) {
      envState = { kind: "error", message: e instanceof Error ? e.message : String(e) };
    }
  }

  async function onEnvSave() {
    if (envState.kind !== "loaded" || busy || !envDirty) return;
    busy = true;
    envMsg = null;
    try {
      const updates: Record<string, string> = {};
      for (const k of envChangedKeys) {
        const v = envValues[k];
        updates[k] = typeof v === "boolean" ? (v ? "1" : "0") : String(v);
      }
      const r = await saveEnv(updates);
      if (r.ok) {
        envBaseline = { ...envValues };
        envMsg = { tone: "ok", text: `Записано в ${r.path ?? "(путь не возвращён)"}. Применится после рестарта.` };
      } else {
        envMsg = { tone: "err", text: r.error ?? "Неизвестная ошибка." };
      }
    } catch (e) {
      envMsg = { tone: "err", text: e instanceof Error ? e.message : String(e) };
    } finally {
      busy = false;
    }
  }

  function onEnvReset() {
    if (busy) return;
    envValues = { ...envBaseline };
    envMsg = null;
  }

  // ---- shared -----------------------------------------------------------
  function switchTab(next: Tab) {
    tab = next;
    if (next === "form" && !formLoaded) void loadForm();
    if (next === "raw" && !rawLoaded) void loadRaw();
    if (next === "env" && !envLoaded) void loadEnv();
  }

  async function onRestart() {
    if (busy) return;
    if (!confirm("Перезапустить сервис doc-rag-mcp? Текущие фоновые задачи (ingest/rebuild) будут прерваны.")) {
      return;
    }
    busy = true;
    restartMsg = null;
    try {
      const r = await restartService();
      if (r.ok) {
        restartMsg = { tone: "ok", text: r.message ?? "Команда перезапуска отправлена." };
      } else {
        restartMsg = {
          tone: "err",
          text: r.error ?? "Перезапуск недоступен. Включите DOC_RAG_UI_RESTART_ENABLED=1 на сервере.",
        };
      }
    } catch (e) {
      restartMsg = { tone: "err", text: e instanceof Error ? e.message : String(e) };
    } finally {
      busy = false;
    }
  }

  loadForm();
</script>

<section>
  <div class="tabs">
    <button type="button" class="tab" class:active={tab === "form"} onclick={() => switchTab("form")}>
      Форма
    </button>
    <button type="button" class="tab" class:active={tab === "env"} onclick={() => switchTab("env")}>
      Сервис (env)
    </button>
    <button type="button" class="tab" class:active={tab === "raw"} onclick={() => switchTab("raw")}>
      Advanced (raw YAML)
    </button>
  </div>

  {#if restartMsg}
    <div class="banner {restartMsg.tone}">{restartMsg.text}</div>
  {/if}

  {#if tab === "form"}
    {#if formState.kind === "loading"}
      <p class="muted">Загружаем конфигурацию…</p>
    {:else if formState.kind === "error"}
      <p class="error">Не удалось загрузить конфиг: <code>{formState.message}</code></p>
      <button type="button" class="btn" onclick={() => void loadForm()}>Попробовать ещё раз</button>
    {:else}
      <div class="meta mono">
        <span class="muted">file:</span>
        <code>{formState.path}</code>
        {#if formDirty}
          <span class="dot dirty" title="несохранённые изменения"></span>
          <span class="muted">несохранённые изменения</span>
        {/if}
      </div>

      {#if formMsg}
        <div class="banner {formMsg.tone}">{formMsg.text}</div>
      {/if}

      <div class="form-scroll">
        {#each CONFIG_SCHEMA as section (section.title)}
          <fieldset class="group">
            <legend>{section.title}</legend>
            {#each section.fields as field (field.path)}
              <div class="row" class:changed={formValues[field.path] !== formBaseline[field.path]}>
                <label for={field.path}>{field.label}</label>
                <div class="control">
                  {#if field.type === "bool"}
                    <input id={field.path} type="checkbox" bind:checked={formValues[field.path]} />
                  {:else if field.type === "select"}
                    <select id={field.path} bind:value={formValues[field.path]}>
                      {#each field.options ?? [] as opt (opt)}
                        <option value={opt}>{opt}</option>
                      {/each}
                    </select>
                  {:else if field.type === "int" || field.type === "float"}
                    <input
                      id={field.path}
                      type="number"
                      step={field.type === "float" ? "any" : "1"}
                      min={field.min}
                      max={field.max}
                      bind:value={formValues[field.path]}
                    />
                  {:else}
                    <input id={field.path} type="text" bind:value={formValues[field.path]} />
                  {/if}
                  {#if field.hint}
                    <p class="hint">{field.hint}</p>
                  {/if}
                </div>
              </div>
            {/each}
          </fieldset>
        {/each}
      </div>

      <div class="actions">
        <button type="button" class="btn primary" disabled={!formDirty || busy} onclick={() => void onFormSave()}>
          Сохранить
        </button>
        <button type="button" class="btn" disabled={!formDirty || busy} onclick={onFormReset}>
          Сбросить изменения
        </button>
        <button type="button" class="btn warn" disabled={busy} onclick={() => void onRestart()}>
          Перезапустить сервис
        </button>
        <span class="muted spacer">Изменения config'а вступят в силу после рестарта сервиса.</span>
      </div>
    {/if}
  {:else if tab === "raw"}
    {#if rawState.kind === "loading"}
      <p class="muted">Загружаем <code>config/config.yaml</code>…</p>
    {:else if rawState.kind === "error"}
      <p class="error">Не удалось загрузить конфиг: <code>{rawState.message}</code></p>
      <button type="button" class="btn" onclick={() => void loadRaw()}>Попробовать ещё раз</button>
    {:else}
      <div class="meta mono">
        <span class="muted">file:</span>
        <code>{rawState.path}</code>
        {#if rawDirty}
          <span class="dot dirty" title="несохранённые изменения"></span>
          <span class="muted">несохранённые изменения</span>
        {/if}
      </div>

      {#if rawMsg}
        <div class="banner {rawMsg.tone}">{rawMsg.text}</div>
      {/if}

      <div class="editor-wrap">
        <textarea class="editor mono" bind:value={text} spellcheck="false" autocomplete="off" autocapitalize="off"></textarea>
      </div>

      <div class="actions">
        <button type="button" class="btn primary" disabled={!rawDirty || busy} onclick={() => void onRawSave()}>
          Сохранить
        </button>
        <button type="button" class="btn" disabled={busy} onclick={() => void onRawReload()}>
          Перезагрузить с диска
        </button>
        <button type="button" class="btn warn" disabled={busy} onclick={() => void onRestart()}>
          Перезапустить сервис
        </button>
        <span class="muted spacer">Изменения config'а вступят в силу после рестарта сервиса.</span>
      </div>
    {/if}
  {:else if envState.kind === "loading"}
    <p class="muted">Загружаем env сервиса…</p>
  {:else if envState.kind === "error"}
    <p class="error">Не удалось загрузить env: <code>{envState.message}</code></p>
    <button type="button" class="btn" onclick={() => void loadEnv()}>Попробовать ещё раз</button>
  {:else}
    <div class="meta mono">
      <span class="muted">file:</span>
      <code>{envState.path}</code>
      {#if envDirty}
        <span class="dot dirty" title="несохранённые изменения"></span>
        <span class="muted">несохранённые изменения</span>
      {/if}
    </div>

    <div class="banner info">
      Эти настройки — runtime сервиса (не пайплайн). Запись идёт в <code>.env</code>,
      который подхватывает <code>run_mcp_http.sh</code>; изменения применяются
      только после перезапуска сервиса.
    </div>

    {#if envMsg}
      <div class="banner {envMsg.tone}">{envMsg.text}</div>
    {/if}

    <div class="form-scroll">
      <fieldset class="group">
        <legend>Runtime / env</legend>
        {#each envFields as f (f.key)}
          <div class="row" class:changed={envValues[f.key] !== envBaseline[f.key]}>
            <label for={f.key}>{envMeta(f.key).label}</label>
            <div class="control">
              {#if f.type === "bool"}
                <input id={f.key} type="checkbox" bind:checked={envValues[f.key]} />
              {:else if f.type === "select"}
                <select id={f.key} bind:value={envValues[f.key]}>
                  {#each f.options ?? [] as opt (opt)}
                    <option value={opt}>{opt}</option>
                  {/each}
                </select>
              {:else if f.type === "int" || f.type === "float"}
                <input id={f.key} type="number" step={f.type === "float" ? "any" : "1"} bind:value={envValues[f.key]} />
              {:else}
                <input id={f.key} type="text" bind:value={envValues[f.key]} />
              {/if}
              <p class="hint">
                <code class="envkey">{f.key}</code>{#if f.source !== "file"} <span class="src">(из {f.source === "env" ? "окружения" : "по умолчанию"})</span>{/if}
                {#if envMeta(f.key).hint}<br />{envMeta(f.key).hint}{/if}
              </p>
            </div>
          </div>
        {/each}
      </fieldset>

      {#if envSecrets.length}
        <fieldset class="group">
          <legend>Секреты (только статус)</legend>
          {#each envSecrets as s (s.key)}
            <div class="row">
              <label for={s.key}>{s.key}</label>
              <div class="control">
                <span class="secret-status {s.set ? 'on' : 'off'}">{s.set ? "задан" : "не задан"}</span>
                <p class="hint">Секрет не редактируется из UI — задайте его в env/systemd на сервере.</p>
              </div>
            </div>
          {/each}
        </fieldset>
      {/if}
    </div>

    <div class="actions">
      <button type="button" class="btn primary" disabled={!envDirty || busy} onclick={() => void onEnvSave()}>
        Сохранить
      </button>
      <button type="button" class="btn" disabled={!envDirty || busy} onclick={onEnvReset}>
        Сбросить изменения
      </button>
      <button type="button" class="btn warn" disabled={busy} onclick={() => void onRestart()}>
        Перезапустить сервис
      </button>
      <span class="muted spacer">Env применяется только после рестарта сервиса.</span>
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
  .tabs {
    display: flex;
    gap: 4px;
    border-bottom: 1px solid var(--border-strong);
  }
  .tab {
    background: transparent;
    color: var(--text-muted);
    border: 1px solid transparent;
    border-bottom: none;
    border-radius: 3px 3px 0 0;
    padding: 6px 14px;
    font-size: 0.85rem;
    font-family: ui-monospace, "JetBrains Mono", "Fira Code", SFMono-Regular, Menlo, Consolas, monospace;
    cursor: pointer;
  }
  .tab:hover {
    color: var(--text-primary);
  }
  .tab.active {
    color: var(--text-primary);
    border-color: var(--border-strong);
    background: var(--bg-elevated);
    margin-bottom: -1px;
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
  .banner.info {
    background: rgba(59, 130, 246, 0.08);
    border-color: var(--accent-info);
    color: var(--text-secondary);
    line-height: 1.4;
  }
  .envkey {
    color: var(--text-muted);
    font-size: 0.78rem;
  }
  .src {
    color: var(--accent-info);
    font-size: 0.75rem;
  }
  .secret-status {
    font-size: 0.85rem;
    padding: 3px 8px;
    border-radius: 3px;
    align-self: flex-start;
  }
  .secret-status.on {
    color: var(--accent-ok);
    border: 1px solid var(--accent-ok);
  }
  .secret-status.off {
    color: var(--text-muted);
    border: 1px solid var(--border-strong);
  }
  .form-scroll {
    flex: 1;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 16px;
    min-height: 0;
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
  .row {
    display: grid;
    grid-template-columns: 240px 1fr;
    gap: 12px;
    align-items: start;
    border-left: 2px solid transparent;
    padding-left: 8px;
  }
  .row.changed {
    border-left-color: var(--accent-warn);
  }
  .row label {
    font-size: 0.85rem;
    padding-top: 5px;
    color: var(--text-secondary);
  }
  .control {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  .control input[type="text"],
  .control input[type="number"],
  .control select {
    background: var(--bg-elevated);
    color: var(--text-primary);
    border: 1px solid var(--border-strong);
    border-radius: 3px;
    padding: 5px 8px;
    font-size: 0.85rem;
    font-family: ui-monospace, "JetBrains Mono", "Fira Code", SFMono-Regular, Menlo, Consolas, monospace;
    max-width: 360px;
  }
  .control input:focus,
  .control select:focus {
    outline: none;
    border-color: var(--accent-info);
  }
  .control input[type="checkbox"] {
    width: 16px;
    height: 16px;
    accent-color: var(--accent-ok);
    margin-top: 4px;
  }
  .hint {
    margin: 0;
    font-size: 0.78rem;
    color: var(--text-muted);
    line-height: 1.35;
    max-width: 520px;
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

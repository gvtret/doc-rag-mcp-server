<script lang="ts">
  import { appState } from "../lib/state.svelte";
  import type { PageKey } from "../lib/types";

  type MenuItem = { key: PageKey; label: string; hint?: string };

  const items: MenuItem[] = [
    { key: "documents", label: "Документы", hint: "Индексированный корпус" },
    { key: "config", label: "Конфигурация", hint: "Парсинг, чанки, эмбеддинги" },
    { key: "manage", label: "Управление", hint: "Ingest, rebuild, аварийные" },
    { key: "logs", label: "Логи", hint: "Ingest log, HTTP log" },
    { key: "docs", label: "Документация", hint: "API, параметры, архитектура" },
  ];

  function select(key: PageKey) {
    appState.page = key;
  }
</script>

<aside class="sidebar" aria-label="Главное меню">
  <div class="brand">
    <span class="logo">doc-rag</span>
    <span class="ver">v3.1</span>
  </div>
  <nav>
    <ul>
      {#each items as item (item.key)}
        <li>
          <button
            type="button"
            class="nav-btn"
            class:active={appState.page === item.key}
            onclick={() => select(item.key)}
            title={item.hint}
          >
            <span class="caret">{appState.page === item.key ? ">" : " "}</span>
            <span class="nav-label">{item.label}</span>
          </button>
          {#if item.hint}
            <div class="nav-hint">{item.hint}</div>
          {/if}
        </li>
      {/each}
    </ul>
  </nav>
</aside>

<style>
  .sidebar {
    grid-area: sidebar;
    background: var(--bg-base);
    color: var(--text-primary);
    padding: 16px 0;
    display: flex;
    flex-direction: column;
    gap: 14px;
    /* Independent scroll so the global footer never gets pushed off
       by a hypothetical long sidebar. */
    overflow-y: auto;
    border-right: 1px solid var(--border-subtle);
  }
  .brand {
    padding: 0 18px 12px;
    display: flex;
    align-items: baseline;
    gap: 8px;
    font-family: ui-monospace, "JetBrains Mono", "Fira Code", SFMono-Regular,
      Menlo, Consolas, monospace;
    border-bottom: 1px solid var(--border-subtle);
  }
  .brand .logo {
    font-weight: 700;
    letter-spacing: 0.05em;
  }
  .brand .ver {
    color: var(--text-muted);
    font-size: 0.78rem;
  }
  nav ul {
    list-style: none;
    margin: 0;
    padding: 0;
  }
  nav li {
    padding: 0 12px;
    margin-bottom: 6px;
  }
  .nav-btn {
    display: flex;
    align-items: center;
    gap: 8px;
    width: 100%;
    padding: 6px 8px;
    background: transparent;
    border: none;
    color: var(--text-secondary);
    text-align: left;
    cursor: pointer;
    font-family: ui-monospace, "JetBrains Mono", "Fira Code", SFMono-Regular,
      Menlo, Consolas, monospace;
    font-size: 0.92rem;
  }
  .nav-btn:hover {
    color: var(--text-primary);
    background: var(--bg-elevated);
    border-radius: 3px;
  }
  .nav-btn.active {
    color: var(--accent-ok);
    background: transparent;
  }
  .caret {
    width: 1ch;
    color: var(--accent-ok);
  }
  .nav-hint {
    font-size: 0.7rem;
    color: var(--text-faint);
    padding: 0 8px 4px 24px;
  }
</style>

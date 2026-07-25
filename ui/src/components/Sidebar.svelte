<script lang="ts">
  import { appState } from "../lib/state.svelte";
  import { t } from "../lib/i18n";
  import type { PageKey } from "../lib/types";

  type MenuItem = { key: PageKey; labelKey: string; hintKey: string };

  const itemDefs: MenuItem[] = [
    { key: "documents", labelKey: "sidebar.documents", hintKey: "sidebar.documents_hint" },
    { key: "stats", labelKey: "sidebar.stats", hintKey: "sidebar.stats_hint" },
    { key: "config", labelKey: "sidebar.config", hintKey: "sidebar.config_hint" },
    { key: "manage", labelKey: "sidebar.manage", hintKey: "sidebar.manage_hint" },
    { key: "logs", labelKey: "sidebar.logs", hintKey: "sidebar.logs_hint" },
    { key: "docs", labelKey: "sidebar.docs", hintKey: "sidebar.docs_hint" },
  ];

  function select(key: PageKey) {
    appState.page = key;
  }
</script>

<aside class="sidebar" aria-label="Главное меню">
  <div class="brand">
    <span class="logo">doc-rag</span>
    <span class="ver">{appState.status?.version ? `v${appState.status.version}` : ""}</span>
  </div>
  <nav>
    <ul>
      {#each itemDefs as item (item.key)}
        <li>
          <button
            type="button"
            class="nav-btn"
            class:active={appState.page === item.key}
            onclick={() => select(item.key)}
            title={t(item.hintKey)}
          >
            <span class="caret">{appState.page === item.key ? ">" : " "}</span>
            <span class="nav-label">{t(item.labelKey)}</span>
          </button>
          <div class="nav-hint">{t(item.hintKey)}</div>
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
    overflow-y: auto;
    border-right: 1px solid var(--border-subtle);
  }
  .brand {
    padding: 0 18px 12px;
    display: flex;
    align-items: baseline;
    gap: 8px;
    border-bottom: 1px solid var(--border-subtle);
  }
  .brand .logo {
    font-weight: 700;
    letter-spacing: 0.02em;
    color: var(--accent-ok);
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
    font-weight: 600;
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

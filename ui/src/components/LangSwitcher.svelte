<script lang="ts">
  import { lang, setLang, type Lang } from '../lib/i18n';

  const options: { value: Lang; label: string }[] = [
    { value: 'ru', label: 'RU' },
    { value: 'en', label: 'EN' },
  ];

  function switchLang(l: Lang) {
    setLang(l);
    // Force re-render by reloading the page
    window.location.reload();
  }
</script>

<div class="lang-switcher">
  {#each options as opt}
    <button
      type="button"
      class="lang-btn"
      class:active={$lang === opt.value}
      onclick={() => switchLang(opt.value)}
    >
      {opt.label}
    </button>
  {/each}
</div>

<style>
  .lang-switcher {
    display: inline-flex;
    gap: 2px;
    margin-left: auto;
    margin-right: 12px;
  }
  .lang-btn {
    background: transparent;
    color: var(--text-muted);
    border: 1px solid var(--border-subtle);
    border-radius: 3px;
    padding: 2px 6px;
    font-size: 0.7rem;
    font-family: ui-monospace, "JetBrains Mono", "Fira Code", SFMono-Regular,
      Menlo, Consolas, monospace;
    cursor: pointer;
    letter-spacing: 0.04em;
  }
  .lang-btn:hover {
    color: var(--text-primary);
    border-color: var(--border-strong);
  }
  .lang-btn.active {
    color: var(--accent-ok);
    border-color: var(--accent-ok);
  }
</style>

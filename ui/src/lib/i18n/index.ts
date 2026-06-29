import { writable, get } from 'svelte/store';

const STORAGE_KEY = 'doc-rag-lang';
export type Lang = 'ru' | 'en';

export const lang = writable<Lang>(
  (typeof localStorage !== 'undefined'
    ? (localStorage.getItem(STORAGE_KEY) as Lang)
    : null) || 'ru'
);

export function setLang(l: Lang) {
  lang.set(l);
  localStorage.setItem(STORAGE_KEY, l);
}

import ru from './ru.json';
import en from './en.json';
const messages: Record<Lang, Record<string, string>> = { ru, en };

export function t(key: string, params?: Record<string, string | number>): string {
  const l = get(lang);
  let val = messages[l]?.[key] ?? messages.ru[key] ?? key;
  if (params) {
    for (const [k, v] of Object.entries(params)) {
      val = val.replace(new RegExp(`\\{${k}\\}`, 'g'), String(v));
    }
  }
  return val;
}

// Human labels + hints for the service env editor. The backend
// (/ui/env) is the source of truth for which keys are editable and their
// types; this map only adds presentation. Keys missing here fall back to
// showing the raw env name.

export type EnvMeta = { label: string; hint?: string };

export const ENV_META: Record<string, EnvMeta> = {
  DOC_RAG_HTTP_HOST: {
    label: "Host привязки",
    hint: "Адрес, на котором слушает сервис. 0.0.0.0 — слушать всю сеть (LAN).",
  },
  DOC_RAG_HTTP_PORT: {
    label: "Port",
    hint: "TCP-порт MCP/UI. По умолчанию 3333.",
  },
  DOC_RAG_ALLOWED_ORIGINS: {
    label: "CORS allow-list",
    hint: "Список Origin через запятую. Нужен, если клиент шлёт заголовок Origin. Не '*'.",
  },
  DOC_RAG_HTTP_LOG: {
    label: "Файл HTTP-лога",
    hint: "Путь для записи лога запросов. Пусто — не писать в файл.",
  },
  DOC_RAG_UI_RESTART_ENABLED: {
    label: "Разрешить рестарт из UI",
    hint: "Включает кнопку «Перезапустить сервис». Требует заданной команды ниже + sudoers.",
  },
  DOC_RAG_UI_RESTART_CMD: {
    label: "Команда рестарта",
    hint: "Например: sudo systemctl restart doc-rag-mcp. Выполняется при нажатии кнопки рестарта.",
  },
  DOC_RAG_UI_MAX_UPLOAD_MB: {
    label: "Лимит загрузки (МБ)",
    hint: "Максимальный размер одного загружаемого файла.",
  },
  DOC_RAG_MAX_CONCURRENCY: {
    label: "Параллелизм инструментов",
    hint: "Максимум одновременно выполняемых MCP-инструментов.",
  },
  DOC_RAG_RATE_LIMIT_RPS: {
    label: "Rate limit (RPS)",
    hint: "Устойчивая частота запросов на клиента. 0 — лимит выключен.",
  },
  DOC_RAG_RATE_LIMIT_BURST: {
    label: "Rate limit (burst)",
    hint: "Ёмкость всплеска токен-бакета.",
  },
  DOC_RAG_LOG_LEVEL: {
    label: "Уровень логов",
  },
  DOC_RAG_LOG_FORMAT: {
    label: "Формат логов",
    hint: "json — для лог-шипперов; text — человекочитаемый.",
  },
};

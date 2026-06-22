// Declarative schema for the structured config form. Each field maps to
// a dotted path in config.yaml; the form reads/writes values by that path
// and sends only changed paths to /ui/config/patch (comment-preserving).
//
// Intentionally omitted (edit via the Advanced/raw tab):
//   - `paths.*`           — filesystem layout, rarely touched
//   - `parsing.edition_year` — nested maps, not a flat field
//   - `server.*`          — does NOT control the production bind (env
//     DOC_RAG_HTTP_HOST/PORT does); host/port land in the Service (env)
//     editor instead.

export type FieldType = "text" | "int" | "float" | "bool" | "select";

export type Field = {
  path: string;
  label: string;
  type: FieldType;
  options?: string[];
  hint?: string;
  min?: number;
  max?: number;
};

export type Section = {
  title: string;
  fields: Field[];
};

export const CONFIG_SCHEMA: Section[] = [
  {
    title: "Парсинг",
    fields: [
      {
        path: "parsing.pdf_backend",
        label: "PDF backend",
        type: "select",
        options: ["docling", "auto", "cascade"],
        hint: "docling — единственный backend; cascade — Docling с fallback на Unstructured при ошибке.",
      },
      {
        path: "parsing.docx_backend",
        label: "DOCX backend",
        type: "select",
        options: ["python-docx", "docling"],
        hint: "python-docx — быстрый (по умолчанию); docling — структурный, дороже по времени и памяти.",
      },
      {
        path: "parsing.normalize_whitespace",
        label: "Нормализация пробелов",
        type: "bool",
      },
      {
        path: "parsing.min_chars_per_page",
        label: "Мин. символов на страницу",
        type: "int",
        min: 0,
        hint: "Порог «почти пустой страницы» для метрик покрытия (извлечённый текст).",
      },
    ],
  },
  {
    title: "Заголовки (sectioning)",
    fields: [
      {
        path: "sectioning.enable_numbered_headings",
        label: "Нумерованные заголовки",
        type: "bool",
      },
      {
        path: "sectioning.enable_allcaps_headings",
        label: "ЗАГОЛОВКИ КАПСОМ",
        type: "bool",
      },
      {
        path: "sectioning.min_heading_len",
        label: "Мин. длина заголовка",
        type: "int",
        min: 1,
      },
      {
        path: "sectioning.max_heading_len",
        label: "Макс. длина заголовка",
        type: "int",
        min: 1,
      },
    ],
  },
  {
    title: "Чанкинг",
    fields: [
      {
        path: "chunking.target_tokens",
        label: "Целевой размер чанка (токены)",
        type: "int",
        min: 1,
      },
      {
        path: "chunking.overlap_tokens",
        label: "Перекрытие (токены)",
        type: "int",
        min: 0,
      },
      {
        path: "chunking.dedup_similarity_threshold",
        label: "Порог дедупликации",
        type: "float",
        min: 0,
        max: 1,
        hint: "Word-bigram Jaccard между документами. 0 = выключено; 0.85 = убирать чанки с ≥85% совпадением. PDF имеет приоритет над DOCX.",
      },
    ],
  },
  {
    title: "Эмбеддинги",
    fields: [
      {
        path: "embeddings.model_name",
        label: "Модель",
        type: "text",
        hint: "Имя модели sentence-transformers, напр. BAAI/bge-large-en-v1.5.",
      },
      {
        path: "embeddings.device",
        label: "Устройство",
        type: "select",
        options: ["cpu", "cuda"],
        hint: "Целевой сервер без видеоускорителя — обычно cpu.",
      },
      {
        path: "embeddings.batch_size",
        label: "Batch size",
        type: "int",
        min: 1,
      },
      {
        path: "embeddings.normalize",
        label: "Нормализация векторов",
        type: "bool",
      },
    ],
  },
  {
    title: "Индекс",
    fields: [
      {
        path: "index.backend",
        label: "Backend",
        type: "select",
        options: ["faiss"],
      },
      {
        path: "index.metric",
        label: "Метрика",
        type: "select",
        options: ["ip", "l2"],
        hint: "ip — скалярное произведение (для нормализованных векторов ≈ косинус).",
      },
      {
        path: "index.top_k",
        label: "Top-K по умолчанию",
        type: "int",
        min: 1,
      },
    ],
  },
];

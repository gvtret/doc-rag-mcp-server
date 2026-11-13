// Type definitions for the `/ui/status` payload and the related
// responses. Kept loose where the backend itself is loose; tighten
// only when a field is load-bearing for the UI.

export type Document = {
  doc_id: string;
  source_file?: string;
  basename?: string;
  chunk_count?: number | null;
  sha256?: string | null;
  title_hint?: string | null;
  edition_year?: number | null;
  coverage?: {
    ocr?: {
      applied?: boolean;
      pages_recognized?: number;
      confidence?: number;
    };
  };
};

export type IndexedCatalog = {
  document_count?: number;
  manifest_present?: boolean;
  chunks_jsonl_present?: boolean;
  semantic_index_present?: boolean;
  lexical_search_ready?: boolean;
  semantic_search_ready?: boolean;
  manifest_generated_at_utc?: string | null;
  pipeline_version?: string | null;
  corpus_content_sha256?: string | null;
  documents?: Document[];
  error?: string;
};

export type StatusPayload = {
  version?: string;
  running?: boolean;
  job?: string | null;
  last_started?: number | null;
  last_finished?: number | null;
  last_ok?: boolean | null;
  last_error?: string | null;
  // Per-document progress fields (added in v2.0.1).
  current_doc?: string | null;
  docs_done?: number;
  docs_total?: number | null;
  current_doc_started_at?: number | null;
  eta_seconds?: number | null;
  eta_human?: string | null;
  // Indexing-phase progress (chunk-based).
  phase?: string | null;
  idx_done?: number;
  idx_total?: number | null;
  phase_elapsed_s?: number | null;
  idx_eta_s?: number | null;
  idx_eta_human?: string | null;
  // Tail logs.
  log_tail?: string[];
  http_log_tail?: string[];
  http_log_file?: string | null;
  rebuild_interrupted?: boolean;
  indexed?: IndexedCatalog;
};

export type HealthReady = {
  status: string;
  ready: boolean;
  has_manifest: boolean;
  job_running: boolean;
  job?: string | null;
  reasons?: string[];
};

export type PageKey = "documents" | "config" | "manage" | "logs" | "docs" | "stats";

export type ConfigRaw = {
  ok: true;
  path: string;
  yaml: string;
};

export type ConfigRawError = {
  ok: false;
  error: string;
};

export type ConfigSaveResponse = {
  ok: boolean;
  path?: string;
  error?: string;
};

// Parsed config.yaml for the structured form editor. `config` is the
// YAML mapping as JSON; the form reads/writes individual fields by path.
export type ConfigParsed = {
  ok: true;
  path: string;
  config: Record<string, unknown>;
};

export type ConfigParsedError = {
  ok: false;
  error: string;
};

// Field-level, comment-preserving write. `updates` is a map of dotted
// paths to values, e.g. { "chunking.target_tokens": 512 }.
export type ConfigPatchResponse = {
  ok: boolean;
  path?: string;
  error?: string;
};

// Service runtime env editor. Editable keys come back as `fields`.
export type EnvField = {
  key: string;
  type: "text" | "int" | "float" | "bool" | "select";
  options?: string[] | null;
  value: string;
  source: "file" | "env" | "default";
};

export type EnvGet = {
  ok: true;
  path: string;
  fields: EnvField[];
};

export type EnvGetError = {
  ok: false;
  error: string;
};

export type EnvSaveResponse = {
  ok: boolean;
  path?: string;
  error?: string;
};

export type RestartResponse = {
  ok: boolean;
  message?: string;
  error?: string;
};

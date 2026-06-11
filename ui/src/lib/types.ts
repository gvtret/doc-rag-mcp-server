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
  // Tail logs.
  log_tail?: string[];
  http_log_tail?: string[];
  http_log_file?: string | null;
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

export type PageKey = "documents" | "config" | "manage" | "logs";

// Thin wrappers around the FastAPI JSON endpoints. Same-origin
// fetches so the existing API key / CORS configuration on the server
// keeps applying transparently.

import type {
  ConfigParsed,
  ConfigParsedError,
  ConfigPatchResponse,
  ConfigRaw,
  ConfigRawError,
  ConfigSaveResponse,
  EnvGet,
  EnvGetError,
  EnvSaveResponse,
  HealthReady,
  RestartResponse,
  StatusPayload,
} from "./types";

async function getJSON<T>(path: string, signal?: AbortSignal): Promise<T> {
  const r = await fetch(path, { credentials: "same-origin", signal });
  if (!r.ok) {
    throw new Error(`${path}: HTTP ${r.status}`);
  }
  return (await r.json()) as T;
}

async function postForm<T>(
  path: string,
  body: Record<string, string>,
  signal?: AbortSignal,
): Promise<T> {
  const fd = new FormData();
  for (const [k, v] of Object.entries(body)) fd.append(k, v);
  const r = await fetch(path, {
    method: "POST",
    body: fd,
    credentials: "same-origin",
    signal,
  });
  // The config endpoints use HTTP status to signal validation
  // failures *and* return a JSON body with `ok: false`. We always
  // parse the body so callers see the server's error message.
  let parsed: T | { ok: false; error?: string };
  try {
    parsed = (await r.json()) as T;
  } catch {
    parsed = { ok: false, error: `${path}: HTTP ${r.status}` };
  }
  return parsed as T;
}

export function fetchStatus(signal?: AbortSignal): Promise<StatusPayload> {
  return getJSON<StatusPayload>("/ui/status", signal);
}

export function fetchHealthReady(signal?: AbortSignal): Promise<HealthReady> {
  return getJSON<HealthReady>("/health/ready", signal);
}

export function fetchConfigRaw(
  signal?: AbortSignal,
): Promise<ConfigRaw | ConfigRawError> {
  return getJSON<ConfigRaw | ConfigRawError>("/ui/config/raw", signal);
}

export function saveConfig(
  content: string,
  signal?: AbortSignal,
): Promise<ConfigSaveResponse> {
  return postForm<ConfigSaveResponse>("/ui/config/save", { content }, signal);
}

export function fetchConfigParsed(
  signal?: AbortSignal,
): Promise<ConfigParsed | ConfigParsedError> {
  return getJSON<ConfigParsed | ConfigParsedError>("/ui/config/parsed", signal);
}

// `updates` maps dotted config paths to values; serialized to JSON for
// the form-encoded POST body.
export function patchConfig(
  updates: Record<string, unknown>,
  signal?: AbortSignal,
): Promise<ConfigPatchResponse> {
  return postForm<ConfigPatchResponse>(
    "/ui/config/patch",
    { updates: JSON.stringify(updates) },
    signal,
  );
}

export function validateConfig(
  updates: Record<string, unknown>,
  signal?: AbortSignal,
): Promise<{ ok: true } | { ok: false; errors: { path: string; error: string }[] }> {
  return postForm(
    "/ui/config/validate",
    { updates: JSON.stringify(updates) },
    signal,
  );
}

export function fetchEnv(signal?: AbortSignal): Promise<EnvGet | EnvGetError> {
  return getJSON<EnvGet | EnvGetError>("/ui/env", signal);
}

export function saveEnv(
  updates: Record<string, string>,
  signal?: AbortSignal,
): Promise<EnvSaveResponse> {
  return postForm<EnvSaveResponse>(
    "/ui/env/save",
    { updates: JSON.stringify(updates) },
    signal,
  );
}

export function restartService(signal?: AbortSignal): Promise<RestartResponse> {
  return postForm<RestartResponse>("/ui/restart", {}, signal);
}

export interface ManageResult {
  ok: boolean;
  message?: string;
  removed?: string[];
  deleted?: string[];
  error?: string;
  saved?: number;
  dups?: number;
}

export async function uploadFiles(
  files: File[],
  signal?: AbortSignal,
): Promise<ManageResult> {
  const fd = new FormData();
  for (const f of files) fd.append("files", f);
  const r = await fetch("/ui/upload", {
    method: "POST",
    body: fd,
    credentials: "same-origin",
    redirect: "manual",
    signal,
  });
  if (r.type === "opaqueredirect" || r.status === 303) {
    return { ok: true, message: `Загружено файлов: ${files.length}` };
  }
  if (!r.ok) {
    const text = await r.text().catch(() => "");
    return { ok: false, error: text || `HTTP ${r.status}` };
  }
  return { ok: true, message: `Загружено файлов: ${files.length}` };
}

export function rebuildIndex(signal?: AbortSignal): Promise<ManageResult> {
  return postForm<ManageResult>("/ui/rebuild", {}, signal);
}

export function wipeIndex(confirm: string, signal?: AbortSignal): Promise<ManageResult> {
  return postForm<ManageResult>("/ui/wipe", { confirm }, signal);
}

export function cleanOrphans(signal?: AbortSignal): Promise<ManageResult> {
  return postForm<ManageResult>("/ui/clean-orphans", {}, signal);
}

export function clearIncoming(signal?: AbortSignal): Promise<ManageResult> {
  return postForm<ManageResult>("/ui/clear-incoming", {}, signal);
}

export interface QualityDoc {
  doc_id: string;
  pages: number;
  score: number;
  warning_count: number;
}

export interface QualitySummary {
  ok: boolean;
  schema_version?: number;
  documents?: QualityDoc[];
  message?: string;
  error?: string;
}

export function fetchQuality(signal?: AbortSignal): Promise<QualitySummary> {
  return getJSON<QualitySummary>("/ui/quality", signal);
}

export type AllDocument = {
  doc_id?: string;
  basename: string;
  source_file?: string;
  path?: string;
  chunk_count?: number | null;
  edition_year?: number | null;
  size?: number;
  indexed: boolean;
};

export type AllDocumentsResponse = {
  ok: boolean;
  indexed: AllDocument[];
  pending: AllDocument[];
  indexed_count: number;
  pending_count: number;
  error?: string;
};

export function fetchAllDocuments(signal?: AbortSignal): Promise<AllDocumentsResponse> {
  return getJSON<AllDocumentsResponse>("/ui/documents", signal);
}

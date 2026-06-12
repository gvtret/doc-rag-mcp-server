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

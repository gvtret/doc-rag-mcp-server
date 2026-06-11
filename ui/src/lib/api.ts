// Thin wrappers around the FastAPI JSON endpoints. Same-origin
// fetches so the existing API key / CORS configuration on the server
// keeps applying transparently.

import type { HealthReady, StatusPayload } from "./types";

async function getJSON<T>(path: string, signal?: AbortSignal): Promise<T> {
  const r = await fetch(path, { credentials: "same-origin", signal });
  if (!r.ok) {
    throw new Error(`${path}: HTTP ${r.status}`);
  }
  return (await r.json()) as T;
}

export function fetchStatus(signal?: AbortSignal): Promise<StatusPayload> {
  return getJSON<StatusPayload>("/ui/status", signal);
}

export function fetchHealthReady(signal?: AbortSignal): Promise<HealthReady> {
  return getJSON<HealthReady>("/health/ready", signal);
}

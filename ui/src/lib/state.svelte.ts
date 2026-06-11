// Central reactive state via Svelte 5 runes. Imported by components
// directly; no Svelte store boilerplate.
//
// The polling loop lives here so multiple components subscribing to
// the same data share one network round-trip.

import { fetchHealthReady, fetchStatus } from "./api";
import type { HealthReady, PageKey, StatusPayload } from "./types";

const POLL_MS = 5000;

class AppState {
  status = $state<StatusPayload | null>(null);
  health = $state<HealthReady | null>(null);
  lastUpdated = $state<Date | null>(null);
  error = $state<string | null>(null);
  page = $state<PageKey>("documents");
  // Sticky “last operation” info derived from /ui/status, updated when
  // a job transitions from running → finished. Used by the Footer.
  lastJobName = $state<string | null>(null);
  lastJobOk = $state<boolean | null>(null);
  lastJobDurationMs = $state<number | null>(null);

  private _polling = false;
  private _pollHandle: number | null = null;
  private _prevRunning = false;

  start() {
    if (this._polling) return;
    this._polling = true;
    void this._tick();
    this._pollHandle = window.setInterval(() => void this._tick(), POLL_MS);
  }

  stop() {
    this._polling = false;
    if (this._pollHandle !== null) {
      window.clearInterval(this._pollHandle);
      this._pollHandle = null;
    }
  }

  private async _tick() {
    try {
      const [s, h] = await Promise.allSettled([fetchStatus(), fetchHealthReady()]);
      if (s.status === "fulfilled") {
        const next = s.value;
        // Detect running → finished transition; remember last job.
        if (this._prevRunning && next.running === false) {
          const start = next.last_started ?? null;
          const end = next.last_finished ?? null;
          if (typeof start === "number" && typeof end === "number") {
            this.lastJobDurationMs = Math.max(0, Math.round((end - start) * 1000));
          }
          this.lastJobOk = next.last_ok ?? null;
        }
        if (typeof next.job === "string" && next.running) {
          this.lastJobName = next.job;
        }
        this._prevRunning = !!next.running;
        this.status = next;
      }
      if (h.status === "fulfilled") {
        this.health = h.value;
      }
      const sOk = s.status === "fulfilled";
      const hOk = h.status === "fulfilled";
      if (sOk || hOk) {
        this.lastUpdated = new Date();
      }
      const firstError = !sOk ? s.reason : !hOk ? h.reason : null;
      this.error = firstError ? String(firstError) : null;
    } catch (e) {
      this.error = e instanceof Error ? e.message : String(e);
    }
  }
}

export const appState = new AppState();

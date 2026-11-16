// Shared size formatting for Stats and Documents: one implementation so
// both pages agree on units, locale and the missing-data dash.

/**
 * Format a byte size with Russian units. Missing data (null/undefined)
 * renders as a dash, matching the table cells; NaN/negative are guarded
 * because the backend only sends non-negative ints, but a stale tab
 * should not print nonsense.
 */
export function formatSize(bytes: number | null | undefined): string {
  if (bytes === null || bytes === undefined || Number.isNaN(bytes) || bytes < 0) {
    return "—";
  }
  if (bytes < 1024) return `${bytes} Б`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} КБ`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} МБ`;
  return `${(bytes / 1024 / 1024 / 1024).toFixed(2)} ГБ`;
}

// Read a nested value out of a parsed-YAML object by a dotted path,
// e.g. getByPath(cfg, "chunking.target_tokens"). Returns undefined if any
// segment is missing.
export function getByPath(obj: unknown, path: string): unknown {
  let node: unknown = obj;
  for (const seg of path.split(".")) {
    if (node == null || typeof node !== "object") return undefined;
    node = (node as Record<string, unknown>)[seg];
  }
  return node;
}

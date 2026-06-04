from __future__ import annotations
import os
import re
from typing import Iterable, List, Optional, Set

def ensure_dir(path: str) -> None:
    if not path:
        return
    os.makedirs(path, exist_ok=True)

def safe_slug(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "doc"

def list_files_recursive(root: str, exts: Optional[Set[str]] = None) -> List[str]:
    out: List[str] = []
    if not os.path.isdir(root):
        return out
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            # Skip common temp/hidden files (MS Office lock files etc.)
            if fn.startswith("~$") or fn.startswith("."):
                continue
            p = os.path.join(dirpath, fn)
            if exts is not None:
                _, e = os.path.splitext(fn.lower())
                if e not in exts:
                    continue
            out.append(p)
    return sorted(out)

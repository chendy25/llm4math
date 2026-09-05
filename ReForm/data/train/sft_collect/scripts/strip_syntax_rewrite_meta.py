#!/usr/bin/env python3
"""Strip legacy ``[syntax_rewrite] ... ```lean4``` `` blocks from collected SFT JSONL.

Collector now silently replaces the last lean fence instead of appending a meta
tail. This cleans already-collected pilot files in place (with .bak once).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lean_syntax_fix import strip_syntax_rewrite_meta  # noqa: E402


def clean_record(row: dict) -> bool:
    """Return True if the record was modified."""
    changed = False
    msgs = row.get("messages")
    if isinstance(msgs, list):
        for m in msgs:
            if not isinstance(m, dict):
                continue
            c = m.get("content")
            if isinstance(c, str) and "[syntax_rewrite]" in c:
                new = strip_syntax_rewrite_meta(c)
                if new != c:
                    m["content"] = new
                    changed = True
    return changed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--targets",
        nargs="+",
        default=None,
        help="JSONL paths relative to data/train (default: sft_collect/{math,physics}/**/*.jsonl)",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.targets:
        targets = [
            (ROOT / p).resolve() if not Path(p).is_absolute() else Path(p)
            for p in args.targets
        ]
    else:
        targets = []
        for sub in ("math", "physics"):
            targets.extend(sorted((ROOT / "sft_collect" / sub).rglob("*.jsonl")))
        targets = [p for p in targets if not str(p).endswith(".bak")]

    total_files = total_rows = total_changed = 0
    for path in targets:
        if not path.is_file():
            continue
        rows = []
        file_changed = 0
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                total_rows += 1
                if clean_record(row):
                    file_changed += 1
                rows.append(row)
        if file_changed:
            total_files += 1
            total_changed += file_changed
            print(f"{path.relative_to(ROOT)}: cleaned {file_changed} rows")
            if not args.dry_run:
                bak = path.with_suffix(path.suffix + ".norewrite.bak")
                if not bak.exists():
                    shutil.copy2(path, bak)
                with path.open("w", encoding="utf-8") as f:
                    for row in rows:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    mode = "DRY-RUN" if args.dry_run else "WROTE"
    print(f"{mode}: files_touched={total_files} rows_changed={total_changed}/{total_rows}")


if __name__ == "__main__":
    main()

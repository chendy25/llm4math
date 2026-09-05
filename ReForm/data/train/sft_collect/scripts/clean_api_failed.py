#!/usr/bin/env python3
"""Remove API-exception rows from collect failed JSONL so resume can retry them."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

API_MARKERS = (
    "Access denied",
    "Error code: 400",
    "overdue-payment",
    "account is in good standing",
    "Insufficient Balance",
    "rate limit",
    "429",
)


def is_api_exception(row: dict) -> bool:
    err = str(row.get("error") or "")
    if not err:
        return False
    if any(m.lower() in err.lower() for m in API_MARKERS):
        return True
    # Exception path: minimal stub without trajectory
    if row.get("messages") is None and row.get("num_rounds") is None:
        return True
    return False


def clean_file(path: Path, *, dry_run: bool) -> tuple[int, int, int]:
    rows = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
    kept = [r for r in rows if not is_api_exception(r)]
    removed = len(rows) - len(kept)
    if removed and not dry_run:
        bak = path.with_suffix(path.suffix + ".pre_api_clean.bak")
        if not bak.exists():
            shutil.copy2(path, bak)
        with path.open("w", encoding="utf-8") as f:
            for r in kept:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows), len(kept), removed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--glob",
        default="sft_collect/math/failed/sft_math_full8k_shard*.failed.jsonl",
        help="Failed JSONL glob relative to data/train",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    paths = sorted(ROOT.glob(args.glob) if not Path(args.glob).is_absolute() else Path(args.glob))
    if not paths:
        # try from sft_collect cwd
        paths = sorted(Path(".").glob(args.glob))
    if not paths:
        raise SystemExit(f"no files match {args.glob}")

    tot_in = tot_kept = tot_rm = 0
    for p in paths:
        n, kept, rm = clean_file(p, dry_run=args.dry_run)
        print(f"{p}: in={n} kept={kept} removed_api={rm}")
        tot_in += n
        tot_kept += kept
        tot_rm += rm
    mode = "DRY-RUN" if args.dry_run else "WROTE"
    print(f"{mode}: total in={tot_in} kept={tot_kept} removed_api={tot_rm}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Backfill missing PhyX question_id by matching informal_statement to seeds.

PhyX collect previously dropped ids because phyx_to_input only read HF ``index``.
This rewrites physics collect JSONL in place (with .bak) after exact statement match.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # .../data/train
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phyx_adapter import _merge_statement  # noqa: E402


def build_statement_to_qid(seed_paths: list[Path]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    conflicts = 0
    for path in seed_paths:
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row.get("dataset_name") != "phyx":
                    continue
                qid = row.get("question_id")
                if qid is None or qid == "":
                    continue
                stmt = _merge_statement(row).strip()
                if not stmt:
                    continue
                qid_s = str(qid)
                if stmt in mapping and mapping[stmt] != qid_s:
                    conflicts += 1
                    continue
                mapping[stmt] = qid_s
    if conflicts:
        print(f"warning: skipped {conflicts} conflicting statement→qid pairs")
    return mapping


def backfill_file(path: Path, mapping: dict[str, str], *, dry_run: bool) -> tuple[int, int, int]:
    """Returns (phyx_rows, filled, unmatched)."""
    rows: list[dict] = []
    phyx = filled = unmatched = 0
    changed = False
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("dataset_name") == "phyx":
                phyx += 1
                qid = row.get("question_id")
                if qid is None or qid == "":
                    stmt = (row.get("informal_statement") or "").strip()
                    new_id = mapping.get(stmt)
                    if new_id is not None:
                        row["question_id"] = new_id
                        filled += 1
                        changed = True
                    else:
                        unmatched += 1
            rows.append(row)

    if changed and not dry_run:
        bak = path.with_suffix(path.suffix + ".bak")
        if not bak.exists():
            shutil.copy2(path, bak)
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return phyx, filled, unmatched


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--seeds",
        nargs="+",
        default=["sft_collect/batches/physics_mm_seed_all.jsonl"],
        help="PhyX-containing seed JSONL paths (relative to data/train)",
    )
    ap.add_argument(
        "--targets",
        nargs="+",
        default=None,
        help="JSONL files to rewrite (default: all under sft_collect/physics/)",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    seed_paths = [(ROOT / p).resolve() if not Path(p).is_absolute() else Path(p) for p in args.seeds]
    mapping = build_statement_to_qid(seed_paths)
    print(f"statement→qid map size: {len(mapping)}")

    if args.targets:
        targets = [(ROOT / p).resolve() if not Path(p).is_absolute() else Path(p) for p in args.targets]
    else:
        targets = sorted((ROOT / "sft_collect" / "physics").rglob("*.jsonl"))
        targets = [p for p in targets if not p.name.endswith(".bak")]

    tot_phyx = tot_filled = tot_unmatched = 0
    for path in targets:
        if not path.is_file():
            print(f"skip missing {path}")
            continue
        phyx, filled, unmatched = backfill_file(path, mapping, dry_run=args.dry_run)
        if phyx:
            print(f"{path.relative_to(ROOT)}: phyx={phyx} filled={filled} unmatched={unmatched}")
        tot_phyx += phyx
        tot_filled += filled
        tot_unmatched += unmatched

    mode = "DRY-RUN" if args.dry_run else "WROTE"
    print(f"{mode}: phyx={tot_phyx} filled={tot_filled} unmatched={tot_unmatched}")
    if tot_unmatched:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

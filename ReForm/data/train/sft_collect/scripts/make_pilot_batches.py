#!/usr/bin/env python3
"""Build smoke / pilot JSONL batches from full seed pools (stratified-ish)."""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def stratified_sample(rows: list[dict], n: int, seed: int) -> list[dict]:
    if n >= len(rows):
        return list(rows)
    by_ds: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_ds[str(r.get("dataset_name") or "unknown")].append(r)
    rng = random.Random(seed)
    for v in by_ds.values():
        rng.shuffle(v)
    keys = sorted(by_ds)
    out: list[dict] = []
    i = 0
    while len(out) < n:
        progressed = False
        for k in keys:
            if by_ds[k]:
                out.append(by_ds[k].pop())
                progressed = True
                if len(out) >= n:
                    break
        if not progressed:
            break
        i += 1
    rng.shuffle(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parents[1]),
        help="sft_collect root",
    )
    args = ap.parse_args()
    root = Path(args.root)
    batches = root / "batches"
    math_all = batches / "math_seed_8k.jsonl"
    phys_all = batches / "physics_mm_seed_all.jsonl"
    if not math_all.is_file() or not phys_all.is_file():
        raise SystemExit(f"missing seed batches under {batches}")

    math_rows = load_jsonl(math_all)
    phys_rows = load_jsonl(phys_all)

    plans = [
        ("smoke_physics10.jsonl", stratified_sample(phys_rows, 10, args.seed), "physics"),
        ("pilot_math500.jsonl", stratified_sample(math_rows, 500, args.seed + 1), "math"),
        ("pilot_physics100.jsonl", stratified_sample(phys_rows, 100, args.seed + 2), "physics"),
    ]
    for name, rows, kind in plans:
        path = batches / name
        write_jsonl(path, rows)
        from collections import Counter

        c = Counter(r.get("dataset_name") for r in rows)
        print(f"{name}: n={len(rows)} kind={kind} datasets={dict(c)}")


if __name__ == "__main__":
    main()

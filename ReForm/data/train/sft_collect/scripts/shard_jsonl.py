#!/usr/bin/env python3
"""Round-robin shard a JSONL into N files for parallel collect workers."""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", "-i", required=True)
    ap.add_argument("--outdir", "-o", required=True)
    ap.add_argument("--shards", "-n", type=int, default=4)
    ap.add_argument("--prefix", default="shard")
    args = ap.parse_args()
    if args.shards < 1:
        raise SystemExit("--shards must be >= 1")

    inp = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    handles = [
        (outdir / f"{args.prefix}_{i:02d}.jsonl").open("w", encoding="utf-8")
        for i in range(args.shards)
    ]
    counts = [0] * args.shards
    try:
        with inp.open(encoding="utf-8") as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                sid = i % args.shards
                handles[sid].write(line if line.endswith("\n") else line + "\n")
                counts[sid] += 1
    finally:
        for h in handles:
            h.close()
    for i, c in enumerate(counts):
        print(f"{args.prefix}_{i:02d}.jsonl\t{c}")
    print(f"total\t{sum(counts)}")


if __name__ == "__main__":
    main()

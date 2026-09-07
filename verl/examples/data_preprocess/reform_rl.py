#!/usr/bin/env python3
"""Build / refresh verl ReForm GRPO parquet (v2 prompts, no tool catalog).

Train = original math + official physics benches (PhyX / SeePhys / PhysUniBench).
physlib_parallel is NOT in the RL set.

Source of truth for the overlap dump is ``train_eval_overlap.parquet``
(the 12836-row file from the first export). ``train.parquet`` is rewritten.

  python3 examples/data_preprocess/reform_rl.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path("/mnt/afs_reason/chendongyang/code/llm4math")
DATA = ROOT / "verl" / "data" / "reform_rl"

PARALLEL_NAME = "physlib_parallel"


def _ds(extra) -> str:
    extra = extra or {}
    return str(extra.get("dataset_name") or "").strip().lower()


def _n_images(r) -> int:
    imgs = r.get("images")
    if imgs is None:
        return 0
    try:
        return int(len(imgs))
    except TypeError:
        return 0


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    overlap = DATA / "train_eval_overlap.parquet"
    current = DATA / "train.parquet"
    src = overlap if overlap.exists() else current
    if not src.exists():
        raise SystemExit(f"missing {src}")

    old = pd.read_parquet(src)
    kept = []
    dropped = Counter()
    for rec in old.to_dict(orient="records"):
        extra = rec.get("extra_info") or {}
        name = _ds(extra)
        if name == PARALLEL_NAME:
            dropped[name] += 1
            continue
        rec["agent_name"] = "reform_tool_agent"
        rec["data_source"] = rec.get("data_source") or "reform"
        kept.append(rec)

    out = pd.DataFrame(kept)
    out.to_parquet(current, index=False)
    recs = out.to_dict(orient="records")
    by = Counter(_ds(r.get("extra_info")) for r in recs)
    n_math = sum(1 for r in recs if r.get("ability") == "math")
    stats = {
        "train": {
            "n": int(len(out)),
            "math": int(n_math),
            "physics": int(len(out) - n_math),
            "with_image": int(sum(1 for r in recs if _n_images(r) > 0)),
        },
        "train_by_dataset": dict(by),
        "dropped_physlib_parallel": dict(dropped),
        "source": str(src),
        "note": (
            "Train = SFT-collection math (official math tests already removed) "
            "plus official physics benches PhyX / SeePhys / PhysUniBench. "
            "physlib_parallel is excluded. Val is AIME 2025 + PhyX-AF. "
            "Those three physics BOTH numbers overlap this train set."
        ),
    }
    (DATA / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))
    print(f"wrote {current}")


if __name__ == "__main__":
    main()

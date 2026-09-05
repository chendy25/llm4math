#!/usr/bin/env python3
"""Verify end-to-end that `--loss_scale ignore_tool_response` masks tool output.

Run inside the training env (the one that has torch/ms-swift installed):

    PYTHONPATH=$ROOT/ms-swift $CONDA/bin/python sft/scripts/check_loss_mask.py

Compares against `ignore_empty_think` (the previous setting) so the delta is the
tool-response span, and asserts the spans we *do* want supervised survive.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_MODEL = "/mnt/afs_reason/chendongyang/code/data/models/Qwen3.5-9B"
DEFAULT_DATA = Path("/mnt/afs_reason/chendongyang/code/llm4math/sft/data/math.jsonl")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--max-length", type=int, default=16384)
    args = ap.parse_args()

    # swift's loader, not AutoProcessor: `get_template` needs the `model_info`
    # attribute that only swift attaches. `load_model=False` under the hood, so
    # no weights are read.
    from swift.model import get_processor
    from swift.template import get_template

    proc = get_processor(args.model)

    row = None
    with args.data.open() as f:
        for line in f:
            rec = json.loads(line)
            if "<tool_response>" in (rec["messages"][-1]["content"] or ""):
                row = rec
                break
    if row is None:
        raise SystemExit(f"no tool-using sample in {args.data}")

    supervised = {}
    for plugin in ("ignore_empty_think", "ignore_tool_response"):
        template = get_template(proc, loss_scale=plugin, max_length=args.max_length)
        template.set_mode("train")
        encoded = template.encode(dict(row))
        labels = encoded["labels"]
        n_sup = sum(1 for x in labels if x != -100)
        supervised[plugin] = n_sup
        print(f"{plugin:>22}: total={len(labels)} supervised={n_sup} "
              f"masked={len(labels) - n_sup}")
        if plugin == "ignore_tool_response":
            text = template.safe_decode([x for x in labels if x != -100])
            checks = {
                "<tool_response> masked": "<tool_response>" not in text,
                "<tool_call> supervised": "<tool_call>" in text,
                "```lean4 supervised": "```lean4" in text,
                "critic analysis supervised": "Math Assertion Analysis" in text,
                "<answer> supervised": "<answer>" in text,
            }
            for name, ok in checks.items():
                print(f"   {'ok ' if ok else 'FAIL'} {name}")
            if not all(checks.values()):
                return 1

    if supervised["ignore_tool_response"] >= supervised["ignore_empty_think"]:
        print("[fail] the new plugin did not mask anything extra")
        return 1
    saved = 1 - supervised["ignore_tool_response"] / supervised["ignore_empty_think"]
    print(f"[ok] supervised tokens down {saved:.1%} on this sample")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

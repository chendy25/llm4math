#!/usr/bin/env python3
"""Token-length census of the ReForm SFT trajectories.

The single-shot ("RL rollout") eval spends one budget on the whole
<think>/<round>.../<answer> answer, so that budget has to be read off the
length distribution the model was actually trained on -- not guessed.

Reports, per split and for the whole mix:
  * total sequence length (user + assistant + ChatML overhead), which is what
    `--max_length 8192 --truncation_strategy delete` filtered on;
  * assistant-only length, which is what a single generate call must cover;
  * the same split by number of <round> blocks and by inline tool responses,
    since a refinement round or a retrieval hit is what makes a trajectory long.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from tokenizers import Tokenizer

DEFAULT_TOKENIZER = (
    "/mnt/afs_reason/chendongyang/code/llm4math/sft/output/"
    "qwen35_9b_reform_merged/checkpoint-284/tokenizer.json"
)
DEFAULT_DATA_DIR = Path("/mnt/afs_reason/chendongyang/code/llm4math/sft/data")

# `<|im_start|>user\n` + `<|im_end|>\n` + `<|im_start|>assistant\n` + `<|im_end|>`
CHATML_OVERHEAD = 10

ROUND_RE = re.compile(r"<round>")
TOOL_RESPONSE_RE = re.compile(r"<tool_response>")


def pct(sorted_vals: list[int], q: float) -> int:
    if not sorted_vals:
        return 0
    i = min(len(sorted_vals) - 1, int(q * (len(sorted_vals) - 1)))
    return sorted_vals[i]


def describe(name: str, vals: list[int]) -> str:
    if not vals:
        return f"  {name:<22} (empty)"
    s = sorted(vals)
    return (
        f"  {name:<22} n={len(s):<6} mean={sum(s)//len(s):<6} "
        f"p50={pct(s,.50):<6} p90={pct(s,.90):<6} p95={pct(s,.95):<6} "
        f"p99={pct(s,.99):<6} max={s[-1]}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--max-length", type=int, default=8192,
                    help="SFT --max_length, to count what truncation dropped.")
    ap.add_argument("--splits", default="math,physics,physlib")
    args = ap.parse_args()

    tok = Tokenizer.from_file(args.tokenizer)

    def n_tokens(text: str) -> int:
        return len(tok.encode(text, add_special_tokens=False).ids)

    all_total: list[int] = []
    all_resp: list[int] = []
    by_rounds: dict[int, list[int]] = {}
    by_tools: dict[int, list[int]] = {}
    dropped = 0
    kept_resp: list[int] = []

    for split in args.splits.split(","):
        path = args.data_dir / f"{split}.jsonl"
        if not path.is_file():
            print(f"[skip] {path}")
            continue
        totals: list[int] = []
        resps: list[int] = []
        split_dropped = 0
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                msgs = rec.get("messages") or []
                user = next((m for m in msgs if m.get("role") == "user"), None)
                asst = next((m for m in msgs if m.get("role") == "assistant"), None)
                if not user or not asst:
                    continue
                u = n_tokens(user.get("content") or "")
                a = n_tokens(asst.get("content") or "")
                # Images occupy real prompt tokens too (IMAGE_MAX_TOKEN_NUM=1024).
                img = 1024 * len(rec.get("images") or [])
                total = u + a + img + CHATML_OVERHEAD
                totals.append(total)
                resps.append(a)
                if total > args.max_length:
                    split_dropped += 1
                else:
                    kept_resp.append(a)
                    body = asst.get("content") or ""
                    r = len(ROUND_RE.findall(body))
                    t = len(TOOL_RESPONSE_RE.findall(body))
                    by_rounds.setdefault(min(r, 4), []).append(a)
                    by_tools.setdefault(min(t, 4), []).append(a)
        print(f"[{split}] {path.name}")
        print(describe("total (u+a+img)", totals))
        print(describe("assistant only", resps))
        print(f"  dropped by max_length={args.max_length}: {split_dropped}"
              f"/{len(totals)} ({split_dropped/max(1,len(totals)):.1%})")
        all_total += totals
        all_resp += resps
        dropped += split_dropped

    print("\n[ALL]")
    print(describe("total (u+a+img)", all_total))
    print(describe("assistant only", all_resp))
    print(f"  dropped by max_length={args.max_length}: {dropped}"
          f"/{len(all_total)} ({dropped/max(1,len(all_total)):.1%})")
    print(describe("assistant, TRAINED only", kept_resp))

    print("\n[assistant length of TRAINED samples, by #<round> blocks]")
    for k in sorted(by_rounds):
        label = f"rounds={k}" + ("+" if k == 4 else "")
        print(describe(label, by_rounds[k]))
    print("\n[assistant length of TRAINED samples, by #<tool_response>]")
    for k in sorted(by_tools):
        label = f"tool_resp={k}" + ("+" if k == 4 else "")
        print(describe(label, by_tools[k]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

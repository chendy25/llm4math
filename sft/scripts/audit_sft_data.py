#!/usr/bin/env python3
"""Audit the ReForm SFT dataset against the failure modes seen at eval time.

Every check here exists because a single-shot rollout exposed a defect that the
scaffolded collection pipeline hid. Run before re-collecting data:

    python scripts/audit_sft_data.py [--test-sets-dir eval/sets]

Checks
  1. Loss budget: how much of the trained signal sits on `<tool_response>`
     text, i.e. tokens the model must never generate at inference time.
  2. Oracle leakage: spans copied verbatim from the Lean server / external
     critic, which the model is asked to reproduce without access to either.
  3. Verdict balance: whether any trajectory ends on "Incorrect".
  4. Format mixture: reflective (`<round>`/`<answer>`) vs bare direct answers.
  5. Tag integrity: unbalanced `<tool_call>`, missing `<answer>`.
  6. Length-filter selection bias under `--max_length` + truncation=delete.
  7. Train/test contamination against the official eval sets.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from tokenizers import Tokenizer

ROOT = Path("/mnt/afs_reason/chendongyang/code/llm4math/sft")
DEFAULT_TOKENIZER = ROOT / "output/qwen35_9b_reform_merged/checkpoint-284/tokenizer.json"
CHATML_OVERHEAD = 10

TOOL_RESP = re.compile(r"<tool_response>(.*?)</tool_response>", re.DOTALL)
ROUND = re.compile(r"<round>")
ANSWER = re.compile(r"<answer>")
EMPTY_THINK = re.compile(r"<think>\s*</think>")
LEAN_ERR = re.compile(r"Lean compiler reported errors:")
VACUOUS = re.compile(r"Vacuous formalization rejected")
UNAVAILABLE = re.compile(r"Lean verification unavailable:")
SYNTAX_REWRITE = re.compile(r"\[syntax_rewrite\]")
VERDICT = re.compile(
    r"outcome of the consistency verification is:\s*(Correct|Incorrect)", re.I
)
# The critic's five-stage analysis, pasted into the assistant turn as if the
# model had produced it.
CRITIC_STAGES = re.compile(r"Math Assertion Analysis")
THINK = re.compile(r"<think>")
# Must match `--loss_scale` in sft_qwen35_9b_lora.sh.
LOSS_SCALE_PLUGIN = "ignore_tool_response"


def norm(text: str) -> str:
    """Aggressive normalisation for contamination matching."""
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def pctl(vals: list[int], q: float) -> int:
    if not vals:
        return 0
    s = sorted(vals)
    return s[min(len(s) - 1, int(q * (len(s) - 1)))]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    ap.add_argument("--splits", default="math,physics,physlib")
    ap.add_argument("--tokenizer", default=str(DEFAULT_TOKENIZER))
    ap.add_argument("--max-length", type=int, default=8192)
    ap.add_argument("--test-sets-dir", type=Path, default=ROOT / "eval/sets")
    args = ap.parse_args()

    tok = Tokenizer.from_file(args.tokenizer)

    def nt(text: str) -> int:
        return len(tok.encode(text, add_special_tokens=False).ids)

    rows: list[dict] = []
    for split in args.splits.split(","):
        path = args.data_dir / f"{split}.jsonl"
        if not path.is_file():
            print(f"[skip] {path}")
            continue
        for line in path.open():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            msgs = rec.get("messages") or []
            user = next((m for m in msgs if m.get("role") == "user"), None)
            asst = next((m for m in msgs if m.get("role") == "assistant"), None)
            if not user or not asst:
                continue
            u = user.get("content") or ""
            a = asst.get("content") or ""
            tool_spans = TOOL_RESP.findall(a)
            a_tok = nt(a)
            tool_tok = sum(nt(s) for s in tool_spans)
            rows.append({
                "split": split,
                "user": u,
                "asst": a,
                "u_tok": nt(u),
                "a_tok": a_tok,
                "tool_tok": tool_tok,
                "img_tok": 1024 * len(rec.get("images") or []),
                "rounds": len(ROUND.findall(a)),
                "has_answer": bool(ANSWER.search(a)),
                "empty_think": bool(EMPTY_THINK.search(a)),
                "lean_err": len(LEAN_ERR.findall(a)),
                "vacuous": bool(VACUOUS.search(a)),
                "unavailable": bool(UNAVAILABLE.search(a)),
                "rewrite": bool(SYNTAX_REWRITE.search(a)),
                "critic_stages": bool(CRITIC_STAGES.search(a)),
                "verdicts": VERDICT.findall(a),
                "unbalanced": a.count("<tool_call>") != a.count("</tool_call>"),
                "n_tool_calls": a.count("<tool_call>"),
            })
    for r in rows:
        r["total"] = r["u_tok"] + r["a_tok"] + r["img_tok"] + CHATML_OVERHEAD
        r["kept"] = r["total"] <= args.max_length

    n = len(rows)
    kept = [r for r in rows if r["kept"]]
    refl = [r for r in rows if r["rounds"] > 0]
    kept_refl = [r for r in kept if r["rounds"] > 0]
    print(f"=== dataset: {n} samples, {len(kept)} survive --max_length "
          f"{args.max_length} (truncation=delete drops {n - len(kept)})")

    print("\n--- 1. where does the training loss actually go? ---")
    for label, pool in (("all", rows), ("trained", kept)):
        a_sum = sum(r["a_tok"] for r in pool)
        t_sum = sum(r["tool_tok"] for r in pool)
        print(f"  [{label}] assistant tokens={a_sum} of which "
              f"<tool_response>={t_sum} ({t_sum/max(1,a_sum):.1%})")
    for label, pool in (("math+physics reflective", kept_refl),):
        a_sum = sum(r["a_tok"] for r in pool)
        t_sum = sum(r["tool_tok"] for r in pool)
        print(f"  [{label}] n={len(pool)} assistant={a_sum} "
              f"tool_response={t_sum} ({t_sum/max(1,a_sum):.1%})")
    a_sum = sum(r["a_tok"] for r in kept)
    t_sum = sum(r["tool_tok"] for r in kept)
    print(f"  --loss_scale {LOSS_SCALE_PLUGIN}: the tool spans above get weight "
          f"0, leaving {a_sum - t_sum} of {a_sum} assistant tokens supervised.")
    print("  Without it ms-swift applies full LM loss to the whole assistant "
          "turn, spliced retrieval output included.")

    print("\n--- 2. oracle leakage (text the model cannot reproduce) ---")
    for key, desc in (
        ("lean_err", "Lean server compile errors pasted in"),
        ("vacuous", "pipeline's own 'Vacuous formalization rejected'"),
        ("unavailable", "'Lean verification unavailable' (infra noise)"),
        ("rewrite", "'[syntax_rewrite]' marker from lean_syntax_fix"),
        ("critic_stages", "external critic's 5-stage analysis"),
    ):
        c = sum(1 for r in kept_refl if r[key])
        print(f"  {desc:<48} {c:>5}/{len(kept_refl)} ({c/max(1,len(kept_refl)):.1%})")

    print("\n--- 3. final verdict balance ---")
    fin = Counter(
        (r["verdicts"][-1].capitalize() if r["verdicts"] else "none") for r in kept_refl
    )
    print(f"  trained reflective trajectories end on: {dict(fin)}")
    mid_incorrect = sum(1 for r in kept_refl if any(
        v.lower() == "incorrect" for v in r["verdicts"][:-1]
    ))
    print(f"  have an 'Incorrect' verdict mid-trajectory: {mid_incorrect}"
          f"/{len(kept_refl)} ({mid_incorrect/max(1,len(kept_refl)):.1%})")

    print("\n--- 4. format mixture ---")
    fmt = Counter()
    for r in kept:
        a = r["asst"]
        has_think = bool(THINK.search(a))
        if r["rounds"] > 0 and r["has_answer"] and has_think:
            fmt["reflective (<think> + <round> + <answer>)"] += 1
        elif has_think and r["has_answer"]:
            fmt["direct (<think> + <answer>, no <round>)"] += 1
        elif r["has_answer"] and not has_think:
            # physlib: Lean lives in <answer>, no canned <think>.
            fmt["<answer> only (no <think>/<round>)"] += 1
        elif not has_think and not r["has_answer"] and "```lean4" in a:
            fmt["bare ```lean4, no <think>/<round>/<answer>"] += 1
        else:
            fmt["other"] += 1
    for k, v in fmt.most_common():
        print(f"  {k:<48} {v:>5}/{len(kept)} ({v/max(1,len(kept)):.1%})")

    print("\n--- 5. tag integrity (tool-using trajectories) ---")
    tool_rows = [r for r in rows if r["n_tool_calls"] > 0]
    unb = sum(1 for r in tool_rows if r["unbalanced"])
    noans = sum(1 for r in refl if not r["has_answer"])
    print(f"  unbalanced <tool_call>/</tool_call>: {unb}/{len(tool_rows)} "
          f"({unb/max(1,len(tool_rows)):.1%})")
    print(f"  reflective but no <answer>:          {noans}/{len(refl)}")

    print("\n--- 6. length-filter selection bias ---")
    dropped = [r for r in rows if not r["kept"]]
    for label, pool in (("trained", kept_refl), ("deleted", [r for r in dropped if r["rounds"] > 0])):
        if not pool:
            continue
        multi = sum(1 for r in pool if r["rounds"] >= 2)
        with_err = sum(1 for r in pool if r["lean_err"])
        print(f"  [{label}] n={len(pool)} rounds>=2 {multi/len(pool):.1%} "
              f"has_lean_error_round {with_err/len(pool):.1%}")
    # Refinement is now purely semantic (the Lean gate no longer writes rounds),
    # so a multi-round trajectory *is* the repair demonstration.
    all_multi = sum(1 for r in refl if r["rounds"] >= 2)
    kept_multi = sum(1 for r in kept_refl if r["rounds"] >= 2)
    print(f"  refinement demonstrations kept: {kept_multi}/{all_multi} "
          f"({1 - kept_multi/max(1,all_multi):.1%} were deleted)")
    tot = [r["total"] for r in rows]
    print(f"  total-length percentiles: p50={pctl(tot,.5)} p90={pctl(tot,.9)} "
          f"p95={pctl(tot,.95)} p99={pctl(tot,.99)} max={max(tot)}")
    for cand in (8192, 12288, 16384, 20480, 24576):
        cov = sum(1 for t in tot if t <= cand) / max(1, len(tot))
        print(f"    max_length={cand:<6} keeps {cov:.1%}")

    print("\n--- 7. train/test contamination ---")
    train_keys = {}
    for r in rows:
        train_keys.setdefault(norm(r["user"])[-400:], 0)
        train_keys[norm(r["user"])[-400:]] += 1
    if args.test_sets_dir.is_dir():
        for tp in sorted(args.test_sets_dir.glob("*_all.jsonl")):
            items = [json.loads(l) for l in tp.open() if l.strip()]
            hits = 0
            for it in items:
                st = it.get("informal_statement") or it.get("question") or ""
                k = norm(st)[-400:]
                if k and k in train_keys:
                    hits += 1
            print(f"  {tp.name:<44} {hits}/{len(items)} statements also in train")
    else:
        print(f"  [skip] {args.test_sets_dir} not found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

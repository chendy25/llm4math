#!/usr/bin/env python3
"""Convert physlib_corpus.jsonl → parallel SFT messages (no tools / critic)."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


OUTPUT_KINDS = {"theorem", "definition", "instance", "abbrev", "opaque", "axiom"}


def _pp_name(name) -> str:
    if isinstance(name, list):
        return ".".join(str(x) for x in name)
    return str(name or "")


def _module_str(module_name) -> str:
    if isinstance(module_name, list):
        return ".".join(str(x) for x in module_name)
    return str(module_name or "")


def row_to_sample(row: dict) -> dict | None:
    kind = (row.get("kind") or "").strip()
    if kind not in OUTPUT_KINDS:
        return None
    informal_name = str(row.get("informal_name") or "").strip()
    informal_desc = str(row.get("informal_description") or "").strip()
    lean_type = str(row.get("type") or "").strip()
    name = _pp_name(row.get("name"))
    if not informal_name or not informal_desc or not lean_type or not name:
        return None
    if len(lean_type) > 12000:
        return None

    module = _module_str(row.get("module_name"))
    user = (
        f"Formalize the following Physlib declaration into Lean 4.\n\n"
        f"Name (informal): {informal_name}\n"
        f"Description: {informal_desc}\n"
    )
    if module:
        user += f"Module hint: {module}\n"

    kw = "theorem" if kind == "theorem" else ("def" if kind in {"definition", "abbrev"} else kind)
    assistant = (
        f"```lean4\n"
        f"{kw} {name} : {lean_type} := by\n"
        f"  sorry\n"
        f"```"
    )
    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "informal_statement": f"{informal_name}\n{informal_desc}",
        "dataset_name": "physlib_parallel",
        "question_id": name,
        "kind": kind,
        "module_name": module,
        "retrieval_domain": "physlib_lib",
        "is_parallel_physlib": True,
        "success": True,
        "syntax_pass": True,
        "tools_used": [],
        "num_tool_calls": 0,
        "num_rounds": 1,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        type=Path,
        default=Path(
            "/mnt/afs_reason/chendongyang/code/llm4math/LeanSearch-v2/"
            "data/corpus/final/physlib_corpus.jsonl"
        ),
    )
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--n", type=int, default=2000, help="How many samples to write (0=all usable)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    usable: list[dict] = []
    skipped = Counter()
    with args.input.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sample = row_to_sample(row)
            if sample is None:
                skipped[(row.get("kind") or "?")] += 1
                continue
            usable.append(sample)

    by_mod: dict[str, list[dict]] = defaultdict(list)
    for s in usable:
        top = (s.get("module_name") or "Unknown").split(".")[:2]
        key = ".".join(top) if top else "Unknown"
        by_mod[key].append(s)

    rng = random.Random(args.seed)
    for v in by_mod.values():
        rng.shuffle(v)

    # Round-robin stratified sample
    chosen: list[dict] = []
    if args.n <= 0 or args.n >= len(usable):
        chosen = usable
        rng.shuffle(chosen)
    else:
        keys = sorted(by_mod.keys())
        idxs = {k: 0 for k in keys}
        while len(chosen) < args.n:
            progress = False
            for k in keys:
                i = idxs[k]
                if i < len(by_mod[k]):
                    chosen.append(by_mod[k][i])
                    idxs[k] = i + 1
                    progress = True
                    if len(chosen) >= args.n:
                        break
            if not progress:
                break

    out = args.output_dir / f"physlib_parallel_{len(chosen)}.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for s in chosen:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    stats = {
        "input": str(args.input),
        "usable_total": len(usable),
        "written": len(chosen),
        "output": str(out),
        "kind_dist": dict(Counter(s["kind"] for s in chosen)),
        "module_top_dist": dict(Counter(s["module_name"].split(".")[0] if s["module_name"] else "?" for s in chosen)),
        "skipped_by_kind": dict(skipped),
    }
    stats_path = args.output_dir / f"physlib_parallel_{len(chosen)}.stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

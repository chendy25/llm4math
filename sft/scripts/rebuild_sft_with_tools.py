#!/usr/bin/env python3
"""Rewrite SFT jsonl user turns so they include the shared tool catalog.

Reads the v2 split files (math / physics / physlib), appends
``prompts.tool_prompt(domain)`` after the existing user instruction, and
writes train_mix_tools.jsonl. Assistant turns are unchanged.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path("/mnt/afs_reason/chendongyang/code/llm4math")
TRAIN = ROOT / "ReForm/data/train"
DATA = ROOT / "sft/data"
sys.path.insert(0, str(TRAIN))

from prompts import REFORM_USER_INSTRUCTION, format_user_prompt, tool_prompt  # noqa: E402

PREFIX = REFORM_USER_INSTRUCTION.split("{informal_statement}")[0]


def _user_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict) and p.get("type") in (None, "text"):
                parts.append(p.get("text") or "")
            elif isinstance(p, str):
                parts.append(p)
        return "".join(parts)
    return str(content or "")


def _rewrite_user(text: str, domain: str) -> str:
    raw = text or ""
    image = raw.startswith("<image>")
    body = raw[len("<image>") :] if image else raw
    body = body.strip()
    if body.startswith(PREFIX.strip()):
        informal = body[len(PREFIX) :].lstrip("\n")
        new = format_user_prompt(informal, domain)
    elif "Formalize the following Physlib" in body:
        new = body.rstrip() + "\n\n" + tool_prompt("physics")
    else:
        new = format_user_prompt(body, domain)
    return ("<image>" + new) if image else new


def _iter(path: Path):
    with path.open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    random.seed(42)
    out_rows = []
    stats = {"math": 0, "physics": 0, "physlib": 0}
    for name, domain in (("math", "math"), ("physics", "physics"), ("physlib", "physics")):
        src = DATA / f"{name}.jsonl"
        for rec in _iter(src):
            msgs = rec.get("messages") or []
            for m in msgs:
                if m.get("role") == "user":
                    m["content"] = _rewrite_user(_user_text(m.get("content")), domain)
            out_rows.append(rec)
            stats[name] += 1
    random.shuffle(out_rows)
    dest = DATA / "train_mix_tools.jsonl"
    with dest.open("w") as f:
        for rec in out_rows:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    smoke = DATA / "train_smoke_tools.jsonl"
    with smoke.open("w") as f:
        for rec in out_rows[:50]:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(json.dumps({"wrote": str(dest), "n": len(out_rows), **stats}, indent=2))


if __name__ == "__main__":
    main()

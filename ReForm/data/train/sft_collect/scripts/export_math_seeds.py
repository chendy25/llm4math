#!/usr/bin/env python3
"""Export math informal_statement seeds from local jsonl + optional HF caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} -> {path}")


def export_local(reform_root: Path, out_dir: Path) -> None:
    mapping = {
        "aime2025": reform_root / "data/test/aime2025.jsonl",
        "minif2f_testset": reform_root / "data/test/minif2f_testset.jsonl",
        "proofnet_testset": reform_root / "data/test/proofnet_testset.jsonl",
        "putnam_v4": reform_root / "data/test/putnam_v4.jsonl",
    }
    all_rows: list[dict] = []
    for ds, path in mapping.items():
        if not path.is_file():
            print(f"skip missing {path}")
            continue
        rows = []
        with path.open() as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                informal = (
                    obj.get("informal_statement")
                    or obj.get("question")
                    or obj.get("text")
                    or obj.get("problem")
                    or ""
                ).strip()
                if not informal:
                    continue
                qid = str(obj.get("question_id") or obj.get("id") or obj.get("name") or i)
                rows.append(
                    {
                        "dataset_name": obj.get("dataset_name") or ds,
                        "question_id": qid,
                        "informal_statement": informal,
                        "retrieval_domain": "math",
                        "attached_statement": obj.get("attached_statement"),
                    }
                )
        _write(out_dir / f"math_{ds}.jsonl", rows)
        all_rows.extend(rows)
    _write(out_dir / "math_local_all.jsonl", all_rows)


def export_numina(cache_dir: Path, out_dir: Path, max_per_source: int | None) -> None:
    """Expect a jsonl dump or datasets arrow export; also supports loading via datasets if installed."""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit(
            "datasets not installed; pip install datasets or pass a pre-exported jsonl"
        ) from e

    sources = {"olympiads", "amc_aime", "aops_forum", "olympiads_ref"}
    ds = load_dataset("AI-MO/NuminaMath-1.5", split="train", cache_dir=str(cache_dir))
    rows: list[dict] = []
    counts: dict[str, int] = {s: 0 for s in sources}
    for i, ex in enumerate(ds):
        src = ex.get("source") or ""
        if src not in sources:
            continue
        if max_per_source is not None and counts[src] >= max_per_source:
            if all(counts[s] >= max_per_source for s in sources):
                break
            continue
        problem = (ex.get("problem") or ex.get("question") or "").strip()
        if not problem:
            continue
        rows.append(
            {
                "dataset_name": f"numina15_{src}",
                "question_id": str(ex.get("id") or f"{src}_{i}"),
                "informal_statement": problem,
                "retrieval_domain": "math",
                "source": src,
            }
        )
        counts[src] += 1
    _write(out_dir / "math_numina_competition.jsonl", rows)
    print("numina counts", counts)


def export_aops(cache_dir: Path, out_dir: Path, limit: int | None) -> None:
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit("datasets not installed") from e
    ds = load_dataset("AI-MO/aops", split="train", cache_dir=str(cache_dir))
    rows = []
    for i, ex in enumerate(ds):
        if limit is not None and i >= limit:
            break
        problem = (ex.get("problem") or "").strip()
        if not problem:
            continue
        rows.append(
            {
                "dataset_name": "aops",
                "question_id": str(ex.get("id") or i),
                "informal_statement": problem,
                "retrieval_domain": "math",
                "tags": ex.get("tags"),
            }
        )
    _write(out_dir / "math_aops.jsonl", rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--reform-root",
        type=Path,
        default=Path("/mnt/afs_reason/chendongyang/code/llm4math/ReForm"),
    )
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--hf-cache", type=Path, default=Path("/mnt/afs_reason/chendongyang/code/data/hf_datasets"))
    ap.add_argument("--local-only", action="store_true")
    ap.add_argument("--numina", action="store_true")
    ap.add_argument("--aops", action="store_true")
    ap.add_argument("--max-per-numina-source", type=int, default=8000)
    ap.add_argument("--aops-limit", type=int, default=30000)
    args = ap.parse_args()

    export_local(args.reform_root, args.out_dir / "local")
    if args.local_only:
        return 0
    if args.numina:
        export_numina(args.hf_cache, args.out_dir, args.max_per_numina_source)
    if args.aops:
        export_aops(args.hf_cache, args.out_dir, args.aops_limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

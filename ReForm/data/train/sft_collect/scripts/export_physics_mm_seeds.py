#!/usr/bin/env python3
"""Export multimodal physics seeds (must keep an image path or bytes ref)."""

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


def export_physunibench(root: Path, out: Path) -> None:
    """PrismaX/PhysUniBench layout: *_{en,zh}_{MCQ,OE}.json + images/."""
    img_dir = root / "images"
    rows: list[dict] = []
    for jf in sorted(root.glob("PhysUniBench_*.json")) + sorted(root.glob("*.json")):
        if jf.name == "README.md":
            continue
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"skip {jf}: {e}")
            continue
        if not isinstance(data, list):
            continue
        for i, ex in enumerate(data):
            q = (
                ex.get("question")
                or ex.get("problem")
                or ex.get("query")
                or ""
            ).strip()
            if not q:
                continue
            img = ex.get("image") or ex.get("image_path") or ex.get("img")
            if isinstance(img, str):
                img_path = Path(img)
                if not img_path.is_file():
                    img_path = img_dir / Path(img).name
            else:
                # common: image id / filename field
                fname = ex.get("image_filename") or ex.get("figure") or f"{ex.get('id', i)}.png"
                img_path = img_dir / str(fname)
            if not img_path.is_file():
                # try id.png
                alt = img_dir / f"{ex.get('id', i)}.png"
                if alt.is_file():
                    img_path = alt
                else:
                    continue
            caption = (ex.get("image_caption") or ex.get("caption") or "").strip()
            desc = (ex.get("question_description") or ex.get("description") or "").strip()
            rows.append(
                {
                    "dataset_name": "physunibench",
                    "question_id": str(ex.get("id") or f"{jf.stem}_{i}"),
                    "question": q,
                    "question_description": desc,
                    "image_caption": caption,
                    "image": str(img_path.resolve()),
                    "category": ex.get("subject") or ex.get("category") or ex.get("discipline"),
                    "options": ex.get("options") or ex.get("choices"),
                    "answer": ex.get("answer"),
                    "retrieval_domain": "physics",
                    "source_file": jf.name,
                }
            )
    _write(out, rows)


def export_phyx_local(root: Path, out: Path) -> None:
    """Prefer parquet/jsonl under Cloudriver/PhyX snapshot; fallback Phyx_100."""
    rows: list[dict] = []
    # try datasets load from local
    try:
        from datasets import load_dataset

        ds = load_dataset(str(root), split="test_mini")
        for i, ex in enumerate(ds):
            img = ex.get("image")
            # save image later at collect time; store HF index
            rows.append(
                {
                    "dataset_name": "phyx",
                    "question_id": str(ex.get("id") or ex.get("index") or i),
                    "question": ex.get("question") or "",
                    "question_description": ex.get("question_description") or "",
                    "image_caption": ex.get("image_caption") or "",
                    "image": None,  # filled by collector via datasets
                    "image_hf_ref": {"repo": str(root), "split": "test_mini", "index": i},
                    "category": ex.get("category"),
                    "options": ex.get("options"),
                    "answer": ex.get("answer"),
                    "retrieval_domain": "physics",
                }
            )
        _write(out, rows)
        return
    except Exception as e:
        print(f"PhyX datasets load failed ({e}); trying Phyx_100 fallback")

    # Multimodal_Autoformalization Phyx_100
    for name, img_subdir in (("data.jsonl", "image"), ("test_100.jsonl", "test_100_image")):
        p = root / name
        if not p.is_file():
            continue
        img_root = root / img_subdir
        with p.open() as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                ex = json.loads(line)
                img = ex.get("image") or ex.get("image_path")
                if not img:
                    continue
                img_path = Path(str(img))
                if not img_path.is_file():
                    img_path = img_root / Path(str(img)).name
                if not img_path.is_file():
                    continue
                rows.append(
                    {
                        "dataset_name": "phyx",
                        "question_id": str(ex.get("index") or ex.get("id") or i),
                        "question": ex.get("question") or "",
                        "question_description": ex.get("question_description") or ex.get("description") or "",
                        "image_caption": ex.get("image_caption") or "",
                        "image": str(img_path.resolve()),
                        "category": ex.get("category"),
                        "options": ex.get("options"),
                        "answer": ex.get("answer"),
                        "retrieval_domain": "physics",
                    }
                )
    _write(out, rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-root", type=Path, default=Path("/mnt/afs_reason/chendongyang/code/data/hf_datasets"))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--phyx100", type=Path, default=Path(
        "/mnt/afs_reason/chendongyang/code/llm4math/Multimodal_Autoformalization/data/Phyx_100_data"
    ))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    pub = args.hf_root / "PrismaX__PhysUniBench"
    if pub.is_dir():
        export_physunibench(pub, args.out_dir / "physics_physunibench.jsonl")
    else:
        print(f"PhysUniBench not found at {pub}")

    phyx = args.hf_root / "Cloudriver__PhyX"
    if phyx.is_dir():
        export_phyx_local(phyx, args.out_dir / "physics_phyx.jsonl")
    elif args.phyx100.is_dir():
        export_phyx_local(args.phyx100, args.out_dir / "physics_phyx100.jsonl")
    else:
        print("PhyX not available yet")

    see = args.hf_root / "SeePhys__SeePhys"
    if see.is_dir():
        # generic: dump whatever jsonl/json present with image fields
        print(f"SeePhys present at {see}; listing:", [p.name for p in see.iterdir()][:30])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

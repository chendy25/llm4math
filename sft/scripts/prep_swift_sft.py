#!/usr/bin/env python3
"""Convert ReForm SFT collect jsonl into ms-swift messages(+images) format."""

from __future__ import annotations

import argparse
import base64
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sanitize_traj  # noqa: E402

DATA_URL_RE = re.compile(r"^data:image/([\w+.-]+);base64,(.+)$", re.DOTALL)
UNSAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")

SRC_DEFAULT = Path(
    "/mnt/afs_reason/chendongyang/code/llm4math/ReForm/data/train/sft_collect"
)
OUT_DEFAULT = Path("/mnt/afs_reason/chendongyang/code/llm4math/sft/data")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=Path, default=SRC_DEFAULT)
    p.add_argument("--out-dir", type=Path, default=OUT_DEFAULT)
    p.add_argument(
        "--filter",
        choices=("reform", "success", "syntax", "all"),
        default="reform",
        help="reform (default)=repaired successes plus a bounded share of "
             "honestly-labelled negatives; success=verified trajectories only.",
    )
    p.add_argument(
        "--negative-ratio",
        type=float,
        default=0.15,
        help="Share of reflective samples that end on an 'Incorrect' verdict. "
             "The first mixture was 4041/4041 Correct, so the model learned to "
             "close every trajectory with 'Correct' regardless of the "
             "statement. Only negatives whose final statement compiles are "
             "eligible, so the target never contains broken Lean.",
    )
    p.add_argument(
        "--no-sanitize",
        action="store_true",
        help="Keep the raw collected trajectories (compiler-error rounds, "
             "[syntax_rewrite] notes, dangling <tool_call>). For A/B only.",
    )
    p.add_argument(
        "--no-excise-prose",
        action="store_true",
        help="Drop a trajectory whose prose narrates a compiler message instead "
             "of editing the offending sentences out. Costs 418 trajectories.",
    )
    p.add_argument(
        "--no-physlib-wrap",
        action="store_true",
        help="Leave physlib answers as bare ```lean4 blocks.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke-n", type=int, default=50)
    return p.parse_args()


def keep_record(rec: dict, mode: str) -> bool:
    if mode == "all":
        return True
    if mode == "syntax":
        return bool(rec.get("syntax_pass") or rec.get("success"))
    if mode == "reform":
        # Negatives are subsampled later; here we only exclude trajectories
        # whose final statement does not compile.
        return bool(rec.get("success") or rec.get("syntax_pass"))
    return bool(rec.get("success"))


def safe_name(text: str, fallback: str) -> str:
    cleaned = UNSAFE_NAME_RE.sub("_", str(text or "")).strip("._")
    return (cleaned or fallback)[:120]


def decode_data_url(url: str) -> tuple[str, bytes] | None:
    if not url:
        return None
    m = DATA_URL_RE.match(url)
    if not m:
        return None
    fmt = m.group(1).lower().split("+")[0]
    ext = "jpg" if fmt in {"jpeg", "jpg"} else ("png" if fmt == "png" else fmt)
    try:
        return ext, base64.b64decode(m.group(2))
    except Exception:
        return None


def extract_image_from_part(part: dict) -> tuple[str, bytes] | Path | None:
    if part.get("type") not in {"image_url", "image"} and "image_url" not in part:
        return None
    raw = part.get("image_url") or part.get("image") or part.get("url")
    if isinstance(raw, dict):
        raw = raw.get("url") or raw.get("path")
    if isinstance(raw, str):
        if raw.startswith("data:"):
            return decode_data_url(raw)
        path = Path(raw)
        if path.is_file():
            return path
    return None


def flatten_user_content(content) -> tuple[str, list]:
    """Return (text, extracted_images) where images are Path or (ext, bytes)."""
    if content is None:
        return "", []
    if isinstance(content, str):
        return content, []
    if not isinstance(content, list):
        return str(content), []

    texts: list[str] = []
    images: list = []
    for part in content:
        if isinstance(part, str):
            texts.append(part)
            continue
        if not isinstance(part, dict):
            continue
        typ = part.get("type")
        if typ == "text" or (typ is None and "text" in part):
            texts.append(part.get("text") or "")
            continue
        img = extract_image_from_part(part)
        if img is not None:
            images.append(img)
    return "\n".join(t for t in texts if t), images


def ensure_image_tokens(text: str, n_images: int) -> str:
    if n_images <= 0:
        return text
    existing = text.count("<image>")
    if existing == 0:
        return ("<image>" * n_images) + (text or "")
    if existing < n_images:
        return text + ("<image>" * (n_images - existing))
    return text


def materialize_images(
    images: list,
    extra_paths: list[str],
    dump_dir: Path,
    stem: str,
) -> list[str]:
    paths: list[str] = []
    for raw in extra_paths:
        if raw and Path(raw).is_file():
            paths.append(str(Path(raw).resolve()))
    # Prefer on-disk paths already present on the record. Dumping the
    # embedded base64 as well would duplicate the same figure.
    if paths:
        return paths
    for i, img in enumerate(images):
        if isinstance(img, Path):
            if img.is_file():
                paths.append(str(img.resolve()))
            continue
        if not isinstance(img, tuple):
            continue
        ext, blob = img
        out = dump_dir / f"{stem}_{i}.{ext}"
        if not out.exists():
            out.write_bytes(blob)
        paths.append(str(out.resolve()))
    # de-dup while preserving order
    seen: set[str] = set()
    uniq: list[str] = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def convert_record(rec: dict, dump_dir: Path, idx: int) -> dict | None:
    messages = rec.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        return None

    converted = []
    collected_images: list = []
    for msg in messages:
        if not isinstance(msg, dict):
            return None
        role = msg.get("role")
        content = msg.get("content")
        if role == "user":
            text, imgs = flatten_user_content(content)
            collected_images.extend(imgs)
            converted.append({"role": "user", "content": text})
        else:
            if isinstance(content, list):
                text, imgs = flatten_user_content(content)
                collected_images.extend(imgs)
            else:
                text = content if isinstance(content, str) else ("" if content is None else str(content))
            if not text.strip():
                return None
            converted.append({"role": role, "content": text})

    roles = [m["role"] for m in converted]
    if "user" not in roles or "assistant" not in roles:
        return None

    extra = []
    if rec.get("image_path"):
        extra.append(rec["image_path"])
    stem = safe_name(
        f"{rec.get('dataset_name', 'unk')}__{rec.get('question_id', idx)}",
        f"rec_{idx}",
    )
    image_paths = materialize_images(collected_images, extra, dump_dir, stem)
    user = next(m for m in converted if m["role"] == "user")
    user["content"] = ensure_image_tokens(user.get("content") or "", len(image_paths))

    if image_paths and user["content"].count("<image>") != len(image_paths):
        # last resort: reset tokens to match files
        body = user["content"].replace("<image>", "")
        user["content"] = ("<image>" * len(image_paths)) + body

    out = {"messages": converted}
    if image_paths:
        out["images"] = image_paths
    return out


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"[warn] skip {path}:{line_no}: {exc}", file=sys.stderr)
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def collect_sources(src: Path) -> dict[str, list[Path]]:
    return {
        "math": sorted(src.glob("math/sft_math_full8k_shard*.jsonl")),
        "physics": sorted(src.glob("physics/sft_physics_mm_full_shard*.jsonl")),
        "physlib": [src / "parallel/physlib_parallel_4000.jsonl"],
    }


def assistant_index(messages: list) -> int:
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], dict) and messages[i].get("role") == "assistant":
            return i
    return -1


def repair_reflective(rec: dict) -> tuple[bool, str]:
    """Rewrite one collected trajectory in place.

    Returns ``(usable, label)``. Unusable means either every round was a
    Lean-gate rejection, or a surviving round narrates a compiler message we
    just removed; the label says which, for the stats line.
    """
    messages = rec.get("messages")
    if not isinstance(messages, list):
        return False, "no_messages"
    idx = assistant_index(messages)
    if idx < 0 or not isinstance(messages[idx].get("content"), str):
        return False, "no_assistant"
    res = sanitize_traj.sanitize_response(messages[idx]["content"])
    if res.text is None:
        return False, res.reason or "unusable"
    messages[idx]["content"] = res.text
    rec["_ends_incorrect"] = res.final_verdict == "Incorrect"
    return True, f"ends_{(res.final_verdict or '?').lower()}"


def rewrite_physlib(rec: dict) -> None:
    messages = rec.get("messages")
    if not isinstance(messages, list):
        return
    idx = assistant_index(messages)
    if idx < 0 or not isinstance(messages[idx].get("content"), str):
        return
    messages[idx]["content"] = sanitize_traj.wrap_physlib_response(
        messages[idx]["content"],
        kind=rec.get("kind") or "",
        module=rec.get("module_name") or "",
    )


def subsample_negatives(
    negatives: list,
    n_positive: int,
    ratio: float,
    rng: random.Random,
) -> list:
    """Take the largest subset of `negatives` that keeps them at `ratio`."""
    if not negatives or ratio <= 0:
        return []
    if ratio >= 1:
        return list(negatives)
    target = round(ratio / (1.0 - ratio) * n_positive)
    if target >= len(negatives):
        return list(negatives)
    return rng.sample(negatives, target)


def main() -> int:
    args = parse_args()
    src = args.src
    out_dir = args.out_dir
    dump_dir = out_dir / "images" / "physics"
    dump_dir.mkdir(parents=True, exist_ok=True)

    sources = collect_sources(src)
    for name, files in sources.items():
        missing = [str(p) for p in files if not p.is_file()]
        if missing:
            raise SystemExit(f"missing {name} files: {missing}")

    stats: dict = {
        "filter": args.filter,
        "negative_ratio": args.negative_ratio,
        "sanitize": not args.no_sanitize,
        "physlib_wrap": not args.no_physlib_wrap,
        "splits": {},
    }
    converted: dict[str, list[dict]] = {}

    rng_repair = random.Random(args.seed)
    # Negatives are pooled across the reflective splits before subsampling:
    # math yields far fewer compiling-but-unfaithful trajectories than physics,
    # so a per-split quota would either starve the target ratio or give the two
    # domains very different "how often do I end on Incorrect" priors.
    pos_by_split: dict[str, list[dict]] = {}
    neg_pool: list[tuple[str, dict]] = []
    for split, files in sources.items():
        raw_n = 0
        kept_n = 0
        fail_n = 0
        with_image = 0
        repair = Counter()
        positives: list[dict] = []
        negatives: list[dict] = []
        for fp in files:
            for rec in load_jsonl(fp):
                raw_n += 1
                if not keep_record(rec, args.filter):
                    continue
                kept_n += 1
                if split == "physlib":
                    if not args.no_physlib_wrap:
                        rewrite_physlib(rec)
                        repair["wrapped"] += 1
                elif not args.no_sanitize:
                    usable, label = repair_reflective(rec)
                    repair[label] += 1
                    if not usable:
                        continue
                item = convert_record(rec, dump_dir, kept_n)
                if item is None:
                    fail_n += 1
                    continue
                if item.get("images"):
                    with_image += 1
                if rec.get("_ends_incorrect"):
                    negatives.append(item)
                else:
                    positives.append(item)

        pos_by_split[split] = positives
        neg_pool.extend((split, item) for item in negatives)
        stats["splits"][split] = {
            "raw": raw_n,
            "filter_kept": kept_n,
            "convert_fail": fail_n,
            "with_image": with_image,
            "ends_correct": len(positives),
            "negatives_available": len(negatives),
            "repair": dict(repair),
        }

    # Denominator is the reflective mixture only: physlib parallel data carries
    # no self-validation verdict, so it cannot dilute the Correct/Incorrect
    # balance the model actually learns.
    n_pos_reflective = sum(
        len(v) for split, v in pos_by_split.items() if split != "physlib"
    )
    kept_negatives = subsample_negatives(
        neg_pool, n_pos_reflective, args.negative_ratio, rng_repair
    )
    neg_by_split: dict[str, list[dict]] = {}
    for split, item in kept_negatives:
        neg_by_split.setdefault(split, []).append(item)

    for split in sources:
        rows = pos_by_split[split] + neg_by_split.get(split, [])
        rng_repair.shuffle(rows)
        converted[split] = rows
        info = stats["splits"][split]
        info["ends_incorrect"] = len(neg_by_split.get(split, []))
        info["converted"] = len(rows)
        write_jsonl(out_dir / f"{split}.jsonl", rows)
        print(
            f"[{split}] raw={info['raw']} filter={info['filter_kept']} "
            f"converted={len(rows)} (+{info['ends_correct']} Correct / "
            f"+{info['ends_incorrect']} Incorrect of "
            f"{info['negatives_available']} eligible) "
            f"images={info['with_image']} fail={info['convert_fail']} "
            f"repair={info['repair']}"
        )

    mix = []
    for split, rows in converted.items():
        mix.extend(rows)
    rng = random.Random(args.seed)
    rng.shuffle(mix)
    write_jsonl(out_dir / "train_mix.jsonl", mix)

    # stratified smoke: prefer image examples from physics
    smoke: list[dict] = []
    budget = {
        "math": max(1, args.smoke_n * 2 // 5),
        "physics": max(1, args.smoke_n * 2 // 5),
        "physlib": max(1, args.smoke_n - (args.smoke_n * 4 // 5)),
    }
    for split, n in budget.items():
        pool = converted[split]
        if split == "physics":
            pool = [r for r in pool if r.get("images")] or pool
        smoke.extend(pool[:n])
    rng.shuffle(smoke)
    write_jsonl(out_dir / "train_smoke.jsonl", smoke[: args.smoke_n])

    stats["mix"] = len(mix)
    stats["smoke"] = min(args.smoke_n, len(smoke))
    stats["image_dump_dir"] = str(dump_dir)
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n")

    # sanity: image files exist
    missing_imgs = 0
    for row in mix:
        for img in row.get("images") or []:
            if not Path(img).is_file():
                missing_imgs += 1
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    if missing_imgs:
        print(f"[warn] missing image files: {missing_imgs}", file=sys.stderr)
        return 2
    print(f"[ok] wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

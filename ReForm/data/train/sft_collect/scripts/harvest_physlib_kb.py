#!/usr/bin/env python3
"""Harvest verified Lean declarations from successful physics SFT trajectories
into a growable Physlib KB corpus (KB-growth, Option A).

Idea (borrowed from autoformalization_search's verified-node accumulation, but
implemented on ReForm without graph decomposition): every physics trajectory
that reached semantic consistency AND compiled produces reusable Lean
declarations. We extract them, re-verify each under the *physics header* (so we
only keep declarations that stand on their own given their sibling bundle),
informalize them for semantic retrieval, dedup, and append to:

  1. physlib_kb_corpus.jsonl  -- 9-field schema consumed by `build_cuvs`
                                 (name, module_name, kind, type, index,
                                  signature, value, informal_name,
                                  informal_description)
  2. physlib_kb_store.jsonl   -- reform-side record: full source bundle, deps,
                                 provenance, dedup hash. Also the live index
                                 source for KB-growth Option B.

Two ways `physsearch` picks these up (selected by `PHYSSEARCH_KB_MODE`):
  - Option B (DEFAULT, `local`): `physlib_kb_local.py` builds a local BM25 index
    over `physlib_kb_store.jsonl` and HOT-RELOADS on append. So just running
    this harvester is enough -- the running collection workers see new KB
    entries on their next `physsearch` call. NO GPU, NO server restart.
  - Option A (`remote`/`both`): re-index the corpus JSONL into a second
    LeanSearch-v2 corpus (see reindex_physlib_kb.sh); needs GPU + server
    restart, so it is batched between rounds, not live.

Usage (from ReForm/data/train):

  PY=/mnt/afs_reason/chendongyang/conda/leansearch/bin/python
  # informalize needs an LLM; source the same env the collector uses:
  source poc_phyx/.llm.env; export VERIFY_URL=... VERIFY_KEY=...
  $PY sft_collect/scripts/harvest_physlib_kb.py \
      --inputs 'sft_collect/physics/sft_physics_mm_*.jsonl' \
      --auto-model "$AUTO_MODEL" --auto-base-url "$TOKENHUB_BASE_URL" \
      --auto-api-key "$TOKENHUB_API_KEY"
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import multiprocessing as mp
import re
import sys
import threading
from pathlib import Path

# ReForm data/train on sys.path for verify_lean / llm_client / physics_header.
_TRAIN_DIR = Path(__file__).resolve().parents[2]
if str(_TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAIN_DIR))

import physics_header  # noqa: E402
import verify_lean  # noqa: E402

try:
    import llm_client  # noqa: E402
except Exception:  # pragma: no cover - llm optional
    llm_client = None


# Surface keywords we treat as top-level declarations. `lemma` maps to theorem;
# def/abbrev map to definition; instance stays instance. structure/inductive/
# class are skipped in v1 (their type/value do not map cleanly onto the corpus
# schema and they are rare in physics statements).
_DECL_RE = re.compile(
    r"^(theorem|lemma|def|abbrev|instance|structure|inductive|class)\b",
)
_KIND_MAP = {
    "theorem": "theorem",
    "lemma": "theorem",
    "def": "definition",
    "abbrev": "definition",
    "instance": "instance",
}
_HARVEST_KINDS = set(_KIND_MAP)  # skip structure/inductive/class in v1

_lock = threading.RLock()


def _norm_src(src: str) -> str:
    """Whitespace-normalized source for content hashing / dedup."""
    return re.sub(r"\s+", " ", src).strip()


def _decl_hash(src: str) -> str:
    return hashlib.sha1(_norm_src(src).encode("utf-8")).hexdigest()


def strip_header(code: str) -> str:
    """Drop import / open / set_option / section / namespace scaffolding lines so
    we are left with declaration bodies."""
    keep: list[str] = []
    for ln in code.splitlines():
        s = ln.strip()
        if s.startswith((
            "import ", "open ", "set_option ", "namespace ", "section",
            "end", "noncomputable section", "attribute ",
        )):
            continue
        keep.append(ln)
    return "\n".join(keep)


def split_top_level_decls(code: str) -> list[dict]:
    """Split Lean source into top-level declaration blocks.

    A block starts at a column-0 line beginning with a decl keyword and runs
    until the next such line (or EOF). Returns dicts with keyword/name/src.
    Tolerant by design: anything that does not re-verify is dropped later."""
    lines = code.splitlines()
    starts: list[int] = []
    for i, ln in enumerate(lines):
        # column-0 (no indentation) declaration start
        if ln[:1] not in (" ", "\t") and _DECL_RE.match(ln.strip()):
            starts.append(i)
    blocks: list[dict] = []
    for idx, start in enumerate(starts):
        end = starts[idx + 1] if idx + 1 < len(starts) else len(lines)
        src = "\n".join(lines[start:end]).strip()
        if not src:
            continue
        m = _DECL_RE.match(src)
        if not m:
            continue
        keyword = m.group(1)
        name = _decl_name(src, keyword)
        blocks.append({"keyword": keyword, "name": name, "src": src})
    return blocks


def _decl_name(src: str, keyword: str) -> str:
    m = re.match(
        r"^" + keyword + r"\s+([A-Za-z_][\w'!?]*(?:\.[A-Za-z_][\w'!?]*)*)",
        src.strip(),
    )
    return m.group(1) if m else ""


def _split_sig_value(src: str, keyword: str, name: str):
    """Return (type_str, value_str) from a decl surface source.

    type_str = everything after the name up to the first top-level ` := `.
    value_str = the body after ` := ` (used only for definition/instance).
    Both best-effort; the corpus row is fine even if value is empty."""
    body = src
    # strip leading "keyword name"
    prefix = keyword + (" " + name if name else "")
    if body.startswith(keyword):
        body = body[len(keyword):].lstrip()
        if name and body.startswith(name):
            body = body[len(name):]
    # split at first top-level assignment
    m = re.search(r":=", body)
    if m:
        type_str = body[: m.start()].strip()
        value_str = body[m.end():].strip()
    else:
        type_str = body.strip()
        value_str = ""
    # type field: drop a leading ':' so it reads as the corpus `type` does
    return type_str, value_str


def _to_ascii_sorry(src: str) -> str:
    """Normalize a declaration to end in `:= by sorry` for re-verify (we only
    care that the *statement/type* elaborates, not the proof)."""
    s = src.rstrip()
    # replace an existing top-level proof with `by sorry` for theorems
    return s


def informalize(client, model: str, kind: str, name: str, type_str: str,
                problem_text: str) -> tuple[str, str]:
    """Produce (informal_name, informal_description) for retrieval.

    Uses one short LLM call when a client is available; otherwise falls back to
    a heuristic derived from the declaration name and problem statement."""
    if client is not None and llm_client is not None:
        sys_prompt = (
            "You write concise, reusable informal metadata for a Lean 4 physics "
            "declaration so it can be found by semantic search. Given the Lean "
            "kind, name and type, output STRICT JSON with keys `informal_name` "
            "(a short concept title, <= 12 words, NO problem-specific numbers) "
            "and `informal_description` (1-2 sentences describing the general "
            "physical/mathematical meaning, NOT the specific problem)."
        )
        user = (
            f"kind: {kind}\nname: {name}\ntype: {type_str}\n"
            f"(context, do not copy specifics) problem: {problem_text[:600]}\n\n"
            "Return only JSON."
        )
        try:
            print(f"[harvest]   informalize {name} ...", flush=True)
            out = llm_client.chat_completion(
                client,
                [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user},
                ],
                model=model,
                temperature=0.2,
                max_tokens=400,
            )
            js = _extract_json(out)
            if js:
                iname = str(js.get("informal_name") or "").strip()
                idesc = str(js.get("informal_description") or "").strip()
                if iname and idesc:
                    return iname, idesc
        except Exception as e:
            print(f"[harvest]   informalize fallback ({e})", flush=True)
    # Heuristic fallback (keeps the row usable even without an LLM).
    iname = _humanize(name) or (kind + " declaration")
    idesc = _norm_src(problem_text)[:240] or iname
    return iname, idesc


def _extract_json(text: str) -> dict | None:
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _humanize(name: str) -> str:
    tail = name.split(".")[-1] if name else ""
    tail = re.sub(r"(?<!^)(?=[A-Z])", " ", tail)
    tail = tail.replace("_", " ").strip()
    return tail


def _final_lean_of(row: dict) -> str:
    """Best-effort extraction of the final Lean code from a collected row."""
    code = row.get("final_lean_code")
    if code and str(code).strip():
        return str(code)
    # Fall back to the last ```lean4 block in the assistant message.
    msgs = row.get("messages") or []
    for m in reversed(msgs):
        if m.get("role") != "assistant":
            continue
        content = m.get("content")
        if isinstance(content, list):
            content = " ".join(
                str(x.get("text", "")) if isinstance(x, dict) else str(x)
                for x in content
            )
        blocks = re.findall(r"```lean4?\s*(.*?)```", str(content or ""), re.DOTALL)
        if blocks:
            return blocks[-1].strip()
    return ""


def _is_physics_success(row: dict) -> bool:
    if not row.get("success"):
        return False
    if row.get("syntax_pass") is False:
        return False
    dom = (row.get("retrieval_domain") or "").lower()
    return dom in ("physics", "phyx", "physlib_lib") or bool(row.get("has_image"))


def load_seen_hashes(store_path: Path) -> set[str]:
    seen: set[str] = set()
    if store_path.is_file():
        for line in store_path.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            h = rec.get("hash")
            if h:
                seen.add(h)
    return seen


def next_corpus_index(corpus_path: Path) -> int:
    if not corpus_path.is_file():
        return 0
    n = 0
    for line in corpus_path.open(encoding="utf-8"):
        if line.strip():
            n += 1
    return n


def _verify_worker(q: "mp.Queue", code: str) -> None:
    """Child process: isolate Kimina verify so a hung REPL cannot stall harvest."""
    try:
        ok, err = verify_lean.verify_sync(code, header=physics_header.physics_header())
        q.put((bool(ok), str(err or "")))
    except Exception as e:  # pragma: no cover
        q.put((False, f"verify_exception: {e}"))


def reverify_bundle(code: str, timeout_s: float = 180.0) -> tuple[bool, str]:
    """Re-verify in a subprocess with a hard timeout (avoids in-process asyncio hangs)."""
    ctx = mp.get_context("spawn")
    q: mp.Queue = ctx.Queue()
    proc = ctx.Process(target=_verify_worker, args=(q, code), daemon=True)
    proc.start()
    proc.join(timeout_s)
    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        if proc.is_alive():
            proc.kill()
            proc.join(2)
        return False, f"verify_timeout>{timeout_s:.0f}s"
    try:
        return q.get_nowait()
    except Exception:
        return False, "verify_no_result"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--inputs",
        nargs="+",
        default=["sft_collect/physics/sft_physics_mm_*.jsonl"],
        help="Glob(s) of collected physics success JSONL (relative to data/train).",
    )
    ap.add_argument("--out-dir", default="sft_collect/physlib_kb")
    ap.add_argument("--corpus-name", default="physlib_kb")
    ap.add_argument("--kinds", default="theorem,definition,instance",
                    help="Corpus kinds to keep (post-map).")
    ap.add_argument("--reverify", dest="reverify", action="store_true", default=True,
                    help="Re-compile the full bundle under the physics header (default on).")
    ap.add_argument("--no-reverify", dest="reverify", action="store_false")
    ap.add_argument("--informalize", dest="informalize", action="store_true", default=True)
    ap.add_argument("--no-informalize", dest="informalize", action="store_false")
    ap.add_argument("--auto-model", default="default")
    ap.add_argument("--auto-base-url", default=None)
    ap.add_argument("--auto-api-key", default=None)
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--verify-timeout", type=float, default=180.0,
                    help="Hard timeout (seconds) per bundle re-verify subprocess.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    keep_kinds = {k.strip() for k in args.kinds.split(",") if k.strip()}

    # Resolve inputs relative to data/train.
    paths: list[Path] = []
    for g in args.inputs:
        gg = g if Path(g).is_absolute() else str(_TRAIN_DIR / g)
        paths.extend(sorted(Path(p) for p in glob.glob(gg)))
    # Exclude failed/ files.
    paths = [p for p in paths if "/failed/" not in str(p) and ".failed." not in p.name]
    if not paths:
        raise SystemExit(f"no input files match: {args.inputs}")

    out_dir = _TRAIN_DIR / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = out_dir / f"{args.corpus_name}_corpus.jsonl"
    store_path = out_dir / f"{args.corpus_name}_store.jsonl"

    client = None
    if args.informalize and llm_client is not None:
        try:
            client = llm_client.get_client(
                base_url=args.auto_base_url, api_key=args.auto_api_key
            )
        except Exception as e:
            print(f"[harvest] informalize client unavailable ({e}); using heuristic.")
            client = None

    seen = load_seen_hashes(store_path)
    corpus_idx = next_corpus_index(corpus_path)

    stats = {
        "files": len(paths), "rows_seen": 0, "rows_success": 0,
        "bundles_reverified": 0, "bundles_reverify_failed": 0,
        "decls_extracted": 0, "decls_kept": 0, "decls_dup": 0,
        "decls_skipped_kind": 0,
    }

    corpus_rows: list[dict] = []
    store_rows: list[dict] = []

    print(
        f"[harvest] inputs={len(paths)} store={store_path} corpus={corpus_path} "
        f"reverify={args.reverify} informalize={args.informalize}",
        flush=True,
    )

    for p in paths:
        print(f"[harvest] reading {p.relative_to(_TRAIN_DIR)}", flush=True)
        for line in p.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            stats["rows_seen"] += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not _is_physics_success(row):
                continue
            stats["rows_success"] += 1
            qid = row.get("question_id")
            print(
                f"[harvest] success#{stats['rows_success']} qid={qid} "
                f"from={p.name}",
                flush=True,
            )

            code = _final_lean_of(row)
            if not code.strip():
                print(f"[harvest]   skip: no lean code", flush=True)
                continue

            # Re-verify the whole bundle under the physics header.
            if args.reverify:
                print(
                    f"[harvest]   reverify lean_chars={len(code)} "
                    f"timeout={args.verify_timeout:.0f}s ...",
                    flush=True,
                )
                ok, err = reverify_bundle(code, timeout_s=args.verify_timeout)
                if not ok:
                    stats["bundles_reverify_failed"] += 1
                    print(f"[harvest]   reverify FAIL: {str(err)[:160]}", flush=True)
                    continue
                stats["bundles_reverified"] += 1
                print("[harvest]   reverify OK", flush=True)

            problem_text = str(row.get("informal_statement") or "")
            body = strip_header(code)
            decls = split_top_level_decls(body)
            sibling_names = [d["name"] for d in decls if d["name"]]
            print(f"[harvest]   decls={len(decls)} names={sibling_names}", flush=True)

            for d in decls:
                stats["decls_extracted"] += 1
                kind = _KIND_MAP.get(d["keyword"])
                if kind is None or kind not in keep_kinds:
                    stats["decls_skipped_kind"] += 1
                    continue
                name = d["name"]
                if not name:
                    continue
                h = _decl_hash(d["src"])
                if h in seen:
                    stats["decls_dup"] += 1
                    continue
                seen.add(h)

                type_str, value_str = _split_sig_value(d["src"], d["keyword"], name)
                deps = [s for s in sibling_names if s and s != name and s in d["src"]]

                iname, idesc = ("", "")
                if not args.dry_run:
                    iname, idesc = informalize(
                        client, args.auto_model, kind, name, type_str, problem_text
                    )
                else:
                    iname, idesc = _humanize(name), _norm_src(problem_text)[:200]

                corpus_row = {
                    "name": name.split("."),
                    "module_name": ["ReformKB", args.corpus_name],
                    "kind": kind,
                    "type": type_str,
                    "index": corpus_idx,
                    "signature": "",
                    "value": value_str if kind in ("definition", "instance") else "",
                    "informal_name": iname,
                    "informal_description": idesc,
                }
                store_row = {
                    "hash": h,
                    "name": name,
                    "kind": kind,
                    "corpus_index": corpus_idx,
                    "type": type_str,
                    "value": value_str,
                    "src": d["src"],
                    "bundle": code,
                    "deps": deps,
                    "informal_name": iname,
                    "informal_description": idesc,
                    "provenance": {
                        "dataset_name": row.get("dataset_name"),
                        "question_id": row.get("question_id"),
                        "source_file": str(p.relative_to(_TRAIN_DIR)),
                        "verified_by": (
                            "kimina" if args.reverify else "collect_syntax_pass"
                        ),
                    },
                }
                # Append-immediately so a mid-run crash / hung LLM still keeps
                # every already-harvested decl (local KB hot-reloads on mtime).
                if not args.dry_run:
                    with _lock:
                        with corpus_path.open("a", encoding="utf-8") as f:
                            f.write(json.dumps(corpus_row, ensure_ascii=False) + "\n")
                            f.flush()
                        with store_path.open("a", encoding="utf-8") as f:
                            f.write(json.dumps(store_row, ensure_ascii=False) + "\n")
                            f.flush()
                    print(
                        f"[harvest]   APPENDED {name} ({kind}) "
                        f"-> store idx={corpus_idx}",
                        flush=True,
                    )
                else:
                    corpus_rows.append(corpus_row)
                    store_rows.append(store_row)
                corpus_idx += 1
                stats["decls_kept"] += 1

                if args.max_rows and stats["decls_kept"] >= args.max_rows:
                    break
            if args.max_rows and stats["decls_kept"] >= args.max_rows:
                break
        if args.max_rows and stats["decls_kept"] >= args.max_rows:
            break

    print(json.dumps({
        "mode": "DRY-RUN" if args.dry_run else "WROTE",
        "corpus_out": str(corpus_path),
        "store_out": str(store_path),
        "kept_this_run": stats["decls_kept"],
        "corpus_total": corpus_idx,
        "stats": stats,
    }, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Load ReForm eval / collect inputs. Eval only needs ``load_input``.

``main()`` is the SFT collection CLI used by ``sft_collect/scripts``.
Eval imports this module for ``load_input`` only and never calls ``main``.
"""

from __future__ import annotations

import json


def _merge_mm_informal(it: dict) -> str:
    """Build informal_statement for generic multimodal physics seeds."""
    parts = []
    q = (it.get("question") or it.get("text") or "").strip()
    if q:
        parts.append(q)
    cap = (it.get("image_caption") or "").strip()
    if cap:
        parts.append(f"[Image description: {cap}]")
    desc = (it.get("question_description") or it.get("description") or "").strip()
    if desc and desc != q:
        parts.append(desc)
    opts = it.get("options")
    if opts:
        if isinstance(opts, list):
            parts.append("Options:\n" + "\n".join(str(o) for o in opts))
        else:
            parts.append(f"Options: {opts}")
    return "\n\n".join(parts).strip()


def _attach_image_if_needed(it: dict, *, include_image: bool) -> dict:
    """Keep a path string; never open PIL during load."""
    if not include_image:
        return it
    row = dict(it)
    img = row.get("image")
    if isinstance(img, str) and img.strip():
        row.setdefault("image_path", img)
    elif isinstance(img, dict) and img.get("path"):
        row.setdefault("image_path", img["path"])
    return row


def load_input(path: str, *, include_image: bool = False) -> list[dict]:
    """Load items with informal_statement from JSON or JSONL.

    PhyX rows without a pre-merged statement go through ``phyx_adapter``.
    Image paths stay on disk; ``run_eval_rollout`` opens them in the worker.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    items = []
    if raw.startswith("["):
        data = json.loads(raw)
        items = data if isinstance(data, list) else [data]
    else:
        lines = [ln.strip() for ln in raw.split("\n") if ln.strip()]
        if len(lines) == 1 and (lines[0].startswith("{") or "informal_statement" in lines[0]):
            items = [json.loads(lines[0])]
        else:
            for line in lines:
                items.append(json.loads(line))

    def _image_path(row: dict):
        img = row.get("image")
        if isinstance(img, str) and img.strip():
            return img
        if isinstance(img, dict):
            p = img.get("path")
            if isinstance(p, str) and p.strip():
                return p
        p = row.get("image_path")
        return p if isinstance(p, str) and p.strip() else None

    def _keep_meta(row: dict, src: dict) -> dict:
        row = dict(row)
        p = _image_path(src) or _image_path(row)
        if p:
            row.setdefault("image_path", p)
        for key in (
            "retrieval_domain",
            "source_subset",
            "source_file",
            "question_id",
            "category",
            "dataset_name",
            "image_caption",
        ):
            if src.get(key) and not row.get(key):
                row[key] = src[key]
        return row

    out = []
    for it in items:
        if not isinstance(it, dict):
            continue
        if it.get("dataset_name") == "phyx" and not it.get("informal_statement"):
            try:
                from . import phyx_adapter  # type: ignore
            except ImportError:
                import phyx_adapter  # type: ignore
            normalized = phyx_adapter.phyx_to_input(it, include_image=False)
            if normalized.get("informal_statement"):
                out.append(_keep_meta(normalized, it))
                continue
        if it.get("informal_statement"):
            out.append(_keep_meta(
                _attach_image_if_needed(it, include_image=include_image), it
            ))
            continue
        st = _merge_mm_informal(it) if (
            it.get("image") or it.get("image_caption") or it.get("question")
        ) else (it.get("question") or it.get("text"))
        if st:
            row = {
                "informal_statement": st,
                **{k: v for k, v in it.items() if k not in ("question", "text")},
            }
            out.append(_keep_meta(
                _attach_image_if_needed(row, include_image=include_image), it
            ))
    return out if out else items


def _item_image_path(item: dict):
    for key in ("image_path", "image"):
        v = item.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
        if isinstance(v, dict) and v.get("path"):
            return str(v["path"])
    return None


def _open_image_lazy(item: dict, *, include_image: bool):
    if not include_image:
        return None
    img = item.get("image")
    try:
        from PIL import Image  # type: ignore
    except ImportError:
        return None
    if img is not None and hasattr(img, "convert"):
        return img
    path = _item_image_path(item)
    if not path:
        return None
    try:
        return Image.open(path)
    except Exception:
        return None


def _close_image(img) -> None:
    if img is None:
        return
    try:
        img.close()
    except Exception:
        pass


def _default_failed_path(output_path: str) -> str:
    import os
    d = os.path.dirname(output_path) or "."
    base = os.path.basename(output_path)
    if base.endswith(".jsonl"):
        name = base[: -len(".jsonl")] + ".failed.jsonl"
    else:
        name = base + ".failed.jsonl"
    return os.path.join(d, "failed", name)


def _load_done_keys(*paths: str) -> set[tuple]:
    import os
    done: set[tuple] = set()
    for path in paths:
        if not path or not os.path.isfile(path):
            continue
        with open(path, "r", encoding="utf-8") as prev:
            for line in prev:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                done.add((str(row.get("dataset_name") or ""), str(row.get("question_id") or "")))
    return done


def _item_key(it: dict) -> tuple:
    return (str(it.get("dataset_name") or ""), str(it.get("question_id") or ""))


def main() -> None:
    import argparse
    import os
    import sys
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    _DIR = os.path.dirname(os.path.abspath(__file__))
    if _DIR not in sys.path:
        sys.path.insert(0, _DIR)

    parser = argparse.ArgumentParser(description="Collect ReForm SFT trajectories")
    parser.add_argument("--input", "-i", default="../test/test.json")
    parser.add_argument("--output", "-o", default="sft_data.jsonl")
    parser.add_argument("--failed-output", default=None)
    parser.add_argument("--write-failures", action="store_true", default=True)
    parser.add_argument("--no-write-failures", action="store_false", dest="write_failures")
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("WORKERS", "1")),
        help="Parallel item workers. Default 1, or $WORKERS.",
    )
    parser.add_argument("--use-lean-verify", action="store_true")
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument(
        "--max-gate-retries",
        type=int,
        default=3,
        help="Extra autoformalization attempts rejected by the Lean gate (not recorded).",
    )
    parser.add_argument("--use-leansearch", action="store_true")
    parser.add_argument("--leansearch-url", default="https://leansearch.net")
    parser.add_argument("--leansearch-top-k", type=int, default=8)
    parser.add_argument("--max-tool-calls", type=int, default=3)
    parser.add_argument("--use-physsearch", action="store_true")
    parser.add_argument("--physsearch-url", default="http://localhost:8000")
    parser.add_argument(
        "--joint-retrieval",
        action="store_true",
        help="Enable leansearch + physsearch. Implies both --use-* flags.",
    )
    parser.add_argument("--use-web-search", action="store_true")
    parser.add_argument("--include-image", action="store_true")
    parser.add_argument("--auto-model", default=None)
    parser.add_argument("--auto-base-url", default=None)
    parser.add_argument("--auto-api-key", default=None)
    parser.add_argument("--critic-model", default=None)
    parser.add_argument("--critic-base-url", default=None)
    parser.add_argument("--critic-api-key", default=None)
    parser.add_argument("--model", default="default")
    parser.add_argument("--base-url", default=None)
    args = parser.parse_args()

    from collect_sft_trajectory import collect_one_trajectory
    import llm_client
    try:
        from pipeline_guard import PipelineHalt, is_arrears_text
    except ImportError:
        PipelineHalt = Exception  # type: ignore

        def is_arrears_text(_text: str) -> bool:
            return False

    if args.joint_retrieval:
        args.use_leansearch = True
        args.use_physsearch = True
        if args.leansearch_url == "https://leansearch.net":
            args.leansearch_url = args.physsearch_url or "http://127.0.0.1:8000"

    auto_model = args.auto_model or args.model
    auto_base_url = args.auto_base_url or args.base_url
    auto_api_key = args.auto_api_key
    critic_model = args.critic_model or auto_model
    critic_base_url = args.critic_base_url
    critic_api_key = args.critic_api_key

    input_path = args.input
    if not os.path.isabs(input_path):
        input_path = os.path.join(_DIR, input_path)
    if not os.path.isfile(input_path):
        raise FileNotFoundError("Input file not found: " + args.input)

    items = load_input(input_path, include_image=args.include_image)
    if not items:
        print("No items with informal_statement found.")
        return

    output_path = args.output
    if not os.path.isabs(output_path):
        output_path = os.path.join(_DIR, output_path)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    failed_path = args.failed_output
    if failed_path is None:
        failed_path = _default_failed_path(output_path)
    if not os.path.isabs(failed_path):
        failed_path = os.path.join(_DIR, failed_path)
    if args.write_failures:
        os.makedirs(os.path.dirname(failed_path) or ".", exist_ok=True)

    done_keys = _load_done_keys(output_path, failed_path if args.write_failures else "")
    if done_keys:
        before = len(items)
        items = [it for it in items if _item_key(it) not in done_keys]
        print(
            f"Resume: skipped {before - len(items)} already-written items "
            f"({len(done_keys)} keys in success/failed)"
        )
    if args.max_items:
        items = items[: args.max_items]

    n_with_img = sum(1 for it in items if _item_image_path(it))
    print(f"Processing {len(items)} item(s) from {input_path} (image paths: {n_with_img}, lazy-load)")
    print(f"Autoformalizer: model={auto_model}, base_url={auto_base_url}")
    print(f"CriticLean:     model={critic_model}, base_url={critic_base_url}")
    print(f"Success out:    {output_path}")
    if args.write_failures:
        print(f"Failed out:     {failed_path}")
    if args.joint_retrieval:
        print(
            f"Joint retrieval: leansearch@{args.leansearch_url} + "
            f"physsearch@{args.physsearch_url}; max_tool_calls={args.max_tool_calls}"
        )
    else:
        if args.use_leansearch:
            print(f"LeanSearch:     url={args.leansearch_url}, max_tool_calls={args.max_tool_calls}")
        if args.use_physsearch:
            print(f"PhysSearch:     url={args.physsearch_url}")
    if args.use_web_search:
        print(f"WebSearch:      key={'set' if os.environ.get('SERPER_API_KEY') else 'MISSING'}")
    if args.include_image:
        print("Vision:         --include-image ON")

    workers = max(1, int(args.workers))
    print(f"Workers: {workers}")
    tls = threading.local()
    write_lock = threading.Lock()

    def _thread_clients():
        if not hasattr(tls, "auto"):
            tls.auto = llm_client.get_client(base_url=auto_base_url, api_key=auto_api_key)
            tls.critic = llm_client.get_critic_client(
                base_url=critic_base_url, api_key=critic_api_key
            )
        return tls.auto, tls.critic

    def _collect_record(it: dict):
        informal = it["informal_statement"]
        auto_client, critic_client = _thread_clients()
        image = _open_image_lazy(it, include_image=args.include_image)
        try:
            traj = collect_one_trajectory(
                informal_statement=informal,
                auto_client=auto_client,
                critic_client=critic_client,
                model=auto_model,
                critic_model=critic_model,
                max_rounds=args.max_rounds,
                max_gate_retries=args.max_gate_retries,
                use_lean_verify=args.use_lean_verify,
                use_leansearch=args.use_leansearch,
                leansearch_url=args.leansearch_url,
                leansearch_top_k=args.leansearch_top_k,
                max_tool_calls=args.max_tool_calls,
                use_physsearch=args.use_physsearch,
                physsearch_url=args.physsearch_url,
                use_web_search=args.use_web_search,
                image=image,
                image_caption=it.get("image_caption") or None,
                dataset_name=it.get("dataset_name"),
                category=it.get("category"),
                force_domain=it.get("retrieval_domain"),
            )
        finally:
            _close_image(image)
        record = {
            "messages": traj["messages"],
            "informal_statement": informal,
            "num_rounds": traj["num_rounds"],
            "num_tool_calls": traj.get("num_tool_calls", 0),
            "tools_used": traj.get("tools_used", []),
            "success": traj["success"],
            "syntax_pass": traj.get("syntax_pass"),
            "verify_error": traj.get("verify_error"),
            "gate_retries": traj.get("gate_retries", 0),
            "syntax_rewrites": traj.get("syntax_rewrites", []),
            "retrieval_domain": traj.get("retrieval_domain"),
            "auto_model": auto_model,
            "critic_model": critic_model,
        }
        if it.get("dataset_name"):
            record["dataset_name"] = it["dataset_name"]
        if it.get("question_id"):
            record["question_id"] = it["question_id"]
        if it.get("category"):
            record["category"] = it["category"]
        if args.include_image:
            path = _item_image_path(it)
            record["has_image"] = path is not None
            if path:
                record["image_path"] = path
        return record, traj

    success_mode = "a" if os.path.isfile(output_path) and done_keys else "w"
    failed_mode = "a" if os.path.isfile(failed_path) and done_keys else "w"
    n_ok = 0
    n_fail = 0

    def _write_one(i: int, it: dict, record: dict, traj: dict) -> None:
        nonlocal n_ok, n_fail
        ok = bool(traj.get("success"))
        dest = out_f if ok else fail_f
        dest.write(json.dumps(record, ensure_ascii=False) + "\n")
        dest.flush()
        if ok:
            n_ok += 1
        else:
            n_fail += 1
        print(
            f"[{i+1}/{len(items)}] {it['informal_statement'][:60]}... "
            f"domain={traj.get('retrieval_domain')}, "
            f"rounds={traj['num_rounds']}, "
            f"tools={traj.get('tools_used', [])}, "
            f"syntax_pass={traj.get('syntax_pass')}, "
            f"success={traj['success']}"
            + (f", VERIFY_ERROR={traj['verify_error'][:80]}" if traj.get("verify_error") else "")
            + ("" if ok else " [failed]"),
            flush=True,
        )

    def _run_one(i: int, it: dict) -> None:
        print(
            f"[{i+1}/{len(items)}] Collecting trajectory for: "
            f"{it['informal_statement'][:60]}...",
            flush=True,
        )
        record, traj = _collect_record(it)
        with write_lock:
            _write_one(i, it, record, traj)

    with open(output_path, success_mode, encoding="utf-8") as out_f, \
         (open(failed_path, failed_mode, encoding="utf-8") if args.write_failures else open(os.devnull, "w")) as fail_f:
        if workers == 1:
            for i, it in enumerate(items):
                try:
                    _run_one(i, it)
                except PipelineHalt as e:
                    print(f"PIPELINE_HALT: {e}", flush=True)
                    raise SystemExit(2)
                except Exception as e:
                    if is_arrears_text(str(e)):
                        print(f"PIPELINE_HALT: {e}", flush=True)
                        raise SystemExit(2)
                    print(f"  -> Error: {e}", flush=True)
                    import traceback
                    traceback.print_exc()
                    if args.write_failures:
                        fail_f.write(json.dumps({
                            "informal_statement": it.get("informal_statement"),
                            "success": False,
                            "error": str(e),
                            "dataset_name": it.get("dataset_name"),
                            "question_id": it.get("question_id"),
                        }, ensure_ascii=False) + "\n")
                        fail_f.flush()
                        n_fail += 1
        else:
            halt = None
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futs = [pool.submit(_run_one, i, it) for i, it in enumerate(items)]
                for fut in as_completed(futs):
                    try:
                        fut.result()
                    except PipelineHalt as e:
                        halt = e
                        pool.shutdown(wait=False, cancel_futures=True)
                        break
                    except Exception as e:
                        if is_arrears_text(str(e)):
                            print(f"PIPELINE_HALT: {e}", flush=True)
                            pool.shutdown(wait=False, cancel_futures=True)
                            raise SystemExit(2)
                        print(f"  -> Error: {e}", flush=True)
                        import traceback
                        traceback.print_exc()
            if halt is not None:
                print(f"PIPELINE_HALT: {halt}", flush=True)
                raise SystemExit(2)

    print(
        f"Wrote success={n_ok} -> {output_path}; "
        f"failed={n_fail} -> {failed_path if args.write_failures else '(discarded)'}"
    )


if __name__ == "__main__":
    main()

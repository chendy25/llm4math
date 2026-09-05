#!/usr/bin/env python3
"""Evaluate the ReForm SFT model with single-shot ("RL rollout") generation.

Unlike `run_collect_sft_data.py`, which re-runs the *data collection* outer loop
(tool-description system prompt, one chat request per round, external verifier
and critic driving the refinement), this runner asks the model for the whole
`<think>/<round>.../<answer>` trajectory in one continuous assistant turn under
a single token budget, and only scores it afterwards. That is the behaviour SFT
actually taught and the behaviour an RL rollout would sample.

  python run_eval_rollout.py \
    --input  .../math_aime2025_all.jsonl \
    --output .../aime2025.jsonl \
    --auto-model qwen35-9b-reform-sft --auto-base-url http://10.x.x.x:8000/v1 \
    --critic-model qwen3.7-plus --critic-base-url https://tokenhub.../v1 \
    --critic-api-key sk-... \
    --use-lean-verify --use-leansearch --leansearch-url http://10.x.x.x:8000 \
    --max-response-tokens 12288 --workers 16
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

import llm_client
import rollout_singleshot
from run_collect_sft_data import load_input


def _open_eval_image(it: dict):
    """Load a figure in the worker. ``load_input`` keeps a path only."""
    img = it.get("image")
    if img is not None and not isinstance(img, (str, dict)):
        return img
    path = it.get("image_path")
    if not path and isinstance(img, str):
        path = img
    if not path and isinstance(img, dict):
        path = img.get("path")
    if not path:
        return None
    try:
        from PIL import Image  # type: ignore
        return Image.open(path)
    except Exception:
        return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", "-i", required=True)
    p.add_argument("--output", "-o", required=True)
    p.add_argument("--max-items", type=int, default=None)
    p.add_argument("--workers", type=int, default=int(os.environ.get("WORKERS", "1")))
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument(
        "--max-response-tokens",
        type=int,
        default=int(os.environ.get("MAX_RESPONSE_TOKENS", "20480")),
        help="One budget for the entire assistant turn, generated tokens AND "
             "spliced <tool_response> tokens (verl's rollout.response_length). "
             "Sized by the retrieval payload, not the model: measured rollouts "
             "generate 0.7k-5k tokens but splice in up to 6k of tool text, and "
             "the worst case is max_tool_calls * max_tool_response_tokens. SFT "
             "ran --max_length 8192 --truncation_strategy delete, so no trained "
             "turn exceeded 8106 tokens and a larger budget buys nothing.",
    )
    p.add_argument("--max-tool-calls", type=int, default=9,
                   help="Cap on real tool executions per rollout (9 = the most "
                        "seen in any SFT trajectory).")
    p.add_argument("--max-tool-response-tokens", type=int, default=1536,
                   help="Per-tool-response cap. SFT blocks were ~865 tokens "
                        "(p95 1530), so this matches training.")
    p.add_argument("--max-tokens-consistency", type=int,
                   default=int(os.environ.get("MAX_TOKENS_CONSISTENCY", "4096")))
    p.add_argument("--use-lean-verify", action="store_true")
    p.add_argument("--use-leansearch", action="store_true")
    p.add_argument("--leansearch-url", default="http://127.0.0.1:8000")
    p.add_argument("--leansearch-top-k", type=int, default=8)
    p.add_argument("--use-physsearch", action="store_true")
    p.add_argument("--physsearch-url", default="http://127.0.0.1:8000")
    p.add_argument("--use-web-search", action="store_true")
    p.add_argument("--auto-model", default="default")
    p.add_argument("--auto-base-url", default=None)
    p.add_argument("--auto-api-key", default=None)
    p.add_argument("--critic-model", default=None)
    p.add_argument("--critic-base-url", default=None)
    p.add_argument("--critic-api-key", default=None)
    p.add_argument("--save-response", action="store_true",
                   help="Persist the full rollout text (large, but the only way "
                        "to see why a rollout failed).")
    p.add_argument(
        "--eval-prompt-patch",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("EVAL_PROMPT_PATCH", "0").lower() in ("1", "true", "yes"),
        help="Append collection-time Lean/Physlib constraints to the user "
             "message (not a system turn). Default off: that is a different "
             "eval column from the trained user prompt. Env: EVAL_PROMPT_PATCH=1.",
    )
    p.add_argument(
        "--include-image",
        action="store_true",
        help="Open each item's image_path (PhyX / SeePhys / PhysUniBench) and "
             "send it as a vision user turn. Paths stay on disk until the "
             "worker runs; do not preload every PIL in the parent.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    items = load_input(args.input, include_image=False)
    if not items:
        print("No items with informal_statement found.")
        return 1
    if args.max_items:
        items = items[: args.max_items]

    out_path = args.output
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    counter = rollout_singleshot.token_counter()
    print(f"Items: {len(items)}  workers: {args.workers}")
    print(f"Auto:   {args.auto_model} @ {args.auto_base_url}")
    print(f"Critic: {args.critic_model or args.auto_model} @ "
          f"{args.critic_base_url or args.auto_base_url}")
    print(f"Budget: response={args.max_response_tokens} "
          f"tool_calls<={args.max_tool_calls} "
          f"tool_response<={args.max_tool_response_tokens} "
          f"tokenizer={'exact' if counter.exact else 'ESTIMATED'}")
    print(f"eval_prompt_patch={int(bool(args.eval_prompt_patch))} "
          f"(appended to user; off = trained prompt only)")

    tls = threading.local()
    write_lock = threading.Lock()

    def _clients():
        if not hasattr(tls, "auto"):
            tls.auto = llm_client.get_client(
                base_url=args.auto_base_url, api_key=args.auto_api_key
            )
            tls.critic = llm_client.get_critic_client(
                base_url=args.critic_base_url, api_key=args.critic_api_key
            )
        return tls.auto, tls.critic

    def _one(i: int, it: dict) -> dict:
        auto_client, critic_client = _clients()
        traj = rollout_singleshot.rollout_one(
            it["informal_statement"],
            auto_client,
            critic_client,
            model=args.auto_model,
            critic_model=args.critic_model or args.auto_model,
            temperature=args.temperature,
            max_response_tokens=args.max_response_tokens,
            max_tool_calls=args.max_tool_calls,
            max_tool_response_tokens=args.max_tool_response_tokens,
            max_tokens_consistency=args.max_tokens_consistency,
            use_lean_verify=args.use_lean_verify,
            use_leansearch=args.use_leansearch,
            leansearch_url=args.leansearch_url,
            leansearch_top_k=args.leansearch_top_k,
            use_physsearch=args.use_physsearch,
            physsearch_url=args.physsearch_url,
            use_web_search=args.use_web_search,
            dataset_name=it.get("dataset_name"),
            category=it.get("category"),
            force_domain=it.get("retrieval_domain"),
            eval_prompt_patch=args.eval_prompt_patch,
            image=_open_eval_image(it) if args.include_image else None,
            image_caption=it.get("image_caption") or None,
        )
        record = {k: v for k, v in traj.items() if k != "response"}
        if args.save_response:
            record["response"] = traj["response"]
        for key in ("dataset_name", "question_id", "category", "source_subset", "retrieval_domain"):
            if it.get(key):
                record[key] = it[key]
        record["auto_model"] = args.auto_model
        record["critic_model"] = args.critic_model or args.auto_model
        record["eval_prompt_patch"] = bool(args.eval_prompt_patch)
        return record

    with open(out_path, "w", encoding="utf-8") as f:
        def _run(i: int, it: dict) -> None:
            rec = _one(i, it)
            with write_lock:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                print(
                    f"[{i+1}/{len(items)}] rounds={rec['num_rounds']} "
                    f"tools={rec['num_tool_calls']} "
                    f"tok={rec['response_tokens']}(gen {rec['gen_tokens']}"
                    f"+tool {rec['tool_tokens']})/{rec['budget']} "
                    f"stop={rec['stop_reason']} answer={rec['answer_tag']} "
                    f"syntax={rec['syntax_pass']} success={rec['success']}"
                    + (f" VERIFY_ERROR={rec['verify_error'][:60]}"
                       if rec.get("verify_error") else ""),
                    flush=True,
                )

        if args.workers <= 1:
            for i, it in enumerate(items):
                try:
                    _run(i, it)
                except Exception as e:
                    print(f"  -> Error on item {i+1}: {type(e).__name__}: {e}", flush=True)
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futs = {pool.submit(_run, i, it): i for i, it in enumerate(items)}
                for fut in as_completed(futs):
                    try:
                        fut.result()
                    except Exception as e:
                        print(f"  -> Error on item {futs[fut]+1}: "
                              f"{type(e).__name__}: {e}", flush=True)

    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

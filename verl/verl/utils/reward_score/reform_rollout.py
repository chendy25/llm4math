"""Score a ReForm autoformalization rollout the same way eval does.

Gates (independent):
  compile  — Kimina Lean verify (Mathlib or Physlib header by domain)
  semantic — TokenHub critic ``<consistency>Correct</consistency>``
  both     — compile AND semantic

GRPO scalar is ``0.5 * compile + 0.5 * semantic`` so BOTH=1.0, one gate=0.5,
neither=0. Infra Lean failures do not count as a compiler reject; they zero
the compile bit and set ``verify_error``.
"""

from __future__ import annotations

import os
import re
import sys
import threading
from pathlib import Path
from typing import Any, Optional

_REFORM_TRAIN = Path(
    os.environ.get(
        "REFORM_TRAIN",
        "/mnt/afs_reason/chendongyang/code/llm4math/ReForm/data/train",
    )
)
if str(_REFORM_TRAIN) not in sys.path:
    sys.path.insert(0, str(_REFORM_TRAIN))

import lean_syntax_fix  # noqa: E402
import llm_client  # noqa: E402
import prompts  # noqa: E402
import verify_lean  # noqa: E402

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
LEAN_BLOCK_RE = re.compile(r"```lean4\s*(.*?)\s*```", re.DOTALL)

_critic_lock = threading.Lock()
_critic_client = None


def extract_final_lean(body: str) -> tuple[Optional[str], bool]:
    m = ANSWER_RE.search(body or "")
    if m:
        blocks = LEAN_BLOCK_RE.findall(m.group(1))
        if blocks:
            return blocks[-1].strip(), True
    blocks = LEAN_BLOCK_RE.findall(body or "")
    if blocks:
        return blocks[-1].strip(), False
    return None, False


def _domain(extra_info: Optional[dict[str, Any]]) -> str:
    extra = extra_info or {}
    raw = str(extra.get("retrieval_domain") or extra.get("domain") or extra.get("ability") or "math")
    raw = raw.lower()
    if raw in {"physics", "physlib", "physlib_lib", "phyx"}:
        return "physics"
    return "math"


def _header(domain: str) -> str:
    if domain == "physics":
        return verify_lean.physics_header()
    return verify_lean.DEFAULT_HEADER


def _critic(informal: str, lean: str) -> tuple[bool, str]:
    global _critic_client
    model = os.environ.get("CRITIC_MODEL", "qwen3.7-plus")
    extra_body = {"enable_thinking": False} if os.environ.get("CRITIC_NO_THINKING", "1") != "0" else None
    with _critic_lock:
        if _critic_client is None:
            _critic_client = llm_client.get_critic_client(
                base_url=os.environ.get("CRITICLEAN_BASE_URL") or os.environ.get("CRITIC_BASE_URL"),
                api_key=os.environ.get("CRITICLEAN_API_KEY")
                or os.environ.get("CRITIC_API_KEY")
                or os.environ.get("TOKENHUB_API_KEY"),
            )
        client = _critic_client
    user = prompts.CONSISTENCY_CHECK_USER_FIRST.format(
        informal_statement=informal,
        formal_statement=lean,
    )
    text = llm_client.chat_completion(
        client,
        [
            {"role": "system", "content": prompts.CONSISTENCY_CHECK_SYSTEM},
            {"role": "user", "content": user},
        ],
        model=model,
        temperature=0.0,
        max_tokens=int(os.environ.get("CRITIC_MAX_TOKENS", "4096")),
        extra_body=extra_body,
        content_only=True,
    )
    ok, comments = llm_client.parse_consistency_result(text)
    return ok, comments


def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    **kwargs,
) -> dict[str, Any]:
    extra = extra_info if isinstance(extra_info, dict) else {}
    informal = str(extra.get("informal_statement") or ground_truth or "").strip()
    lean, _from_answer = extract_final_lean(solution_str or "")
    if not lean:
        return {
            "score": 0.0,
            "compile": 0.0,
            "semantic": 0.0,
            "both": 0.0,
            "verify_error": 0.0,
        }

    lean, _notes = lean_syntax_fix.rewrite_lean4_statement(lean)
    domain = _domain(extra)
    canonical = os.environ.get("VERIFY_CANONICAL_HEADER", "1") != "0"
    passed, _err, status = verify_lean.verify_sync_status(
        lean,
        header=_header(domain),
        canonical_header=canonical,
    )
    compile_bit = 1.0 if (passed and status == verify_lean.OK) else 0.0
    infra = 1.0 if status == verify_lean.INFRA_ERROR else 0.0

    semantic_bit = 0.0
    if informal:
        try:
            ok, _comments = _critic(informal, lean)
            semantic_bit = 1.0 if ok else 0.0
        except Exception:
            semantic_bit = 0.0

    both = 1.0 if (compile_bit == 1.0 and semantic_bit == 1.0) else 0.0
    return {
        "score": 0.5 * compile_bit + 0.5 * semantic_bit,
        "compile": compile_bit,
        "semantic": semantic_bit,
        "both": both,
        "verify_error": infra,
    }

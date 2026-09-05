"""Single-shot ("RL rollout") autoformalization for evaluation.

The SFT target is ONE assistant turn:

    <|im_start|>user
    Think step by step to translate ... {informal_statement}<|im_end|>
    <|im_start|>assistant
    <think>
    # Round 1 (Autoformalization)
    <round>
    ... reasoning, <tool_call>/<tool_response> pairs, ```lean4 ...``` ...
    # Round 1 (Semantic Self-validation)
    ... critique ... Therefore, the outcome ... is: Incorrect
    </round>
    # Round 2 (Autoformalization)
    ...
    </think>
    <answer>
    ```lean4
    ...
    ```
    </answer>

`collect_sft_trajectory.collect_one_trajectory` *manufactures* that string from
an outer Python loop: a tool-description system prompt, one chat request per
round with a rewritten history, an external Lean verifier and an external
critic deciding when to refine. Evaluating through it measures the scaffolding,
not the model.

This module instead does what an RL rollout does: one growing assistant turn on
the raw completion endpoint, with a single token budget for the whole
trajectory. Generation stops on `</tool_call>`, the tool really runs, its
`<tool_response>` is spliced into the same sequence, and generation resumes from
the extended prefix. The model decides on its own when to refine and when to
close `</answer>`. Lean verify and the critic run once afterwards, for scoring
only, and never feed back into generation.
"""

from __future__ import annotations

import os
import re
from typing import Optional

try:
    from . import prompts
    from . import llm_client
    from . import verify_lean
    from . import retrieval_tool
    from . import lean_syntax_fix
    from .collect_sft_trajectory import MATH_IDIOM_POLICY, _build_tools, infer_retrieval_domain
except ImportError:
    import prompts
    import llm_client
    import verify_lean
    import retrieval_tool
    import lean_syntax_fix
    from collect_sft_trajectory import MATH_IDIOM_POLICY, _build_tools, infer_retrieval_domain


IM_START = "<|im_start|>"
IM_END = "<|im_end|>"

# ms-swift trained with `default_system: None`, so the rendered sequence has no
# system block at all. Reproducing it exactly matters: an injected system turn
# is enough to push the model off the trajectory format it was trained on.
PROMPT_TEMPLATE = (
    f"{IM_START}user\n" "{user}" f"{IM_END}\n" f"{IM_START}assistant\n"
)

DEFAULT_TOKENIZER = (
    "/mnt/afs_reason/chendongyang/code/llm4math/sft/output/"
    "qwen35_9b_reform_merged_v3/checkpoint-312/tokenizer.json"
)

ROUND_OPEN_RE = re.compile(r"<round>")
ANSWER_RE = re.compile(r"<answer>(.*?)(?:</answer>|$)", re.DOTALL)
LEAN_BLOCK_RE = re.compile(r"```lean4\s*(.*?)\s*```", re.DOTALL)
SELF_VERDICT_RE = re.compile(
    r"outcome of the consistency verification is:\s*(Correct|Incorrect)", re.IGNORECASE
)


class _TokenCounter:
    """Counts tokens of the text we splice in (tool responses).

    Generated tokens come from the API's `usage.completion_tokens`, which is
    authoritative. Injected text has no usage record, so it is counted locally
    with the model's own tokenizer; a char heuristic is only a fallback, and it
    is cross-checked against the prompt-token delta the server reports on the
    next call so a silent drift cannot quietly inflate the budget.
    """

    def __init__(self, path: Optional[str] = None) -> None:
        self._tok = None
        self.exact = False
        path = path or os.environ.get("ROLLOUT_TOKENIZER") or DEFAULT_TOKENIZER
        try:
            from tokenizers import Tokenizer  # type: ignore

            self._tok = Tokenizer.from_file(path)
            self.exact = True
        except Exception:
            self._tok = None

    def count(self, text: str) -> int:
        if not text:
            return 0
        if self._tok is not None:
            return len(self._tok.encode(text, add_special_tokens=False).ids)
        # Qwen tokenizers land near 3.3 chars/token on this mixed Lean/English
        # text; rounding down would understate the budget, so round up.
        return len(text) // 3 + 1


_COUNTER: Optional[_TokenCounter] = None


def token_counter() -> _TokenCounter:
    global _COUNTER
    if _COUNTER is None:
        _COUNTER = _TokenCounter()
    return _COUNTER


def eval_user_patch(domain: str) -> str:
    """Constraints that lived on the *collection* system prompt.

    SFT never saw them (no system turn). Appending to the user message keeps
    the trained shape (one user + one assistant) and is the least-bad place
    to inject them at eval. Off by default; see ``--eval-prompt-patch``.
    """
    if domain == "physics":
        try:
            from . import physics_header as _phys_hdr
        except ImportError:
            import physics_header as _phys_hdr
        header = _phys_hdr.prompt_header_block()
        return (
            "\n\n# Lean 4.29 / Physlib (required)\n"
            "Call `physsearch` at least once before the final ```lean4 block. "
            "`leansearch` is optional for a pure Mathlib lemma.\n"
            "The ```lean4 block MUST begin with EXACTLY this header (verbatim, "
            "these lines first, in this order):\n"
            f"```lean4\n{header}\n```\n"
            "You MAY append more specific Physlib / PhysLean / QuantumInfo "
            "imports discovered via physsearch AFTER these lines. Do NOT drop "
            "the header, do NOT invent top-level imports, and do NOT fall back "
            "to a bare Mathlib-only header.\n"
            "\n# Physics formalization fidelity (required)\n"
            "- Encode THIS problem's asked quantity, not a free-floating "
            "textbook identity. If the question asks for a value/expression "
            "given data, the theorem conclusion must determine that quantity.\n"
            "- Only use Physlib names that appeared in a `physsearch` result. "
            "If retrieval is empty, keep `import Physlib` and formalize with "
            "`ℝ` / `Real`. Do not invent modules. End with `:= by sorry`.\n"
            "- Every identifier must be a binder, a Mathlib/Physlib name from "
            "tools, or a numeric literal.\n"
        )
    return (
        "\n\n# Lean 4.29 / Mathlib (required)\n"
        "Call `leansearch` at least once before the final ```lean4 block. "
        "Do not call `physsearch` on a pure math problem.\n"
        + MATH_IDIOM_POLICY
    )


def build_prompt(
    informal_statement: str,
    *,
    domain: str = "math",
    eval_prompt_patch: bool = False,
) -> str:
    user = prompts.format_user_prompt(informal_statement, domain)
    if eval_prompt_patch:
        user = user.rstrip() + eval_user_patch(domain)
    return PROMPT_TEMPLATE.format(user=user)


def _choice_text(choice) -> str:
    msg = getattr(choice, "message", None)
    if msg is not None:
        return llm_client._assistant_text(msg)
    return choice.text or ""


def _complete(
    client,
    prompt: str,
    *,
    model: str,
    max_tokens: int,
    temperature: float,
    stop: list[str],
    vision_user: Optional[dict] = None,
    body: str = "",
):
    """One generation chunk.

    Text items stay on `/v1/completions` so a half-written assistant turn can
    continue. Vision items use chat with the figure on the first chunk, and
    `continue_final_message` for later chunks so LightLLM does not start a
    new assistant turn. If that knob is rejected, later chunks fall back to
    text completions (image already consumed on the first call).
    """
    if vision_user is None:
        return client.completions.create(
            model=model or "default",
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
        )
    messages = [vision_user]
    extra: Optional[dict] = None
    if body:
        messages.append({"role": "assistant", "content": body})
        extra = {"continue_final_message": True, "add_generation_prompt": False}
    kwargs = dict(
        model=model or "default",
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=stop,
    )
    try:
        if extra:
            return client.chat.completions.create(**kwargs, extra_body=extra)
        return client.chat.completions.create(**kwargs)
    except Exception:
        if extra is None:
            raise
        return client.completions.create(
            model=model or "default",
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
        )


def _truncate_tool_output(text: str, max_tokens: int) -> str:
    """Cap one tool response.

    Retrieval blocks in the SFT data run ~865 tokens (p95 1530), so verl's
    256-*character* default would be a train/eval mismatch. The cap here only
    exists to stop one pathological payload from eating the whole budget.
    """
    counter = token_counter()
    if counter.count(text) <= max_tokens:
        return text
    # Cut on result boundaries ("[1] ...", "[2] ...") so a truncated block is
    # still parseable rather than ending mid-identifier.
    parts = re.split(r"(?m)^(?=\[\d+\])", text)
    kept: list[str] = []
    used = 0
    for part in parts:
        n = counter.count(part)
        if used + n > max_tokens:
            break
        kept.append(part)
        used += n
    if not kept:
        return text[: max_tokens * 3] + "\n...(truncated)"
    return "".join(kept).rstrip() + "\n...(truncated)"


def _first_json_object(text: str) -> tuple[Optional[dict], int]:
    """Return (parsed object, index just past its closing brace).

    Brace-matching rather than a regex: a chunk can hold several `<tool_call>`
    blocks, and a greedy `\\{.*\\}` would span two of them and fail to parse.
    """
    import json

    start = text.find("{")
    if start < 0:
        return None, -1
    depth = 0
    in_str = False
    escaped = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if escaped:
                escaped = False
            elif c == "\\":
                escaped = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(text[start : i + 1])
                except Exception:
                    return None, -1
                return (obj if isinstance(obj, dict) else None), i + 1
    return None, -1


def trailing_tool_call(text: str) -> Optional[dict]:
    """The tool call this chunk *ended on*, or None.

    Generation stops on `</tool_call>`, so a serviceable call is always the
    last thing in the chunk. A `<tool_call>` that the model wrote and then
    generated straight past (SFT trajectories do contain such dangling tags,
    because the collector only re-appended `</tool_call>` when it actually
    dispatched) must NOT be dispatched: its response would land far below the
    text that already moved on, corrupting the sequence.
    """
    if not text or retrieval_tool.TOOL_CALL_OPEN not in text:
        return None
    tail = text.rsplit(retrieval_tool.TOOL_CALL_OPEN, 1)[1]
    call, end = _first_json_object(tail)
    if call is None:
        return None
    rest = tail[end:].strip()
    if rest and rest != retrieval_tool.TOOL_CALL_CLOSE:
        return None
    return call


def extract_final_lean(body: str) -> tuple[Optional[str], bool]:
    """Return (lean_code, from_answer_tag).

    The trained format puts the final statement in `<answer>`; a rollout that
    ran out of budget mid-`<think>` has none, and then the last ```lean4 block
    it did produce is the fairest thing to score.
    """
    m = ANSWER_RE.search(body)
    if m:
        blocks = LEAN_BLOCK_RE.findall(m.group(1))
        if blocks:
            return blocks[-1].strip(), True
    blocks = LEAN_BLOCK_RE.findall(body)
    if blocks:
        return blocks[-1].strip(), False
    return None, False


def rollout_one(
    informal_statement: str,
    auto_client,
    critic_client=None,
    *,
    model: str = "default",
    critic_model: str = "default",
    temperature: float = 0.6,
    # One budget for the whole assistant turn: model tokens AND spliced tool
    # tokens, exactly like verl's `rollout.response_length`.
    max_response_tokens: int = 12288,
    max_tool_calls: int = 9,
    max_tool_response_tokens: int = 1536,
    max_tokens_consistency: int = 4096,
    use_lean_verify: bool = False,
    use_leansearch: bool = True,
    leansearch_url: str = "http://127.0.0.1:8000",
    leansearch_top_k: int = 8,
    use_physsearch: bool = False,
    physsearch_url: str = "http://127.0.0.1:8000",
    use_web_search: bool = False,
    tool_timeout: float = 60.0,
    dataset_name: Optional[str] = None,
    category: Optional[str] = None,
    force_domain: Optional[str] = None,
    eval_prompt_patch: bool = False,
    image=None,
    image_caption: Optional[str] = None,
) -> dict:
    """Roll out one problem end to end, then score it. See module docstring."""
    if critic_client is None:
        critic_client = auto_client
    if critic_model == "default" and model != "default":
        critic_model = model

    if force_domain in ("math", "physics"):
        domain = force_domain
    else:
        domain = infer_retrieval_domain(
            informal_statement, dataset_name=dataset_name, category=category
        )
    physsearch_for_item = use_physsearch and (
        domain == "physics" or os.environ.get("PHYSSEARCH_ON_MATH") == "1"
    )
    tools = _build_tools(
        use_leansearch=use_leansearch,
        use_physsearch=physsearch_for_item,
        leansearch_url=leansearch_url,
        physsearch_url=physsearch_url,
        use_web_search=use_web_search,
    )

    counter = token_counter()
    base_prompt = build_prompt(
        informal_statement,
        domain=domain,
        eval_prompt_patch=eval_prompt_patch,
    )
    vision_user = None
    if image is not None:
        user = prompts.format_user_prompt(informal_statement, domain)
        if eval_prompt_patch:
            user = user.rstrip() + eval_user_patch(domain)
        vision_user = llm_client.build_vision_user_message(
            user, image=image, image_caption=image_caption,
        )
    body = ""
    gen_tokens = 0
    tool_tokens = 0
    n_tool_calls = 0
    tools_used: list[str] = []
    tool_errors = 0
    stop_reason = "eos"
    prompt_tokens_first: Optional[int] = None
    prompt_tokens_last: Optional[int] = None
    completion_tokens_last = 0

    # +2 so a rollout that spends its last tool call still gets a turn to write
    # the answer; the token budget is the real limit.
    for _ in range(max_tool_calls + 2):
        remaining = max_response_tokens - gen_tokens - tool_tokens
        if remaining <= 0:
            stop_reason = "budget"
            break
        resp = _complete(
            auto_client,
            base_prompt + body,
            model=model,
            max_tokens=remaining,
            temperature=temperature,
            stop=[retrieval_tool.TOOL_CALL_CLOSE],
            vision_user=vision_user,
            body=body,
        )
        choice = resp.choices[0]
        text = _choice_text(choice)
        usage = getattr(resp, "usage", None)
        if usage is not None:
            if prompt_tokens_first is None:
                prompt_tokens_first = usage.prompt_tokens
            prompt_tokens_last = usage.prompt_tokens
            completion_tokens_last = usage.completion_tokens
            gen_tokens += usage.completion_tokens
        else:
            gen_tokens += counter.count(text)
        body += text

        if choice.finish_reason == "length":
            stop_reason = "length"
            break

        call = trailing_tool_call(text)
        if call is None:
            stop_reason = "eos"
            break
        if n_tool_calls >= max_tool_calls:
            # Do not silently drop the call: tell the model, so it can still
            # close `<answer>` instead of waiting for a response forever.
            body += retrieval_tool.TOOL_CALL_CLOSE + (
                "\n"
                + retrieval_tool.TOOL_RESPONSE_OPEN
                + "\n(Tool call budget exhausted. Write the final Lean statement now.)\n"
                + retrieval_tool.TOOL_RESPONSE_CLOSE
                + "\n"
            )
            stop_reason = "tool_budget"
            continue

        # `stop` is not echoed back by the server, so the close tag has to be
        # re-appended or the recorded trajectory would be malformed.
        if not body.rstrip().endswith(retrieval_tool.TOOL_CALL_CLOSE):
            body = body.rstrip() + "\n" + retrieval_tool.TOOL_CALL_CLOSE

        name = call.get("name") or call.get("tool")
        try:
            output = llm_client._dispatch_tool_call(
                call, tools, default_num_results=leansearch_top_k, timeout=tool_timeout,
            )
        except Exception as e:
            output = f"(Tool error: {type(e).__name__}: {e})"
            tool_errors += 1
        output = _truncate_tool_output(output or "(No results found.)", max_tool_response_tokens)
        block = (
            "\n"
            + retrieval_tool.TOOL_RESPONSE_OPEN
            + "\n"
            + output
            + "\n"
            + retrieval_tool.TOOL_RESPONSE_CLOSE
            + "\n"
        )
        body += block
        tool_tokens += counter.count(block)
        n_tool_calls += 1
        if name:
            tools_used.append(str(name))
    else:
        stop_reason = "turn_limit"

    final_lean, from_answer = extract_final_lean(body)
    syntax_rewrites: list[str] = []
    if final_lean:
        final_lean, notes = lean_syntax_fix.rewrite_lean4_statement(final_lean)
        syntax_rewrites = sorted(set(notes or []))

    syntax_pass: Optional[bool] = None
    verify_error: Optional[str] = None
    if use_lean_verify and final_lean:
        header = verify_lean.physics_header() if domain == "physics" else None
        passed, err, status = verify_lean.verify_sync_status(
            final_lean,
            header=header,
            canonical_header=os.environ.get("VERIFY_CANONICAL_HEADER") == "1",
        )
        if status == verify_lean.INFRA_ERROR:
            verify_error = err
        else:
            syntax_pass = passed

    success = False
    critic_comments = ""
    if final_lean:
        user_cons = prompts.CONSISTENCY_CHECK_USER_FIRST.format(
            informal_statement=informal_statement, formal_statement=final_lean,
        )
        critic_extra = (
            {"enable_thinking": False}
            if os.environ.get("CRITIC_NO_THINKING", "1") == "1"
            else None
        )
        critic_text = llm_client.chat_completion(
            critic_client,
            [
                {"role": "system", "content": prompts.CONSISTENCY_CHECK_SYSTEM},
                {"role": "user", "content": user_cons},
            ],
            model=critic_model,
            temperature=0.3,
            max_tokens=max_tokens_consistency,
            extra_body=critic_extra,
            content_only=bool(critic_extra),
        )
        success, critic_comments = llm_client.parse_consistency_result(critic_text)

    self_verdicts = SELF_VERDICT_RE.findall(body)
    total_response = gen_tokens + tool_tokens
    observed_total = None
    if prompt_tokens_first is not None and prompt_tokens_last is not None:
        # Server-side ground truth for the whole assistant turn, used to audit
        # the local tool-token estimate.
        observed_total = prompt_tokens_last + completion_tokens_last - prompt_tokens_first

    return {
        "response": body,
        "final_lean_code": final_lean or "",
        "answer_tag": from_answer,
        "num_rounds": len(ROUND_OPEN_RE.findall(body)),
        "num_tool_calls": n_tool_calls,
        "tools_used": sorted(set(tools_used)),
        "tool_errors": tool_errors,
        "self_verdict": (self_verdicts[-1].lower() == "correct") if self_verdicts else None,
        "success": bool(success),
        "syntax_pass": syntax_pass,
        "verify_error": verify_error,
        "syntax_rewrites": syntax_rewrites,
        "critic_comments": critic_comments,
        "gen_tokens": gen_tokens,
        "tool_tokens": tool_tokens,
        "response_tokens": total_response,
        "response_tokens_server": observed_total,
        "prompt_tokens": prompt_tokens_first,
        "budget": max_response_tokens,
        "stop_reason": stop_reason,
        "truncated": stop_reason in ("length", "budget", "turn_limit"),
        "retrieval_domain": domain,
        "eval_prompt_patch": bool(eval_prompt_patch),
        "used_vision": image is not None,
        "informal_statement": informal_statement,
    }

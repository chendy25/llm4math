#!/usr/bin/env python3
"""Repair already-collected ReForm trajectories so they match the fixed pipeline.

The first collection run baked three things into the assistant target that the
model can never reproduce at inference time:

* rounds whose "critique" is the Kimina compiler's error text, or one of the
  pipeline's own rejections (`Vacuous formalization rejected`,
  `Lean verification unavailable`). ReForm keeps only semantic critiques in the
  refinement history (paper §3.1: H_t = {(S_j, C_j)}); the compiler is a gate
  that decides *whether* to refine (App. Step 2) and semantic evaluation runs
  only on statements that already compile (Step 3).
* a `[syntax_rewrite] ...` note plus a duplicated ```lean4 block, which is an
  artefact of `lean_syntax_fix` running between the model and the verifier.
* dangling `<tool_call>` tags, because generation stopped on `</tool_call>` and
  the collector only re-appended it on the dispatch path.

Re-collecting is expensive, so this module parses the built trajectory back into
rounds, drops/cleans them, and rebuilds. Round numbering is regenerated, so a
trajectory whose first two attempts were gate rejections comes out as a clean
single-round example.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

TRAIN_DIR = Path(__file__).resolve().parents[2] / "ReForm/data/train"
if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))

import lean_syntax_fix  # noqa: E402
import retrieval_tool  # noqa: E402
from trajectory_builder import build_reflective_response  # noqa: E402

ROUND_RE = re.compile(
    r"# Round (\d+) \(Autoformalization\)\n\n<round>\n\n(.*?)"
    r"\n\n# Round \1 \(Semantic Self-validation\)\n\n(.*?)"
    r"\n\nTherefore, the outcome of the consistency verification is: (Correct|Incorrect)"
    r"\n\n</round>",
    re.DOTALL,
)
ANSWER_RE = re.compile(r"<answer>\s*```lean4\n(.*?)\n```\s*</answer>", re.DOTALL)

# Critiques that are not critiques: the gate's own verdict, pasted in verbatim.
GATE_CRITIQUE_PREFIXES = (
    "Lean compiler reported errors:",
    "Lean verification unavailable:",
    "Vacuous formalization rejected",
)

SYNTAX_REWRITE_APPENDIX = re.compile(
    r"\n*\[syntax_rewrite\][^\n]*\n+```lean4\n.*?\n```\n*", re.DOTALL
)
LEAN_BLOCK_RE = re.compile(r"(```lean4\n)(.*?)(\n```)", re.DOTALL)
TOOL_RESPONSE_RE = re.compile(r"<tool_response>.*?</tool_response>", re.DOTALL)

# Prose that only makes sense if the writer had seen a compiler message. Checked
# on rounds that directly follow a dropped gate round, since that is where the
# collector was reacting to an error we are now hiding.
COMPILER_PROSE_RE = re.compile(
    r"\b(?:the|this|that|above|previous|following|first|original|reported|"
    r"compil\w+|parse|syntax|type)\s+errors?\b"
    r"|\berrors?\s+(?:message|says?|indicates?|shows?|is|was|occurs?)\b"
    r"|\blooking at the error\b"
    r"|\bunknown (?:identifier|constant)\b"
    r"|\btype mismatch\b"
    r"|\bfailed to (?:compile|elaborate|parse|synthesize)\b"
    r"|\bdid ?n[o']t compile\b"
    r"|\bdoes not (?:compile|elaborate|typecheck)\b"
    r"|\bcompilation (?:error|failure|failed)\b"
    r"|\bsyntactic verification\b",
    re.IGNORECASE,
)


# Prose that presupposes an attempt the repaired trajectory no longer contains.
# Harmless mid-trajectory (there really is a previous round), incoherent once the
# round is renumbered to Round 1 because the gate round before it was dropped.
PRIOR_ATTEMPT_RE = re.compile(
    r"\bprevious (?:formal )?(?:statement|attempt|formalization|version|code)\b"
    r"|\bearlier (?:attempt|version|statement)\b"
    r"|\blet me (?:fix|correct|revise|rewrite) (?:this|it|that|the \w+( \w+)?)\b"
    r"|\b(?:my|the) (?:last|first) attempt\b"
    r"|\binstead of the (?:previous|last)\b",
    re.IGNORECASE,
)

# Leading clause that only points at the gate outcome. Everything after it is
# usually retrieval-grounded Lean reasoning worth keeping, so try removing just
# this before falling back to deleting the whole sentence.
#
# Restricted on purpose to forms whose remainder is an independent clause: a
# comma-terminated adverbial, or a `... that|because <clause>` complement.
# `The error is in the Finset.prod binder syntax` would leave the bare noun
# phrase `The Finset.prod binder syntax.`, so it falls through to sentence
# deletion instead.
GATE_MARKER_RE = re.compile(
    r"^(?:now\s+)?looking at the (?:compiler\s+)?errors?(?:\s+messages?)?"
    r"(?:\s+and[^,:\n]*)?,\s*"
    r"|^the errors?\s+(?:is|are|was|were)\s+that\s+"
    r"|^the errors?\s+occurs?\s+because\s+"
    r"|^(?:based on|from)\s+the errors?(?:\s+messages?)?,\s*",
    re.IGNORECASE,
)

# Regions whose text must never be edited: injected tool output and Lean code.
PROTECTED_RE = re.compile(
    r"<tool_response>.*?</tool_response>|```[a-zA-Z0-9]*\n.*?\n```",
    re.DOTALL,
)
_TICKS_RE = re.compile(r"`[^`\n]*`")


def _sentence_bounds(text: str, lo: int, hi: int) -> tuple[int, int]:
    """Expand [lo, hi) to sentence boundaries, ignoring dots inside backticks.

    `Nat.digits` and `1.5` would fool a plain `[.!?]` split, so backtick spans
    are blanked out before looking for boundaries.
    """
    masked = _TICKS_RE.sub(lambda m: " " * len(m.group(0)), text)
    start = 0
    for m in re.finditer(r"(?:[.!?:]\s+|\n)", masked[:lo]):
        start = m.end()
    m = re.search(r"(?:[.!?]\s+|\n)", masked[hi:])
    end = hi + m.end() if m else len(text)
    return start, end


def _excise_in_segment(seg: str) -> str:
    """Drop compiler narration from one editable prose segment."""
    for _ in range(6):  # a segment can hold several narrating sentences
        m = COMPILER_PROSE_RE.search(seg)
        if not m:
            return seg
        lo, hi = _sentence_bounds(seg, m.start(), m.end())
        sentence = seg[lo:hi]
        trimmed = GATE_MARKER_RE.sub("", sentence.lstrip(), count=1)
        if trimmed and not COMPILER_PROSE_RE.search(trimmed):
            # Only the discourse marker was the problem; keep the diagnosis,
            # which is type-level Lean reasoning the model can derive itself.
            trimmed = trimmed[0].upper() + trimmed[1:]
            seg = seg[:lo] + trimmed + seg[hi:]
        else:
            seg = seg[:lo] + seg[hi:]
    return seg


def _drop_prior_refs(seg: str) -> str:
    for _ in range(6):
        m = PRIOR_ATTEMPT_RE.search(seg)
        if not m:
            return seg
        lo, hi = _sentence_bounds(seg, m.start(), m.end())
        seg = seg[:lo] + seg[hi:]
    return seg


def excise_compiler_prose(text: str, *, is_first_round: bool) -> Optional[str]:
    """Remove compiler narration from a round, or None if nothing can be saved.

    Tool output and Lean blocks are left byte-identical. The result is re-checked
    with the same detector used to reject rounds outright, so a repaired round
    clears exactly the bar that naturally-clean rounds clear.
    """
    out: list[str] = []
    pos = 0
    for m in PROTECTED_RE.finditer(text):
        seg = text[pos:m.start()]
        seg = _excise_in_segment(seg)
        if is_first_round:
            seg = _drop_prior_refs(seg)
        out.append(seg)
        out.append(m.group(0))
        pos = m.end()
    tail = _excise_in_segment(text[pos:])
    if is_first_round:
        tail = _drop_prior_refs(tail)
    out.append(tail)

    result = re.sub(r"\n{3,}", "\n\n", "".join(out)).strip()
    if not result or "```lean4" not in result:
        return None
    if _narrates_compiler(result):
        return None
    if is_first_round and PRIOR_ATTEMPT_RE.search(PROTECTED_RE.sub("", result)):
        return None
    return result


@dataclass
class Sanitized:
    text: Optional[str]
    rounds_in: int
    rounds_out: int
    dropped_gate: int
    final_verdict: Optional[str]
    stripped_rewrite: bool
    closed_tool_call: bool
    reason: Optional[str] = None  # set when text is None
    excised_prose: int = 0


def _strip_syntax_rewrite(auto_text: str) -> tuple[str, bool]:
    """Drop the `[syntax_rewrite]` appendix, keeping the round's own Lean block.

    The appendix duplicated the block with `lean_syntax_fix` applied, so the
    same fixes are re-applied to the surviving block instead — otherwise the
    round would show `f₁` / `λ` / `in` while `<answer>` shows the rewritten
    form. The rewrites are surface-level and idempotent.
    """
    cleaned, n = SYNTAX_REWRITE_APPENDIX.subn("\n", auto_text)
    if not n:
        return auto_text, False

    def fix(m: re.Match) -> str:
        body, _ = lean_syntax_fix.rewrite_lean4_statement(m.group(2))
        return m.group(1) + body + m.group(3)

    return LEAN_BLOCK_RE.sub(fix, cleaned).rstrip(), True


def _narrates_compiler(auto_text: str) -> bool:
    # Retrieval payloads legitimately talk about "elaboration errors" in Mathlib
    # docstrings; only the model's own prose counts.
    return bool(COMPILER_PROSE_RE.search(TOOL_RESPONSE_RE.sub("", auto_text)))


def sanitize_response(
    asst: str,
    *,
    excise_prose: bool = True,
) -> Sanitized:
    """Repair one built assistant trajectory. Returns text=None if unusable.

    ``excise_prose=False`` drops any trajectory whose prose narrates a compiler
    message instead of editing it, for A/B comparison.
    """
    rounds = ROUND_RE.findall(asst)
    answer = ANSWER_RE.search(asst)
    if not rounds or not answer:
        return Sanitized(None, len(rounds), 0, 0, None, False, False,
                         reason="unparseable")

    kept: list[dict] = []
    dropped_gate = 0
    stripped = False
    closed = False
    excised = 0
    prev_was_gate = False
    for _no, auto_text, critique, verdict in rounds:
        if critique.lstrip().startswith(GATE_CRITIQUE_PREFIXES):
            dropped_gate += 1
            prev_was_gate = True
            continue
        auto_text, did_strip = _strip_syntax_rewrite(auto_text)
        stripped = stripped or did_strip
        # Only a round that followed a gate rejection can be reacting to a
        # compiler message; elsewhere the words are the model's own. Such a
        # round is always run through the repair, because even when it does not
        # narrate the compiler it may still point at an attempt that the gate
        # drop removed ("the previous formal statement...").
        if prev_was_gate:
            narrates = _narrates_compiler(auto_text)
            if narrates and not excise_prose:
                return Sanitized(None, len(rounds), 0, dropped_gate, None,
                                 stripped, closed, reason="compiler_prose")
            repaired = excise_compiler_prose(auto_text, is_first_round=not kept)
            if repaired is None:
                if narrates:
                    return Sanitized(None, len(rounds), 0, dropped_gate, None,
                                     stripped, closed, reason="compiler_prose")
                # Nothing to leak, only an awkward reference we could not
                # excise cleanly. Keep the round as collected.
            elif repaired != auto_text:
                auto_text = repaired
                excised += 1
        prev_was_gate = False
        balanced = retrieval_tool.close_tool_calls(auto_text)
        closed = closed or (balanced != auto_text)
        kept.append({
            "autoformalization_text": balanced,
            "consistency_comments": critique,
            "is_correct": verdict == "Correct",
        })

    if not kept:
        return Sanitized(None, len(rounds), 0, dropped_gate, None,
                         stripped, closed, reason="no_semantic_round")

    text = build_reflective_response(kept, answer.group(1))
    return Sanitized(
        text,
        len(rounds),
        len(kept),
        dropped_gate,
        "Correct" if kept[-1]["is_correct"] else "Incorrect",
        stripped,
        closed,
        excised_prose=excised,
    )


# --- physlib parallel data -------------------------------------------------

DECL_KEYWORDS = ("theorem", "lemma", "def", "abbrev", "instance", "opaque", "axiom",
                 "structure", "class", "noncomputable def")
PHYSLIB_KIND_WORD = {
    "definition": "definition",
    "theorem": "theorem",
    "instance": "instance",
    "abbrev": "abbreviation",
    "opaque": "opaque constant",
}


def _decl_name(lean: str) -> Optional[str]:
    m = re.search(
        r"^\s*(?:noncomputable\s+|private\s+|protected\s+)*"
        r"(?:theorem|lemma|def|abbrev|instance|opaque|axiom|structure|class)\s+"
        r"([A-Za-z_][\w'.«»]*)",
        lean,
        re.MULTILINE,
    )
    return m.group(1) if m else None


def wrap_physlib_response(lean_code: str, *, kind: str = "", module: str = "") -> str:
    """Wrap a bare ```lean4 block in `<answer>` only.

    Parallel Physlib data has no critique to reflect on. A canned `<think>`
    (name / module / binder counts) is 35 sentence skeletons over 4000
    samples and would teach a fake thinking style. Keep the tag the
    reflective trajectories use for the final Lean (`<answer>`), and leave
    `<think>` off so the model is not trained to emit boilerplate.
    `kind` / `module` are accepted for call-site compatibility and unused.
    """
    del kind, module
    lean = lean_code.strip()
    if lean.startswith("```"):
        m = LEAN_BLOCK_RE.search(lean)
        lean = m.group(2).strip() if m else lean.strip("`").strip()
    return "<answer>\n\n```lean4\n" + lean + "\n```\n\n</answer>"

"""Build ReForm-format assistant response from collection rounds (Paper D.4, Case Study C)."""

from typing import List

try:  # package or flat import, matching the rest of data/train
    from . import retrieval_tool
except ImportError:  # pragma: no cover
    import retrieval_tool


def build_reflective_response(
    rounds: List[dict],
    final_lean_code: str,
) -> str:
    """
    Build the full model response for SFT: <think> + <round>...</round> blocks + <answer>...</answer>.

    Each round dict should have:
      - "autoformalization_text": str  (reasoning + lean4 block for this round)
      - "consistency_comments": str    (critique from consistency check)
      - "is_correct": bool             (consistency result)
    """
    parts = ["<think>\n"]
    for i, r in enumerate(rounds, 1):
        parts.append("# Round {} (Autoformalization)\n\n".format(i))
        parts.append("<round>\n\n")
        # Last line of defence for tag balance: whatever produced the round
        # text, an unclosed <tool_call> in the target trains the model to
        # deadlock a single-shot rollout.
        parts.append(retrieval_tool.close_tool_calls(
            r.get("autoformalization_text", "").strip()
        ))
        parts.append("\n\n# Round {} (Semantic Self-validation)\n\n".format(i))
        parts.append(r.get("consistency_comments", "").strip())
        result = "Correct" if r.get("is_correct") else "Incorrect"
        parts.append("\n\nTherefore, the outcome of the consistency verification is: ")
        parts.append(result)
        parts.append("\n\n</round>\n\n")
    parts.append("</think>\n\n")
    parts.append("<answer>\n\n")
    parts.append("```lean4\n")
    parts.append(final_lean_code.strip())
    parts.append("\n```\n\n")
    parts.append("</answer>")
    return "".join(parts)

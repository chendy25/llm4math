"""Conservative Lean 4.29 syntax rewrites before Kimina verify.

These transforms only fix well-known parse failures from teacher models that
still emit Mathlib-3 / older Mathlib-4 idioms. They intentionally avoid
semantic rewrites.
"""

from __future__ import annotations

import re
from typing import Tuple

# Unicode digit subscripts → ASCII (f₁ → f1). Common failure: Lean rejects
# unexpected tokens around subscripted binders like `f₁`.
_SUBSCRIPT_DIGITS = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")

# ∑ / ∃ / ∀ binders: obsolete `in` → `∈`
_BINDER_IN = re.compile(
    r"(∑\s*[^\s,∈]+|∃\s*[^\s,∈]+|∀\s*[^\s,∈]+)\s+in\s+"
)

# Also: `∑ b in Finset...` where binder is a single identifier
_SUM_IN_IDENT = re.compile(r"(∑\s+)([A-Za-z_][\w']*)\s+in\s+")
_EXISTS_IN_IDENT = re.compile(r"([∃∀]\s+)([A-Za-z_][\w']*)\s+in\s+")

_LAMBDA = re.compile(r"\bλ\s+")


def rewrite_lean4_statement(code: str) -> Tuple[str, list[str]]:
    """Apply conservative Lean 4.29 fixes. Returns ``(new_code, change_notes)``."""
    if not code or code.strip() == "(no valid lean4 block)":
        return code, []

    notes: list[str] = []
    out = code

    new = _SUM_IN_IDENT.sub(r"\1\2 ∈ ", out)
    if new != out:
        notes.append("∑ binder: `in` → `∈`")
        out = new

    new = _EXISTS_IN_IDENT.sub(r"\1\2 ∈ ", out)
    if new != out:
        notes.append("∃/∀ binder: `in` → `∈`")
        out = new

    # Fallback for denser binder shapes matched by _BINDER_IN
    new = _BINDER_IN.sub(r"\1 ∈ ", out)
    if new != out:
        notes.append("binder: `in` → `∈` (fallback)")
        out = new

    new = _LAMBDA.sub("fun ", out)
    if new != out:
        notes.append("`λ` → `fun`")
        out = new

    new = out.translate(_SUBSCRIPT_DIGITS)
    if new != out:
        notes.append("unicode subscript digits → ASCII")
        out = new

    # Normalize `:=` / `by sorry` split across lines: `:=\nby sorry` → `:= by sorry`
    new = re.sub(r":=\s*\n\s*by\s+sorry\b", ":= by sorry", out)
    if new != out:
        notes.append("normalize `:= by sorry`")
        out = new

    return out, notes

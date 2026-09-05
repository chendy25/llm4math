"""Single source of truth for the Physlib import header used across the physics
ReForm pipeline.

The exact same header must be shown to the model (prompt), used to compile the
statement (Kimina verify), and used to re-verify harvested KB declarations,
otherwise "what the model was told" drifts from "what we actually compiled" and
we get spurious syntax failures (e.g. physics statements silently compiled under
the Mathlib-only header).

The three lines below were smoke-tested against the live Kimina server
(`import Physlib` + `set_option maxHeartbeats 400000` + `open Real` all compile).
Add PhysLean submodules / extra `open`s here ONLY after confirming they compile
on the verify server — do NOT copy autoformalization_search's `import PhysLean.*`
list, which targets a different library layout.
"""

from __future__ import annotations

PHYSICS_IMPORTS: list[str] = ["import Physlib"]
PHYSICS_OPTIONS: list[str] = ["set_option maxHeartbeats 400000"]
PHYSICS_OPENS: list[str] = ["open Real"]


def header_lines() -> list[str]:
    """Ordered header lines (imports, then options, then opens)."""
    return list(PHYSICS_IMPORTS) + list(PHYSICS_OPTIONS) + list(PHYSICS_OPENS)


def physics_header() -> str:
    """Header string suitable for prepending to a bare statement before compile.

    Trailing blank line so a statement can be concatenated directly.
    """
    return "\n".join(header_lines()) + "\n\n"


def prompt_header_block() -> str:
    """The exact ```lean4 header the model is told to begin its statement with."""
    return "\n".join(header_lines())

"""SFT collection helpers shared with eval rollout.

Eval only needs ``MATH_IDIOM_POLICY``, ``infer_retrieval_domain``, and
``_build_tools``. The full collect-one-trajectory loop lived here historically;
restore it from transcripts if collection is needed again.
"""

from __future__ import annotations

from typing import Optional

try:
    from . import leansearch_tool
except ImportError:
    import leansearch_tool


_PHYSICS_DATASET_MARKERS = (
    "phyx",
    "seephys",
    "physunibench",
    "phyx_af",
    "phyx-af",
    "phyxaf",
    "physlib",
    "physlib_lib",
    "physics",
)

_PHYSICS_CATEGORY_MARKERS = (
    "mechanics",
    "optics",
    "electromagnetism",
    "thermodynamics",
    "modern physics",
    "waves",
    "acoustics",
    "quantum",
    "relativity",
    "stat mech",
    "statistical",
)
_PHYSICS_TEXT_MARKERS = (
    "physlib",
    "physlean",
    "newton",
    "lagrangian",
    "hamiltonian",
    "schrodinger",
    "schrödinger",
    "wavelength",
    "magnetic field",
    "electric field",
    "focal length",
    "thin lens",
    "capacitor",
    "resistor",
    "thermodynamic",
    "entropy",
)

# Each rule below fixes a failure class observed on the AIME 2025 eval: missing
# typeclass instances (5/30), invented Mathlib names (3/30), Fin/ℕ coercion
# mismatches (2/30), and statements so long they exceeded the token budget
# before the block closed (5/30).
MATH_IDIOM_POLICY = (
    "\n# Mathlib idioms that must be respected (elaboration will fail otherwise)\n"
    "- Counting a set: if the predicate is not decidable, do NOT use "
    "`Finset.filter`. Use `Set.ncard {x | ...}` (or `Nat.card`), or add "
    "`open Classical` right after the imports.\n"
    "- `Finset.Icc` / `Finset.range` need a `LocallyFiniteOrder`, which `ℝ` does "
    "NOT have. For real intervals use `Set.Icc` / `Set.Ioo` with `Set.ncard`.\n"
    "- `Fintype ↑S` for a `Set` is not automatic. Prefer `Set.ncard` over "
    "`Fintype.card` unless the carrier is already a `Finset` or a `Fin n`.\n"
    "- Probability: do NOT introduce a `MeasurableSpace`/`PMF` for a finite "
    "counting problem. Express it as a ratio of `Finset.card`s or `Set.ncard`s "
    "over `ℚ`/`ℝ`.\n"
    "- Never mix `Fin n` with `ℕ` arithmetic (`i + 1` where `i : Fin n` will not "
    "elaborate against `ℕ`). Quantify over `ℕ` with an explicit bound instead.\n"
    "- Keep numeric types consistent: distances, areas and roots live in `ℝ`, so "
    "do not equate them to a `ℚ` or `ℕ` expression. Insert casts deliberately.\n"
    "- Every declaration name must have appeared in a `leansearch` result. If "
    "retrieval gave you nothing usable, encode the problem from first principles "
    "with binders, arithmetic and `Finset`/`Set` primitives. Do NOT invent names "
    "like `signedArea`, `List.ofFinset` or `AffineSubspace.circumcenter`.\n"
    "- Do not put `let` bindings inside the theorem type when they can be "
    "hypotheses or plain substitutions; nested `let` chains are a common parse "
    "failure.\n"
    "\n# Keep the statement short\n"
    "Aim for a statement that fits in about 20 lines. State the asked quantity "
    "directly; do not restate the whole solution as nested existentials, and do "
    "not re-derive intermediate steps. A long statement risks being cut off "
    "before the closing ``` and then nothing compiles.\n"
)


def infer_retrieval_domain(
    informal_statement: str = "",
    *,
    dataset_name: Optional[str] = None,
    category: Optional[str] = None,
) -> str:
    """Return ``\"physics\"`` or ``\"math\"`` for joint-retrieval tool routing."""
    ds = (dataset_name or "").strip().lower()
    if ds in _PHYSICS_DATASET_MARKERS:
        return "physics"
    cat = (category or "").strip().lower()
    if cat and any(m in cat for m in _PHYSICS_CATEGORY_MARKERS):
        return "physics"
    text = (informal_statement or "").lower()
    if any(m in text for m in _PHYSICS_TEXT_MARKERS):
        return "physics"
    return "math"


def _build_tools(
    use_leansearch: bool,
    use_physsearch: bool,
    leansearch_url: str,
    physsearch_url: str,
    *,
    preferred_tool: Optional[str] = None,
    use_web_search: bool = False,
) -> list:
    """Construct the ordered tool list for eval / collection dispatch."""
    tools = []
    lean_mod = None
    phys_mod = None
    if use_leansearch:
        leansearch_tool.DEFAULT_URL = leansearch_url
        if "leansearch.net" not in (leansearch_url or ""):
            leansearch_tool.DEFAULT_CORPUS = "mathlib"
        else:
            leansearch_tool.DEFAULT_CORPUS = None
        lean_mod = leansearch_tool
    if use_physsearch:
        try:
            from . import physsearch_tool
        except ImportError:
            import physsearch_tool
        physsearch_tool.DEFAULT_URL = physsearch_url
        phys_mod = physsearch_tool

    if preferred_tool == "physsearch" and phys_mod is not None:
        tools.append(phys_mod)
        if lean_mod is not None:
            tools.append(lean_mod)
    else:
        if lean_mod is not None:
            tools.append(lean_mod)
        if phys_mod is not None:
            tools.append(phys_mod)
    if use_web_search:
        try:
            from . import web_search_tool
        except ImportError:
            import web_search_tool
        tools.append(web_search_tool)
    return tools

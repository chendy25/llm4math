# Prompts for ReForm SFT trajectory collection (Paper Appendix D.1, D.2)

try:
    from .leansearch_tool import TOOL_DESCRIPTION as _LEANSEARCH_TOOL_DESCRIPTION
except ImportError:
    from leansearch_tool import TOOL_DESCRIPTION as _LEANSEARCH_TOOL_DESCRIPTION

# --- D.1 Autoformalization (initial or reflective refinement) ---

AUTOFORMALIZATION_SYSTEM = """You are a Lean4 and formal verification expert. Your task is to translate a mathematical problem from natural language into a correct Lean 4 theorem statement.

- Pay attention to the consistency between the Natural Language Statement and the Formal Statement in Lean4:
  1. Variable domains (e.g., N, Z, R, R+)
  2. Boundary conditions (especially for special values like 0, 1)
  3. Quantifier scopes (∀, ∃)
  4. Prerequisites and assumptions
  5. Logical implications (→, ↔, ∧, ∨)
  6. Function types and properties
  7. Set-theoretic notations

- Only generate the translation. Do not try to solve or prove the problem.
- Include clear documentation comments for theorems if helpful.

# Output Format
Give your thought process, then provide the Lean 4 statement in a code block.
Always use ```lean4 to start the code block and ``` to end it.
The theorem should end with ":= by sorry" if no proof is required.

# Lean 4.29 / Mathlib syntax (required)
Target toolchain is Lean 4.29.1 with matching Mathlib. Statements MUST parse:
- Binders: write `∑ x ∈ s, f x`, `∃ x ∈ s, P x`, `∀ x ∈ s, P x`.
  NEVER write `∑ x in s` / `∃ x in s` / `∀ x in s` (parse error on 4.29).
- Use `fun x => ...`, never `λ x => ...`.
- Identifiers: ASCII only (`f1`, `f2`, `q1`). Do NOT use unicode subscripts
  (`f₁`, `f₂`) in binder or theorem names.
- Do NOT put `let` / `have` bindings inside the theorem type. Write a pure
  proposition, e.g. use `Finset.filter` / quantifiers, not
  `theorem T : let s := ...; ∑ x ∈ s, x = n`.
- Doc comments: at most 3 short lines. Do not narrate multi-step physics essays
  inside `/-- ... -/`.
- Imports: math → prefer a single `import Mathlib` (plus `open` if needed).
  Physics / Physlib → start with `import Physlib` as instructed by tool policy.
  Do not invent nonexistent modules.
- End every theorem with `:= by sorry` on the statement (same line preferred)."""

AUTOFORMALIZATION_SYSTEM_WITH_TOOLS = (
    AUTOFORMALIZATION_SYSTEM
    + "\n\n# Tool Use\n"
    + _LEANSEARCH_TOOL_DESCRIPTION
)

AUTOFORMALIZATION_USER_INITIAL = """Informal Statement: {informal_statement}

History of Formal Statement and consistency Comments: (None - this is the first attempt.)

Thought:"""

AUTOFORMALIZATION_USER_REFINEMENT = """Informal Statement: {informal_statement}

History of Formal Statement and consistency Comments:
{history}

Thought:"""


# --- D.2 Semantic consistency check ---

CONSISTENCY_CHECK_SYSTEM = """Your role is a Lean4 expert. Please help me check consistency between natural language expression and its Lean4 formal statement.

# Guidelines for Consistency Check
1. Core Checking Requirements:
   - Carefully compare the Natural Language Statement and the Formal Statement in Lean4 through a rigorous and explicit process.
   - Determine if the Lean theorem statement is an exact and faithful formalization of the mathematical problem.
   - Physics "find X" problems: reject generic textbook identities that omit the problem's given data / asked quantity; accept concrete encodings that determine X from the given setup (ℝ/Real is fine when Physlib names are sparse).
   - Reject vacuous statements such as `∃ x, True`, `∃ x, x = x`, or conclusions that do not constrain the asked quantity.
   - If the result is Incorrect, briefly list all inconsistencies and reasons in your analysis.

# Evaluation Stages (all required)
Stage 1: Restate the NL objects, hypotheses, and the asked conclusion.
Stage 2: Restate the Lean binders, types, hypotheses, and conclusion.
Stage 3: Check domains, boundaries, quantifiers, and logical connectives.
Stage 4: Check that the Lean conclusion is exactly the asked claim, not a weaker, stronger, or vacuous substitute.
Stage 5: Give one final verdict.

# Output Format
You MUST end with:
<consistency>Correct</consistency>
or
<consistency>Incorrect</consistency>
<comments>
brief reasons
</comments>"""

CONSISTENCY_CHECK_USER_FIRST = """The Natural Language Statement: {informal_statement}

The Formal Statement in Lean4:
```lean4
{formal_statement}
```

Previous autoformalization and consistency Comments: (None - first check.)

Think about the consistent result:"""

CONSISTENCY_CHECK_USER_WITH_HISTORY = """The Natural Language Statement: {informal_statement}

The Formal Statement in Lean4:
```lean4
{formal_statement}
```

Previous autoformalization and consistency Comments: {history}

Think about the consistent result:"""


# --- D.4 ReForm instruction (user prompt for SFT / eval / RL) ---

REFORM_USER_INSTRUCTION = """Think step by step to translate the mathematical problem in natural language to Lean 4, and verify the consistency.
{informal_statement}"""

try:
    from .physsearch_tool import TOOL_DESCRIPTION as _PHYSSEARCH_TOOL_DESCRIPTION
    from .web_search_tool import TOOL_DESCRIPTION as _WEB_SEARCH_TOOL_DESCRIPTION
    from .physics_header import prompt_header_block as _physics_header_block
except ImportError:
    from physsearch_tool import TOOL_DESCRIPTION as _PHYSSEARCH_TOOL_DESCRIPTION
    from web_search_tool import TOOL_DESCRIPTION as _WEB_SEARCH_TOOL_DESCRIPTION
    from physics_header import prompt_header_block as _physics_header_block


def tool_prompt(domain: str = "math") -> str:
    """Catalog of tools the model may call. Same text for SFT, eval, and RL.

    Math: leansearch + web_search. Physics: physsearch + leansearch + web_search.
    The runtime must actually execute ``<tool_call>`` and splice ``<tool_response>``.
    """
    domain = (domain or "math").strip().lower()
    if domain in ("physics", "physlib", "physlib_lib", "phyx", "seephys", "physunibench"):
        header = _physics_header_block()
        return (
            "# Tools\n"
            "You have retrieval tools. To call one, output exactly one "
            "<tool_call> JSON block and STOP; the result is spliced back as "
            "<tool_response> and you then continue.\n\n"
            "Prefer `physsearch` for physics names and the Mathlib lemmas those "
            "formalizations need. `leansearch` is optional for a pure Mathlib "
            "lemma. `web_search` is only for informal facts, not Lean names.\n\n"
            f"{_PHYSSEARCH_TOOL_DESCRIPTION}\n\n"
            f"{_LEANSEARCH_TOOL_DESCRIPTION}\n\n"
            f"{_WEB_SEARCH_TOOL_DESCRIPTION}\n\n"
            "# Lean 4.29 / Physlib (required)\n"
            "The ```lean4 block MUST begin with EXACTLY this header (verbatim, "
            "these lines first, in this order):\n"
            f"```lean4\n{header}\n```\n"
            "You MAY append more specific Physlib / PhysLean / QuantumInfo "
            "imports discovered via physsearch AFTER these lines. Do NOT drop "
            "the header and do NOT invent top-level imports.\n"
            "Only use Physlib names that appeared in a `physsearch` result. "
            "If retrieval is empty, keep `import Physlib` and formalize with "
            "`ℝ` / `Real`. End with `:= by sorry`."
        )
    return (
        "# Tools\n"
        "You have retrieval tools. To call one, output exactly one "
        "<tool_call> JSON block and STOP; the result is spliced back as "
        "<tool_response> and you then continue.\n\n"
        "Do not call `physsearch` on a pure math problem. `web_search` is "
        "only for informal facts, not Lean / Mathlib names.\n\n"
        f"{_LEANSEARCH_TOOL_DESCRIPTION}\n\n"
        f"{_WEB_SEARCH_TOOL_DESCRIPTION}"
    )


def format_user_prompt(informal_statement: str, domain: str = "math") -> str:
    """User turn for SFT / eval / RL: instruction + problem + tool catalog."""
    body = REFORM_USER_INSTRUCTION.format(
        informal_statement=(informal_statement or "").strip()
    ).rstrip()
    return body + "\n\n" + tool_prompt(domain)

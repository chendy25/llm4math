"""SFT collection helpers shared with eval rollout.

Eval imports ``MATH_IDIOM_POLICY``, ``infer_retrieval_domain``, and
``_build_tools`` only. Collection-only imports stay inside
``collect_one_trajectory`` so eval's import graph stays as light as HEAD.
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


def _autoformalize_system_prompt(tools: list, *, domain: str = "math") -> str:
    """Compose the autoformalization system prompt from every enabled tool's
    description. Empty tool list => plain autoformalization prompt (no tools)."""
    try:
        from . import prompts
    except ImportError:
        import prompts
    if not tools:
        return prompts.AUTOFORMALIZATION_SYSTEM
    names = [getattr(t, "TOOL_NAME", "?") for t in tools]
    has_lean = "leansearch" in names
    has_phys = "physsearch" in names
    has_web = "web_search" in names

    lean_contract = (
        "# Lean output contract (required)\n"
        "After tool responses (or if you already have enough information):\n"
        "- You MUST emit a complete ```lean4 ... ``` block in this same turn.\n"
        "- The theorem must end with `:= by sorry`.\n"
        "- Do not stop after a tool call if you already have enough names to write "
        "the statement."
    )
    if has_lean and has_phys:
        policy = (
            "# Joint retrieval routing\n"
            "You have Lean corpus retrieval tools. Choose by problem domain — do not "
            "use the wrong corpus as the primary search.\n"
            "- Physics / Physlib / PhyX problems: you MUST call `physsearch` at "
            "least once first.\n"
            "- Math / Mathlib problems: you MUST call `leansearch` at least once first.\n"
            f"Preferred first tool: **{'physsearch' if domain == 'physics' else 'leansearch'}**.\n"
            f"Enabled tools: {', '.join(names)}.\n"
        )
    else:
        policy = (
            "# Tool Use\n"
            "You have retrieval tools available. Call the enabled Lean corpus tool "
            "at least once with a concise English query BEFORE writing the "
            f"```lean4 block. Enabled tools: {', '.join(names)}.\n"
        )
        if has_phys:
            policy += (
                "For physics / Physlib-related problems, you MUST call `physsearch` "
                "before guessing declaration names.\n"
            )
        if has_lean:
            policy += (
                "For Mathlib problems, you MUST call `leansearch` before guessing "
                "declaration names.\n"
            )
    if has_web:
        policy += (
            "\n# Web search (optional, secondary — math and physics)\n"
            "`web_search` looks up informal facts only (constants, units, textbook "
            "formulas, clarifications). Do NOT use it to find Lean declaration names; "
            "use leansearch / physsearch for that.\n"
        )
    if domain == "physics":
        try:
            from .physics_header import prompt_header_block as _phys_hdr
        except ImportError:
            from physics_header import prompt_header_block as _phys_hdr
        _hdr_block = _phys_hdr()
        policy += (
            "\n# Physlib import convention\n"
            "Physics formalizations MUST begin the ```lean4 block with EXACTLY this "
            "header (verbatim, these lines first, in this order):\n"
            f"```\n{_hdr_block}\n```\n"
            "You MAY append more specific Physlib / PhysLean / QuantumInfo imports "
            "discovered via physsearch AFTER these lines. Do NOT drop the header, "
            "and do NOT start with `import Mathlib` on a physics problem.\n"
            "- Only use Physlib names/namespaces that appeared in `physsearch` "
            "hits, or a numeric literal. Prefer short pure propositions.\n"
        )
    policy += "\n" + MATH_IDIOM_POLICY
    descs = "\n\n".join(t.TOOL_DESCRIPTION for t in tools)
    return prompts.AUTOFORMALIZATION_SYSTEM + "\n\n" + lean_contract + "\n\n" + policy + "\n\n" + descs


def collect_one_trajectory(
    informal_statement: str,
    auto_client,
    critic_client=None,
    model: str = "default",
    critic_model: str = "default",
    max_rounds: int = 3,
    max_gate_retries: int = 3,
    use_lean_verify: bool = False,
    temperature: float = 0.6,
    max_tokens_auto: int = 4096,
    max_tokens_consistency: int = 2048,
    use_leansearch: bool = False,
    leansearch_url: str = leansearch_tool.DEFAULT_URL,
    leansearch_top_k: int = 8,
    max_tool_calls: int = 3,
    use_physsearch: bool = False,
    physsearch_url: str = "http://localhost:8000",
    use_web_search: bool = False,
    image=None,
    image_caption: Optional[str] = None,
    dataset_name: Optional[str] = None,
    category: Optional[str] = None,
    force_domain: Optional[str] = None,
) -> dict:
    """Run the reflective pipeline for one problem.

    Lean compiler failures are a gate: they resample and do not become a
    recorded round. Eval/SFT do not call this function.
    """
    import os
    import re
    try:
        from . import prompts
        from . import llm_client
        from . import verify_lean
        from . import trajectory_builder
        from . import lean_syntax_fix
    except ImportError:
        import prompts
        import llm_client
        import verify_lean
        import trajectory_builder
        import lean_syntax_fix

    if critic_client is None:
        critic_client = auto_client
    domain = (force_domain or "").strip().lower()
    if domain not in ("math", "physics"):
        domain = infer_retrieval_domain(
            informal_statement,
            dataset_name=dataset_name,
            category=category,
        )
    preferred_tool = "physsearch" if domain == "physics" else "leansearch"
    # Offering physsearch on a math item pulls the model toward `import Physlib`.
    physsearch_for_item = use_physsearch and (
        domain == "physics" or os.environ.get("PHYSSEARCH_ON_MATH") == "1"
    )
    tools = _build_tools(
        use_leansearch=use_leansearch,
        use_physsearch=physsearch_for_item,
        leansearch_url=leansearch_url,
        physsearch_url=physsearch_url,
        preferred_tool=preferred_tool if (use_leansearch and physsearch_for_item) else None,
        use_web_search=use_web_search,
    )
    auto_system = _autoformalize_system_prompt(tools, domain=domain)
    if image is not None:
        auto_system += (
            "\n\n# Multimodal input\n"
            "You will also be given an image of the problem. Use it to extract "
            "the physical setup (diagrams, circuits, geometry, field directions, etc.). "
            "The caption is a hint; if it conflicts with the figure, trust the figure."
        )

    rounds = []
    history_critic = ""
    gate_feedback = ""
    current_formal = None
    total_tool_calls = 0
    tools_used: list[str] = []
    last_syntax_ok: Optional[bool] = None
    syntax_rewrites: list[str] = []
    verify_infra_error: Optional[str] = None
    gate_retries = 0

    for _attempt in range(max_rounds + max_gate_retries):
        if len(rounds) >= max_rounds:
            break
        round_no = len(rounds) + 1
        refine_history = "\n\n".join(h for h in (history_critic, gate_feedback) if h)
        if not refine_history:
            user_auto_text = prompts.AUTOFORMALIZATION_USER_INITIAL.format(
                informal_statement=informal_statement
            )
        else:
            user_auto_text = prompts.AUTOFORMALIZATION_USER_REFINEMENT.format(
                informal_statement=informal_statement,
                history=refine_history,
            )
        if tools:
            first = preferred_tool if any(
                getattr(t, "TOOL_NAME", None) == preferred_tool for t in tools
            ) else getattr(tools[0], "TOOL_NAME", "leansearch")
            user_auto_text = (
                user_auto_text
                + f"\n\nIMPORTANT: This is a {domain} problem. Call `{first}` "
                + "<tool_call> FIRST (before any ```lean4 block)."
            )
            if use_web_search:
                user_auto_text += (
                    " `web_search` is optional: use it only if corpus retrieval "
                    "still leaves constants/units/formulas unclear."
                )
        if image is not None and not refine_history:
            messages_auto = [
                {"role": "system", "content": auto_system},
                llm_client.build_vision_user_message(
                    user_auto_text, image=image, image_caption=image_caption,
                ),
            ]
        else:
            messages_auto = [
                {"role": "system", "content": auto_system},
                {"role": "user", "content": user_auto_text},
            ]

        if tools:
            auto_text, lean_block, n_calls = llm_client.autoformalize_with_tools(
                auto_client,
                messages_auto,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens_auto,
                leansearch_url=leansearch_url,
                leansearch_top_k=leansearch_top_k,
                max_tool_calls=max_tool_calls,
                tools=tools,
            )
            for t in tools:
                if t.TOOL_NAME in auto_text:
                    tools_used.append(t.TOOL_NAME)
            total_tool_calls += n_calls
        else:
            auto_text = llm_client.chat_completion(
                auto_client,
                messages_auto,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens_auto,
            )
            lean_block = llm_client.extract_lean4_block(auto_text)

        if not lean_block:
            gate_feedback = (
                "(no valid lean4 block)\n\nWrite a complete ```lean4 theorem "
                "... := by sorry``` block."
            )
            gate_retries += 1
            if gate_retries > max_gate_retries:
                break
            continue

        rewritten, fixes = lean_syntax_fix.rewrite_lean4_statement(lean_block)
        if fixes:
            syntax_rewrites.extend(fixes)
            current_formal = rewritten
        else:
            current_formal = lean_block

        if re.search(r"exists[^\n]*,\s*True\b", current_formal, re.I):
            gate_feedback = (
                "[Vacuous formalization]: conclusion is unconstrained "
                "(`∃ _, True`). Encode the asked quantity with the problem "
                "data (e.g. `v0b = ...` or conservation-law equations)."
                "\n\nCurrent formal statement:\n```lean4\n"
                + current_formal
                + "\n```"
            )
            last_syntax_ok = False
            gate_retries += 1
            if gate_retries > max_gate_retries:
                break
            continue

        if use_lean_verify:
            verify_header = verify_lean.physics_header() if domain == "physics" else None
            passed, err, verify_status = verify_lean.verify_sync_status(
                current_formal,
                header=verify_header,
                canonical_header=os.environ.get("VERIFY_CANONICAL_HEADER") == "1",
            )
            if verify_status == verify_lean.INFRA_ERROR:
                verify_infra_error = err
                last_syntax_ok = None
                break
            last_syntax_ok = passed
            if not passed:
                phys_fix = ""
                if domain == "physics":
                    phys_fix = (
                        " For physics: drop invented identifiers/namespaces that are "
                        "not in prior `physsearch` hits; restate with `import Physlib` "
                        "+ `ℝ`/`Real` binders; bind every free name; keep the asked "
                        "quantity in the conclusion."
                    )
                if use_web_search:
                    phys_fix += (
                        " If informal constants/units/formulas are still unclear "
                        "after corpus retrieval, you may call `web_search` once "
                        "before rewriting the ```lean4 block."
                    )
                gate_feedback = (
                    "[Syntactic verification failed]: " + err
                    + "\n\nCurrent formal statement:\n```lean4\n" + current_formal + "\n```"
                    + "\n\nFix Lean 4.29 parse/type errors. Remember: use `∈` not `in` "
                    + "binders; ASCII ids (`f1` not `f₁`); no `let` inside the theorem type; "
                    + "end with `:= by sorry`."
                    + phys_fix
                )
                gate_retries += 1
                if gate_retries > max_gate_retries:
                    break
                continue

        gate_feedback = ""
        if round_no == 1:
            user_cons = prompts.CONSISTENCY_CHECK_USER_FIRST.format(
                informal_statement=informal_statement,
                formal_statement=current_formal,
            )
        else:
            user_cons = prompts.CONSISTENCY_CHECK_USER_WITH_HISTORY.format(
                informal_statement=informal_statement,
                formal_statement=current_formal,
                history=history_critic,
            )
        messages_cons = [
            {"role": "system", "content": prompts.CONSISTENCY_CHECK_SYSTEM},
            {"role": "user", "content": user_cons},
        ]
        critic_extra = (
            {"enable_thinking": False}
            if os.environ.get("CRITIC_NO_THINKING", "1") == "1"
            else None
        )
        consistency_text = llm_client.chat_completion(
            critic_client,
            messages_cons,
            model=critic_model,
            temperature=0.3,
            max_tokens=max_tokens_consistency,
            extra_body=critic_extra,
            content_only=True,
        )
        is_correct, comments = llm_client.parse_consistency_result(consistency_text)
        history_critic = (
            "Previous formal statement:\n```lean4\n" + current_formal + "\n```\n\n"
            + "Consistency check result: " + ("Correct" if is_correct else "Incorrect") + "\n\n"
            + comments
        )
        rounds.append({
            "autoformalization_text": auto_text,
            "consistency_comments": comments + "\n\nResult: " + ("Correct" if is_correct else "Incorrect"),
            "is_correct": is_correct,
            "syntax_ok": True if use_lean_verify else None,
        })
        if is_correct:
            break

    final_lean = current_formal or ""
    response = trajectory_builder.build_reflective_response(rounds, final_lean)
    user_prompt = prompts.REFORM_USER_INSTRUCTION.format(informal_statement=informal_statement)
    if image is not None:
        user_msg = llm_client.build_vision_user_message(
            user_prompt, image=image, image_caption=image_caption,
        )
    else:
        user_msg = {"role": "user", "content": user_prompt}
    messages = [
        user_msg,
        {"role": "assistant", "content": response},
    ]
    return {
        "rounds": rounds,
        "final_lean_code": final_lean,
        "num_rounds": len(rounds),
        "num_tool_calls": total_tool_calls,
        "tools_used": sorted(set(tools_used)),
        "success": rounds[-1].get("is_correct", False) if rounds else False,
        "syntax_pass": last_syntax_ok,
        "verify_error": verify_infra_error,
        "gate_retries": gate_retries,
        "syntax_rewrites": sorted(set(syntax_rewrites)),
        "messages": messages,
        "informal_statement": informal_statement,
        "retrieval_domain": domain,
    }

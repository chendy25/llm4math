"""Physlib retrieval tool for autoformalization (LeanSearch-v2 /search endpoint
with ``?corpus=physlib``).

Physlib is a community Lean 4 library for physics (ClassicalMechanics,
QuantumMechanics, QuantumInfo, ...). The tool hits the exact same FastAPI
endpoint as `leansearch_tool`; only the URL (defaults to a local
`scripts/serve_physlib.sh` instance) and the ``corpus=physlib`` query
parameter differ. The server routes the request to the Physlib cuVS index.

Wire format is identical to `leansearch_tool`:

    <tool_call>
    {"name": "physsearch", "arguments": {"query": "...", "num_results": 8}}
    </tool_call>

Returns results wrapped in `<tool_response>...</tool_response>` exactly like
`leansearch`. From the model's perspective the two tools are siblings; the
only way it can tell them apart is the tool name and the description
below.
"""

from __future__ import annotations

import json
from typing import Any

try:
    from .retrieval_tool import (
        format_results as _shared_format_results,
        parse_tool_call,
        post_json,
    )
except ImportError:  # running as a flat script under data/train/
    from retrieval_tool import (
        format_results as _shared_format_results,
        parse_tool_call,
        post_json,
    )


TOOL_NAME = "physsearch"

# Default URL points at the local multi-corpus server that `serve_physlib.sh`
# starts. Override via ``--physsearch-url`` in run_collect_sft_data.py when
# pointing at a remote self-hosted Physlib instance (e.g. a GPU box you run
# `serve_physlib.sh` on); the public ``https://leansearch.net`` only serves
# mathlib, so it cannot be used as a physsearch backend.
DEFAULT_URL = "http://localhost:8000"

# The Physlib cuVS index always lives behind `?corpus=physlib` regardless of
# which physical LeanSearch-v2 server URL we hit. Keeping this as a constant
# here means the rest of the code doesn't need to know about LeanSearch-v2's
# multi-corpus routing.
DEFAULT_CORPUS = "physlib"

# KB-growth: verified declarations harvested from successful physics
# trajectories augment `physsearch`, so each collection round can retrieve
# concepts formalized (and Kimina-verified) in the previous round. Two backends
# coexist, selected by env `PHYSSEARCH_KB_MODE`:
#
#   local  (DEFAULT) -- Option B: a local BM25 index over physlib_kb_store.jsonl
#                       (physlib_kb_local.py). No GPU, no server restart; picks
#                       up harvest appends live via mtime hot-reload.
#   remote           -- Option A: a second LeanSearch-v2 corpus `physlib_kb`
#                       (needs GPU reindex + server restart to update).
#   both             -- query local AND remote KB (dedup).
#   off              -- KB augmentation disabled (plain `physlib` only).
import os as _os

KB_MODE = (_os.environ.get("PHYSSEARCH_KB_MODE") or "local").strip().lower()
# How many top KB (Option B/A) hits to reserve at the front of the result list.
try:
    KB_QUOTA = int(_os.environ.get("PHYSSEARCH_KB_QUOTA", "2"))
except ValueError:
    KB_QUOTA = 2

# Also query Mathlib on the same LeanSearch-v2 server (dual-corpus). Distances
# are NOT comparable across corpora, so Mathlib hits are quota-merged (not
# globally re-ranked with physlib). Set PHYSSEARCH_INCLUDE_MATHLIB=0 to disable.
INCLUDE_MATHLIB = (_os.environ.get("PHYSSEARCH_INCLUDE_MATHLIB") or "1").strip().lower() not in (
    "0", "false", "no", "off",
)
try:
    MATHLIB_QUOTA = int(_os.environ.get("PHYSSEARCH_MATHLIB_QUOTA", "3"))
except ValueError:
    MATHLIB_QUOTA = 3
MATHLIB_CORPUS = "mathlib"

# Option A remote KB corpora (only used when KB_MODE in {remote, both}).
_env_extra = _os.environ.get("PHYSSEARCH_EXTRA_CORPORA")
if _env_extra is None:
    EXTRA_CORPORA: list[str] = ["physlib_kb"]
else:
    EXTRA_CORPORA = [c.strip() for c in _env_extra.split(",") if c.strip()]

# Corpora that returned an error (e.g. 404 not-yet-built) are remembered so we
# don't hammer the server with the same failing request every tool call.
_UNAVAILABLE_CORPORA: set[str] = set()

# Description injected into the autoformalization system prompt. Mirrors
# `leansearch_tool.TOOL_DESCRIPTION` in shape so the model emits parseable
# tool calls, with a Physlib-flavoured preamble.
TOOL_DESCRIPTION = """You can use the `physsearch` tool to retrieve relevant Lean 4 declarations for physics formalization. One call searches THREE sources together:
1. Physlib / PhysLean / QuantumInfo (primary physics corpus)
2. A local grown KB of previously verified physics decls (when available)
3. Mathlib (pure-math lemmas often needed by physics: Real, analysis, algebra, topology, ...)

Use it to find EXACT declaration names and signatures before committing to a formalization — do not guess deep namespaces like `PhysLean.ClassicalMechanics` or Mathlib lemma names.

Prefer `physsearch` for:
- Classical / quantum mechanics, electromagnetism, thermodynamics, relativity
- AND the Mathlib lemmas those formalizations need (e.g. `Real` algebra, continuity, inequalities)

To call the tool, output exactly one tool call block and then STOP and wait for the result:
<tool_call>
{"name": "physsearch", "arguments": {"query": "your search query here", "num_results": 8}}
</tool_call>

The retrieved declarations will be returned to you wrapped in:
<tool_response>
...
</tool_response>

Guidelines:
- Query in concise English (physics concept OR the Mathlib lemma you need).
- `num_results` is optional (default 8, max 20). Results are tagged by corpus.
- You may call the tool multiple times with different queries (physics vs pure-math).
- After you have enough information, write your final Lean 4 statement in a ```lean4 code block as instructed. Physics statements MUST begin with the Physlib import header; Mathlib names from tool hits may still be used in the statement body."""

def _search_one(
    query: str,
    num_results: int,
    url: str,
    rerank: bool,
    timeout: float,
    corpus: str,
) -> list[dict[str, Any]]:
    """Query a single corpus. Returns [] on any error (and, for augmentation
    corpora, records them as unavailable so we stop retrying)."""
    endpoint = url.rstrip("/") + "/search"
    body = {"query": [query], "num_results": num_results, "rerank": rerank}
    import urllib.parse
    qs = urllib.parse.urlencode({"corpus": corpus})
    try:
        data = post_json(f"{endpoint}?{qs}", body, timeout=timeout)
    except Exception:
        if corpus != DEFAULT_CORPUS:
            _UNAVAILABLE_CORPORA.add(corpus)
        return []
    if not isinstance(data, list) or not data:
        return []
    batch = data[0] if isinstance(data[0], list) else data
    out: list[dict[str, Any]] = []
    for item in batch:
        if not isinstance(item, dict):
            continue
        res = item.get("result", item)
        if isinstance(res, dict):
            merged = dict(res)
            if "distance" in item:
                merged["distance"] = item["distance"]
            merged.setdefault("corpus", corpus)
            out.append(merged)
    return out


def _dotted_name(item: dict[str, Any]) -> str:
    name = item.get("name")
    if isinstance(name, list):
        return ".".join(str(x) for x in name)
    return str(name or "")


def _search_remote(
    query: str, num_results: int, url: str, rerank: bool, timeout: float,
    corpus: str, extra_corpora: list[str] | None,
) -> list[dict[str, Any]]:
    """Query `corpus` (+ optional remote KB corpora for Option A) and merge by
    ascending distance (closer first), deduped by dotted name."""
    corpora = [corpus]
    if extra_corpora:
        for c in extra_corpora:
            if c and c != corpus and c not in corpora and c not in _UNAVAILABLE_CORPORA:
                corpora.append(c)
    merged: list[dict[str, Any]] = []
    for c in corpora:
        try:
            merged.extend(_search_one(query, num_results, url, rerank, timeout, c))
        except Exception:
            if c != DEFAULT_CORPUS:
                _UNAVAILABLE_CORPORA.add(c)
    if not merged:
        return []
    best: dict[str, dict[str, Any]] = {}
    for item in merged:
        key = _dotted_name(item) or id(item)
        prev = best.get(key)
        if prev is None or item.get("distance", 1e9) < prev.get("distance", 1e9):
            best[key] = item
    return sorted(best.values(), key=lambda r: r.get("distance", 1e9))


def _local_kb_search(query: str, num_results: int) -> list[dict[str, Any]]:
    """Option B: local BM25 KB retrieval. Never raises; [] if unavailable."""
    try:
        try:
            from . import physlib_kb_local
        except ImportError:
            import physlib_kb_local
        return physlib_kb_local.search(query, num_results=num_results)
    except Exception:
        return []


def search(
    query: str,
    num_results: int = 8,
    url: str = DEFAULT_URL,
    rerank: bool = True,
    timeout: float = 60.0,
    corpus: str = DEFAULT_CORPUS,
    extra_corpora: list[str] | None = None,
    include_mathlib: bool | None = None,
) -> list[dict[str, Any]]:
    """Retrieve decls for a physics query from physlib + local KB + Mathlib.

    Merge order (distances are NOT cross-corpus comparable, so we use quotas):
      1. Local KB (Option B) — up to ``KB_QUOTA`` (front)
      2. Remote ``physlib`` (+ optional remote ``physlib_kb``) — fills the middle
      3. Remote ``mathlib`` — up to ``MATHLIB_QUOTA`` (tail), when enabled

    Default: ``PHYSSEARCH_INCLUDE_MATHLIB=1``. Set to ``0`` to disable Mathlib.
    Dedup by dotted name; truncated to ``num_results``. Degrades silently if a
    backend is unavailable."""
    if not query or not query.strip():
        return []
    num_results = max(1, min(int(num_results), 20))

    use_local = KB_MODE in ("local", "both")
    use_remote_kb = KB_MODE in ("remote", "both")
    use_mathlib = INCLUDE_MATHLIB if include_mathlib is None else include_mathlib
    remote_extra = (
        (extra_corpora if extra_corpora is not None else EXTRA_CORPORA)
        if use_remote_kb else None
    )

    phys_hits = _search_remote(
        query, num_results, url, rerank, timeout, corpus, remote_extra
    )
    local_hits = _local_kb_search(query, KB_QUOTA) if use_local else []
    math_hits: list[dict[str, Any]] = []
    if use_mathlib and MATHLIB_CORPUS not in _UNAVAILABLE_CORPORA:
        try:
            math_hits = _search_one(
                query, max(MATHLIB_QUOTA, num_results), url, rerank, timeout,
                MATHLIB_CORPUS,
            )
        except Exception:
            _UNAVAILABLE_CORPORA.add(MATHLIB_CORPUS)
            math_hits = []

    # Quota merge: KB → physlib → mathlib.
    out: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _take(items: list[dict[str, Any]], limit: int | None = None) -> None:
        n = 0
        for item in items:
            if limit is not None and n >= limit:
                return
            if len(out) >= num_results:
                return
            key = _dotted_name(item)
            if key and key in seen:
                continue
            if key:
                seen.add(key)
            out.append(item)
            n += 1

    _take(local_hits, max(0, KB_QUOTA))
    # Reserve slots for Mathlib when enabled so physlib cannot crowd it out.
    math_reserve = min(max(0, MATHLIB_QUOTA), max(0, num_results - len(out))) if use_mathlib else 0
    phys_budget = max(0, num_results - len(out) - math_reserve)
    _take(phys_hits, phys_budget)
    if use_mathlib:
        _take(math_hits, max(0, MATHLIB_QUOTA))
    # If Mathlib under-filled, backfill with remaining physlib hits.
    if len(out) < num_results:
        _take(phys_hits, None)
    return out


def format_results(results: list[dict[str, Any]], max_chars_per_item: int = 600) -> str:
    """Render search results; tag each hit with its corpus (physlib/mathlib/kb)."""
    if not results:
        return "(No results found.)"
    # Prefer shared formatter, then annotate corpus on the header line.
    base = _shared_format_results(results, max_chars_per_item=max_chars_per_item)
    # Rebuild with corpus tags so the model can tell Mathlib vs Physlib hits.
    lines: list[str] = []
    for i, r in enumerate(results, 1):
        name = r.get("name")
        if isinstance(name, list):
            name = ".".join(str(x) for x in name)
        else:
            name = str(name or "")
        kind = r.get("kind") or ""
        corpus = r.get("corpus") or ""
        header = f"[{i}] {name}"
        if kind:
            header += f"  ({kind})"
        if corpus:
            header += f"  [{corpus}]"
        lines.append(header)
        signature = (r.get("signature") or r.get("type") or "").strip()
        if signature:
            lines.append(f"    {signature[:max_chars_per_item]}")
        desc = (r.get("informal_description") or r.get("docstring") or "").strip()
        if desc:
            desc = " ".join(desc.split())
            if len(desc) > max_chars_per_item:
                desc = desc[:max_chars_per_item].rstrip() + "..."
            lines.append(f"    {desc}")
    return "\n".join(lines) if lines else base


def run_tool_call(
    call: dict[str, Any],
    url: str | None = None,
    default_num_results: int = 8,
    timeout: float = 60.0,
    corpus: str | None = None,
) -> str:
    """Dispatch a parsed `physsearch` tool call and return a formatted string.

    Foreign tool names (anything other than `physsearch`) get a clear
    "(Unknown tool: ...)" reply so the model can correct itself rather than
    silently dropping the call.

    ``url`` / ``corpus`` resolve to module-level ``DEFAULT_URL`` /
    ``DEFAULT_CORPUS`` at call time so ``--physsearch-url`` patching works.
    """
    name = call.get("name") or call.get("tool")
    args = call.get("arguments") or call.get("args") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            args = {"query": args}
    if name not in (None, TOOL_NAME):
        return f"(Unknown tool: {name})"
    query = args.get("query") or args.get("q") or ""
    num_results = args.get("num_results") or args.get("top_k") or default_num_results
    results = search(
        query,
        num_results=num_results,
        url=DEFAULT_URL if url is None else url,
        timeout=timeout,
        corpus=DEFAULT_CORPUS if corpus is None else corpus,
    )
    return format_results(results)
"""LeanSearch-v2 retrieval tool for autoformalization (standard /search endpoint).

Standalone and dependency-light (stdlib urllib by default; uses `requests` if
available). It deliberately does NOT import the `leansearchv2` package, so this
module can be reused as-is by the SFT data collection pipeline AND by a future
RL rollout (e.g. verl) without pulling in GPU/torch dependencies.

The tool is one of several `RetrievalTool` implementations (`physsearch_tool`
serves Physlib, etc.); shared wire-format helpers live in
`retrieval_tool.py` so every tool exposes the same `<tool_call>` /
`<tool_response>` protocol.

Protocol for inline tool calls inside a single assistant message:

    <tool_call>
    {"name": "leansearch", "arguments": {"query": "...", "num_results": 8}}
    </tool_call>

The harness executes the call and appends:

    <tool_response>
    [1] <name> : <signature>
        <informal description>
    ...
    </tool_response>
"""

from __future__ import annotations

import json
from typing import Any, Optional

try:
    from .retrieval_tool import (
        TOOL_CALL_CLOSE,
        TOOL_CALL_OPEN,
        TOOL_RESPONSE_CLOSE,
        TOOL_RESPONSE_OPEN,
        format_results as _shared_format_results,
        parse_tool_call,
        post_json,
    )
except ImportError:  # running as a flat script under data/train/
    from retrieval_tool import (
        TOOL_CALL_CLOSE,
        TOOL_CALL_OPEN,
        TOOL_RESPONSE_CLOSE,
        TOOL_RESPONSE_OPEN,
        format_results as _shared_format_results,
        parse_tool_call,
        post_json,
    )


TOOL_NAME = "leansearch"
DEFAULT_URL = "https://leansearch.net"
# Used when talking to a multi-corpus LeanSearch-v2 server (`?corpus=`).
# Left as None for the public leansearch.net endpoint (single-corpus wire format).
DEFAULT_CORPUS: str | None = None

# Tags used to delimit inline tool calls / responses in the trajectory.
# Kept as aliases so any external importer still works.
TOOL_CALL_OPEN = TOOL_CALL_OPEN
TOOL_CALL_CLOSE = TOOL_CALL_CLOSE
TOOL_RESPONSE_OPEN = TOOL_RESPONSE_OPEN
TOOL_RESPONSE_CLOSE = TOOL_RESPONSE_CLOSE

# Description injected into the autoformalization system prompt. Keep the JSON
# schema exact so the model emits parseable tool calls.
TOOL_DESCRIPTION = """You can use the `leansearch` tool to retrieve relevant Lean 4 / Mathlib declarations (definitions, theorems, lemmas) by a natural-language or mathematical query. Use it to find the EXACT Mathlib names, signatures, and notation before committing to a formalization, instead of guessing names.

To call the tool, output exactly one tool call block and then STOP and wait for the result:
<tool_call>
{"name": "leansearch", "arguments": {"query": "your search query here", "num_results": 8}}
</tool_call>

The retrieved declarations will be returned to you wrapped in:
<tool_response>
...
</tool_response>

Guidelines:
- Query in concise English describing the concept/lemma you need (e.g. "order of a group element divides the order of the group").
- `num_results` is optional (default 8, max 20).
- You may call the tool multiple times, but keep it to at most a few calls.
- After you have enough information, write your final Lean 4 statement in a ```lean4 code block as instructed."""


def search(
    query: str,
    num_results: int = 8,
    url: str = DEFAULT_URL,
    rerank: bool = True,
    timeout: float = 60.0,
    corpus: str | None = None,
) -> list[dict[str, Any]]:
    """Call the standard-mode /search endpoint. Returns a list of result dicts
    (each is the `result` payload merged with its `distance`). Returns [] on error.

    ``corpus`` selects which LeanSearch-v2 corpus to query via the
    ``?corpus=`` query parameter. ``None`` (default) keeps the legacy
    single-corpus wire format — the server falls back to its default corpus
    (mathlib)."""
    if not query or not query.strip():
        return []
    num_results = max(1, min(int(num_results), 20))
    endpoint = url.rstrip("/") + "/search"
    body = {"query": [query], "num_results": num_results, "rerank": rerank}
    params: dict[str, Any] = {}
    if corpus:
        params["corpus"] = corpus
    try:
        if params:
            import urllib.request
            import urllib.parse
            qs = urllib.parse.urlencode(params)
            data = post_json(f"{endpoint}?{qs}", body, timeout=timeout)
        else:
            data = post_json(endpoint, body, timeout=timeout)
    except Exception:
        return []
    # Response shape: list (per query) of list (per result) of {result, distance}.
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
            out.append(merged)
    return out


def format_results(results: list[dict[str, Any]], max_chars_per_item: int = 600) -> str:
    """Render search results as a compact numbered text block for the model."""
    return _shared_format_results(results, max_chars_per_item=max_chars_per_item)


def run_tool_call(
    call: dict[str, Any],
    url: str | None = None,
    default_num_results: int = 8,
    timeout: float = 60.0,
    corpus: str | None = None,
) -> str:
    """Dispatch a parsed tool call to the retriever and return a formatted string.

    ``url`` / ``corpus`` default to the *current* module-level ``DEFAULT_URL`` /
    ``DEFAULT_CORPUS`` (resolved at call time) so the SFT collector can patch
    them for a local multi-corpus server without rebuilding the function.
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

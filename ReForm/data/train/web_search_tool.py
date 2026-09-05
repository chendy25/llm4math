"""Web search tool for ReForm autoformalization (Serper Google Search API).

Use for informal facts (constants, units, problem context). Do NOT use for
Lean / Mathlib / Physlib declaration names — those stay on leansearch /
physsearch.

Wire format matches leansearch / physsearch:

    <tool_call>
    {"name": "web_search", "arguments": {"query": "...", "num_results": 5}}
    </tool_call>
"""

from __future__ import annotations

import json
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

TOOL_NAME = "web_search"

# Public Serper by default (login / ACP nodes cannot reach sandbox intranet).
DEFAULT_BASE_URL = "https://google.serper.dev"
DEFAULT_NUM_RESULTS = 5

TOOL_DESCRIPTION = """Optional tool for math or physics: look up informal web facts (constants, unit conversions, standard textbook formulas, clarifying wording/diagram captions). It returns web snippets — NOT Lean declarations.

Do NOT use `web_search` to find Mathlib / Physlib theorem names; use `leansearch` or `physsearch` for that. Skip `web_search` when corpus retrieval already gives you enough to formalize.

To call the tool, output exactly one tool call block and then STOP and wait for the result:
<tool_call>
{"name": "web_search", "arguments": {"query": "your search query here", "num_results": 5}}
</tool_call>

Results are returned wrapped in:
<tool_response>
...
</tool_response>

Guidelines:
- Prefer short factual queries (e.g. "speed of sound in air at 20C", "thin lens equation two lenses").
- `num_results` is optional (default 5, max 10).
- After corpus retrieval (`leansearch` / `physsearch`) you may call `web_search` if constants or informal context are still missing.
- Web snippets are hints only; the final ```lean4 statement must still be a valid Lean 4 theorem ending with `:= by sorry`."""


def _api_key() -> str:
    return (os.environ.get("SERPER_API_KEY") or "").strip()


def _base_url() -> str:
    return (os.environ.get("SERPER_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")


def search(
    query: str,
    num_results: int = DEFAULT_NUM_RESULTS,
    timeout: float = 30.0,
) -> list[dict[str, Any]]:
    """Call Serper `/search`. Returns a list of organic hit dicts (may be empty)."""
    if not query or not query.strip():
        return []
    key = _api_key()
    if not key:
        return [{"title": "(web_search error)", "snippet": "SERPER_API_KEY is not set", "link": ""}]

    num_results = max(1, min(int(num_results), 10))
    url = _base_url() + "/search"
    body = json.dumps({"q": query.strip(), "num": num_results}).encode("utf-8")
    req = Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "X-API-KEY": key,
        },
        method="POST",
    )
    try:
        with urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
    except HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:300] if e.fp else str(e)
        return [{"title": "(web_search HTTP error)", "snippet": detail, "link": ""}]
    except (URLError, TimeoutError, json.JSONDecodeError) as e:
        return [{"title": "(web_search error)", "snippet": str(e)[:300], "link": ""}]

    organic = data.get("organic") if isinstance(data, dict) else None
    if not isinstance(organic, list):
        return [{"title": "(web_search empty)", "snippet": str(data)[:300], "link": ""}]
    out: list[dict[str, Any]] = []
    for item in organic[:num_results]:
        if not isinstance(item, dict):
            continue
        out.append(
            {
                "title": item.get("title") or "",
                "snippet": item.get("snippet") or item.get("description") or "",
                "link": item.get("link") or item.get("url") or "",
            }
        )
    return out


def format_results(results: list[dict[str, Any]], max_chars_per_item: int = 400) -> str:
    if not results:
        return "(no web search results)"
    lines: list[str] = []
    for i, r in enumerate(results, 1):
        title = (r.get("title") or "").strip()
        snippet = (r.get("snippet") or "").strip()
        link = (r.get("link") or "").strip()
        block = f"[{i}] {title}"
        if snippet:
            if len(snippet) > max_chars_per_item:
                snippet = snippet[: max_chars_per_item - 3] + "..."
            block += f"\n    {snippet}"
        if link:
            block += f"\n    {link}"
        lines.append(block)
    return "\n".join(lines)


def run_tool_call(
    call: dict[str, Any],
    default_num_results: int = DEFAULT_NUM_RESULTS,
    timeout: float = 30.0,
    **_kwargs: Any,
) -> str:
    """Dispatch a parsed `web_search` tool call; return formatted snippets."""
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
    return format_results(search(query, num_results=num_results, timeout=timeout))

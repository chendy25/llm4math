"""Shared <tool_call> / <tool_response> wire format for ReForm tools."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"
TOOL_RESPONSE_OPEN = "<tool_response>"
TOOL_RESPONSE_CLOSE = "</tool_response>"


def _json_object_end(text: str, start: int) -> int:
    open_at = text.find("{", start)
    if open_at < 0:
        return -1
    depth = 0
    in_str = False
    escaped = False
    for i in range(open_at, len(text)):
        c = text[i]
        if in_str:
            if escaped:
                escaped = False
            elif c == "\\":
                escaped = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i + 1
    return -1


def parse_tool_call(text: str) -> dict[str, Any] | None:
    if not text or TOOL_CALL_OPEN not in text:
        return None
    start = text.rfind(TOOL_CALL_OPEN)
    chunk = text[start + len(TOOL_CALL_OPEN) :]
    end_tag = chunk.find(TOOL_CALL_CLOSE)
    if end_tag >= 0:
        chunk = chunk[:end_tag]
    obj_end = _json_object_end(chunk, 0)
    if obj_end < 0:
        return None
    try:
        obj = json.loads(chunk[:obj_end])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def format_results(results: list[dict[str, Any]], max_chars_per_item: int = 600) -> str:
    if not results:
        return "(No results found.)"
    lines = []
    for i, item in enumerate(results, 1):
        name = item.get("name") or item.get("formal_name") or item.get("title") or ""
        kind = item.get("kind") or item.get("type") or ""
        sig = item.get("signature") or item.get("type_signature") or ""
        desc = item.get("informal_description") or item.get("doc") or item.get("description") or ""
        corpus = item.get("corpus") or ""
        head = f"[{i}] {name}".rstrip()
        if kind:
            head += f"  ({kind})"
        if corpus:
            head += f"  [{corpus}]"
        block = head
        if sig:
            block += f"\n    {sig}"
        if desc:
            d = str(desc).replace("\n", " ").strip()
            if len(d) > max_chars_per_item:
                d = d[: max_chars_per_item - 3] + "..."
            block += f"\n    {d}"
        lines.append(block)
    return "\n".join(lines)


def post_json(url: str, body: dict[str, Any], timeout: float = 60.0) -> Any:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except (urllib.error.URLError, TimeoutError, OSError):
        raise
    return json.loads(raw.decode("utf-8") or "null")

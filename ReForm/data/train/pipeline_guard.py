"""Fatal-stop helpers for SFT collection (DashScope arrears / search / verify)."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

ARREARS_MARKERS = (
    "Arrearage",
    "Access denied",
    "overdue-payment",
    "account is in good standing",
    "Insufficient Balance",
    "InsufficientBalance",
)


class PipelineHalt(Exception):
    """Stop the collect worker immediately; do not continue remaining items."""


def is_arrears_text(text: str) -> bool:
    if not text:
        return False
    low = text.lower()
    return any(m.lower() in low for m in ARREARS_MARKERS)


SOFT_DASHSCOPE_MARKERS = (
    "timed out",
    "timeout",
    "temporarily",
    "connection reset",
    "broken pipe",
    "network is unreachable",
    "connection aborted",
    "remote end closed",
    "http=429",
    "rate limit",
)


def is_soft_dashscope_text(text: str) -> bool:
    """Transient DashScope probe failures that must not stop the pipeline."""
    if not text:
        return False
    low = text.lower()
    return any(m in low for m in SOFT_DASHSCOPE_MARKERS)


def raise_if_arrears(exc: BaseException) -> None:
    msg = str(exc)
    if is_arrears_text(msg):
        raise PipelineHalt(f"DashScope arrears/access-denied: {msg}") from exc


def check_search(url: str, timeout: float = 45.0) -> tuple[bool, str]:
    base = (url or "").rstrip("/")
    if not base:
        return False, "search url empty"
    try:
        with urllib.request.urlopen(base + "/health", timeout=min(timeout, 10.0)) as resp:
            health = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        return False, f"search /health failed: {e}"
    if not isinstance(health, dict) or health.get("status") != "ok":
        return False, f"search /health not ok: {health}"
    body = json.dumps(
        {"query": ["group order divides the group order"], "num_results": 1, "rerank": True}
    ).encode("utf-8")
    req = urllib.request.Request(
        base + "/search?corpus=mathlib",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        return False, f"search /search failed: {e}"
    hits = data[0] if isinstance(data, list) and data else []
    if not hits:
        return False, "search /search returned 0 hits"
    return True, "ok"


def check_verify(url: str, timeout: float = 8.0) -> tuple[bool, str]:
    base = (url or "").rstrip("/")
    if not base:
        return False, "verify url empty"
    try:
        with urllib.request.urlopen(base + "/", timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            code = resp.status
    except Exception as e:
        return False, f"verify failed: {e}"
    if code != 200:
        return False, f"verify http={code} body={raw[:180]}"
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return True, "ok"
    if isinstance(data, dict) and data.get("status") not in (None, "ok"):
        return False, f"verify not ok: {data}"
    return True, "ok"


def check_llm(
    base_url: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    timeout: float | None = None,
) -> tuple[bool, str]:
    """Probe a local OpenAI-compatible /v1 endpoint (models + 1-token chat)."""
    if timeout is None:
        timeout = float(os.environ.get("LLM_PROBE_TIMEOUT", "90"))
    base = (
        base_url
        or os.environ.get("LLM_BASE_URL")
        or os.environ.get("TOKENHUB_BASE_URL")
        or ""
    ).rstrip("/")
    if not base:
        return False, "llm base url empty"
    model = model or os.environ.get("AUTO_MODEL") or os.environ.get("LLM_MODEL") or "Qwen3.8-27B"
    key = api_key or os.environ.get("TOKENHUB_API_KEY") or os.environ.get("LLM_API_KEY") or "EMPTY"
    try:
        req = urllib.request.Request(
            base + "/models",
            headers={"Authorization": f"Bearer {key}"},
        )
        with urllib.request.urlopen(req, timeout=min(timeout, 15.0)) as resp:
            listing = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        msg = str(e)
        if is_soft_dashscope_text(msg):
            return False, f"llm soft timeout: {e}"
        return False, f"llm /models failed: {e}"
    ids = [m.get("id") for m in (listing.get("data") or []) if isinstance(m, dict)]
    if ids and model not in ids:
        return False, f"llm model {model!r} not in {ids[:8]}"
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 1,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        base + "/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "ignore") if e.fp else str(e)
        if is_soft_dashscope_text(raw) or e.code == 429:
            return False, f"llm soft http={e.code}: {raw[:240] or e}"
        return False, f"llm http={e.code}: {raw[:240] or e}"
    except Exception as e:
        msg = str(e)
        if is_soft_dashscope_text(msg):
            return False, f"llm soft timeout: {e}"
        return False, f"llm failed: {e}"
    if isinstance(data, dict) and data.get("error"):
        msg = json.dumps(data["error"], ensure_ascii=False)
        if is_soft_dashscope_text(msg):
            return False, f"llm soft error: {msg}"
        return False, f"llm error: {msg}"
    return True, f"ok chat model={model}"


def check_dashscope(api_key: str | None = None, timeout: float | None = None) -> tuple[bool, str]:
    if timeout is None:
        timeout = float(os.environ.get("DASHSCOPE_PROBE_TIMEOUT", "90"))
    key = api_key or os.environ.get("TOKENHUB_API_KEY") or os.environ.get("DASHSCOPE_API_KEY") or ""
    if not key:
        return False, "missing DashScope API key"
    model = os.environ.get("AUTO_MODEL") or os.environ.get("LLM_MODEL") or "qwen3.5-plus"
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 1,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "ignore") if e.fp else str(e)
        if is_arrears_text(raw) or is_arrears_text(str(e)):
            return False, f"dashscope arrears: {raw[:240] or e}"
        if e.code == 429 or is_soft_dashscope_text(raw):
            return False, f"dashscope soft http={e.code}: {raw[:240] or e}"
        if 400 <= int(e.code) < 500:
            return False, f"dashscope http={e.code}: {raw[:240] or e}"
        return False, f"dashscope soft http={e.code}: {raw[:240] or e}"
    except Exception as e:
        msg = str(e)
        if is_arrears_text(msg):
            return False, f"dashscope arrears: {msg}"
        if is_soft_dashscope_text(msg):
            return False, f"dashscope soft timeout: {e}"
        return False, f"dashscope failed: {e}"
    if isinstance(data, dict) and data.get("error"):
        msg = json.dumps(data["error"], ensure_ascii=False)
        if is_arrears_text(msg):
            return False, f"dashscope arrears: {msg}"
        return False, f"dashscope error: {msg}"
    return True, f"ok chat model={model}"

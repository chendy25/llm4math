"""Kimina Lean 4 verifier used as a compile gate (eval / collect / harvest).

Compiler errors must never enter the model's generation history. Eval scores
``syntax_pass`` separately from the critic; ``INFRA_ERROR`` is a service
failure, not a model failure.
"""

from __future__ import annotations

import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Optional, Tuple

_REFORM_ROOT = Path(__file__).resolve().parents[2]
if str(_REFORM_ROOT) not in sys.path:
    sys.path.insert(0, str(_REFORM_ROOT))

# Kimina keys its REPL cache on the *import* lines only, so every distinct
# import set costs a cold `import Mathlib` load. Keep this to the single
# canonical import. `set_option`/`open` land in the body and are cheap.
DEFAULT_HEADER = (
    "import Mathlib\n"
    "set_option maxHeartbeats 0\n"
    "open BigOperators Real Nat Topology Rat\n\n"
)

try:
    from .physics_header import physics_header as _physics_header
except ImportError:
    from physics_header import physics_header as _physics_header


def physics_header() -> str:
    """Physlib verify/compile header (single source: physics_header.py)."""
    return _physics_header()


PHYSLIB_HEADER = _physics_header()

OK = "ok"
SYNTAX_ERROR = "syntax_error"
INFRA_ERROR = "infra_error"
SKIPPED = "skipped"

_verify_sema: Optional[threading.Semaphore] = None
_sema_lock = threading.Lock()

_lean_client = None
_leanclient_available = False
try:
    from leanclient.client import Lean4Client
    _leanclient_available = True
except ImportError:
    Lean4Client = None


def _get_client():
    """Build/reuse Lean4Client from current VERIFY_URL / VERIFY_KEY env."""
    global _lean_client
    verify_url = os.environ.get("VERIFY_URL")
    verify_key = os.environ.get("VERIFY_KEY")
    if not verify_url or not verify_key:
        return None
    if not _leanclient_available:
        return None
    if _lean_client is not None:
        return _lean_client
    try:
        _lean_client = Lean4Client(base_url=verify_url, api_key=verify_key)
        return _lean_client
    except Exception:
        return None


def _verify_limit() -> threading.Semaphore:
    """Cap in-process Lean verify fan-out so many workers do not 429."""
    global _verify_sema
    with _sema_lock:
        if _verify_sema is None:
            n = max(1, int(os.environ.get("VERIFY_MAX_CONCURRENCY", "8")))
            _verify_sema = threading.Semaphore(n)
        return _verify_sema


def _fmt_exc(e: BaseException) -> str:
    """Timeout exceptions stringify to '', which used to be indistinguishable
    from a Lean compile failure with no message. Always keep the class name."""
    msg = str(e).strip()
    name = type(e).__name__
    return f"{name}: {msg}" if msg else name


_TRANSIENT_MARKERS = (
    "429",
    "too many requests",
    "502",
    "503",
    "504",
    "timeout",
    "timed out",
    "temporar",
    "connection reset",
    "connecterror",
    "connectionerror",
    "remoteprotocolerror",
    "readerror",
    "writeerror",
    "no result",
)


def _transient_verify_error(err: str) -> bool:
    e = (err or "").lower()
    if not e:
        return True
    return any(s in e for s in _TRANSIENT_MARKERS)


def process_formal_statement(formal_statement: str) -> str:
    """Strip markdown fences so Kimina sees raw Lean."""
    statement = (formal_statement or "").strip()
    if not statement:
        return ""
    m = re.search(r"```(?:lean4?)?\s*(.*?)\s*```", statement, re.DOTALL)
    if m:
        return m.group(1).strip()
    if statement.startswith("```"):
        lines = statement.split("\n")
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        statement = "\n".join(lines).strip()
    return statement


def strip_imports(statement: str) -> str:
    """Drop leading import lines so a canonical header can be substituted."""
    lines = statement.splitlines()
    i = 0
    while i < len(lines) and (
        not lines[i].strip() or lines[i].strip().startswith("import ")
    ):
        i += 1
    return "\n".join(lines[i:])


def verify_sync_status(
    formal_statement: str,
    header: Optional[str] = None,
    canonical_header: bool = False,
) -> Tuple[bool, str, str]:
    """Verify a Lean 4 statement. Returns (passed, error_message, status).

    ``status`` separates a real compiler rejection (SYNTAX_ERROR) from a Lean
    service/transport failure (INFRA_ERROR); callers must not charge the latter
    to the model.

    ``canonical_header`` replaces whatever imports the model emitted with
    ``header``. Kimina caches one REPL per import set and a cold Mathlib load
    costs minutes, so a stray `import Physlib` on a math item would evict the
    warm REPL and stall the whole run.
    """
    client = _get_client()
    if client is None:
        return True, "", SKIPPED

    header_content = header or DEFAULT_HEADER
    content = process_formal_statement(formal_statement)
    if not content:
        return False, "Empty statement", SYNTAX_ERROR
    if canonical_header:
        check_content = header_content + strip_imports(content)
    elif "import" in content:
        check_content = content
    else:
        check_content = header_content + content

    retries = max(1, int(os.environ.get("VERIFY_RETRIES", "8")))
    call_timeout = max(30, int(os.environ.get("VERIFY_TIMEOUT", "240")))
    last_err = ""
    for attempt in range(retries):
        try:
            import asyncio
            import concurrent.futures

            def _run():
                return asyncio.run(
                    _verify_async_impl(client, check_content, call_timeout)
                )

            with _verify_limit():
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    fut = pool.submit(_run)
                    result = fut.result(timeout=call_timeout + 120)
        except Exception as e:
            last_err = _fmt_exc(e)
            if _transient_verify_error(last_err) and attempt + 1 < retries:
                time.sleep(min(30.0, 1.5 * (2 ** attempt)))
                continue
            return False, last_err, INFRA_ERROR

        if isinstance(result, dict) and "results" in result:
            results = result["results"]
        else:
            results = result if isinstance(result, list) else [result]
        if not results:
            last_err = "No result"
            if attempt + 1 < retries:
                time.sleep(min(30.0, 1.5 * (2 ** attempt)))
                continue
            return False, last_err, INFRA_ERROR
        r = results[0]
        if r.get("error"):
            last_err = str(r["error"]).strip() or "empty verify error"
            if _transient_verify_error(last_err) and attempt + 1 < retries:
                time.sleep(min(30.0, 1.5 * (2 ** attempt)))
                continue
            return False, last_err, INFRA_ERROR
        msgs = (r.get("response") or {}).get("messages") or []
        errors = [m.get("data", "") for m in msgs if m.get("severity") == "error"]
        if not errors:
            return True, "", OK
        detail = "; ".join(d for d in errors if d).strip()
        if not detail:
            last_err = "empty verify error"
            if attempt + 1 < retries:
                time.sleep(min(30.0, 1.5 * (2 ** attempt)))
                continue
            return False, last_err, INFRA_ERROR
        return False, detail, SYNTAX_ERROR
    return False, last_err or "verify failed after retries", INFRA_ERROR


def verify_sync(formal_statement: str, header: Optional[str] = None) -> Tuple[bool, str]:
    """Backwards-compatible wrapper around :func:`verify_sync_status`."""
    passed, err, _status = verify_sync_status(formal_statement, header=header)
    return passed, err


async def _verify_async_impl(client, check_content: str, timeout: int = 240):
    req = [{"proof": check_content, "custom_id": "0"}]
    return await client.async_verify(req, timeout=timeout)

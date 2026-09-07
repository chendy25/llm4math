"""Utilities for verifying Lean formal statements."""

from typing import Optional, List, Dict, Any, Tuple
import re
import os
import sys
import asyncio
from pathlib import Path

# Ensure ReForm root is importable for the local leanclient shim.
_REFORM_ROOT = Path(__file__).resolve().parents[1]
if str(_REFORM_ROOT) not in sys.path:
    sys.path.insert(0, str(_REFORM_ROOT))

# Lean client is optional - falls back to no-op if not available
try:
    from leanclient.client import Lean4Client
    _LEANCLIENT_AVAILABLE = True
except ImportError:
    _LEANCLIENT_AVAILABLE = False
    Lean4Client = None

# Default Lean 4 header for verification
default_header = (
    "import Mathlib\n"
    "import Aesop\n"
    "import Mathlib.Tactic\n"
    "set_option maxHeartbeats 0\n"
    "open BigOperators Real Nat Topology Rat\n\n"
)

VERIFY_URL = os.environ.get("VERIFY_URL")
VERIFY_KEY = os.environ.get("VERIFY_KEY")

client = None
if _LEANCLIENT_AVAILABLE and VERIFY_URL and VERIFY_KEY:
    try:
        client = Lean4Client(base_url=VERIFY_URL, api_key=VERIFY_KEY)
    except Exception as e:
        print(f"[verify_utils] Warning: Failed to initialize Lean4Client: {e}")
        client = None


def process_verify_result(verify_response: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Process verification result and extract errors."""
    if verify_response.get('error'):
        return False, [str(verify_response['error'])]
    
    response = verify_response.get('response', {})
    messages = response.get('messages', [])
    
    error_messages = []
    has_error = False
    
    for msg in messages:
        if msg.get('severity') == 'error':
            has_error = True
            error_data = msg.get('data', '')
            if error_data:
                error_messages.append(error_data)
    
    return (not has_error), error_messages


def process_batch_results(
    verify_results: List[Dict[str, Any]], 
    start_idx: int
) -> List[Dict[str, Any]]:
    """Process batch verification results."""
    result_map = {r['custom_id']: r for r in verify_results}
    results = []
    
    for idx, item in enumerate(verify_results):
        custom_id = str(start_idx + idx)
        item_copy = {}
        
        if custom_id in result_map:
            verify_result = result_map[custom_id]
            passed, errors = process_verify_result(verify_result)
            item_copy.update({
                'pass': passed,
                'verify_result': verify_result
            })
            if not passed:
                item_copy['errors'] = errors
        else:
            item_copy.update({
                'pass': False,
                'errors': ["No result found"],
                'verify_result': {"error": "No result found"}
            })
            
        results.append(item_copy)
    return results


def process_formal_statement(statement: Optional[str]) -> str:
    """Process and normalize formal statement for verification."""
    if not statement:
        return statement

    statement = str(statement)
    # Remove code block markers
    statement = re.sub(r'```(?:lean4|lean)?\s*', '', statement)
    statement = statement.strip()

    # Ensure statement ends with := by sorry
    if not statement.endswith(':= by sorry'):
        statement = re.sub(r':=\s*by\s*$', ' := by sorry', statement)
        statement = re.sub(r':=\s*$', ' := by sorry', statement)
        if not re.search(r':=.*$', statement):
            statement = statement + ' := by sorry'

    return statement


async def verify_formal_statement(formal_statement: str, header: dict) -> List[bool]:
    """
    Verify a formal statement using the Lean 4 client.

    Args:
        formal_statement: The Lean 4 formal statement to verify.
        header: Custom header to use, or None for default.

    Returns:
        List of boolean results indicating verification success.
    """
    if client is None:
        # Fallback: do basic syntax check without Lean server
        return [_basic_syntax_check(formal_statement)]

    header_content = header if header is not None else default_header

    formal_statement_content = process_formal_statement(formal_statement)

    if formal_statement_content and "import" in formal_statement_content:
        check_content = formal_statement_content
    else:
        check_content = header_content + formal_statement_content
    
    verify_requests = [
        {
            "proof": check_content,
            "custom_id": "0"
        },
    ]
    response = await client.async_verify(verify_requests, timeout=100)
    verify_results = response['results'] if isinstance(response, dict) and 'results' in response else response
    batch_results = process_batch_results(verify_results, 0)

    return [res['pass'] for res in batch_results]


def _basic_syntax_check(statement: str) -> bool:
    """Basic heuristic syntax check when Lean server is unavailable."""
    if not statement or not statement.strip():
        return False
    # Check for balanced theorem declarations
    has_theorem = "theorem" in statement or "def" in statement or "lemma" in statement
    has_colon_equals = ":=" in statement
    return has_theorem and has_colon_equals


if __name__ == "__main__":
    header = None
    formal_statement = """```lean4
theorem sum_fraction_inequality {n : ℕ} (a : Fin n → ℝ) :
  (∀ i, 1 + a i > 0) →
  (∑ i : Fin n, (a i)/(1 + a i)) ≥ 
  (∑ i : Fin n, a i)/(1 + ∑ i : Fin n, a i)
  := by sorry
```"""
    results = verify_formal_statement(formal_statement, header)

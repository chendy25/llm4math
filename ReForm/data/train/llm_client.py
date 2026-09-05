"""OpenAI-compatible LLM client for SFT trajectory collection and eval.

Environment variables:
  - OPENAI_BASE_URL / OPENAI_API_KEY:      Autoformalizer model endpoint
  - CRITICLEAN_BASE_URL / CRITICLEAN_API_KEY: CriticLean model endpoint
"""

from __future__ import annotations

import base64
import io
import os
import re
from typing import Optional

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from . import leansearch_tool
    from . import retrieval_tool
except ImportError:
    import leansearch_tool
    import retrieval_tool


def get_client(base_url: Optional[str] = None, api_key: Optional[str] = None):
    """Get OpenAI-compatible client for Autoformalizer."""
    base_url = base_url or os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
    api_key = api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")
    if OpenAI is None:
        raise ImportError("Please install openai: pip install openai")
    return OpenAI(base_url=base_url, api_key=api_key)


def get_critic_client(base_url: Optional[str] = None, api_key: Optional[str] = None):
    """Get OpenAI-compatible client for CriticLean.

    Falls back to autoformalizer settings if not configured.
    """
    base_url = (
        base_url
        or os.environ.get("CRITICLEAN_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
    )
    api_key = (
        api_key
        or os.environ.get("CRITICLEAN_API_KEY")
        or os.environ.get("OPENAI_API_KEY", "EMPTY")
    )
    if OpenAI is None:
        raise ImportError("Please install openai: pip install openai")
    return OpenAI(base_url=base_url, api_key=api_key)


def _assistant_text(msg) -> str:
    """LightLLM/Qwen3 reasoning_parser puts the body in `reasoning` and
    leaves `content` empty. ReForm needs the full text to find ```lean4 /
    <tool_call>."""
    content = getattr(msg, "content", None) or ""
    reasoning = (
        getattr(msg, "reasoning_content", None)
        or getattr(msg, "reasoning", None)
        or ""
    )
    if not reasoning:
        dumped = {}
        if hasattr(msg, "model_dump"):
            dumped = msg.model_dump() or {}
        elif hasattr(msg, "to_dict"):
            dumped = msg.to_dict() or {}
        reasoning = dumped.get("reasoning_content") or dumped.get("reasoning") or ""
        if not content:
            content = dumped.get("content") or ""
    if reasoning and content:
        return f"{reasoning}\n{content}"
    return content or reasoning or ""


def _assistant_content(msg) -> str:
    """Only the final answer channel, falling back to reasoning if empty."""
    content = getattr(msg, "content", None) or ""
    if not content and hasattr(msg, "model_dump"):
        content = (msg.model_dump() or {}).get("content") or ""
    if content.strip():
        return content
    return _assistant_text(msg)


def chat_completion(
    client,
    messages: list,
    model: str = "default",
    temperature: float = 0.6,
    max_tokens: int = 8192,
    stop=None,
    extra_body: Optional[dict] = None,
    content_only: bool = False,
) -> str:
    """Return assistant content from chat completion. `stop` is passed through
    to the API when provided (used to stop generation on the tool-call close tag).

    ``extra_body`` carries vendor knobs such as ``enable_thinking``; a backend
    that rejects them is retried without so a self-judge fallback still works.
    ``content_only`` skips the reasoning trace, which matters for the critic:
    its thinking often contains intermediate `<consistency>` verdicts that must
    not be mistaken for the final one.
    """
    kwargs = {}
    if stop is not None:
        kwargs["stop"] = stop

    def _create(with_extra: bool):
        call_kwargs = dict(kwargs)
        if with_extra and extra_body:
            call_kwargs["extra_body"] = dict(extra_body)
        return client.chat.completions.create(
            model=model or "default",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **call_kwargs,
        )

    try:
        try:
            resp = _create(True)
        except Exception:
            if not extra_body:
                raise
            resp = _create(False)
    except Exception as e:
        try:
            from pipeline_guard import raise_if_arrears
        except ImportError:
            raise
        raise_if_arrears(e)
        raise
    msg = resp.choices[0].message
    return _assistant_content(msg) if content_only else _assistant_text(msg)


def _dispatch_tool_call(
    call: dict,
    tools: list,
    default_num_results: int,
    timeout: float,
) -> str:
    """Dispatch a parsed tool call to the matching tool by name."""
    target_name = call.get("name") or call.get("tool")
    for tool in tools:
        tool_name = getattr(tool, "TOOL_NAME", getattr(tool, "name", None))
        if target_name is None or target_name == tool_name:
            run = getattr(tool, "run_tool_call", None)
            if run is None:
                continue
            return run(call, default_num_results=default_num_results, timeout=timeout)
    return f"(Unknown tool: {target_name})"


def _pil_to_data_url(image, max_side: int = 1024, jpeg_quality: int = 85) -> str:
    """Convert a PIL image to an OpenAI-vision-compatible data URL."""
    try:
        from PIL import Image  # type: ignore
    except ImportError as e:
        raise ImportError(
            "Pillow is required for vision messages: pip install Pillow"
        ) from e
    if not isinstance(image, Image.Image):
        raise TypeError(f"Expected PIL.Image, got {type(image).__name__}")
    img = image.convert("RGB")
    w, h = img.size
    longest = max(w, h)
    if longest > max_side:
        scale = max_side / float(longest)
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=jpeg_quality)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def build_vision_user_message(
    text: str,
    image=None,
    image_caption: str | None = None,
) -> dict:
    """Build an OpenAI-vision user message from text + (optional) PIL image."""
    parts: list[dict] = []
    if image_caption and image_caption.strip() and image_caption.strip() not in text:
        parts.append({"type": "text", "text": f"[Image description: {image_caption.strip()}]\n\n"})
    parts.append({"type": "text", "text": text})
    if image is not None:
        parts.append({
            "type": "image_url",
            "image_url": {"url": _pil_to_data_url(image)},
        })
    return {"role": "user", "content": parts}


def extract_lean4_block(text: str) -> Optional[str]:
    """Extract first ```lean4 ... ``` block from model output."""
    pattern = r"```lean4\s*(.*?)\s*```"
    m = re.search(pattern, text, re.DOTALL)
    return m.group(1).strip() if m else None


def parse_consistency_result(text: str) -> tuple[bool, str]:
    """Parse <consistency>Correct|Incorrect</consistency> and comments.

    Take the LAST verdict: a thinking trace can weigh up "Correct" before
    settling on "Incorrect".
    """
    text = text.strip()
    correct = False
    comments = ""
    verdicts = re.findall(
        r"<consistency>\s*(Correct|Incorrect)\s*</consistency>", text, re.IGNORECASE
    )
    if verdicts:
        correct = verdicts[-1].strip().lower() == "correct"
    c = re.findall(r"<comments>\s*(.*?)\s*</comments>", text, re.DOTALL | re.IGNORECASE)
    if c:
        comments = c[-1].strip()
    return correct, comments


def autoformalize_with_tools(
    client,
    base_messages: list,
    model: str = "default",
    temperature: float = 0.6,
    max_tokens: int = 4096,
    leansearch_url: str = leansearch_tool.DEFAULT_URL,
    leansearch_top_k: int = 8,
    max_tool_calls: int = 3,
    tools: Optional[list] = None,
    tool_top_k: Optional[int] = None,
    tool_timeout: float = 60.0,
) -> tuple[str, Optional[str], int]:
    """Run an agentic ReAct loop for one autoformalization step (collection)."""
    if tools is None:
        tools = [leansearch_tool]
    top_k = tool_top_k if tool_top_k is not None else leansearch_top_k
    default_num_results = top_k

    messages = list(base_messages)
    round_text = ""
    num_tool_calls = 0

    for _ in range(max_tool_calls + 1):
        text = chat_completion(
            client,
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=[retrieval_tool.TOOL_CALL_CLOSE],
        )
        if hasattr(retrieval_tool, "close_tool_calls"):
            text = retrieval_tool.close_tool_calls(text)

        call = retrieval_tool.parse_tool_call(text)
        has_lean = extract_lean4_block(text) is not None

        if call is None or has_lean or num_tool_calls >= max_tool_calls:
            round_text += text
            break

        segment = text
        if retrieval_tool.TOOL_CALL_OPEN in segment and not segment.rstrip().endswith(
            retrieval_tool.TOOL_CALL_CLOSE
        ):
            segment = segment.rstrip() + "\n" + retrieval_tool.TOOL_CALL_CLOSE

        tool_output = _dispatch_tool_call(
            call, tools, default_num_results=default_num_results, timeout=tool_timeout,
        )
        num_tool_calls += 1
        response_block = (
            "\n"
            + retrieval_tool.TOOL_RESPONSE_OPEN
            + "\n"
            + tool_output
            + "\n"
            + retrieval_tool.TOOL_RESPONSE_CLOSE
            + "\n"
        )

        round_text += segment + response_block
        messages = messages + [
            {"role": "assistant", "content": segment},
            {"role": "user", "content": response_block},
        ]

    return round_text, extract_lean4_block(round_text), num_tool_calls

"""PhyX -> ReForm SFT input adapter.

PhyX (https://phyx-bench.github.io/, HF: Cloudriver/PhyX) is a multimodal
benchmark of 3000 physics problems across 6 domains (mechanics,
thermodynamics, electromagnetism, modern physics, optics, wave &
acoustics). Each problem has both a textual statement and a visual
context (PIL image), and the public HF repo exposes both the image and
an `image_caption` textual description of it.

We support two input modes:

- ``adapter="text"`` (default): only the textual fields are used
  (question + description + image_caption). This works with any
  text-only LLM that the upstream `collect_sft_trajectory.py` already
  expects.
- ``adapter="vision"``: the original PIL image is passed through (as a
  base64 data URL or a file path, depending on the LLM client) so a
  vision-capable model can see the figure. The textual statement is
  still included as a primary source of truth.

Both adapters emit a dict shaped like every other ReForm input item:

    {
        "informal_statement": "<merged problem text>",
        "question_id": <int>,
        "dataset_name": "phyx",
        "image": <PIL.Image | None>,
        "image_caption": <str>,
        "category": <str>,
        "subfield": <str>,
        "reasoning_type": <list[str]>,
        "answer": <str>,
        "options": <list[str]>,
    }

The downstream `run_collect_sft_data.py::load_input` recognises
``dataset_name == "phyx"`` and routes through `phyx_to_input`.
"""

from __future__ import annotations

import io
from typing import Any, Iterable, Optional

try:
    from PIL import Image  # type: ignore
except ImportError:  # pragma: no cover
    Image = None


def _merge_statement(record: dict[str, Any]) -> str:
    """Build the natural-language problem statement from PhyX's text fields.

    Priority: question + image_caption (always included) + question_description
    (when the simplified setting isn't being used). The image_caption
    substitute gives the model the visual context even in text-only mode.
    """
    parts: list[str] = []
    q = (record.get("question") or "").strip()
    if q:
        parts.append(q)
    caption = (record.get("image_caption") or "").strip()
    if caption:
        parts.append(f"[Image description: {caption}]")
    desc = (record.get("question_description") or "").strip()
    if desc and desc != q:
        parts.append(desc)
    options = record.get("options")
    if isinstance(options, (list, tuple)) and options:
        parts.append("Options:\n" + "\n".join(f"  - {o}" for o in options))
    return "\n\n".join(parts).strip()


def _record_to_image(record: dict[str, Any]):
    """Return the PIL image from a PhyX record, or None if not present.

    PhyX stores the image as either a PIL.Image, a path string, or a
    ``{"path": ..., "bytes": ...}``-like dict; we accept all three so the
    adapter works whether the user loaded the dataset via HF `datasets`
    or via the raw TSV files."""
    img = record.get("image")
    if img is None:
        return None
    if Image is not None and isinstance(img, Image.Image):
        return img
    if isinstance(img, dict):
        if "bytes" in img and img["bytes"]:
            return Image.open(io.BytesIO(img["bytes"])) if Image else None
        if "path" in img and img["path"]:
            return Image.open(img["path"]) if Image else None
    if isinstance(img, str):
        # Image filename (PhyX ships with a separate image archive).
        return Image.open(img) if Image else None
    return None


def _resolve_question_id(
    record: dict[str, Any],
    *,
    question_id_key: str | None = "index",
) -> Any:
    """Resolve PhyX question id from seed / HF / export schemas.

    Prefer an explicit ``question_id_key`` when that field is present and
    non-empty; otherwise fall back to ``question_id`` → ``index`` → ``id``.
    SFT export seeds store ``question_id`` (not HF ``index``), so a bare
    ``record.get("index")`` used to drop ids for the whole PhyX collect path.
    """
    if question_id_key:
        v = record.get(question_id_key)
        if v is not None and v != "":
            return v
    for key in ("question_id", "index", "id"):
        v = record.get(key)
        if v is not None and v != "":
            return v
    return None


def phyx_to_input(
    record: dict[str, Any],
    *,
    include_image: bool = False,
    question_id_key: str | None = "index",
) -> dict[str, Any]:
    """Convert one PhyX record (HF row or raw TSV row) into a ReForm input item.

    Args:
        record: one PhyX example. Required text fields: ``question``,
            ``question_description``, ``question_simplified`` / ``question_simply``.
            Optional: ``image``, ``image_caption``, ``options``, ``answer``,
            ``category``, ``subfield``, ``reasoning_type``.
        include_image: when True, attach the PIL image (or None) under
            ``"image"`` so a vision-capable model can see it. When False
            the textual ``image_caption`` is still included in
            ``informal_statement``.
        question_id_key: preferred key for ``question_id`` (HF PhyX uses
            ``index``). If missing/empty, falls back to ``question_id`` /
            ``index`` / ``id`` so export seeds keep their ids.
    """
    informal_statement = _merge_statement(record)
    out: dict[str, Any] = {
        "informal_statement": informal_statement,
        "dataset_name": "phyx",
        "question_id": _resolve_question_id(record, question_id_key=question_id_key),
        "image_caption": (record.get("image_caption") or "").strip(),
        "category": record.get("category") or "",
        "subfield": record.get("subfield") or "",
        "reasoning_type": record.get("reasoning_type") or [],
        "answer": record.get("answer") or "",
        "options": record.get("options") or [],
    }
    if include_image:
        out["image"] = _record_to_image(record)
    return out


def iter_phyx(dataset: Any, *, include_image: bool = False, limit: Optional[int] = None) -> Iterable[dict[str, Any]]:
    """Iterate over a loaded PhyX dataset (HF ``datasets.Dataset`` or a list
    of dicts) yielding ReForm input items.

    Args:
        dataset: anything indexable by integer that yields dict-like rows.
        include_image: forwarded to ``phyx_to_input``.
        limit: optional max rows to emit.
    """
    n = len(dataset) if hasattr(dataset, "__len__") else None
    upto = min(n, limit) if (n is not None and limit is not None) else limit
    for i, row in enumerate(dataset):
        if upto is not None and i >= upto:
            break
        yield phyx_to_input(dict(row), include_image=include_image)
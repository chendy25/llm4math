#!/usr/bin/env python3
"""Load ReForm eval / collect inputs. Eval only needs ``load_input``."""

from __future__ import annotations

import json


def _merge_mm_informal(it: dict) -> str:
    """Build informal_statement for generic multimodal physics seeds."""
    parts = []
    q = (it.get("question") or it.get("text") or "").strip()
    if q:
        parts.append(q)
    cap = (it.get("image_caption") or "").strip()
    if cap:
        parts.append(f"[Image description: {cap}]")
    desc = (it.get("question_description") or it.get("description") or "").strip()
    if desc and desc != q:
        parts.append(desc)
    opts = it.get("options")
    if opts:
        if isinstance(opts, list):
            parts.append("Options:\n" + "\n".join(str(o) for o in opts))
        else:
            parts.append(f"Options: {opts}")
    return "\n\n".join(parts).strip()


def _attach_image_if_needed(it: dict, *, include_image: bool) -> dict:
    """Keep a path string; never open PIL during load."""
    if not include_image:
        return it
    row = dict(it)
    img = row.get("image")
    if isinstance(img, str) and img.strip():
        row.setdefault("image_path", img)
    elif isinstance(img, dict) and img.get("path"):
        row.setdefault("image_path", img["path"])
    return row


def load_input(path: str, *, include_image: bool = False) -> list[dict]:
    """Load items with informal_statement from JSON or JSONL.

    PhyX rows without a pre-merged statement go through ``phyx_adapter``.
    Image paths stay on disk; ``run_eval_rollout`` opens them in the worker.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    items = []
    if raw.startswith("["):
        data = json.loads(raw)
        items = data if isinstance(data, list) else [data]
    else:
        lines = [ln.strip() for ln in raw.split("\n") if ln.strip()]
        if len(lines) == 1 and (lines[0].startswith("{") or "informal_statement" in lines[0]):
            items = [json.loads(lines[0])]
        else:
            for line in lines:
                items.append(json.loads(line))

    def _image_path(row: dict):
        img = row.get("image")
        if isinstance(img, str) and img.strip():
            return img
        if isinstance(img, dict):
            p = img.get("path")
            if isinstance(p, str) and p.strip():
                return p
        p = row.get("image_path")
        return p if isinstance(p, str) and p.strip() else None

    def _keep_meta(row: dict, src: dict) -> dict:
        row = dict(row)
        p = _image_path(src) or _image_path(row)
        if p:
            row.setdefault("image_path", p)
        for key in (
            "retrieval_domain",
            "source_subset",
            "source_file",
            "question_id",
            "category",
            "dataset_name",
            "image_caption",
        ):
            if src.get(key) and not row.get(key):
                row[key] = src[key]
        return row

    out = []
    for it in items:
        if not isinstance(it, dict):
            continue
        if it.get("dataset_name") == "phyx" and not it.get("informal_statement"):
            try:
                from . import phyx_adapter  # type: ignore
            except ImportError:
                import phyx_adapter  # type: ignore
            normalized = phyx_adapter.phyx_to_input(it, include_image=False)
            if normalized.get("informal_statement"):
                out.append(_keep_meta(normalized, it))
                continue
        if it.get("informal_statement"):
            out.append(_keep_meta(
                _attach_image_if_needed(it, include_image=include_image), it
            ))
            continue
        st = _merge_mm_informal(it) if (
            it.get("image") or it.get("image_caption") or it.get("question")
        ) else (it.get("question") or it.get("text"))
        if st:
            row = {
                "informal_statement": st,
                **{k: v for k, v in it.items() if k not in ("question", "text")},
            }
            out.append(_keep_meta(
                _attach_image_if_needed(row, include_image=include_image), it
            ))
    return out if out else items

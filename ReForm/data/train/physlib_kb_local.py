"""Local, live-updatable Physlib KB retrieval (KB-growth Option B).

Why this exists
---------------
Option A (re-index the KB as a second LeanSearch-v2 corpus) needs a GPU rebuild
and a server restart per KB update, which stalls data collection. Option B keeps
a *local* index over the harvested KB store (`physlib_kb_store.jsonl`) that:

  - needs NO GPU and NO external service,
  - hot-reloads when the store file changes (harvest appends are picked up on the
    next query by the running collection workers — no restart),
  - queries ONLY the KB (it augments, never replaces, the remote `physlib`).

Backend: a small self-contained BM25 (no third-party dependency). The KB is
small and high-precision (verified + informalized), so lexical BM25 over
`informal_name + informal_description + name + type` gives good recall cheaply.
A future embedding/FAISS backend can be dropped in behind the same `search()`.

Result dicts match the shape `physsearch_tool.format_results` expects:
``{name: list[str], kind, type, signature, informal_description, distance,
   corpus}``.
"""
from __future__ import annotations

import json
import math
import os
import re
import threading
from pathlib import Path
from typing import Any

_TRAIN_DIR = Path(__file__).resolve().parent
_DEFAULT_STORE = _TRAIN_DIR / "sft_collect" / "physlib_kb" / "physlib_kb_store.jsonl"

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> list[str]:
    if not text:
        return []
    # Split camelCase / snake so `thermalEfficiency` matches "thermal efficiency".
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
    return [t.lower() for t in _TOKEN_RE.findall(text) if len(t) >= 2]


def _doc_text(rec: dict) -> str:
    parts = [
        str(rec.get("informal_name") or ""),
        str(rec.get("informal_description") or ""),
        str(rec.get("name") or ""),
        str(rec.get("type") or ""),
    ]
    return " ".join(p for p in parts if p)


class _BM25:
    """Minimal BM25 (Okapi) over a list of tokenized documents."""

    def __init__(self, docs_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_tokens = docs_tokens
        self.N = len(docs_tokens)
        self.doc_len = [len(d) for d in docs_tokens]
        self.avgdl = (sum(self.doc_len) / self.N) if self.N else 0.0
        self.tf: list[dict[str, int]] = []
        df: dict[str, int] = {}
        for toks in docs_tokens:
            counts: dict[str, int] = {}
            for t in toks:
                counts[t] = counts.get(t, 0) + 1
            self.tf.append(counts)
            for t in counts:
                df[t] = df.get(t, 0) + 1
        self.idf: dict[str, float] = {}
        for t, n in df.items():
            # BM25 idf with +1 to stay positive.
            self.idf[t] = math.log(1 + (self.N - n + 0.5) / (n + 0.5))

    def score(self, query_tokens: list[str], idx: int) -> float:
        if not query_tokens or self.N == 0:
            return 0.0
        counts = self.tf[idx]
        dl = self.doc_len[idx]
        s = 0.0
        for qt in query_tokens:
            f = counts.get(qt)
            if not f:
                continue
            idf = self.idf.get(qt, 0.0)
            denom = f + self.k1 * (1 - self.b + self.b * (dl / self.avgdl if self.avgdl else 0))
            s += idf * (f * (self.k1 + 1)) / (denom or 1.0)
        return s


class LocalKBIndex:
    """BM25 index over the KB store JSONL with mtime-based hot reload."""

    def __init__(self, store_path: str | os.PathLike | None = None):
        env_path = os.environ.get("PHYSSEARCH_KB_PATH")
        self.store_path = Path(store_path or env_path or _DEFAULT_STORE)
        self._lock = threading.RLock()
        self._mtime: float | None = None
        self._records: list[dict] = []
        self._bm25: _BM25 | None = None
        self._build_if_stale()

    # -- loading / building -------------------------------------------------
    def _current_mtime(self) -> float | None:
        try:
            return self.store_path.stat().st_mtime
        except OSError:
            return None

    def _build_if_stale(self) -> None:
        mt = self._current_mtime()
        if mt == self._mtime and self._bm25 is not None:
            return
        with self._lock:
            mt = self._current_mtime()
            if mt == self._mtime and self._bm25 is not None:
                return
            records: list[dict] = []
            if mt is not None:
                try:
                    with self.store_path.open(encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                records.append(json.loads(line))
                            except json.JSONDecodeError:
                                # Tolerate a partially-written trailing line
                                # (harvest appends concurrently).
                                continue
                except OSError:
                    records = []
            docs = [_tokenize(_doc_text(r)) for r in records]
            self._records = records
            self._bm25 = _BM25(docs) if records else None
            self._mtime = mt

    # -- query --------------------------------------------------------------
    def search(self, query: str, num_results: int = 8) -> list[dict[str, Any]]:
        self._build_if_stale()
        if not query or not query.strip() or not self._bm25 or not self._records:
            return []
        qt = _tokenize(query)
        if not qt:
            return []
        scored: list[tuple[float, int]] = []
        for i in range(len(self._records)):
            sc = self._bm25.score(qt, i)
            if sc > 0:
                scored.append((sc, i))
        if not scored:
            return []
        scored.sort(key=lambda x: x[0], reverse=True)
        out: list[dict[str, Any]] = []
        for sc, i in scored[: max(1, num_results)]:
            rec = self._records[i]
            name = rec.get("name")
            name_list = name.split(".") if isinstance(name, str) else (name or [])
            out.append({
                "name": name_list,
                "kind": rec.get("kind") or "",
                "type": rec.get("type") or "",
                "signature": "",
                "informal_description": rec.get("informal_description") or "",
                # Synthetic distance (smaller = better) purely for display/debug;
                # cross-source ranking uses a KB quota, not this value.
                "distance": 1.0 / (1.0 + sc),
                "bm25_score": sc,
                "corpus": "physlib_kb_local",
            })
        return out

    def __len__(self) -> int:
        self._build_if_stale()
        return len(self._records)


# Module-level singleton so all tool calls in a process share one index.
_INDEX: LocalKBIndex | None = None
_INDEX_LOCK = threading.Lock()


def get_index() -> LocalKBIndex:
    global _INDEX
    if _INDEX is None:
        with _INDEX_LOCK:
            if _INDEX is None:
                _INDEX = LocalKBIndex()
    return _INDEX


def search(query: str, num_results: int = 8) -> list[dict[str, Any]]:
    """Convenience wrapper over the process-wide local KB index."""
    try:
        return get_index().search(query, num_results=num_results)
    except Exception:
        return []

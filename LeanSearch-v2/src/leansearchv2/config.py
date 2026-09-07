"""Tiny config helper.

Reads ``config.yaml`` from the repository root, then layers ``config.local.yaml``
on top if it exists. Each setting can also be overridden by an environment
variable of the same name; environment variables win over both files.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]

# Matches `org/name` HF repo ids (single slash, no path separators).
_HF_REPO_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")


def _deep_merge(base: dict, overlay: dict) -> dict:
    out = dict(base)
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


_CACHE: dict[str, Any] | None = None


def _config() -> dict[str, Any]:
    global _CACHE
    if _CACHE is None:
        base_path = Path(os.environ.get("LEANSEARCHV2_CONFIG", REPO_ROOT / "config.yaml"))
        cfg = _load_yaml(base_path)
        local_path = REPO_ROOT / "config.local.yaml"
        if local_path.exists():
            cfg = _deep_merge(cfg, _load_yaml(local_path))
        _CACHE = cfg
    return _CACHE


def _resolve(value: str) -> str:
    if _HF_REPO_ID.match(value):
        return value
    p = Path(value)
    if not p.is_absolute():
        p = REPO_ROOT / p
    return str(p)


def get(env_key: str | None, *yaml_path: str, default: Any = None) -> Any:
    """Return ``$env_key`` if set, else look up ``yaml_path`` in config files, else ``default``.
    Pass ``env_key=None`` to skip the env-var lookup entirely."""
    if env_key is not None and env_key in os.environ:
        return os.environ[env_key]
    node: Any = _config()
    for key in yaml_path:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node if node is not None else default


def get_path(env_key: str, *yaml_path: str, default: str | None = None) -> str:
    value = get(env_key, *yaml_path, default=default)
    if value is None:
        raise KeyError(
            f"Missing path: set ${env_key} or config.yaml:{'.'.join(yaml_path)}"
        )
    return _resolve(value)


# ---------------------------------------------------------------------------
# Multi-corpus helpers
# ---------------------------------------------------------------------------
#
# The retriever serves one or more named corpora from a single FastAPI process.
# Each corpus owns:
#   - a cuVS index directory (loaded into memory on first request to that corpus)
#   - a Lean source checkout (used by the corpus build pipeline)
#   - a per-corpus workdir subdirectory for jixia raw / merged / final JSONL
#
# The legacy `paths.cuvs_db` / `paths.mathlib4` keys are still honored so
# existing single-corpus deployments and tooling keep working.
DEFAULT_CORPUS = "mathlib"


def corpus_config(corpus_name: str) -> dict[str, Any]:
    """Return the ``paths.corpora.<corpus_name>`` block, falling back to the
    legacy single-corpus fields if no ``corpora`` block is configured."""
    corpora = get(None, "paths", "corpora")
    if isinstance(corpora, dict) and corpus_name in corpora:
        return corpora[corpus_name] or {}
    # Legacy fallback: synthesize a mathlib entry from the old keys.
    if corpus_name == DEFAULT_CORPUS:
        return {
            "cuvs_db": get(None, "paths", "cuvs_db"),
            "source": get(None, "paths", "mathlib4"),
            "source_subdir": "Mathlib",
            "workdir_subdir": "mathlib",
        }
    raise KeyError(
        f"Unknown corpus {corpus_name!r}: configure paths.corpora.{corpus_name} "
        f"in config.yaml (or config.local.yaml)."
    )


def corpus_cuvs_dir(corpus_name: str) -> str:
    cfg = corpus_config(corpus_name)
    key = f"CORPUS_{corpus_name.upper()}_CUVS_DB"
    if "cuvs_db" in cfg and "corpora" in (get(None, "paths") or {}):
        return get_path(key, "paths", "corpora", corpus_name, "cuvs_db")
    # Legacy fallback (mathlib uses paths.cuvs_db).
    return get_path(key, "paths", "cuvs_db")


def corpus_source_dir(corpus_name: str) -> str:
    cfg = corpus_config(corpus_name)
    key = f"CORPUS_{corpus_name.upper()}_SOURCE"
    corpora = get(None, "paths", "corpora")
    if "source" in cfg and isinstance(corpora, dict) and corpus_name in corpora:
        return get_path(key, "paths", "corpora", corpus_name, "source")
    # Legacy fallback.
    return get_path(key, "paths", "mathlib4")


def corpus_source_subdir(corpus_name: str) -> str:
    cfg = corpus_config(corpus_name)
    val = cfg.get("source_subdir") or "."
    return str(val) if val else "."


def corpus_workdir(corpus_name: str) -> str:
    """Return the per-corpus jixia raw / merged / final parent directory."""
    workdir = get_path("CORPUS_WORKDIR", "paths", "corpus_workdir")
    cfg = corpus_config(corpus_name)
    sub = cfg.get("workdir_subdir") or corpus_name
    return str(Path(workdir) / str(sub))

#!/usr/bin/env bash
# Start the LeanSearch v2 FastAPI server with both mathlib and physlib corpora
# registered. The mathlib cuVS index is loaded eagerly (so the legacy
# `?corpus=` default keeps working without a cold start), and the physlib
# index is loaded lazily on the first request to `?corpus=physlib`.
#
# Paths and tuning come from ../config.yaml; override any of them by
# exporting the matching env var before running this script
# (VECTORDB_DIR, EMBEDDING_MODEL_PATH, RERANKER_MODEL_PATH, NUM_GPUS,
# GPU_MEMORY_UTILIZATION, HOST, PORT, CORPORA).
#
# To switch to "physlib only" mode (saves startup time when mathlib is
# unnecessary), run with `CORPORA=physlib ./scripts/serve_physlib.sh`.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

# Allow running without `pip install -e .`
export PYTHONPATH="${REPO_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-8000}"

# Default to registering both corpora. Override via CORPORA env var if needed.
export CORPORA="${CORPORA:-mathlib,physlib}"

exec python -m uvicorn leansearchv2.server:app --host "$HOST" --port "$PORT"
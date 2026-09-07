#!/usr/bin/env bash
# End-to-end bootstrap for the *local* dual-corpus retrieval path.
#
# What this does (each step is skippable if already done):
#   1. Download prebuilt Mathlib cuVS  (~5.7 GB; needs vpn / HF access)
#   2. Clone Physlib + lake cache get
#   3. Build jixia for mathlib (submodule, Lean 4.28) and physlib (Lean 4.32)
#   4. Optionally run scripts/build_corpus.sh physlib  (LLM + GPU for last step)
#
# This machine may not have a GPU. Serving (`scripts/serve_physlib.sh`) and
# `build_cuvs` require GPUs with the Qwen3 embedding/reranker models.
#
# Usage:
#   vpn   # required on this cluster for github / huggingface
#   ./scripts/setup_local_retrieval.sh              # steps 1-3
#   ./scripts/setup_local_retrieval.sh --with-physlib-corpus   # also run build_corpus
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

WITH_PHYS_CORPUS=0
for arg in "$@"; do
  case "$arg" in
    --with-physlib-corpus) WITH_PHYS_CORPUS=1 ;;
    -h|--help)
      sed -n '2,20p' "$0"
      exit 0
      ;;
  esac
done

if [[ -z "${http_proxy:-}${https_proxy:-}" ]]; then
  echo "WARNING: http(s)_proxy unset. On this cluster run:  vpn" >&2
fi

chmod +x \
  scripts/download_mathlib_cuvs.sh \
  scripts/setup_physlib_source.sh \
  scripts/build_jixia_for_corpus.sh \
  scripts/build_corpus.sh \
  scripts/serve_physlib.sh \
  2>/dev/null || true

echo "======== [1/4] Mathlib cuVS ========"
if [[ -f data/cuvs/mathlib-v4.28.0-rc1/cuvs_index.bin \
   && -f data/cuvs/mathlib-v4.28.0-rc1/metadata.pkl \
   && -f data/cuvs/mathlib-v4.28.0-rc1/texts.pkl ]]; then
  echo "Mathlib cuVS already present."
  ls -lh data/cuvs/mathlib-v4.28.0-rc1/{cuvs_index.bin,metadata.pkl,texts.pkl}
else
  ./scripts/download_mathlib_cuvs.sh
fi

echo "======== [2/4] Physlib source ========"
if [[ -d external/physlib/.git ]]; then
  echo "Physlib checkout present at external/physlib"
else
  ./scripts/setup_physlib_source.sh
fi

echo "======== [3/4] jixia binaries ========"
./scripts/build_jixia_for_corpus.sh mathlib
./scripts/build_jixia_for_corpus.sh physlib

echo "======== [4/4] Physlib corpus (optional) ========"
if [[ "$WITH_PHYS_CORPUS" -eq 1 ]]; then
  JIXIA_BIN="$(cat data/corpus/jixia_physlib.bin.path)"
  export JIXIA_BIN
  echo "Using JIXIA_BIN=${JIXIA_BIN}"
  ./scripts/build_corpus.sh physlib
else
  echo "Skipped. When ready (needs LLM for informalize + GPU for build_cuvs):"
  echo "  export JIXIA_BIN=\$(cat data/corpus/jixia_physlib.bin.path)"
  echo "  ./scripts/build_corpus.sh physlib"
fi

echo
echo "Next (on a GPU node):"
echo "  CORPORA=mathlib,physlib ./scripts/serve_physlib.sh"
echo "  curl -X POST 'http://localhost:8000/search?corpus=mathlib' -H 'Content-Type: application/json' \\"
echo "    -d '{\"query\":[\"group order divides\"],\"num_results\":5}'"
echo "  curl -X POST 'http://localhost:8000/search?corpus=physlib' -H 'Content-Type: application/json' \\"
echo "    -d '{\"query\":[\"Noether theorem\"],\"num_results\":5}'"

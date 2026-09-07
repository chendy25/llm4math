#!/usr/bin/env bash
# Build a LeanSearch v2 corpus from scratch.
#
# Pipeline (all four sub-steps; comment / split as needed):
#   1a. jixia_extract     — Lean *.lean -> raw .sym/.decl/.mod JSON  (lake env; CPU is enough)
#   1b. merge_to_jsonl    — raw JSON -> merged JSONL                  (CPU-only)
#   1c. informalize       — merged JSONL -> final JSONL (with informal) (CPU + LLM API)
#   2.  build_cuvs        — final JSONL -> cuVS index + metadata.pkl   (requires GPU)
#
# Usage:
#   scripts/build_corpus.sh                # build the default mathlib corpus
#   scripts/build_corpus.sh physlib        # build the physlib corpus
#
# Most users SHOULD NOT need to run this. The default workflow is:
#   - download a pre-built cuVS DB from HuggingFace into
#     `data/cuvs/<corpus-name>/`
#   - run scripts/serve.sh (mathlib) or scripts/serve_physlib.sh (physlib)
#
# Settings come from config.yaml (paths.corpora.<name>.*, jixia_bin, models.*,
# llm.*). Override per invocation via env vars (see each module's main).

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

# Allow running without `pip install -e .`
export PYTHONPATH="${REPO_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

PY="${PY:-python}"
CORPUS_NAME="${1:-mathlib}"

# Prefer a corpus-matched jixia binary if setup scripts produced one.
# (Mathlib submodule = Lean 4.28; Physlib HEAD = Lean 4.32 → different builds.)
if [[ -z "${JIXIA_BIN:-}" && -f "data/corpus/jixia_${CORPUS_NAME}.bin.path" ]]; then
  export JIXIA_BIN="$(cat "data/corpus/jixia_${CORPUS_NAME}.bin.path")"
  echo "Using JIXIA_BIN=${JIXIA_BIN}"
fi

echo "== Step 1a: jixia extract (corpus=${CORPUS_NAME}) =="
$PY -m leansearchv2.corpus.jixia_extract --corpus-name "$CORPUS_NAME"

echo "== Step 1b: merge to JSONL (corpus=${CORPUS_NAME}) =="
$PY -m leansearchv2.corpus.merge_to_jsonl --corpus-name "$CORPUS_NAME"

echo "== Step 1c: informalize (corpus=${CORPUS_NAME}) =="
$PY -m leansearchv2.corpus.informalize --corpus-name "$CORPUS_NAME"

echo "== Step 2: build cuVS index (corpus=${CORPUS_NAME}) =="
$PY -m leansearchv2.corpus.build_cuvs --corpus-name "$CORPUS_NAME"

echo "Corpus build complete: ${CORPUS_NAME}"

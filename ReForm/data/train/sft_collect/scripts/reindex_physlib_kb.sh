#!/usr/bin/env bash
# Re-index the harvested Physlib KB corpus into LeanSearch-v2 as a SECOND corpus
# `physlib_kb`, so `physsearch` can retrieve KB-grown concepts via
# `?corpus=physlib_kb` (KB-growth Option A).
#
# NOTE: Option A is OPTIONAL. The DEFAULT retrieval path is Option B (local BM25
# over physlib_kb_store.jsonl, hot-reloaded live by physlib_kb_local.py) which
# needs neither GPU nor a restart. Use THIS script only when you also want the
# heavier remote embedding corpus (set PHYSSEARCH_KB_MODE=remote|both to use it).
#
# Run this ON THE GPU BOX that serves LeanSearch-v2 (needs CUDA for build_cuvs).
# It:
#   1. copies the reform-side KB corpus into LeanSearch-v2's final/ dir
#   2. builds the cuVS index at data/cuvs/physlib_kb
#   3. ensures config.local.yaml registers the corpus (idempotent)
#   4. prints the server restart + smoke commands
#
# It does NOT restart the running server (that would drop in-flight requests);
# the restart command is printed for you to run when ready.
set -euo pipefail

REFORM_TRAIN="${REFORM_TRAIN:-/mnt/afs_reason/chendongyang/code/llm4math/ReForm/data/train}"
LS2="${LS2:-/mnt/afs_reason/chendongyang/code/llm4math/LeanSearch-v2}"
CORPUS_NAME="${CORPUS_NAME:-physlib_kb}"
KB_CORPUS_SRC="${KB_CORPUS_SRC:-$REFORM_TRAIN/sft_collect/physlib_kb/${CORPUS_NAME}_corpus.jsonl}"
EMBEDDER="${EMBEDDER:-/mnt/afs_reason/chendongyang/code/data/models/Qwen3-Embedding-8B}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_GPUS="${NUM_GPUS:-}"

PY="${PY:-python}"

if [[ ! -f "$KB_CORPUS_SRC" ]]; then
  echo "!! KB corpus not found: $KB_CORPUS_SRC" >&2
  echo "   Run harvest_physlib_kb.py first." >&2
  exit 1
fi

n_rows=$(grep -c . "$KB_CORPUS_SRC" || true)
echo "[reindex] KB corpus rows: $n_rows  ($KB_CORPUS_SRC)"
if [[ "$n_rows" -eq 0 ]]; then
  echo "!! KB corpus is empty; nothing to index." >&2
  exit 1
fi

cd "$LS2"
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"

# 1. Stage the corpus at the path build_cuvs expects (also keeps a copy in-repo).
FINAL_DIR="$LS2/data/corpus/final"
mkdir -p "$FINAL_DIR"
DEST="$FINAL_DIR/${CORPUS_NAME}_corpus.jsonl"
cp -f "$KB_CORPUS_SRC" "$DEST"
echo "[reindex] staged corpus -> $DEST"

# 2. Build the cuVS index (explicit paths so config is not strictly required).
OUT_DIR="$LS2/data/cuvs/${CORPUS_NAME}"
echo "[reindex] building cuVS index -> $OUT_DIR"
build_args=(
  -m leansearchv2.corpus.build_cuvs
  --corpus-name "$CORPUS_NAME"
  --jsonl "$DEST"
  --output-dir "$OUT_DIR"
  --embedder "$EMBEDDER"
  --batch-size "$BATCH_SIZE"
)
if [[ -n "$NUM_GPUS" ]]; then
  build_args+=(--num-gpus "$NUM_GPUS")
fi
"$PY" "${build_args[@]}"

ls -lh "$OUT_DIR"/{cuvs_index.bin,metadata.pkl,texts.pkl}

# 3. Register the corpus in config.local.yaml (idempotent) so the server can
#    resolve paths.corpora.physlib_kb.cuvs_db at startup.
"$PY" - "$LS2" "$CORPUS_NAME" <<'PY'
import sys, pathlib
ls2, name = sys.argv[1], sys.argv[2]
cfg_path = pathlib.Path(ls2) / "config.local.yaml"
cuvs_db = f"data/cuvs/{name}"
try:
    import yaml
except Exception:
    yaml = None
if yaml is not None:
    cfg = {}
    if cfg_path.is_file():
        cfg = yaml.safe_load(cfg_path.read_text()) or {}
    paths = cfg.setdefault("paths", {})
    corpora = paths.setdefault("corpora", {})
    entry = corpora.setdefault(name, {})
    entry["cuvs_db"] = cuvs_db
    entry.setdefault("workdir_subdir", name)
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    print(f"[reindex] registered corpus '{name}' -> {cfg_path}")
else:
    print("!! pyyaml not available; add this to config.local.yaml manually:")
    print(f"    paths:\n      corpora:\n        {name}:\n          cuvs_db: {cuvs_db}")
PY

echo
echo "======================================================================"
echo "[reindex] DONE building index for corpus '$CORPUS_NAME'."
echo "Restart the server so it registers the new corpus (no hot reload):"
echo
echo "    cd $LS2"
echo "    CORPORA=mathlib,physlib,$CORPUS_NAME ./scripts/serve_physlib.sh"
echo
echo "Then smoke test:"
echo "    curl -s http://localhost:8000/health | grep -o '$CORPUS_NAME'"
echo "    curl -s -X POST 'http://localhost:8000/search?corpus=$CORPUS_NAME' \\"
echo "      -H 'Content-Type: application/json' \\"
echo "      -d '{\"query\":[\"thermal efficiency of a heat engine\"],\"num_results\":5}'"
echo "======================================================================"

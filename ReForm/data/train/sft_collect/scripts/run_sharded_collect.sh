#!/usr/bin/env bash
# Launch N parallel collect workers over round-robin shards.
# Usage:
#   bash run_sharded_collect.sh math|physics [shards=4] [input_jsonl] [out_prefix]
set -euo pipefail
ROOT=/mnt/afs_reason/chendongyang/code/llm4math/ReForm/data/train
cd "$ROOT"
KIND="${1:?math|physics}"
SHARDS="${2:-4}"
export MAX_TOOL_CALLS="${MAX_TOOL_CALLS:-3}"

PY=/mnt/afs_reason/chendongyang/conda/leansearch/bin/python
SCRIPTS=sft_collect/scripts

case "$KIND" in
  math)
    IN="${3:-sft_collect/batches/pilot_math500.jsonl}"
    OUT_PREFIX="${4:-sft_collect/math/sft_math_pilot500}"
    RUNNER="$SCRIPTS/run_collect_math.sh"
    PREFIX=math
    ;;
  physics)
    IN="${3:-sft_collect/batches/pilot_physics100.jsonl}"
    OUT_PREFIX="${4:-sft_collect/physics/sft_physics_mm_pilot100}"
    RUNNER="$SCRIPTS/run_collect_physics_mm.sh"
    PREFIX=phys
    ;;
  *)
    echo "KIND must be math|physics" >&2
    exit 1
    ;;
esac

# Shard dir named after output prefix basename to avoid clobbering other runs.
OUT_BASE=$(basename "$OUT_PREFIX")
SHARD_DIR="sft_collect/batches/shards_${OUT_BASE}"

mkdir -p "$SHARD_DIR" sft_collect/logs "$(dirname "$OUT_PREFIX")"
rm -f "$SHARD_DIR"/${PREFIX}_*.jsonl
$PY "$SCRIPTS/shard_jsonl.py" --input "$IN" --outdir "$SHARD_DIR" --shards "$SHARDS" --prefix "$PREFIX"

for ((i=0; i<SHARDS; i++)); do
  sid=$(printf '%02d' "$i")
  shard="$SHARD_DIR/${PREFIX}_${sid}.jsonl"
  out="${OUT_PREFIX}_shard${sid}.jsonl"
  log="sft_collect/logs/collect_${KIND}_shard${sid}.log"
  : > "$log"
  echo "start $KIND shard=$sid in=$shard out=$out"
  setsid bash "$RUNNER" "$shard" "$out" "$log" < /dev/null &
  echo $! > "sft_collect/logs/collect_${KIND}_shard${sid}.pid"
done

echo "launched $SHARDS $KIND workers; logs under sft_collect/logs/collect_${KIND}_shard*.log"

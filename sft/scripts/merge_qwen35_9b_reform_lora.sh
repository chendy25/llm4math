#!/usr/bin/env bash
# Merge ReForm LoRA into standalone HF weights. Run on a CCI GPU node.
set -euo pipefail

ROOT=/mnt/afs_reason/chendongyang/code
ADAPTERS=${ADAPTERS:-$ROOT/llm4math/sft/output/qwen35_9b_reform_lora_v5/v0-20260906-113745/checkpoint-312}
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT/llm4math/sft/output/qwen35_9b_reform_merged_v5/checkpoint-312}
CONDA_ENV=${CONDA_ENV:-/mnt/afs_reason/chendongyang/conda/ms-swift}

export PATH=$CONDA_ENV/bin:$PATH
export CONDA_PREFIX=$CONDA_ENV
export PYTHONPATH=$ROOT/ms-swift${PYTHONPATH:+:$PYTHONPATH}
PYTHON=${CONDA_ENV}/bin/python

swift() {
    "$PYTHON" -c "import sys; from swift.cli.main import cli_main; sys.argv=['swift']+sys.argv[1:]; raise SystemExit(cli_main())" "$@"
}

if [ ! -d "$ADAPTERS" ]; then
    echo "[error] adapters not found: $ADAPTERS" >&2
    exit 1
fi

echo "============================================================"
echo "  Merge ReForm Qwen3.5-9B LoRA"
echo "  Adapters: $ADAPTERS"
echo "  Output:   $OUTPUT_DIR"
echo "  hostname=$(hostname) ips=$(hostname -I 2>/dev/null || true)"
echo "============================================================"

mkdir -p "$(dirname "$OUTPUT_DIR")"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
swift export \
    --adapters "$ADAPTERS" \
    --merge_lora true \
    --output_dir "$OUTPUT_DIR"

BASE=$ROOT/data/models/Qwen3.5-9B
for f in preprocessor_config.json video_preprocessor_config.json chat_template.jinja tokenizer_config.json; do
    if [ -f "$BASE/$f" ] && [ ! -f "$OUTPUT_DIR/$f" ]; then
        cp -f "$BASE/$f" "$OUTPUT_DIR/$f"
        echo "[merge] copied $f from base"
    fi
done

if [ ! -f "$OUTPUT_DIR/config.json" ] || ! compgen -G "$OUTPUT_DIR/*.safetensors" > /dev/null; then
    echo "[error] merge output missing HF files: $OUTPUT_DIR" >&2
    exit 1
fi

echo "[merge] done: $OUTPUT_DIR"
ls -lh "$OUTPUT_DIR" | head -30
if [ "${KEEP_ALIVE:-0}" = "1" ]; then
    sleep infinity
fi

#!/usr/bin/env bash
# Download the pre-built Mathlib cuVS index from HuggingFace into
# data/cuvs/mathlib-v4.28.0-rc1/ (matches config.yaml paths.corpora.mathlib.cuvs_db).
#
# Artifacts (~5.7 GB total):
#   metadata.pkl ~310 MB, texts.pkl ~240 MB, cuvs_index.bin ~4.8 GB
#
# Network tips (this cluster often needs one of):
#   vpn                                   # enable http(s)_proxy from ~/.zshrc
#   export HF_ENDPOINT=https://hf-mirror.com
#
# Uses wget with resume (-c). Prefer this over huggingface_hub parallel
# snapshot_download which has been flaky through the corporate proxy.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

DEST="${DEST:-data/cuvs/mathlib-v4.28.0-rc1}"
REPO_ID="${REPO_ID:-FrenzyMath/lsv2-mathlib-v4.28.0-rc1-cuvs}"
BASE_URL="${HF_ENDPOINT:-https://huggingface.co}/datasets/${REPO_ID}/resolve/main"

mkdir -p "$DEST"

if [[ -z "${http_proxy:-}${https_proxy:-}${HF_ENDPOINT:-}" ]]; then
  echo "WARNING: no http(s)_proxy / HF_ENDPOINT set; download may fail." >&2
  echo "         Try: vpn   or   export HF_ENDPOINT=https://hf-mirror.com" >&2
fi

# Drop stale HF hub cache (partial .incomplete files confuse later checks).
if [[ -d "${DEST}/.cache" ]]; then
  echo "Removing incomplete HF cache under ${DEST}/.cache ..."
  rm -rf "${DEST}/.cache"
fi

need=()
for f in metadata.pkl texts.pkl cuvs_index.bin; do
  if [[ -f "${DEST}/${f}" ]]; then
    echo "Already present: ${DEST}/${f} ($(du -h "${DEST}/${f}" | awk '{print $1}'))"
  else
    need+=("$f")
  fi
done

if [[ ${#need[@]} -eq 0 ]]; then
  echo "Mathlib cuVS already complete at ${DEST}"
  exit 0
fi

if ! command -v wget >/dev/null 2>&1; then
  echo "ERROR: wget not found" >&2
  exit 1
fi

echo "Downloading from ${BASE_URL}"
for f in "${need[@]}"; do
  echo "=== ${f} ==="
  # Follow redirects to CDN; resume supported.
  wget -c --timeout=60 --tries=0 --retry-connrefused \
    -O "${DEST}/${f}" "${BASE_URL}/${f}"
  ls -lh "${DEST}/${f}"
done

echo "Verifying artifacts..."
missing=0
for f in cuvs_index.bin metadata.pkl texts.pkl; do
  if [[ -f "${DEST}/${f}" ]]; then
    ls -lh "${DEST}/${f}"
  else
    echo "MISSING: ${DEST}/${f}"
    missing=1
  fi
done
# Sanity: cuvs_index.bin should be multi-GB.
size=$(stat -c%s "${DEST}/cuvs_index.bin" 2>/dev/null || echo 0)
if [[ "$size" -lt 1000000000 ]]; then
  echo "ERROR: cuvs_index.bin looks too small (${size} bytes). Re-run download." >&2
  exit 1
fi
if [[ "$missing" -ne 0 ]]; then
  echo "ERROR: cuVS download incomplete." >&2
  exit 1
fi
echo "Mathlib cuVS ready at ${DEST}"

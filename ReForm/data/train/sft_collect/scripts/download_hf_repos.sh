#!/usr/bin/env bash
# Download HF datasets needed for SFT seed prep. Uses vpn proxy if set.
set -euo pipefail

ROOT="${HF_DATA_ROOT:-/mnt/afs_reason/chendongyang/code/data/hf_datasets}"
PY="${PYTHON:-/mnt/afs_reason/chendongyang/conda/leansearch/bin/python}"
mkdir -p "$ROOT"
export HF_HOME="${HF_HOME:-$ROOT/hf_home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE"

# Prefer vpn proxy when caller did not set one.
if [[ -z "${http_proxy:-}${HTTP_PROXY:-}" ]]; then
  export http_proxy="http://sysagent:c08400bf@10.119.176.202:3128"
  export https_proxy="http://sysagent:c08400bf@10.119.176.202:3128"
  echo "enabled default vpn proxy"
fi
echo "proxy http_proxy=${http_proxy:-} https_proxy=${https_proxy:-}"

download_one() {
  local repo_id="$1"
  local local_dir="$2"
  shift 2
  echo "=== snapshot_download $repo_id -> $local_dir ==="
  mkdir -p "$local_dir"
  "$PY" - <<PY
from huggingface_hub import snapshot_download
print(snapshot_download(
    repo_id="$repo_id",
    repo_type="dataset",
    local_dir="$local_dir",
    local_dir_use_symlinks=False,
))
PY
}

# Math
download_one "AI-MO/NuminaMath-1.5" "$ROOT/AI-MO__NuminaMath-1.5"
download_one "AI-MO/aops" "$ROOT/AI-MO__aops"

# Physics multimodal / NL
download_one "PrismaX/PhysUniBench" "$ROOT/PrismaX__PhysUniBench"
download_one "SeePhys/SeePhys" "$ROOT/SeePhys__SeePhys"
download_one "Cloudriver/PhyX" "$ROOT/Cloudriver__PhyX"
download_one "IUTVanguard/PhysicsEval" "$ROOT/IUTVanguard__PhysicsEval"

echo "DONE all downloads under $ROOT"
ls -lah "$ROOT"

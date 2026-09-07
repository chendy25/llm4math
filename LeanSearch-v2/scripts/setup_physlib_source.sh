#!/usr/bin/env bash
# Clone Physlib and fetch Lean build cache so jixia_extract can run.
#
# Usage:
#   ./scripts/setup_physlib_source.sh
#   PHYS_LIB_REF=<git-ref> ./scripts/setup_physlib_source.sh   # pin a commit/tag
#
# After this succeeds, update config.yaml `paths.corpora.physlib.source_subdir`
# if the top-level Lean dirs differ from what jixia expects (see echo at end).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

DEST="${DEST:-external/physlib}"
URL="${PHYS_LIB_URL:-https://github.com/leanprover-community/physlib.git}"
# Default pins to the last Lean v4.29.1 commit before the v4.30 bump.
# jixia@main currently tracks Lean v4.29 and does not yet compile on v4.32
# (Analyzer.Process.Declaration API break). Override with PHYS_LIB_REF=HEAD
# only after a matching jixia port exists.
REF="${PHYS_LIB_REF:-7ea4fda8110811f0b921da99212d464ba638ab4f}"

if [[ ! -d "${DEST}/.git" ]]; then
  echo "Cloning ${URL} -> ${DEST} (ref=${REF})"
  git clone "$URL" "$DEST"
  git -C "$DEST" checkout "$REF"
else
  echo "Physlib already present at ${DEST}; fetching + checking out ${REF}..."
  git -C "$DEST" fetch --depth 1 origin "$REF" 2>/dev/null \
    || git -C "$DEST" fetch --deepen=500 origin
  git -C "$DEST" checkout --force "$REF"
fi

echo "Lean toolchain: $(cat "${DEST}/lean-toolchain" 2>/dev/null || echo unknown)"
echo "Top-level Lean dirs / libs:"
ls -la "$DEST" | head -40
echo "---"
# Prefer lake cache; full `lake build` is optional for jixia (jixia uses lake env).
cd "$DEST"
echo "Running lake exe cache get (may take a while)..."
lake exe cache get
echo "Physlib source ready at ${DEST}"
echo "Candidate source_subdir roots (adjust config.yaml if needed):"
find . -maxdepth 2 -type d \( -name PhysLean -o -name Physlib -o -name QuantumInfo -o -name PhysLib \) 2>/dev/null || true

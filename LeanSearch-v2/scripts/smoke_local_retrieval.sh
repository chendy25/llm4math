#!/usr/bin/env bash
# Offline + optional online smoke checks for the local dual-corpus path.
#
# Offline (no GPU): verifies Mathlib cuVS artifacts + jixia binaries exist.
# Online (GPU + serve running): hits /search?corpus=mathlib and /search?corpus=physlib.
#
# Usage:
#   ./scripts/smoke_local_retrieval.sh              # offline only
#   ./scripts/smoke_local_retrieval.sh --online      # also curl localhost:8000
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

ONLINE=0
for arg in "$@"; do
  [[ "$arg" == "--online" ]] && ONLINE=1
done

fail=0
check_file() {
  local f="$1" min_bytes="${2:-1}"
  if [[ -f "$f" ]]; then
    local sz
    sz=$(stat -c%s "$f")
    if [[ "$sz" -ge "$min_bytes" ]]; then
      echo "OK  $f ($(du -h "$f" | awk '{print $1}'))"
    else
      echo "BAD $f too small ($sz < $min_bytes)"
      fail=1
    fi
  else
    echo "MISSING $f"
    fail=1
  fi
}

echo "======== Offline: Mathlib cuVS ========"
check_file data/cuvs/mathlib-v4.28.0-rc1/cuvs_index.bin 1000000000
check_file data/cuvs/mathlib-v4.28.0-rc1/metadata.pkl 100000000
check_file data/cuvs/mathlib-v4.28.0-rc1/texts.pkl 100000000

echo "======== Offline: jixia binaries ========"
check_file external/jixia/.lake/build/bin/jixia 1000000
if [[ -f data/corpus/jixia_mathlib.bin.path ]]; then
  echo "OK  data/corpus/jixia_mathlib.bin.path -> $(cat data/corpus/jixia_mathlib.bin.path)"
else
  echo "WARN data/corpus/jixia_mathlib.bin.path missing (optional)"
fi
if [[ -x external/jixia-physlib/.lake/build/bin/jixia ]]; then
  check_file external/jixia-physlib/.lake/build/bin/jixia 1000000
else
  echo "MISSING external/jixia-physlib/.lake/build/bin/jixia  (run: ./scripts/build_jixia_for_corpus.sh physlib)"
  fail=1
fi

echo "======== Offline: Physlib source / cuVS ========"
if [[ -d external/physlib/.git ]]; then
  echo "OK  external/physlib ($(cat external/physlib/lean-toolchain))"
else
  echo "MISSING external/physlib"
  fail=1
fi
if [[ -f data/cuvs/physlib/cuvs_index.bin ]]; then
  check_file data/cuvs/physlib/cuvs_index.bin 1000000
else
  echo "WARN data/cuvs/physlib not built yet (needs: JIXIA_BIN=... ./scripts/build_corpus.sh physlib + GPU)"
fi

if [[ "$ONLINE" -eq 1 ]]; then
  echo "======== Online: /search ========"
  BASE="${RETRIEVER_URL:-http://localhost:8000}"
  for corpus in mathlib physlib; do
    echo "-- corpus=${corpus}"
    code=$(curl -s -o /tmp/ls_smoke_${corpus}.json -w '%{http_code}' \
      -X POST "${BASE}/search?corpus=${corpus}" \
      -H 'Content-Type: application/json' \
      -d '{"query":["group order divides the group order"],"num_results":3}' \
      --max-time 120 || echo 000)
    echo "HTTP ${code}"
    head -c 400 "/tmp/ls_smoke_${corpus}.json" 2>/dev/null; echo
    [[ "$code" == "200" ]] || fail=1
  done
fi

echo
if [[ "$fail" -eq 0 ]]; then
  echo "SMOKE_OK"
else
  echo "SMOKE_INCOMPLETE (see MISSING/BAD/WARN above)"
  exit 1
fi

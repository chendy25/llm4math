#!/usr/bin/env bash
# Single Kimina for math+physics: PROJECT_DIR=Physlib (includes Mathlib), port 8001.
# LeanSearch retrieval stays on :8000. Kill any previous Kimina on :8001 first.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

PHYS_DIR="${PHYS_DIR:-/tmp/chendongyang/lean-local/physlib}"
if [[ ! -f "$PHYS_DIR/Physlib.lean" && ! -f "$PHYS_DIR/lakefile.lean" ]]; then
  echo "Physlib project missing at $PHYS_DIR" >&2
  exit 1
fi
if [[ ! -x /tmp/kimina_repl_4291 ]]; then
  if [[ -x "$ROOT/repl/.lake/build/bin/repl" ]]; then
    cp -f "$ROOT/repl/.lake/build/bin/repl" /tmp/kimina_repl_4291
    chmod +x /tmp/kimina_repl_4291
  else
    echo "REPL binary missing; build repl with lean-toolchain v4.29.1 first" >&2
    exit 1
  fi
fi

set -a
# shellcheck disable=SC1091
. ./.env
set +a
export LEAN_SERVER_PORT=8001
export LEAN_SERVER_LEAN_VERSION=v4.29.1
export LEAN_SERVER_PROJECT_DIR="$PHYS_DIR"
export LEAN_SERVER_REPL_PATH=/tmp/kimina_repl_4291
export PATH="${HOME}/.elan/toolchains/leanprover--lean4---v4.29.1/bin:${HOME}/.elan/bin:${PATH}"
export PYTHONUNBUFFERED=1
LOG="${KIMINA_LOG:-/tmp/kimina_physlib_8001.log}"

# stop previous kimina on 8001 if any
if curl -fsS -m 2 "http://127.0.0.1:${LEAN_SERVER_PORT}/health" >/dev/null 2>&1; then
  echo "already healthy on :${LEAN_SERVER_PORT} — restarting with Physlib project"
  pkill -f '/mnt/afs_reason/chendongyang/conda/kimina/bin/python -u -m server' || true
  sleep 2
fi

setsid /mnt/afs_reason/chendongyang/conda/kimina/bin/python -u -m server </dev/null >>"$LOG" 2>&1 &
echo "started pid=$! port=${LEAN_SERVER_PORT} project=${LEAN_SERVER_PROJECT_DIR} log=$LOG"

#!/usr/bin/env bash
# Start Kimina Lean Server (detached). Uses /tmp REPL + toolchain + mathlib.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
set -a
# shellcheck disable=SC1091
. ./.env
set +a
export PATH="/tmp/lean-v4.29.0-rc6/bin:${HOME}/.elan/bin:/mnt/afs_reason/chendongyang/conda/kimina/bin:${PATH}"
export PYTHONUNBUFFERED=1
LOG="${KIMINA_LOG:-/tmp/kimina_server.log}"
if curl -fsS -m 2 "http://127.0.0.1:${LEAN_SERVER_PORT:-8000}/health" >/dev/null 2>&1; then
  echo "already healthy on :${LEAN_SERVER_PORT:-8000}"
  exit 0
fi
if [[ ! -x /tmp/kimina_repl ]]; then
  cp -f "$ROOT/repl/.lake/build/bin/repl" /tmp/kimina_repl
  chmod +x /tmp/kimina_repl
fi
setsid /mnt/afs_reason/chendongyang/conda/kimina/bin/python -u -m server </dev/null >>"$LOG" 2>&1 &
echo "started pid=$! log=$LOG"

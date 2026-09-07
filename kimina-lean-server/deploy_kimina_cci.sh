#!/usr/bin/env bash
# Stage Physlib oleans onto local /tmp, then serve Kimina on :8001 (foreground).
# AFS oleans are too slow for REPL; copy once per pod then run from /tmp.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
RUNTIME="$ROOT/.runtime"
AFS_PHYS="${AFS_PHYS:-$RUNTIME/physlib}"
AFS_REPL="${AFS_REPL:-$RUNTIME/kimina_repl_4291}"
AFS_LAKE_ENV="${AFS_LAKE_ENV:-$RUNTIME/kimina_physlib_lake_env.txt}"
PHYS_DIR="${PHYS_DIR:-/tmp/chendongyang/lean-local/physlib}"
REPL_DST="${REPL_DST:-/tmp/kimina_repl_4291}"
LAKE_ENV_DST="${LAKE_ENV_DST:-/tmp/kimina_physlib_lake_env.txt}"
PY="${PY:-/mnt/afs_reason/chendongyang/conda/kimina/bin/python}"
PORT="${PORT:-8001}"
LOG_DIR="${LOG_DIR:-$ROOT/logs}"
mkdir -p "$LOG_DIR" "$(dirname "$PHYS_DIR")"
LOG="$LOG_DIR/cci_kimina_$(date +%Y%m%d_%H%M%S).log"

if [[ ! -x "$PY" ]]; then
  chmod +x "$PY" 2>/dev/null || true
fi
if [[ ! -x "$PY" ]]; then
  echo "kimina python missing: $PY" >&2
  exit 1
fi
echo "[deploy] waiting for AFS Physlib backup at $AFS_PHYS"
for i in $(seq 1 240); do
  if [[ -f "$AFS_PHYS/.rsync_complete" ]] && { [[ -f "$AFS_PHYS/lakefile.lean" ]] || [[ -f "$AFS_PHYS/Physlib.lean" ]]; }; then
    break
  fi
  echo "[deploy] AFS Physlib not ready yet (${i}/240)"
  sleep 30
done
if [[ ! -f "$AFS_PHYS/.rsync_complete" ]]; then
  echo "AFS Physlib backup never became ready: $AFS_PHYS" >&2
  exit 1
fi
if [[ ! -x "$AFS_REPL" && ! -f "$AFS_REPL" ]]; then
  echo "AFS REPL missing: $AFS_REPL" >&2
  exit 1
fi

echo "[deploy] staging Physlib $AFS_PHYS -> $PHYS_DIR"
mkdir -p "$PHYS_DIR"
if command -v rsync >/dev/null 2>&1; then
  rsync -a --delete "$AFS_PHYS/" "$PHYS_DIR/"
else
  echo "[deploy] rsync missing; using cp -a (image has no rsync)"
  # Fresh dest so leftover files cannot shadow oleans.
  rm -rf "$PHYS_DIR"
  mkdir -p "$PHYS_DIR"
  cp -a "$AFS_PHYS/." "$PHYS_DIR/"
fi
cp -f "$AFS_REPL" "$REPL_DST"
chmod +x "$REPL_DST"
if [[ -f "$AFS_LAKE_ENV" ]]; then
  cp -f "$AFS_LAKE_ENV" "$LAKE_ENV_DST"
fi

export HOME="${HOME:-/mnt/afs_reason/chendongyang}"
export PATH="/mnt/afs_reason/chendongyang/.elan/toolchains/leanprover--lean4---v4.29.1/bin:/mnt/afs_reason/chendongyang/.elan/bin:${PATH}"
export PYTHONUNBUFFERED=1
export LEAN_SERVER_HOST=0.0.0.0
export LEAN_SERVER_PORT="$PORT"
export LEAN_SERVER_LOG_LEVEL=INFO
export LEAN_SERVER_ENVIRONMENT=prod
export LEAN_SERVER_LEAN_VERSION=v4.29.1
export LEAN_SERVER_MAX_REPLS="${LEAN_SERVER_MAX_REPLS:-6}"
export LEAN_SERVER_MAX_REPL_USES=-1
# Kimina applies this as RLIMIT_AS (virtual address space), not RSS.
# Lean mmaps the full ~8G Physlib/Mathlib oleans; 4G kills the REPL
# immediately and /verify returns empty JSON / HTTP 500.
export LEAN_SERVER_MAX_REPL_MEM="${LEAN_SERVER_MAX_REPL_MEM:-24G}"
export LEAN_SERVER_MAX_WAIT=180
export LEAN_SERVER_INIT_REPLS='{}'
export LEAN_SERVER_API_KEY="${LEAN_SERVER_API_KEY:-reform-lean-key}"
export LEAN_SERVER_REPL_PATH="$REPL_DST"
export LEAN_SERVER_PROJECT_DIR="$PHYS_DIR"
export KIMINA_LAKE_ENV_FILE="$LAKE_ENV_DST"
export KIMINA_USE_LAKE_ENV=0

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy || true

echo "[deploy] hostname=$(hostname) ips=$(hostname -I 2>/dev/null || true)"
echo "[deploy] python=$PY port=$PORT project=$PHYS_DIR repls=$LEAN_SERVER_MAX_REPLS mem=$LEAN_SERVER_MAX_REPL_MEM"
echo "[deploy] log=$LOG"
echo "$LOG" > "$LOG_DIR/cci_kimina.logpath"

# Cold import Physlib is several minutes; warm once so the first client is not 500/timeout.
(
  for i in $(seq 1 60); do
    if curl -sf --max-time 2 "http://127.0.0.1:${PORT}/health" >/dev/null; then
      break
    fi
    sleep 2
  done
  VERIFY_URL="http://127.0.0.1:${PORT}" VERIFY_KEY="${LEAN_SERVER_API_KEY}" \
    "$PY" - <<'PY' >>"$LOG_DIR/cci_kimina_warmup.log" 2>&1 || true
import json, os, time, urllib.request
url = os.environ["VERIFY_URL"].rstrip("/") + "/verify"
key = os.environ["VERIFY_KEY"]
code = "import Physlib\n#check Nat\nexample : True := trivial\n"
payload = json.dumps({
    "codes": [{"custom_id": "warmup", "proof": code}],
    "timeout": 600,
    "disable_cache": False,
}).encode()
headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
for i in (1, 2):
    t0 = time.time()
    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=660) as r:
        body = r.read().decode()
    print(f"warmup{i} elapsed={time.time()-t0:.1f}s {body[:240]}", flush=True)
PY
) &

cd "$ROOT"
exec "$PY" -u -m server 2>&1 | tee "$LOG"

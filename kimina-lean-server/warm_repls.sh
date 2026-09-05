#!/usr/bin/env bash
# Warm the Kimina REPL pool so eval runs do not pay a cold `import Mathlib`
# (~40s+) per request. Kimina keys REPL reuse on the import lines only, so one
# warm REPL per distinct header is what matters; N should match the concurrency
# the eval will drive (VERIFY_MAX_CONCURRENCY).
set -euo pipefail

PORT=${PORT:-8001}
KEY=${VERIFY_KEY:-reform-lean-key}
N=${N:-4}
HEADER=${HEADER:-import Mathlib}

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

echo "warming $N REPL(s) with header: $HEADER"
pids=()
for i in $(seq 1 "$N"); do
  (
    body=$(printf '%s\n\ntheorem warm_%s : 1 + 1 = 2 := by sorry' "$HEADER" "$i")
    payload=$(python3 -c '
import json,sys
print(json.dumps({"codes":[{"custom_id":sys.argv[1],"proof":sys.argv[2]}],"timeout":600}))' "warm$i" "$body")
    t0=$SECONDS
    code=$(curl -s -m 900 -o "/tmp/kimina_warm_$i.json" -w '%{http_code}' \
      -X POST "http://127.0.0.1:${PORT}/verify" \
      -H 'Content-Type: application/json' \
      -H "Authorization: Bearer ${KEY}" \
      -d "$payload")
    echo "  warm$i http=$code in $((SECONDS - t0))s"
  ) &
  pids+=($!)
  # Stagger so N cold Mathlib loads do not thrash the 8-CPU cgroup at once.
  sleep 20
done
for p in "${pids[@]}"; do wait "$p"; done

echo "--- pool state ---"
grep -oE '# Free = [0-9]+ \| # Busy = [0-9]+ \| # Max = [0-9]+' \
  "${KIMINA_LOG:-/tmp/kimina_physlib_8001.log}" | tail -1 || true

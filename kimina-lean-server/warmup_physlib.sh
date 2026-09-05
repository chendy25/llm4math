#!/usr/bin/env bash
# Warm Kimina REPL header cache with `import Physlib` (run on the Kimina host).
# Cold start can take several minutes; a second call should be near-instant.
set -euo pipefail
BASE="${VERIFY_URL:-http://127.0.0.1:8001}"
KEY="${VERIFY_KEY:-reform-lean-key}"
BASE="${BASE%/}"
case "$BASE" in
  */verify) URL="$BASE" ;;
  *) URL="${BASE}/verify" ;;
esac
export WARMUP_URL="$URL" WARMUP_KEY="$KEY"
python3 - <<'PY'
import json, os, time, urllib.request
url = os.environ["WARMUP_URL"]
key = os.environ["WARMUP_KEY"]
code = "import Physlib\n#check Nat\nexample : True := trivial\n"
payload = json.dumps({
    "codes": [{"custom_id": "warmup", "proof": code}],
    "timeout": 600,
    "disable_cache": False,
}).encode()
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {key}",
}
for i in (1, 2):
    t0 = time.time()
    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=660) as r:
        body = r.read().decode()
    print(f"warmup{i} elapsed={time.time()-t0:.1f}s {body[:240]}")
PY

#!/usr/bin/env bash
# Run the same eval config N times and report mean/spread.
#
#   REPEATS=3 SUITES=aime MODEL_URL=http://10.x.x.x:8000/v1 \
#     bash scripts/eval_repeat.sh
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
EVAL_DIR=$(cd "$HERE/.." && pwd)/eval
EVAL_SCRIPT=${EVAL_SCRIPT:-eval_rollout.sh}
REPEATS=${REPEATS:-3}
TAG=${TAG:-repeat}
STAMP=$(date +%Y%m%d_%H%M%S)
GROUP="$EVAL_DIR/runs/${TAG}_${STAMP}"
mkdir -p "$GROUP"
echo "group: $GROUP"

# Runs share one warm Lean REPL, so they must not overlap.
for i in $(seq 1 "$REPEATS"); do
  out="$GROUP/run$i"
  mkdir -p "$out"
  echo "==== repeat $i/$REPEATS -> $out ===="
  OUT_DIR="$out" bash "$HERE/$EVAL_SCRIPT" > "$out.driver.log" 2>&1 || true
  grep -hE '^\[[a-z0-9_-]+\] n=' "$out.driver.log" || echo "  (no summary line)"
done

echo
echo "==== aggregate ===="
python3 - "$GROUP" <<'PY'
import json, statistics, sys
from pathlib import Path

group = Path(sys.argv[1])
per_suite: dict[str, list[dict]] = {}
for run in sorted(group.glob("run*")):
    for f in sorted(run.glob("*.jsonl")):
        rows = [json.loads(l) for l in f.open()]
        if not rows:
            continue
        infra = sum(1 for r in rows if r.get("verify_error"))
        per_suite.setdefault(f.stem, []).append({
            "n": len(rows),
            "success": sum(1 for r in rows if r.get("success")),
            "syntax": sum(1 for r in rows if r.get("syntax_pass")),
            "verify_error": infra,
        })

for suite, runs in per_suite.items():
    n = runs[0]["n"]
    succ = [r["success"] for r in runs]
    syn = [r["syntax"] for r in runs]
    infra = [r["verify_error"] for r in runs]
    sd = statistics.stdev(succ) if len(succ) > 1 else 0.0
    print(f"[{suite}] n={n} over {len(runs)} run(s)")
    print(f"  success per run: {succ}  mean={statistics.mean(succ):.1f} sd={sd:.1f} "
          f"rate={statistics.mean(succ)/n:.3f}")
    print(f"  syntax  per run: {syn}   mean={statistics.mean(syn):.1f}")
    print(f"  verify_error:    {infra}")
PY

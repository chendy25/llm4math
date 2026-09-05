# SFT collect status

Updated: 2026-07-27 (paused full run; pilot plan)

## Paused
- Full math 8k / physics 6.5k stopped.
- Partial math archived under `archive/sft_math_collect_partial_*.jsonl`.

## Pipeline changes
- Lazy image load (no batch PIL preload)
- `max_tool_calls=3`
- Success → main jsonl; failures → `*/failed/*.failed.jsonl`
- Pilot batches + sharded workers

## Current runs
1. Physics smoke10 → `physics/sft_physics_mm_smoke10.jsonl`
2. Math pilot500 ×4 shards → `math/sft_math_pilot500_shardXX.jsonl`
3. Physics pilot100 ×2 shards after smoke OK → `physics/sft_physics_mm_pilot100_shardXX.jsonl`

Monitor:
```bash
wc -l math/sft_math_pilot500_shard*.jsonl physics/sft_physics_mm_*.jsonl
tail -f logs/collect_physics_smoke10.log logs/collect_math_shard*.log
```

## Auto
- Watcher will start physics pilot100 ×2 shards after smoke10 exits (`logs/watch_physics_smoke_then_pilot.pid`).

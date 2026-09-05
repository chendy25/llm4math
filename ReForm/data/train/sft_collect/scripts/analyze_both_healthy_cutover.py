#!/usr/bin/env python3
"""Analyze when LeanSearch vs DashScope died; propose cutover without deleting."""
from __future__ import annotations

import json
import os
import re
import glob
from collections import Counter
from datetime import datetime, timezone, timedelta

TZ = timezone(timedelta(hours=8))
NO = "(No results found.)"
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # sft_collect/
TRAIN = os.path.dirname(BASE)  # data/train/


def ts(path: str) -> datetime:
    return datetime.fromtimestamp(os.path.getmtime(path), TZ)


def last_run_meta(log_path: str) -> dict:
    run = -1
    writes_per_run: Counter = Counter()
    processing: dict = {}
    skipped: dict = {}
    pending_skip = None
    first_arrear_run = None
    writes_before_arrear = None
    first_arrear_prog = None
    prog = None
    with open(log_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("Resume:"):
                m = re.search(r"skipped (\d+)", line)
                pending_skip = int(m.group(1)) if m else None
            if line.startswith("Processing "):
                run += 1
                m = re.search(r"Processing (\d+) item", line)
                processing[run] = int(m.group(1)) if m else None
                if pending_skip is not None:
                    skipped[run] = pending_skip
                    pending_skip = None
            m = re.match(r"\[(\d+)/(\d+)\]", line)
            if m:
                prog = int(m.group(1))
            if "-> domain=" in line and "success=" in line and run >= 0:
                if first_arrear_run is None:
                    writes_per_run[run] += 1
            if "Arrearage" in line and first_arrear_run is None:
                first_arrear_run = run
                writes_before_arrear = writes_per_run[run]
                first_arrear_prog = prog
    return {
        "n_runs": run + 1,
        "arrear_run": first_arrear_run,
        "writes_before_arrear": writes_before_arrear,
        "processing": processing.get(first_arrear_run) if first_arrear_run is not None else None,
        "skipped": skipped.get(first_arrear_run, 0) if first_arrear_run is not None else 0,
        "first_arrear_prog": first_arrear_prog,
        "writes_per_run": dict(writes_per_run),
    }


def load_rows(path: str) -> list[dict]:
    rows = []
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        asst = next(
            (
                m.get("content")
                for m in (r.get("messages") or [])
                if m.get("role") == "assistant" and isinstance(m.get("content"), str)
            ),
            "",
        ) or ""
        rows.append(
            {
                "success": bool(r.get("success")),
                "syntax_pass": r.get("syntax_pass"),
                "no_results": NO in asst,
                "dataset": r.get("dataset_name"),
                "qid": str(r.get("question_id")),
                "domain": r.get("retrieval_domain"),
                "tools": list(r.get("tools_used") or []),
                "num_tool_calls": r.get("num_tool_calls") or 0,
            }
        )
    return rows


def onset_streak(flags: list[bool], k: int = 10) -> int | None:
    run = 0
    for i, f in enumerate(flags):
        run = run + 1 if f else 0
        if run >= k:
            return i - k + 1
    return None


def onset_roll(flags: list[bool], window: int = 20, thresh: int = 15) -> int | None:
    for i in range(0, len(flags) - window + 1):
        if sum(flags[i : i + window]) >= thresh:
            return i
    return None


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {
            "n": 0,
            "success": 0,
            "fail": 0,
            "rate": 0.0,
            "syntax_true": 0,
            "no_results": 0,
            "tools": {},
        }
    ok = sum(1 for r in rows if r["success"])
    syn = sum(1 for r in rows if r["syntax_pass"] is True)
    no = sum(1 for r in rows if r["no_results"])
    tools: Counter = Counter()
    for r in rows:
        for t in r["tools"]:
            tools[t] += 1
    return {
        "n": n,
        "success": ok,
        "fail": n - ok,
        "rate": ok / n * 100,
        "syntax_true": syn,
        "no_results": no,
        "tools": dict(tools),
    }


def fmt_sum(label: str, rows_lists: list[list[dict]]) -> dict:
    rows = [r for rs in rows_lists for r in rs]
    s = summarize(rows)
    no_pct = (s["no_results"] / s["n"] * 100) if s["n"] else 0.0
    print(
        f"{label}: n={s['n']} success={s['success']} fail={s['fail']} "
        f"rate={s['rate']:.1f}% syntax_pass={s['syntax_true']} "
        f"no_results={s['no_results']} ({no_pct:.1f}%) tools={s['tools']}"
    )
    return s


def main() -> None:
    os.chdir(TRAIN)
    cut_plan = []
    search_times = []
    api_times = []

    print("=== Refined per-shard timeline (interp within final run only) ===\n")

    for kind, pattern in [
        ("math", "sft_collect/math/sft_math_full8k_shard*.jsonl"),
        ("physics", "sft_collect/physics/sft_physics_mm_full_shard*.jsonl"),
    ]:
        for path in sorted(glob.glob(pattern)):
            if ".bak_" in path:
                continue
            sid = re.search(r"shard(\d+)", path).group(1)
            log = f"sft_collect/logs/collect_{kind}_shard{sid}.log"
            pid = f"sft_collect/logs/collect_{kind}_shard{sid}.pid"
            t_launch, t_end = ts(pid), ts(path)
            meta = last_run_meta(log)
            rows = load_rows(path)
            n = len(rows)
            S = meta["skipped"] or 0
            W = meta["writes_before_arrear"] or 0
            flags = [r["no_results"] for r in rows]
            st = onset_streak(flags, 10)
            rl = onset_roll(flags, 20, 15)
            if kind == "physics" and st is None:
                st = onset_roll(flags, 20, 8)
            onset = st if st is not None else rl

            t_search = None
            if onset is not None and W > 1:
                final_start = n - W
                if onset >= final_start:
                    write_i = onset - final_start
                    t_search = t_launch + (t_end - t_launch) * (write_i / (W - 1))
                else:
                    t_search = t_launch
            elif onset is not None and n > 1:
                t_search = t_launch + (t_end - t_launch) * (onset / (n - 1))

            t_api = t_end
            if t_search:
                search_times.append(t_search)
            api_times.append(t_api)

            print(
                f"{kind} shard{sid}: n={n} S_skip={S} W_before_arrear={W} "
                f"S+W={S+W} drift={n - (S+W)}"
            )
            print(
                f"  launch={t_launch:%Y-%m-%d %H:%M:%S}  "
                f"last_write/API≈{t_api:%Y-%m-%d %H:%M:%S}"
            )
            print(
                f"  no_results onset streak10={st} roll20/15={rl} chosen={onset}"
            )
            if t_search:
                print(
                    f"  LeanSearch≈{t_search:%Y-%m-%d %H:%M:%S}  (file_idx={onset})"
                )
            else:
                print("  LeanSearch≈N/A (no clear empty-onset)")
            who = (
                "LeanSearch"
                if (t_search and t_search < t_api)
                else ("DashScope" if not t_search else "DashScope")
            )
            print(f"  who_first≈{who}")
            cut_plan.append(
                {
                    "kind": kind,
                    "sid": sid,
                    "path": path,
                    "onset": onset,
                    "t_search": t_search,
                    "t_api": t_api,
                    "t_launch": t_launch,
                    "S": S,
                    "W": W,
                    "rows": rows,
                }
            )
            print()

    T_search_min = min(search_times)
    T_search_max = max(search_times)
    T_search_med = sorted(search_times)[len(search_times) // 2]
    T_api_min = min(api_times)
    T_api_max = max(api_times)
    T_cut = T_search_min

    print(
        f"LeanSearch death estimates: min={T_search_min:%Y-%m-%d %H:%M:%S}  "
        f"med={T_search_med:%Y-%m-%d %H:%M:%S}  max={T_search_max:%Y-%m-%d %H:%M:%S}"
    )
    print(
        f"DashScope death (last writes): min={T_api_min:%Y-%m-%d %H:%M:%S}  "
        f"max={T_api_max:%Y-%m-%d %H:%M:%S}"
    )
    print(
        f"VERDICT: LeanSearch died FIRST, "
        f"~{T_search_min:%Y-%m-%d %H:%M}–{T_search_max:%Y-%m-%d %H:%M}; "
        f"DashScope ~{((T_api_min - T_search_max).total_seconds()/3600):.1f}–"
        f"{((T_api_min - T_search_min).total_seconds()/3600):.1f}h later on "
        f"{T_api_min:%Y-%m-%d} ~{T_api_min:%H:%M}–{T_api_max:%H:%M}"
    )
    print(
        f"\nProposed global both-healthy end "
        f"(conservative=earliest LeanSearch death): {T_cut:%Y-%m-%d %H:%M:%S %z}"
    )

    final_cuts = {}
    for item in cut_plan:
        kind, sid = item["kind"], item["sid"]
        rows, n, W = item["rows"], len(item["rows"]), item["W"]
        onset = item["onset"]
        t_launch, t_api = item["t_launch"], item["t_api"]

        if kind == "math" and onset is not None:
            c = onset
            reason = "math_no_results_streak10"
        else:
            if W > 1 and t_api > t_launch:
                if T_cut <= t_launch:
                    c = n - W
                elif T_cut >= t_api:
                    c = n
                else:
                    frac = (T_cut - t_launch) / (t_api - t_launch)
                    write_i = int(round(frac * (W - 1)))
                    c = (n - W) + write_i
            else:
                c = onset if onset is not None else n
            reason = (
                "physics_time_aligned_to_earliest_math_search_death"
                if kind == "physics"
                else "fallback"
            )
        c = max(0, min(n, c))
        final_cuts[(kind, sid)] = {
            **item,
            "cut": c,
            "reason": reason,
            "n": n,
        }

    print("\n=== CUT INDICES (keep rows[0:cut], discard rows[cut:]) ===")
    for k, v in sorted(final_cuts.items()):
        print(
            f"  {k[0]} shard{k[1]}: keep 0..{v['cut']-1} ({v['cut']} rows) "
            f"discard {v['n']-v['cut']}  reason={v['reason']}"
        )

    print("\n========== REGIME STATS (written trajectories only) ==========")
    keep_math, disc_math, keep_phys, disc_phys = [], [], [], []
    for (kind, sid), v in sorted(final_cuts.items()):
        rows, c = v["rows"], v["cut"]
        if kind == "math":
            keep_math.append(rows[:c])
            disc_math.append(rows[c:])
        else:
            keep_phys.append(rows[:c])
            disc_phys.append(rows[c:])

    print("-- MATH --")
    agg_keep_m = fmt_sum("  BOTH healthy (keep)", keep_math)
    agg_disc_m = fmt_sum("  AFTER cut (LeanSearch bad, API still ok)", disc_math)
    print("-- PHYSICS --")
    agg_keep_p = fmt_sum("  BOTH healthy (keep)", keep_phys)
    agg_disc_p = fmt_sum("  AFTER cut (time-aligned / degraded)", disc_phys)
    print("-- ALL --")
    agg_keep = fmt_sum("  BOTH healthy (keep)", keep_math + keep_phys)
    agg_disc = fmt_sum("  AFTER cut discard-candidate", disc_math + disc_phys)

    print("\n========== UNWRITTEN due to DashScope Arrearage (not in JSONL) ==========")
    math_unwritten = phys_unwritten = 0
    for kind in ["math", "physics"]:
        for path in sorted(glob.glob(f"sft_collect/logs/collect_{kind}_shard*.log")):
            first = False
            err = 0
            for line in open(path, encoding="utf-8", errors="ignore"):
                if "Arrearage" in line:
                    first = True
                if first and "-> Error:" in line:
                    err += 1
            print(f"  {os.path.basename(path)}: unwritten_errors_after_arrear≈{err}")
            if kind == "math":
                math_unwritten += err
            else:
                phys_unwritten += err
    print(f"  MATH unwritten total≈{math_unwritten}")
    print(f"  PHYSICS unwritten total≈{phys_unwritten}")
    print(
        "  NOTE: never written; resume retries automatically. "
        "Only AFTER-cut WRITTEN rows need discarding before resume."
    )

    print("\n========== DISCARD PLAN (do NOT delete yet) ==========")
    total_discard = total_keep = 0
    shard_plan = []
    for (kind, sid), v in sorted(final_cuts.items()):
        d = v["n"] - v["cut"]
        k = v["cut"]
        total_discard += d
        total_keep += k
        ds = summarize(v["rows"][v["cut"] :])
        ks = summarize(v["rows"][: v["cut"]])
        print(
            f"  {kind} shard{sid}: KEEP {ks['n']} "
            f"(ok={ks['success']}, fail={ks['fail']}, rate={ks['rate']:.1f}%) | "
            f"DISCARD {ds['n']} "
            f"(ok={ds['success']}, fail={ds['fail']}, rate={ds['rate']:.1f}%)"
        )
        shard_plan.append(
            {
                "kind": kind,
                "shard": sid,
                "path": v["path"],
                "n": v["n"],
                "keep": k,
                "discard": d,
                "reason": v["reason"],
                "t_search": v["t_search"].isoformat() if v["t_search"] else None,
                "t_api": v["t_api"].isoformat(),
                "keep_success": ks["success"],
                "keep_fail": ks["fail"],
                "discard_success": ds["success"],
                "discard_fail": ds["fail"],
            }
        )
    print(f"  TOTAL KEEP written={total_keep}  DISCARD written={total_discard}")

    print("\n========== SENSITIVITY: cut at median LeanSearch death ==========")
    alt_keep = alt_disc = alt_keep_ok = alt_disc_ok = 0
    for (kind, sid), v in sorted(final_cuts.items()):
        n, W, rows = v["n"], v["W"], v["rows"]
        t_launch, t_api = v["t_launch"], v["t_api"]
        if W > 1 and t_api > t_launch:
            if T_search_med <= t_launch:
                c = n - W
            elif T_search_med >= t_api:
                c = n
            else:
                frac = (T_search_med - t_launch) / (t_api - t_launch)
                c = (n - W) + int(round(frac * (W - 1)))
        else:
            c = v["cut"]
        c = max(0, min(n, c))
        alt_keep += c
        alt_disc += n - c
        alt_keep_ok += sum(1 for r in rows[:c] if r["success"])
        alt_disc_ok += sum(1 for r in rows[c:] if r["success"])
        print(f"  {kind} shard{sid}: keep={c} discard={n-c}")
    print(
        f"  ALT TOTAL keep={alt_keep} (ok={alt_keep_ok}) "
        f"discard={alt_disc} (ok={alt_disc_ok})"
    )

    # Abruptness check: no_results rate in windows around math onset
    print("\n========== MATH no_results around onset (abruptness) ==========")
    for (kind, sid), v in sorted(final_cuts.items()):
        if kind != "math" or v["onset"] is None:
            continue
        rows, o = v["rows"], v["onset"]
        for tag, sl in [
            ("[-50,-1]", rows[max(0, o - 50) : o]),
            ("[0,+50]", rows[o : o + 50]),
        ]:
            if not sl:
                continue
            no = sum(1 for r in sl if r["no_results"])
            ok = sum(1 for r in sl if r["success"])
            print(
                f"  math shard{sid} {tag}: n={len(sl)} no_results={no} "
                f"({no/len(sl)*100:.0f}%) success={ok}/{len(sl)} "
                f"({ok/len(sl)*100:.0f}%)"
            )

    report = {
        "verdict": {
            "who_first": "LeanSearch",
            "leansearch_death_min": T_search_min.isoformat(),
            "leansearch_death_med": T_search_med.isoformat(),
            "leansearch_death_max": T_search_max.isoformat(),
            "dashscope_death_min": T_api_min.isoformat(),
            "dashscope_death_max": T_api_max.isoformat(),
            "conservative_T_cut": T_cut.isoformat(),
            "method": (
                "LeanSearch death = first 10-streak of '(No results found.)' "
                "in assistant messages; time interpolated within final run "
                "writes between pid mtime and jsonl mtime. "
                "DashScope death ≈ jsonl last-write mtime (first Arrearage "
                "immediately after; 0 writes after)."
            ),
            "notes": [
                "Rows have no timestamps; wall-clock times are interpolated.",
                "leansearch_tool swallows errors -> empty -> '(No results found.)'.",
                "Physics remote share same host; cut aligned to earliest math search death.",
                "Files NOT deleted; this is analysis only.",
            ],
        },
        "regime_written": {
            "both_healthy_keep": agg_keep,
            "after_cut_discard_candidate": agg_disc,
            "math_keep": agg_keep_m,
            "math_discard": agg_disc_m,
            "physics_keep": agg_keep_p,
            "physics_discard": agg_disc_p,
        },
        "unwritten_arrearage": {
            "math": math_unwritten,
            "physics": phys_unwritten,
        },
        "shards": shard_plan,
        "sensitivity_median_cut": {
            "keep": alt_keep,
            "keep_success": alt_keep_ok,
            "discard": alt_disc,
            "discard_success": alt_disc_ok,
        },
    }
    out = "sft_collect/logs/both_healthy_cutover_report.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()

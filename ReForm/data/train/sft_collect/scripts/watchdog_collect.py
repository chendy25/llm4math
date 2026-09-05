#!/usr/bin/env python3
"""Stop SFT collect workers if DashScope is in arrears or search/verify is down."""

from __future__ import annotations

import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline_guard import (
    check_llm,
    check_search,
    check_verify,
    is_arrears_text,
    is_soft_dashscope_text,
)

LOGDIR = ROOT / "sft_collect" / "logs"
HALT_FILE = LOGDIR / "pipeline_halted.reason"
WATCH_LOG = LOGDIR / "watchdog_collect.log"
PID_GLOBS = (
    "collect_math_shard*.pid",
    "collect_physics_shard*.pid",
    "collect_math_*_shard*.pid",
    "collect_physics_*_shard*.pid",
    "harvest_physlib_kb.pid",
)
LOG_GLOBS = (
    "collect_math_qwen38_27b_shard*.log",
    "collect_physics_qwen38_27b_shard*.log",
    "harvest_physlib_kb.log",
)
INTERVAL = int(os.environ.get("WATCHDOG_INTERVAL", "45"))
FAIL_LIMIT = int(os.environ.get("WATCHDOG_FAIL_LIMIT", "2"))


def log(msg: str) -> None:
    line = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)


def _pids_from_files() -> list[int]:
    pids: list[int] = []
    for pat in PID_GLOBS:
        for path in LOGDIR.glob(pat):
            try:
                pid = int(path.read_text().strip())
            except (OSError, ValueError):
                continue
            if pid > 1:
                pids.append(pid)
    return pids


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\x00", b" ").decode("utf-8", "replace")
    except OSError:
        return ""


def collect_pids() -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for pid in _pids_from_files():
        if pid in seen or not _alive(pid):
            continue
        cmd = _cmdline(pid)
        if "watchdog_collect" in cmd:
            continue
        seen.add(pid)
        out.append(pid)
    proc = Path("/proc")
    if proc.is_dir():
        for entry in proc.iterdir():
            if not entry.name.isdigit():
                continue
            pid = int(entry.name)
            if pid in seen:
                continue
            cmd = _cmdline(pid)
            if "watchdog_collect" in cmd:
                continue
            if "run_collect_sft_data.py" in cmd or "harvest_physlib_kb.py" in cmd:
                seen.add(pid)
                out.append(pid)
    return out


def stop_pipeline(reason: str) -> None:
    LOGDIR.mkdir(parents=True, exist_ok=True)
    HALT_FILE.write_text(reason + "\n", encoding="utf-8")
    pids = collect_pids()
    log(f"HALT {reason}; killing {len(pids)} worker(s): {pids}")
    for pid in pids:
        try:
            os.killpg(pid, signal.SIGTERM)
        except OSError:
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                pass
    time.sleep(2)
    for pid in collect_pids():
        try:
            os.killpg(pid, signal.SIGKILL)
        except OSError:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
    log("pipeline stopped")


class LogTailer:
    """Only inspect bytes appended after watchdog start (ignore historical arrears)."""

    def __init__(self) -> None:
        self.offsets: dict[Path, int] = {}
        self._refresh_offsets(init=True)

    def _refresh_offsets(self, *, init: bool) -> None:
        for pat in LOG_GLOBS:
            for path in LOGDIR.glob(pat):
                if path in self.offsets:
                    continue
                try:
                    self.offsets[path] = path.stat().st_size if init else 0
                except OSError:
                    self.offsets[path] = 0

    def scan(self) -> str | None:
        self._refresh_offsets(init=False)
        for path, start in list(self.offsets.items()):
            try:
                size = path.stat().st_size
            except OSError:
                continue
            if size < start:
                start = 0
            if size <= start:
                continue
            try:
                with path.open("rb") as f:
                    f.seek(start)
                    chunk = f.read().decode("utf-8", "ignore")
            except OSError:
                continue
            self.offsets[path] = size
            if is_arrears_text(chunk) or "PIPELINE_HALT" in chunk:
                return f"arrears/halt marker in {path.name}"
        return None


def probe(tailer: LogTailer) -> tuple[str, str, bool]:
    """Return (status, reason, immediate). status is ok|soft|fail."""
    hit = tailer.scan()
    if hit:
        return "fail", hit, True
    search_url = os.environ.get("SEARCH_URL", "http://10.120.5.252:8000")
    verify_url = os.environ.get("VERIFY_URL", "http://10.120.1.3:8001")
    ok, msg = check_search(search_url)
    if not ok:
        return "fail", msg, False
    ok, msg = check_verify(verify_url)
    if not ok:
        return "fail", msg, False
    ok, msg = check_llm()
    if ok:
        return "ok", msg, False
    if is_arrears_text(msg):
        return "fail", msg, True
    if is_soft_dashscope_text(msg) or msg.startswith("llm soft"):
        return "soft", msg, False
    return "fail", msg, False


def main() -> int:
    LOGDIR.mkdir(parents=True, exist_ok=True)
    if HALT_FILE.exists():
        HALT_FILE.unlink()
    log(
        f"watchdog start interval={INTERVAL}s fail_limit={FAIL_LIMIT} "
        f"search={os.environ.get('SEARCH_URL', 'http://10.120.5.252:8000')} "
        f"verify={os.environ.get('VERIFY_URL', 'http://10.120.1.3:8001')} "
        f"llm={os.environ.get('LLM_BASE_URL') or os.environ.get('TOKENHUB_BASE_URL')}"
    )
    tailer = LogTailer()
    streak = 0
    last_reason = ""
    while True:
        workers = collect_pids()
        if not workers:
            log("no collect/harvest workers left; watchdog exit")
            return 0
        status, reason, immediate = probe(tailer)
        if status == "ok":
            if streak:
                log(f"recovered after {streak} fail(s)")
            streak = 0
            last_reason = ""
        elif status == "soft":
            log(f"SOFT ignore (no halt): {reason}")
        else:
            streak += 1
            last_reason = reason
            log(f"FAIL {streak}/{FAIL_LIMIT} immediate={immediate}: {reason}")
            if immediate or streak >= FAIL_LIMIT:
                stop_pipeline(reason)
                return 2
        time.sleep(INTERVAL)


if __name__ == "__main__":
    raise SystemExit(main())

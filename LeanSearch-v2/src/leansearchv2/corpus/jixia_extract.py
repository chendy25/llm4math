"""Step 1a: run jixia on every `*.lean` under a corpus source tree, producing
one `.sym.json` + `.decl.json` + `.mod.json` triple per file under
`<workdir>/raw/<corpus-name>/`.

Works for Mathlib (`source_subdir=Mathlib` under `external/Mathlib4`) and
Physlib (`source_subdir=.` walks the lake root directly), as well as any
other Lean 4 library registered in `config.yaml::paths.corpora`.

Requires:
- `lake` + `lean` on PATH (point at the elan-installed binaries).
- `paths.corpora.<corpus-name>.source` (config.yaml) pointing at the
  corpus checkout with `lake exe cache get` already run.
- `paths.jixia_bin` pointing at a built jixia executable. By default this
  is `external/jixia/.lake/build/bin/jixia` (the submodule).

Output triples live under `<workdir>/raw/<corpus-name>/`, mirroring the
Lean source tree (e.g. `raw/mathlib/Mathlib/Data/List/Basic.sym.json`).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

from tqdm import tqdm

from ..config import (
    corpus_source_dir,
    corpus_source_subdir,
    corpus_workdir,
    get_path,
)


log = logging.getLogger("leansearchv2.corpus.jixia_extract")


def _setup_logging(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        force=True,
    )


def _output_paths(raw_dir: Path, mathlib_dir: Path, lean_file: Path) -> dict[str, Path]:
    rel = lean_file.relative_to(mathlib_dir)
    base = (raw_dir / rel).with_suffix("")
    return {
        "sym": base.with_suffix(".sym.json"),
        "decl": base.with_suffix(".decl.json"),
        "mod": base.with_suffix(".mod.json"),
    }


def _run_jixia_on_file(
    lean_file: Path,
    mathlib_dir: Path,
    raw_dir: Path,
    jixia_bin: Path,
    timeout_s: int,
    resume: bool,
    lake_root: Optional[Path] = None,
) -> Tuple[str, Path, Optional[str]]:
    # Output layout mirrors paths relative to the corpus source (`mathlib_dir`).
    # `lake env` cwd may be a different lake project (e.g. Physlib) that already
    # vendors this corpus under `.lake/packages/<name>/`.
    source_root = Path(mathlib_dir).resolve()
    lake_cwd = Path(lake_root).resolve() if lake_root else source_root
    lean_path = Path(lean_file).resolve()
    try:
        lean_rel_for_jixia = lean_path.relative_to(lake_cwd)
    except ValueError:
        try:
            lean_rel_for_jixia = lean_path.relative_to(source_root)
        except ValueError:
            lean_rel_for_jixia = lean_path

    outputs = _output_paths(Path(raw_dir).resolve(), source_root, lean_path)
    # jixia runs with cwd=lake_cwd; relative -s/-d/-m paths would land
    # inside the Lean package tree. Always pass absolute output paths.
    outputs = {k: v.resolve() for k, v in outputs.items()}
    if resume and all(p.exists() for p in outputs.values()):
        return ("skip", lean_file, None)
    outputs["sym"].parent.mkdir(parents=True, exist_ok=True)
    try:
        cmd = [
            "lake", "env", str(jixia_bin),
            "-s", str(outputs["sym"]),
            "-d", str(outputs["decl"]),
            "-m", str(outputs["mod"]),
            "-i",
            str(lean_rel_for_jixia),
        ]
        env = os.environ.copy()
        result = subprocess.run(
            cmd,
            cwd=str(lake_cwd),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=env,
        )
        if result.returncode != 0:
            return ("error", lean_file, f"exit={result.returncode}\n{result.stderr.strip()[:2000]}")
        if not all(p.exists() for p in outputs.values()):
            return ("error", lean_file, "outputs incomplete")
        return ("success", lean_file, None)
    except subprocess.TimeoutExpired:
        return ("error", lean_file, f"timeout {timeout_s}s")
    except Exception as e:  # pragma: no cover
        return ("error", lean_file, f"{type(e).__name__}: {e}")


def extract(
    corpus_name: str,
    source_dir: Path,
    raw_dir: Path,
    jixia_bin: Path,
    workers: int = 32,
    timeout_s: int = 300,
    resume: bool = True,
    lake_root: Optional[Path] = None,
) -> dict[str, int]:
    """Run jixia over `source_dir[/<source_subdir>]/**/*.lean`.

    `source_subdir` is read from `paths.corpora.<corpus-name>.source_subdir`
    via the caller (default `main()`); pass an explicit subdir here if you
    are calling `extract` directly from Python.

    If `lake_root` is set, `lake env jixia` runs with that cwd (e.g. Physlib)
    while still walking / writing relative to `source_dir`. This avoids using
    a broken nested lake under a vendored Mathlib package.
    """
    if not source_dir.exists():
        raise FileNotFoundError(f"Corpus source dir not found: {source_dir}")
    if not jixia_bin.exists():
        raise FileNotFoundError(f"jixia binary not found: {jixia_bin}")
    if lake_root is not None and not Path(lake_root).exists():
        raise FileNotFoundError(f"Lake root not found: {lake_root}")

    sub = corpus_source_subdir(corpus_name)
    lean_root = source_dir if sub in ("", ".") else source_dir / sub
    if not lean_root.exists():
        raise FileNotFoundError(f"Lean source root not found: {lean_root}")

    raw_dir.mkdir(parents=True, exist_ok=True)
    log_file = raw_dir.parent / "logs" / f"jixia_extract.{corpus_name}.log"
    err_file = raw_dir.parent / "logs" / f"jixia_errors.{corpus_name}.txt"
    _setup_logging(log_file)

    # When source_subdir is `.` (e.g. Physlib lake root), a naive rglob would
    # descend into `.lake/packages/mathlib` (thousands of files + slow FUSE/AFS).
    # Prune those dirs while walking instead of filtering after the fact.
    _skip_dir_names = {".lake", ".git", ".github", ".vscode", "__pycache__"}
    lean_files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(lean_root):
        dirnames[:] = [d for d in dirnames if d not in _skip_dir_names]
        for name in filenames:
            if name.endswith(".lean"):
                lean_files.append(Path(dirpath) / name)
    lean_files.sort()
    lake_cwd = Path(lake_root).resolve() if lake_root else Path(source_dir).resolve()
    log.info(
        f"[corpus={corpus_name}] Found {len(lean_files)} .lean files under {lean_root} "
        f"(pruned dirs={sorted(_skip_dir_names)}); lake_root={lake_cwd}"
    )
    stats = {"total": len(lean_files), "success": 0, "skip": 0, "error": 0}

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(
                _run_jixia_on_file,
                f, source_dir, raw_dir, jixia_bin, timeout_s, resume, lake_root,
            ): f for f in lean_files
        }
        with tqdm(total=len(futures), desc=f"jixia[{corpus_name}]", unit="file") as pbar:
            for fut in as_completed(futures):
                status, lean_file, err = fut.result()
                stats[status] += 1
                pbar.set_postfix(ok=stats["success"], skip=stats["skip"], err=stats["error"])
                pbar.update(1)
                if status == "error":
                    with err_file.open("a") as out:
                        out.write(f"[{datetime.now().isoformat()}] {lean_file}\n{err}\n\n")
                    log.warning(f"FAIL {lean_file.relative_to(source_dir)}: {err.splitlines()[0] if err else ''}")

    log.info(
        f"[corpus={corpus_name}] Done: total={stats['total']} ok={stats['success']} "
        f"skip={stats['skip']} err={stats['error']}"
    )
    if stats["error"]:
        log.warning(f"See {err_file} for error details.")
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run jixia over a Lean corpus (Mathlib / Physlib / ...)."
    )
    parser.add_argument("--corpus-name", type=str, default="mathlib",
                        help="Corpus name (default: mathlib). Must be configured "
                             "under paths.corpora.<name> in config.yaml.")
    parser.add_argument("--source-dir", type=str, default=None,
                        help="Lean source checkout (default: paths.corpora.<name>.source)")
    parser.add_argument("--lake-root", type=str, default=None,
                        help="Directory for `lake env` cwd. Defaults to --source-dir. "
                             "Use a parent lake project (e.g. Physlib) when the corpus "
                             "is vendored under .lake/packages/ and its nested lake is broken.")
    parser.add_argument("--workdir", type=str, default=None,
                        help="Corpus workdir parent (default: paths.corpus_workdir)")
    parser.add_argument("--jixia-bin", type=str, default=None,
                        help="jixia executable (default: paths.jixia_bin)")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    args = parser.parse_args()

    source = Path(args.source_dir) if args.source_dir else Path(corpus_source_dir(args.corpus_name))
    lake_root = Path(args.lake_root) if args.lake_root else None
    parent_workdir = Path(args.workdir) if args.workdir else Path(
        get_path("CORPUS_WORKDIR", "paths", "corpus_workdir")
    )
    raw = parent_workdir / "raw" / args.corpus_name
    jixia = Path(args.jixia_bin) if args.jixia_bin else Path(get_path("JIXIA_BIN", "paths", "jixia_bin"))

    stats = extract(
        args.corpus_name, source, raw, jixia,
        workers=args.workers, timeout_s=args.timeout, resume=args.resume,
        lake_root=lake_root,
    )
    return 0 if stats["error"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

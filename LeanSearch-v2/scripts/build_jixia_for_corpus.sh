#!/usr/bin/env bash
# Build a jixia binary whose Lean toolchain matches a target corpus.
#
# The git submodule at external/jixia is pinned to Lean v4.28.0-rc1 (Mathlib
# / LeanSearch-v2 paper release). Physlib HEAD currently tracks Lean v4.32.0,
# so extracting Physlib requires a separate jixia checkout.
#
# Usage:
#   ./scripts/build_jixia_for_corpus.sh mathlib          # uses submodule (v4.28)
#   ./scripts/build_jixia_for_corpus.sh physlib          # clones jixia@main, bumps to physlib's toolchain
#   JIXIA_LEAN=leanprover/lean4:v4.32.0 ./scripts/build_jixia_for_corpus.sh physlib
#
# On success prints the absolute path of the binary (and writes it to
# data/corpus/jixia_<corpus>.bin.path for later steps).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

CORPUS="${1:-mathlib}"
JIXIA_URL="${JIXIA_URL:-https://github.com/frenzymath/jixia.git}"

if [[ -z "${http_proxy:-}${https_proxy:-}" ]]; then
  echo "WARNING: no http(s)_proxy set; github/releases.lean-lang.org may time out. Try: vpn" >&2
fi

case "$CORPUS" in
  mathlib)
    DEST="external/jixia"
    if [[ ! -f "${DEST}/lean-toolchain" ]]; then
      git submodule update --init --recursive "$DEST"
    fi
    # Keep the pinned submodule toolchain (paper: v4.28.0-rc1).
    ;;
  physlib)
    DEST="external/jixia-physlib"
    # Keep jixia@main's native toolchain (currently Lean v4.29). Forcing it to
    # Physlib HEAD (v4.32) breaks Analyzer.Process.Declaration. Pair with
    # scripts/setup_physlib_source.sh which defaults to a v4.29.1 Physlib tip.
    if [[ ! -d "${DEST}/.git" ]]; then
      echo "Cloning jixia (main) -> ${DEST}"
      git clone --depth 1 -b main "$JIXIA_URL" "$DEST"
    else
      git -C "$DEST" fetch --depth 1 origin main
      git -C "$DEST" checkout -f main
      git -C "$DEST" reset --hard origin/main
    fi
    TARGET_TC="$(cat "${DEST}/lean-toolchain")"
    # Pin lean4-cli to the same Lean tag as jixia's toolchain. Leaving Cli on
    # `main` makes `lake build` rewrite lean-toolchain (observed: -> v4.33).
    CLI_TAG="${TARGET_TC##*:}"
    python3 - <<PY
from pathlib import Path
import re
p = Path("${DEST}/lakefile.toml")
text = p.read_text()
text2, n = re.subn(
    r'(name = "Cli"\ngit = "https://github.com/leanprover/lean4-cli.git"\n)rev = "[^"]+"',
    r'\1rev = "${CLI_TAG}"',
    text, count=1)
if n != 1:
    raise SystemExit(f"failed to pin Cli rev (n={n})")
p.write_text(text2)
print(f"Pinned Cli rev -> ${CLI_TAG}")
PY
    echo "Using jixia toolchain: ${TARGET_TC} (Cli=${CLI_TAG})"
    if [[ -f external/physlib/lean-toolchain ]]; then
      PHYS_TC="$(cat external/physlib/lean-toolchain)"
      echo "Physlib toolchain: ${PHYS_TC}"
      if [[ "${PHYS_TC#*:v}" != "${TARGET_TC#*:v}"* && "${PHYS_TC}" != "${TARGET_TC}" ]]; then
        echo "WARNING: Physlib (${PHYS_TC}) vs jixia (${TARGET_TC}) minor mismatch." >&2
      fi
    fi
    stamp="${DEST}/.leansearch_jixia_pin"
    want_pin="native:${TARGET_TC}|Cli=${CLI_TAG}"
    if [[ -f "$stamp" ]] && [[ "$(cat "$stamp")" == "$want_pin" ]] && [[ -d "${DEST}/.lake" ]]; then
      echo "Resuming existing ${DEST}/.lake (pin unchanged)"
    else
      echo "Pin changed or no prior build — cleaning ${DEST}/.lake"
      rm -f "${DEST}/lake-manifest.json"
      rm -rf "${DEST}/.lake"
      echo "$want_pin" > "$stamp"
    fi
    ;;
  *)
    echo "Usage: $0 mathlib|physlib" >&2
    exit 2
    ;;
esac

echo "Building jixia in ${DEST} (toolchain=$(cat "${DEST}/lean-toolchain"))"
(
  cd "$DEST"
  # Do NOT run `lake update` here: dependency manifests on jixia@main can
  # silently bump lean-toolchain (e.g. to v4.33) and break corpus matching.
  # Re-assert the desired toolchain right before build.
  if [[ -n "${TARGET_TC:-}" ]]; then
    echo "$TARGET_TC" > lean-toolchain
  fi
  lake build
  # Guard against Lake rewriting the toolchain during the build.
  if [[ -n "${TARGET_TC:-}" ]]; then
    got="$(cat lean-toolchain)"
    if [[ "$got" != "$TARGET_TC" ]]; then
      echo "ERROR: lean-toolchain drifted to ${got} (wanted ${TARGET_TC})" >&2
      exit 1
    fi
  fi
)

BIN="${DEST}/.lake/build/bin/jixia"
if [[ ! -x "$BIN" ]]; then
  echo "ERROR: expected binary missing: ${BIN}" >&2
  exit 1
fi
ABS="$(cd "$(dirname "$BIN")" && pwd)/$(basename "$BIN")"
mkdir -p data/corpus
echo "$ABS" > "data/corpus/jixia_${CORPUS}.bin.path"
echo "jixia ready: ${ABS}"
echo "For build_corpus, export:  JIXIA_BIN=${ABS}"

#!/usr/bin/env bash
# Move Physlib checkout + ~/.cache/mathlib onto container-local disk (/tmp)
# and replace the AFS paths with symlinks. Speeds up lake cache get I/O.
#
# Usage:
#   ./scripts/migrate_physlib_to_local.sh
#   LOCAL_ROOT=/tmp/$USER/lean-local ./scripts/migrate_physlib_to_local.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

LOCAL_ROOT="${LOCAL_ROOT:-/tmp/${USER:-chendongyang}/lean-local}"
AFS_PHYS="${AFS_PHYS:-$REPO_DIR/external/physlib}"
AFS_CACHE="${AFS_CACHE:-$HOME/.cache/mathlib}"
STAMP="$(date +%Y%m%d%H%M%S)"

mkdir -p "$LOCAL_ROOT"
echo "LOCAL_ROOT=$LOCAL_ROOT ($(df -Th "$LOCAL_ROOT" | tail -1))"

migrate_dir() {
  local src="$1" dest="$2" label="$3"
  if [[ -L "$src" ]]; then
    echo "[$label] already symlink -> $(readlink -f "$src" || readlink "$src")"
    return 0
  fi
  if [[ ! -d "$src" ]]; then
    echo "[$label] source missing: $src" >&2
    return 1
  fi
  echo "[$label] rsync $src -> $dest"
  rm -rf "$dest"
  mkdir -p "$dest"
  rsync -a "$src"/ "$dest"/
  echo "[$label] size $(du -sh "$dest" | awk '{print $1}')"
  mv "$src" "${src}.afs.bak.${STAMP}"
  ln -s "$dest" "$src"
  echo "[$label] linked $src -> $dest"
}

migrate_dir "$AFS_CACHE" "$LOCAL_ROOT/mathlib-cache" "mathlib-cache"
migrate_dir "$AFS_PHYS" "$LOCAL_ROOT/physlib" "physlib"

echo
echo "=== verify ==="
ls -ld "$AFS_CACHE" "$AFS_PHYS"
readlink -f "$AFS_CACHE"
readlink -f "$AFS_PHYS"
cat "$AFS_PHYS/lean-toolchain"
git -C "$AFS_PHYS" rev-parse --short HEAD
df -h "$LOCAL_ROOT" | tail -1
echo
echo "AFS backups kept as *.afs.bak.${STAMP}"
echo
echo "NOTE: /tmp is container-local and may be wiped on restart."
echo "After cache get finishes, sync products back to AFS, e.g.:"
echo "  rsync -a $LOCAL_ROOT/physlib/.lake/ ${AFS_PHYS}.afs.bak.*/.lake/   # or a permanent AFS path"
echo "  rsync -a $LOCAL_ROOT/mathlib-cache/ \$HOME/.cache/mathlib.afs.bak.*/"
echo
echo "Next:"
echo "  vpn   # if needed"
echo "  cd $AFS_PHYS && lake exe cache get"

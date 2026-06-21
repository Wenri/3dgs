#!/usr/bin/env bash
# scripts/fetch_datasets.sh
# Download dataset release assets from Wenri/3dgs and restore them in place.
# All archives store repo-root-relative paths, so they extract at the repo root.
#
#   bash scripts/fetch_datasets.sh                       # all dataset releases
#   bash scripts/fetch_datasets.sh datasets-dtu-3views   # one release
#
# Requires: gh (authenticated), tar, sha256sum.
set -euo pipefail

REPO="Wenri/3dgs"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DL="${DL:-$ROOT/.release_dl}"

ALL=( datasets-dtu-3views datasets-llff-3views datasets-custom-scenes
      datasets-benchmark-source datasets-misc-aux )
TAGS=( "${@:-${ALL[@]}}" )
[ "$#" -gt 0 ] && TAGS=( "$@" )

mkdir -p "$DL"; cd "$DL"
for tag in "${TAGS[@]}"; do
  echo ">>> $tag"
  gh release download "$tag" --repo "$REPO" --clobber
  if [ -f SHA256SUMS ]; then
    echo "    verifying checksums..."
    sha256sum -c --ignore-missing SHA256SUMS
  fi
  for f in *.tar.gz; do
    [ -e "$f" ] || continue
    echo "    extract $f -> repo root"
    tar -xzf "$f" -C "$ROOT"
    rm -f "$f"
  done
done
echo "Done. Datasets restored under $ROOT"

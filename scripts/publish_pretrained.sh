#!/usr/bin/env bash
# scripts/publish_pretrained.sh
# ---------------------------------------------------------------------------
# Compress each pre/<scene> with `xz -9e` and publish as assets in a GitHub
# release. A scene is shipped whole unless its .tar.xz would exceed 2 GiB, in
# which case it is split by training iteration. Idempotent; DRY-RUN by default.
#
#   bash scripts/publish_pretrained.sh            # build + verify, no upload
#   DRYRUN=0 bash scripts/publish_pretrained.sh   # build + verify + upload
#
# ⚠ LICENSING: pre/ holds Inria's OFFICIAL 3DGS pretrained models, trained on
# Mip-NeRF360 / Tanks&Temples / Deep Blending. These are third-party, non-
# commercial-research only. This script mirrors them WITH attribution; do not
# use commercially. See the generated MANIFEST.md / release notes.
# ---------------------------------------------------------------------------
set -euo pipefail
REPO="Wenri/3dgs"
TAG="pretrained-3dgs-models"
TITLE="3DGS pretrained models — Mip-NeRF360 / Tanks&Temples / Deep Blending (mirror)"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE="${STAGE:-$ROOT/.release_stage}"
DRYRUN="${DRYRUN:-1}"
HARD=$(( 2*1024*1024*1024 ))           # 2 GiB GitHub hard cap (abort)
SAFE=$(( 1950*1024*1024 ))             # whole-scene split threshold (~1.9 GiB, headroom under cap)
mkdir -p "$STAGE"; : > "$STAGE/SHA256SUMS"
log(){ printf '[pre] %s\n' "$*" >&2; }
human(){ numfmt --to=iec "$1"; }

# scene -> source dataset (for the manifest)
dataset_of(){ case "$1" in
  bicycle|bonsai|counter|garden|kitchen|room|stump|flowers|treehill) echo "Mip-NeRF360";;
  train|truck) echo "Tanks&Temples";;
  drjohnson|playroom) echo "DeepBlending";;
  *) echo "unknown";; esac; }
license_of(){ case "$(dataset_of "$1")" in
  "Mip-NeRF360")   echo "Mip-NeRF360 (Barron+ 2022) — research use";;
  "Tanks&Temples") echo "Tanks and Temples (Knapitsch+ 2017) — CC-BY-NC-SA 4.0";;
  "DeepBlending")  echo "Deep Blending (Hedman+ 2018) — research use";;
  *) echo "see upstream";; esac; }

checksum(){ ( cd "$STAGE" && sha256sum "$(basename "$1")" >> SHA256SUMS ); }
finish(){ local f="$1" b; b=$(stat -c%s "$f")
  if (( b > HARD )); then log "FATAL $(basename "$f")=$(human "$b") > 2 GiB"; exit 1; fi
  log "ok   $(basename "$f")=$(human "$b")"; xz -t "$f"; checksum "$f"; }

xz_tar(){ local name="$1"; shift; log "xz $name <- $*"
  tar -C "$ROOT" -c "$@" | xz -9e -T0 -c > "$STAGE/$name"; finish "$STAGE/$name"; }

# compress a scene whole; if > 2 GiB, fall back to split-by-iteration
build_scene(){
  local s="$1" d="$ROOT/pre/$1" out="$STAGE/$1.tar.xz"
  log "compress scene $s ($(du -sh "$d" | cut -f1) raw) ..."
  tar -C "$ROOT" -c "pre/$s" | xz -9e -T0 -c > "$out"
  local b; b=$(stat -c%s "$out")
  if (( b <= SAFE )); then log "ok   $s.tar.xz=$(human "$b")"; xz -t "$out"; checksum "$out"; return; fi
  log "WARN $s.tar.xz=$(human "$b") > 1.9 GiB → split by iteration (cap headroom)"; rm -f "$out"
  # meta = everything except point_cloud/
  local meta=(); for e in "$d"/*; do [ "$(basename "$e")" = point_cloud ] || meta+=( "pre/$s/$(basename "$e")" ); done
  xz_tar "${s}_meta.tar.xz" "${meta[@]}"
  for it in "$d"/point_cloud/iteration_*/; do
    xz_tar "${s}_$(basename "$it").tar.xz" "pre/$s/point_cloud/$(basename "$it")"
  done
}

# ------------------------------ build --------------------------------------
log "ROOT=$ROOT STAGE=$STAGE DRYRUN=$DRYRUN"
scenes=(); for d in "$ROOT"/pre/*/; do scenes+=( "$(basename "$d")" ); done
log "scenes (${#scenes[@]}): ${scenes[*]}"
for s in "${scenes[@]}"; do build_scene "$s"; done

# ------------------------------ manifest -----------------------------------
M="$STAGE/MANIFEST.md"
{
  echo "# 3DGS pretrained models — manifest & licensing"
  echo
  echo "**These are Inria's official 3D Gaussian Splatting pretrained models, mirrored here.**"
  echo "Models trained by Inria (Kerbl et al., SIGGRAPH 2023) under the **Gaussian-Splatting"
  echo "non-commercial research license** (see repo \`LICENSE.md\`). Source scenes belong to their"
  echo "original datasets — **non-commercial / research use only**. Cite the original works."
  echo
  echo "| scene | source dataset | license |"
  echo "|---|---|---|"
  for s in "${scenes[@]}"; do echo "| \`$s\` | $(dataset_of "$s") | $(license_of "$s") |"; done
  echo
  echo "Restore any asset at the repo root: \`tar -xJf <scene>.tar.xz\` → \`pre/<scene>/\`."
  echo "Split scenes (e.g. garden): fetch all \`<scene>_*.tar.xz\` parts and extract each."
  echo "Verify: \`sha256sum -c SHA256SUMS\`."
} > "$M"
( cd "$STAGE" && sha256sum "MANIFEST.md" >> SHA256SUMS )

cat > "$STAGE/notes_pretrained.md" <<EOF
**Mirror of Inria's official 3D Gaussian Splatting pretrained models** — Mip-NeRF360,
Tanks & Temples, and Deep Blending scenes. One \`xz -9e\` archive per scene (large scenes
split by training iteration to stay under GitHub's 2 GiB asset limit).

⚠ **Non-commercial / research use only.** Models trained by Inria (Kerbl et al., 2023)
under the Gaussian-Splatting non-commercial research license. Source datasets retain their
own licenses (Tanks & Temples = CC-BY-NC-SA 4.0; Mip-NeRF360 / Deep Blending = research).
See \`MANIFEST.md\` for per-scene attribution. Cite the original papers.

Restore: \`tar -xJf <scene>.tar.xz\` at the repo root → \`pre/<scene>/\`.
EOF

log "===== staged ====="
( cd "$STAGE" && for f in *.tar.xz; do printf '  %9s  %s\n' "$(human "$(stat -c%s "$f")")" "$f"; done ) >&2
log "total staged: $(du -sh "$STAGE" | cut -f1)"

# ------------------------------ publish ------------------------------------
if (( DRYRUN )); then
  log "DRYRUN: would publish $(ls "$STAGE"/*.tar.xz | wc -l) assets + MANIFEST.md + SHA256SUMS to release '$TAG' on $REPO"
  exit 0
fi
gh release view "$TAG" -R "$REPO" >/dev/null 2>&1 \
  || gh release create "$TAG" -R "$REPO" --title "$TITLE" --notes-file "$STAGE/notes_pretrained.md"
gh release upload "$TAG" "$STAGE"/*.tar.xz "$M" "$STAGE/SHA256SUMS" -R "$REPO" --clobber
log "published $TAG"

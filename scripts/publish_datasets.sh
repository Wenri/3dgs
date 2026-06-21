#!/usr/bin/env bash
# scripts/publish_datasets.sh
# ---------------------------------------------------------------------------
# Package local (git-ignored) datasets into < 2 GiB tarballs and publish them
# as GitHub Release assets on Wenri/3dgs, grouped by data type, with trained
# OUTPUTS separated from INPUTS. Idempotent. Defaults to DRY-RUN.
#
#   bash scripts/publish_datasets.sh            # DRY-RUN: build + verify, no upload
#   DRYRUN=0 bash scripts/publish_datasets.sh   # build + verify + upload
#   STAGE=/big/tmp DRYRUN=0 bash scripts/...     # override staging dir
#
# Every archive stores repo-root-relative paths, so `tar xzf <asset>` at the
# repo root restores files exactly in place. All individual files are < 2 GiB,
# so scenes are grouped whole; no `split` binary is ever needed.
# ---------------------------------------------------------------------------
set -euo pipefail

REPO="Wenri/3dgs"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE="${STAGE:-$ROOT/.release_stage}"
DRYRUN="${DRYRUN:-1}"
HARD=$(( 2 * 1024*1024*1024 ))     # 2 GiB GitHub hard cap
SOFT=$(( 1843*1024*1024 ))         # ~1.8 GiB soft target (warn)
BUDGET=$(( 1600*1024*1024 ))       # ~1.56 GiB per bin-packed part

mkdir -p "$STAGE"
: > "$STAGE/SHA256SUMS"
log(){ printf '[publish] %s\n' "$*" >&2; }
human(){ numfmt --to=iec "$1"; }

verify_size(){
  local f="$1" b; b=$(stat -c%s "$f")
  if   (( b > HARD )); then log "FATAL $(basename "$f") = $(human "$b") > 2 GiB — must split"; exit 1
  elif (( b > SOFT )); then log "WARN  $(basename "$f") = $(human "$b") > 1.8 GiB (under cap, ok)"
  else                       log "ok    $(basename "$f") = $(human "$b")"; fi
}
verify_tar(){ tar -tzf "$1" >/dev/null && log "valid $(basename "$1")"; }
checksum(){ ( cd "$STAGE" && sha256sum "$(basename "$1")" >> SHA256SUMS ); }

finish(){ local f="$1"; verify_size "$f"; verify_tar "$f"; checksum "$f"; }

# tar explicit members (paths relative to repo root)
make_tar(){
  local name="$1"; shift
  local present=(); for m in "$@"; do [ -e "$ROOT/$m" ] && present+=("$m"); done
  [ "${#present[@]}" -gt 0 ] || { log "skip $name (no members on disk)"; return; }
  log "build $name <- ${present[*]}"
  tar -C "$ROOT" -czf "$STAGE/$name" "${present[@]}"
  finish "$STAGE/$name"
}

# inputs tarball: scene dirs minus their output*/ and dense/ subdirs
make_inputs(){
  local name="$1"; shift
  local ex=() present=()
  for d in "$@"; do
    [ -d "$ROOT/$d" ] || continue
    present+=("$d"); ex+=( --exclude="$d/output*" --exclude="$d/dense" )
  done
  ex+=( --exclude='*/.DS_Store' --exclude='*/__pycache__' --exclude='*/.ipynb_checkpoints' )
  [ "${#present[@]}" -gt 0 ] || { log "skip $name (no input dirs)"; return; }
  log "build $name (inputs) <- ${present[*]}"
  tar -C "$ROOT" "${ex[@]}" -czf "$STAGE/$name" "${present[@]}"
  finish "$STAGE/$name"
}

# greedy first-fit-decreasing bin-pack of output dirs into <prefix>_partN.tar.gz
binpack_outputs(){
  local prefix="$1" budget="$2"; shift 2
  local dirs=() sizes=()
  for d in "$@"; do
    [ -d "$ROOT/$d" ] || continue
    local b; b=$(du -sb "$ROOT/$d" | cut -f1)
    (( b > 65536 )) || { log "skip $d (no real output, $(human "$b"))"; continue; }
    dirs+=("$d"); sizes+=("$b")
  done
  [ "${#dirs[@]}" -gt 0 ] || { log "skip $prefix (no outputs)"; return; }
  local order; order=$(for i in "${!sizes[@]}"; do echo "${sizes[$i]} $i"; done | sort -rn | awk '{print $2}')
  local -a bin_load=() bin_members=()
  for i in $order; do
    local d="${dirs[$i]}" s="${sizes[$i]}" placed=0 n=${#bin_load[@]} j
    for ((j=0;j<n;j++)); do
      if (( bin_load[j] + s <= budget )); then
        bin_load[j]=$(( bin_load[j] + s )); bin_members[j]="${bin_members[j]} $d"; placed=1; break
      fi
    done
    (( placed )) || { bin_load+=("$s"); bin_members+=("$d"); }
  done
  local part=1 j
  for ((j=0;j<${#bin_members[@]};j++)); do
    make_tar "${prefix}_part${part}.tar.gz" ${bin_members[j]}
    part=$(( part+1 ))
  done
}

scans(){ ls -1 "$ROOT/$1"; }   # list per-scene subdirs of a family

# --------------------------- release notes ---------------------------------
write_notes(){
  cat > "$STAGE/notes_datasets-dtu-3views.md" <<'EOF'
DTU few-view (3-view) COLMAP reconstructions + trained 3D Gaussians (23 scans).
Inputs (`*_inputs`) and trained outputs (`*_outputs_part*`) are split so you can
fetch only what you need. Extract at the repo root: `tar xzf <asset>`.

Provenance: derived from the **DTU MVS dataset** (Aanæs et al.) — academic/research
use only. COLMAP + Gaussian outputs are ours, under repo LICENSE.md (Inria 3DGS
non-commercial research license). See DATA.md.
EOF
  cat > "$STAGE/notes_datasets-llff-3views.md" <<'EOF'
LLFF forward-facing reconstructions: 3-view COLMAP + trained Gaussians, plus our
full-view LLFF reconstructions from `data/`. Inputs vs outputs split.

Provenance: scenes from **NeRF-LLFF** (Mildenhall et al.) — public/research; cite
NeRF & LLFF. COLMAP + Gaussian outputs are ours (repo LICENSE.md). See DATA.md.
EOF
  cat > "$STAGE/notes_datasets-custom-scenes.md" <<'EOF'
Our own captures: custom multi-view scans (scan2–scan5), in-the-wild DJI scenes
(room, trex, horns), few-view derivatives (3views, horns_15views) and a PatchMatch
MVS intermediate. Inputs vs trained outputs split; large scenes chunked < 2 GiB.

Provenance: own data + reconstructions, repo LICENSE.md (Inria 3DGS non-commercial
research). See DATA.md.
EOF
  cat > "$STAGE/notes_datasets-benchmark-source.md" <<'EOF'
Third-party benchmark SOURCE data, mirrored for convenience. NOT our data — see
upstream licenses before use.

- nerf_llff_data_part{1,2}: NeRF-LLFF (Mildenhall et al.) — public/research.
- tandt_db: Tanks&Temples (train, truck) = CC-BY-NC-SA 4.0 (non-commercial,
  attribution, share-alike); DeepBlending (drjohnson, playroom) = Inria/Deep
  Blending terms. The redundant `tandt_db.zip` is intentionally NOT included.

Inria 3DGS pretrained models are deliberately NOT redistributed. See DATA.md.
EOF
  cat > "$STAGE/notes_datasets-misc-aux.md" <<'EOF'
Loose COLMAP scratch from `data/` (distorted/, images/, input/, sparse/, stereo/,
new/). Extract at repo root. See DATA.md.
EOF
}

# ------------------------------- build -------------------------------------
log "ROOT=$ROOT STAGE=$STAGE DRYRUN=$DRYRUN"

# DTU 3-view --------------------------------------------------------------
make_inputs dtu_3views_inputs.tar.gz $(for s in $(scans dtu_3views_colmap); do echo "dtu_3views_colmap/$s"; done)
binpack_outputs dtu_3views_outputs "$BUDGET" $(for s in $(scans dtu_3views_colmap); do echo "dtu_3views_colmap/$s/output"; done)

# LLFF 3-view + our data/ LLFF recons -------------------------------------
make_inputs llff_3views_inputs.tar.gz $(for s in $(scans llff_3views_colmap); do echo "llff_3views_colmap/$s"; done)
make_tar    llff_3views_outputs.tar.gz $(for s in $(scans llff_3views_colmap); do echo "llff_3views_colmap/$s/output"; done)
make_inputs data_llff_inputs.tar.gz data/fern data/flower_15views data/fortress data/leaves data/orchids
binpack_outputs data_llff_outputs "$BUDGET" \
  data/fern/output data/flower_15views/output data/fortress/output data/leaves/output data/orchids/output

# Custom scenes -----------------------------------------------------------
make_inputs custom_scans_inputs.tar.gz data/scan2 data/scan3 data/scan4 data/scan5
make_inputs custom_djiraw_inputs.tar.gz data/room data/trex
make_inputs horns_few_inputs.tar.gz data/horns data/horns_3views horns_15views 3views
make_tar scan4_outputs.tar.gz       data/scan4/output
make_tar scan23_outputs.tar.gz      data/scan2/output data/scan3/output
make_tar scan5_outputs_part1.tar.gz data/scan5/output15views
make_tar scan5_outputs_part2.tar.gz data/scan5/output3views data/scan5/output3viewscamera
make_tar horns_few_outputs.tar.gz   data/horns_3views/output horns_15views/output \
                                    3views/output 3views/output00 3views/output000 3views/dense
make_tar PM_patchmatch.tar.gz       PM/scan1

# Benchmark source --------------------------------------------------------
make_tar nerf_llff_data_part1.tar.gz \
  llff/nerf_llff_data/fern llff/nerf_llff_data/flower llff/nerf_llff_data/fortress llff/nerf_llff_data/horns
make_tar nerf_llff_data_part2.tar.gz \
  llff/nerf_llff_data/leaves llff/nerf_llff_data/orchids llff/nerf_llff_data/room llff/nerf_llff_data/trex
# tandt_db: drop the redundant zip
if [ -d "$ROOT/dbco" ]; then
  log "build tandt_db.tar.gz (excluding tandt_db.zip)"
  tar -C "$ROOT" --exclude='dbco/tandt_db.zip' -czf "$STAGE/tandt_db.tar.gz" dbco/db dbco/tandt
  finish "$STAGE/tandt_db.tar.gz"
fi

# Misc aux ----------------------------------------------------------------
make_tar data_aux_loose.tar.gz data/distorted data/images data/input data/sparse data/stereo data/new

# ------------------------- summary + publish -------------------------------
log "===== staged assets ====="
( cd "$STAGE" && for f in *.tar.gz; do printf '  %8s  %s\n' "$(human "$(stat -c%s "$f")")" "$f"; done ) >&2
log "total staged: $(du -sh "$STAGE" | cut -f1)"

write_notes
publish(){
  local tag="$1" title="$2"; shift 2
  local files=()
  for g in "$@"; do for f in $STAGE/$g; do [ -f "$f" ] && files+=("$f"); done; done
  [ "${#files[@]}" -gt 0 ] || { log "no assets for $tag"; return; }
  if (( DRYRUN )); then
    log "DRYRUN would publish $tag (${#files[@]} assets): $(for f in "${files[@]}"; do basename "$f"; done | tr '\n' ' ')"
    return
  fi
  gh release view "$tag" --repo "$REPO" >/dev/null 2>&1 \
    || gh release create "$tag" --repo "$REPO" --title "$title" --notes-file "$STAGE/notes_${tag}.md"
  gh release upload "$tag" "${files[@]}" "$STAGE/SHA256SUMS" --repo "$REPO" --clobber
  log "published $tag"
}

publish datasets-dtu-3views       "DTU 3-view COLMAP + trained Gaussians"        'dtu_3views_*.tar.gz'
publish datasets-llff-3views      "LLFF 3-view + LLFF reconstructions"           'llff_3views_*.tar.gz' 'data_llff_*.tar.gz'
publish datasets-custom-scenes    "Custom multi-view + in-the-wild scenes"       'custom_*.tar.gz' 'scan*_outputs*.tar.gz' 'horns_few_*.tar.gz' 'PM_*.tar.gz'
publish datasets-benchmark-source "Third-party benchmark source (see licenses)"  'nerf_llff_data_*.tar.gz' 'tandt_db.tar.gz'
publish datasets-misc-aux         "Loose COLMAP scratch"                         'data_aux_loose.tar.gz'

log "DONE (DRYRUN=$DRYRUN)"

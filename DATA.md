# Datasets — release manifest, restore guide & licensing

The datasets used by this repo are **not tracked in git** (tens of GB). They are
published as **GitHub Release assets** on [`Wenri/3dgs`](https://github.com/Wenri/3dgs/releases),
grouped by data type, with trained **outputs** separated from **inputs**.

Every archive stores **repo-root-relative paths**, so it restores in place with
`tar xzf <asset>` run at the repo root. Each release ships a `SHA256SUMS` file.

## Quick restore

```bash
# everything:
bash scripts/fetch_datasets.sh
# or one group:
bash scripts/fetch_datasets.sh datasets-dtu-3views
# manual:
gh release download datasets-llff-3views -R Wenri/3dgs --clobber
sha256sum -c --ignore-missing SHA256SUMS
for f in *.tar.gz; do tar -xzf "$f"; done    # run at repo root
```

Packaging is reproducible via `scripts/publish_datasets.sh` (dry-run by default;
`DRYRUN=0` to upload). All assets are < 2 GiB (GitHub's per-file cap).

## Releases & assets

### `datasets-dtu-3views` — DTU 3-view COLMAP + trained Gaussians
| Asset | Restores to | Side |
|---|---|---|
| `dtu_3views_inputs.tar.gz` | `dtu_3views_colmap/<scan>/{images,sparse,input,distorted,cams_1,...}` (23 scans) | inputs |
| `dtu_3views_outputs_part1..3.tar.gz` | `dtu_3views_colmap/<scan>/output/` (20 scans with trained models) | outputs |

### `datasets-llff-3views` — LLFF few-view + our LLFF reconstructions
| Asset | Restores to | Side |
|---|---|---|
| `llff_3views_inputs.tar.gz` | `llff_3views_colmap/<scene>/{images,sparse,...}` (8 scenes) | inputs |
| `llff_3views_outputs.tar.gz` | `llff_3views_colmap/<scene>/output/` | outputs |
| `data_llff_inputs.tar.gz` | `data/{fern,flower_15views,fortress,leaves,orchids}` (no outputs) | inputs |
| `data_llff_outputs_part1..N.tar.gz` | `data/<scene>/output/` | outputs |

### `datasets-custom-scenes` — our captures + few-view derivatives
| Asset | Restores to | Side |
|---|---|---|
| `custom_scans_inputs.tar.gz` | `data/{scan2,scan3,scan4,scan5}` (no outputs) | inputs |
| `custom_djiraw_inputs.tar.gz` | `data/{room,trex}` (source-only DJI) | inputs |
| `horns_few_inputs.tar.gz` | `data/{horns,horns_3views}` + `horns_15views/` + `3views/` (no outputs) | inputs |
| `scan4_outputs.tar.gz` | `data/scan4/output/` | outputs |
| `scan23_outputs.tar.gz` | `data/{scan2,scan3}/output/` | outputs |
| `scan5_outputs_part1.tar.gz` | `data/scan5/output15views/` | outputs |
| `scan5_outputs_part2.tar.gz` | `data/scan5/{output3views,output3viewscamera}/` | outputs |
| `horns_few_outputs.tar.gz` | `data/horns_3views/output/`, `horns_15views/output/`, `3views/{output*,dense}` | outputs |
| `PM_patchmatch.tar.gz` | `PM/scan1/` (PatchMatch MVS intermediate) | — |

### `datasets-benchmark-source` — third-party benchmark SOURCE (mirror)
| Asset | Restores to | Upstream / license |
|---|---|---|
| `nerf_llff_data_part1.tar.gz` | `llff/nerf_llff_data/{fern,flower,fortress,horns}` | NeRF-LLFF — public/research |
| `nerf_llff_data_part2.tar.gz` | `llff/nerf_llff_data/{leaves,orchids,room,trex}` | NeRF-LLFF — public/research |
| `tandt_db.tar.gz` | `dbco/{db,tandt}` (redundant `tandt_db.zip` omitted) | T&T = CC-BY-NC-SA 4.0; DeepBlending = Inria/DB terms |
| `nerf_llff_data.zip` | `ds/nerf_llff_data/` (original archive; `tar`-extract elsewhere) | NeRF-LLFF — public/research |
| `nerf_synthetic.zip` | `ds/nerf_synthetic/` (original archive) | NeRF synthetic (Blender) — research; per-scene attribution in its `README.txt` |

### `datasets-misc-aux` — loose COLMAP scratch
| Asset | Restores to |
|---|---|
| `data_aux_loose.tar.gz` | `data/{distorted,images,input,sparse,stereo,new}` |

### `data-snapshot-20260621` (experiment outputs + few-view snapshots)
`data.tar.gz` (= `data/dtuscan114_3views`), `data_output.tar.gz` (experiment outputs),
`dtu_15views_colmap_7000.tar.gz`(+part2,part3), `dtu_3views.tar.gz`, `rgbd-patch-diffusion.pt`.
| Asset | Restores to | Notes |
|---|---|---|
| `output.tar.xz` | `output/` (43 training runs) | full default output dir, `xz -9e` (1.91 GiB); supersedes partial `data_output.tar.gz`. `tar -xJf` |
| `output_manifest.tsv` | — | maps each `output/<hash>` run → source scene + iterations |
| `output.tar.xz.sha256` | — | checksum (`sha256sum -c`) |
| `nerf_synthetic_outputs.tar.gz` | `ds/nerf_synthetic/<scene>/{output,points3d.ply}` | our trained 3DGS on ficus/hotdog/lego/materials/mic/ship + 8 `points3d.ply` (690 MiB raw); not in `nerf_synthetic.zip` |

### `pretrained-3dgs-models` — Inria pretrained models (MIRROR, ⚠ non-commercial)
Per-scene `xz -9e` archives of the official 3DGS pretrained models (content-verified `==` `pre/models.zip`). Large scenes split by training iteration to stay < 2 GiB. Restore: `tar -xJf <asset>` → `pre/<scene>/`.
| Asset | Restores to |
|---|---|
| `<scene>.tar.xz` ×11 (bonsai, counter, drjohnson, flowers, kitchen, playroom, room, stump, train, treehill, truck) | `pre/<scene>/` |
| `bicycle_meta.tar.xz` + `bicycle_iteration_{7000,30000}.tar.xz` | `pre/bicycle/` (split) |
| `garden_meta.tar.xz` + `garden_iteration_{7000,30000}.tar.xz` | `pre/garden/` (split) |
| `MANIFEST.md` | per-scene source dataset + license |
⚠ **Mirror of Inria's pretrained models — non-commercial / research use only.** Cite the original works; comply with each source dataset's license (see `MANIFEST.md` and below).

## Provenance & licensing

- **Our reconstructions/outputs** (COLMAP models, trained Gaussian `point_cloud.ply`,
  metrics, renders): released under this repo's `LICENSE.md` — the **Gaussian-Splatting
  / Inria–MPII non-commercial research license**. Research use only, no commercial use.
- **DTU** (`dtu_3views_colmap/`, `data/dtuscan114_3views`): DTU MVS dataset
  (Aanæs et al.) — academic/research use; cite the DTU dataset.
- **NeRF-LLFF** (`llff/`, LLFF scenes in `data/`): Mildenhall et al. — public/research;
  cite **NeRF** and **LLFF**.
- **Tanks & Temples** (`dbco/tandt/{train,truck}`): **CC-BY-NC-SA 4.0** — non-commercial,
  attribution, share-alike.
- **Deep Blending** (`dbco/db/{drjohnson,playroom}`): Inria / Deep Blending dataset terms,
  research only.

- **Inria 3DGS pretrained models** (`pre/`, == `pre/models.zip`): mirrored in the
  **`pretrained-3dgs-models`** release for convenience — **non-commercial / research use only**.
  Trained by Inria (Kerbl et al. 2023); source scenes are Mip-NeRF360 (research), Tanks & Temples
  (CC-BY-NC-SA 4.0), and Deep Blending (research). Per-scene attribution in that release's
  `MANIFEST.md`. Original source: [Gaussian-Splatting project page](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/).

### Not redistributed (deliberately)
- **`ds/nerf_synthetic.zip`, `ds/nerf_llff_data.zip`**: standard public datasets — get them
  from their original sources (already mirrored in `datasets-benchmark-source` for LLFF).

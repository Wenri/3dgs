# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository actually is

A research fork of **3D Gaussian Splatting (3DGS)** that couples the Gaussian optimizer with
**DiffusioNeRF-style RGBD patch-diffusion regularization** to make few-view / sparse-view
reconstruction work. The target benchmark is **DTU** in sparse settings (`dtu_3views/`,
`dtu_15views_colmap_7000/`, etc.).

Important: `README.md` is the **unmodified upstream Inria README**. It documents the original
dense-view method, the SIBR viewers, `full_eval.py`, and MipNeRF360/Tanks&Temples — most of which
is *not* what this fork does. When `README.md` and the code disagree, **trust the code**. The real
training entry point (`train.py`) and the diffusion machinery (`arguments/diffusion.py`) have
diverged substantially from upstream.

## Environment & build

The conda env is named `gaussian_splatting` (see `environment.yml`):

```shell
conda env create --file environment.yml   # python 3.7, torch 1.12.1, cudatoolkit 11.6
conda activate gaussian_splatting
```

That installs two of the four submodules via pip (`submodules/diff-gaussian-rasterization`,
`submodules/simple-knn`). **This is not sufficient to run `train.py`.** Two extra requirements:

1. **The rasterizer must be the depth-enabled fork.** `.gitmodules` points
   `diff-gaussian-rasterization` at `github.com/Wenri/diff-gaussian-rasterization`, whose
   `rasterizer(...)` returns **four** values `(image, radii, depth, alpha)`. The stock Inria
   rasterizer returns only `(image, radii)` and will break `gaussian_renderer/__init__.py`. Do not
   swap in upstream.
2. **The `diffusionerf` submodule and its CUDA extensions are imported at startup.**
   `arguments/diffusion.py` imports `submodules.diffusionerf.main_nerf`, which transitively imports
   `raymarching`, `gridencoder`, `shencoder`, and `tiny-cuda-nn` (`nerf/network_tcnn.py`,
   `nerf/renderer.py`). These must be built/installed (see `submodules/diffusionerf/environment.yml`
   and its README) or `train.py` won't even import. Run submodules are checked out from
   `github.com/Wenri/diffusionerf`.

The pretrained diffusion checkpoint `models/rgbd-patch-diffusion.pt` (~580 MB, Niantic) must be
present — `DiffusionTrainer` asserts on it. It is already in `models/` here.

## Common commands

```shell
# Train one DTU scan (COLMAP layout: <scan>/sparse, <scan>/images). --eval holds out a test split.
python train.py -s dtu_15views_colmap_7000/scan24 -m <output_dir> --eval

# Render train+test sets from a trained model (reads cfg_args from the model dir)
python render.py -m <output_dir>

# Compute PSNR/SSIM/LPIPS over rendered images; writes results.json / per_view.json
python metrics.py -m <output_dir>          # -m accepts multiple model dirs
```

There is no test suite, linter, or CI. Verification is by training a scan and reading the
PSNR/SSIM/LPIPS metrics.

`full_eval.py` is upstream and targets MipNeRF360/T&T/Deep Blending — it is **not** the DTU
few-view pipeline and is effectively unused here.

## Architecture

### Argument flow (this fork's own system)
`arguments/gaussian.py::parse_args(*ParamGroups)` builds one `ArgumentParser`, lets each
`ParamGroup` register its args, then returns `(args, *extracted_namespaces)`. Each `ParamGroup`
(`arguments/__init__.py`) is a `SimpleNamespace` subclass; `extract(args)` pulls just that group's
keys back out into its own namespace. `train.py`'s `main()` receives them positionally as
`(dataset, opt, pipe, diffusion)`. Param groups:
- `ModelParams` / `PipelineParams` / `OptimizationParams` — `arguments/__init__.py` (upstream-ish).
- `DiffusionParams` — `arguments/diffusion.py` (patch weights, reg schedule, diffusion time, the
  `patch_regulariser_path`).

`render.py` instead uses `get_combined_args`, which `eval()`s the `cfg_args` file written into the
model dir at train time and overlays CLI args — so render-time model/pipeline params are inherited
from training automatically.

### Training loop — `train.py::GBCTrainer`
`GBCTrainer` **subclasses `GaussianModel`** (it *is* the model) and additionally owns the `Scene`,
the `DiffusionTrainer`, and the optimizer. Per iteration it:
1. Renders the picked camera → `RenderData` (see below).
2. Computes the standard loss `(1-λ)·L1 + λ·(1-SSIM)`.
3. Adds the diffusion patch loss: `loss += weight * patch_outputs.loss`, where
   `weight`/`time` follow a schedule that ramps off at `patch_reg_finish_step` (default 8000).
4. Runs upstream densify/prune/opacity-reset on the schedule in `OptimizationParams`.

The diffusion call is wrapped in `redirect_stdout` and its captured text is piped into the tqdm
progress bar description.

### Render contract — `gaussian_renderer/__init__.py`
`render(...)` returns a **`RenderData` namedtuple**:
`(image, viewspace_points, visibility_filter, radii, depth, alpha)`. The added `depth`/`alpha`
fields are what the RGBD diffusion regularizer consumes, and they only exist because of the forked
rasterizer. Callers use attribute access (`render_pkg.image`, `render_pkg.depth`) and
`render_pkg._asdict()`.

### Diffusion regularizer — `arguments/diffusion.py`
`DiffusionTrainer` subclasses `PatchRegulariser` from the `diffusionerf` submodule and bridges the
two camera conventions:
- `IntrinsicsCamera` adapts a 3DGS `scene.cameras.Camera` to diffusionerf's `Intrinsics` (handles
  the c2w↔w2v transform and FoV↔focal conversion via `utils/graphics_utils`).
- `RandomCameraGenerator` perturbs training poses to synthesize novel patch viewpoints.

Each step `patch_regulariser(global_step, data)` stochastically picks (p≈0.25) between:
- **rendered patch** at a perturbed novel pose (`get_diffusion_loss_with_rendered_patch`), or
- **sampled patch** from a real training image (`get_diffusion_loss_with_sampled_patch`).

The denoising RGBD diffusion model scores the (rendered RGB + rendered depth) patch; that score
becomes the regularization gradient. `PlotDebugger` gives optional matplotlib visualization (off by
default).

### Scene & data — `scene/`
`Scene.__init__` auto-detects format by source path: `sparse/` present → COLMAP reader; else
`transforms_train.json` → Blender reader (`scene/dataset_readers.py`,
`sceneLoadTypeCallbacks`). DTU scans use the **COLMAP** path. With `--eval`, COLMAP scenes hold out
every 8th image (`llffhold=8`) as the test set.

`GaussianModel` (`scene/gaussian_model.py`) holds the optimizable tensors and all densify/prune
logic. Note `create_from_pcd(..., max_points=10000)` — the initial point cloud is **capped at
10 000 points** (upstream allows millions), a deliberate choice for the sparse-view regime. The
`capture`/`restore` pair drives checkpointing.

### Submodules (`.gitmodules`)
- `diff-gaussian-rasterization` → **Wenri fork** (adds depth/alpha forward+backward).
- `diffusionerf` → **Wenri fork** of Niantic DiffusioNeRF; supplies the patch regularizer, ray
  utils, and CUDA extensions (`raymarching`, `gridencoder`, `shencoder`, `ffmlp`).
- `simple-knn`, `SIBR_viewers` → upstream (SIBR viewers are not part of the training pipeline).

## Conventions & gotchas

- **Cameras are CUDA-resident.** `data_device` defaults to `cuda`; switch to `cpu` for larger
  inputs. `bg_color` is white only when `--white_background`.
- **Output layout per run:** `<model_dir>/{cfg_args, cameras.json, input.ply, point_cloud/iteration_*/point_cloud.ply, train/, test/, results.json, per_view.json}` plus TensorBoard `events.*`. `render.py` reads `cfg_args`; don't delete it.
- **Two coordinate conventions coexist.** 3DGS uses `world_view_transform` (w2v, column-vector
  convention with `.T` in places); diffusionerf uses c2w poses + an `Intrinsics` dataclass. The
  `IntrinsicsCamera` / `RandomCameraGenerator` adapters are where they meet — changes there are
  the usual source of "renders look transformed/mirrored" bugs (cf. git history: "align c2w and w2v
  camera intrinsics", "solving name field conflict").
- `scripts/` holds **one-off DTU data-prep helpers** (`colmap_from_json.py`, `addinfo.py`,
  `main.py`, `plytotxt.py`), not part of the train/render/metrics flow. They contain hardcoded
  paths and are run manually.
- The many top-level `dtu_*` and `data_output/` directories are **datasets and experiment outputs**
  (gitignored / untracked), not source.

import dataclasses
import random
from contextlib import nullcontext, AbstractContextManager
from itertools import groupby
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from einops import rearrange

from gaussian_renderer import render
from scene.cameras import Camera
from submodules.diffusionerf.main_nerf import run
from submodules.diffusionerf.nerf.learned_regularisation.intrinsics import Intrinsics
from submodules.diffusionerf.nerf.learned_regularisation.patch_pose_generator import PatchPoseGenerator, \
    unpack_4x4_transform
from submodules.diffusionerf.nerf.learned_regularisation.patch_regulariser import PatchRegulariser, \
    load_patch_diffusion_model, make_random_patch_intrinsics, sample_patch_from_img
from submodules.diffusionerf.nerf.utils import get_rays
from utils.graphics_utils import focal2fov, fov2focal
from . import ParamGroup


class DiffusionParams(ParamGroup):
    def __init__(self, parser=None):
        super().__init__(
            parser, "Diffusion Parameters",
            ckpt='latest',
            patch_regulariser_path='models/rgbd-patch-diffusion.pt',
            patch_sample_downscale_factor=4,
            patch_weight_start=0.2,
            patch_weight_finish=0.3,
            patch_reg_start_step=0,
            patch_reg_finish_step=8000,
            # reg_ramp_start_step=3000,
            # reg_ramp_finish_step=8000,
            initial_diffusion_time=0.1,
            normalise_diffusion_losses=True,
            # apply_geom_reg_to_patches=False,
            # spread_loss_strength=1.5e-05,
        )


class RandomCameraGenerator(PatchPoseGenerator):
    def __init__(self, cameras):
        super().__init__(poses=cameras,
                         spatial_perturbation_magnitude=0.2,
                         angular_perturbation_magnitude_rads=0.2 * np.pi,
                         no_perturb_prob=0.,
                         frustum_checker=None)

    def _perturb_pose(self, camera):
        pose_to_perturb = camera.world_view_transform.T.cpu()
        pose_to_perturb = super()._perturb_pose(pose_to_perturb)
        return pose_to_perturb


class IntrinsicsCamera(Camera, Intrinsics):
    def __init__(self, pose, ref=None, *args, **kwargs):
        self.call_super_init = True
        R, t = unpack_4x4_transform(pose.inverse().cpu())
        super().__init__(
            *args, **kwargs,
            uid=None, colmap_id=None, image_name=None, gt_alpha_mask=None, R=R.T, T=t,
            image=torch.empty((0, ref.height, ref.width), dtype=torch.float32),
            FoVx=focal2fov(ref.fx, ref.width), FoVy=focal2fov(ref.fy, ref.height),
        )
        assert ref.cx * 2 == ref.width and ref.cy * 2 == ref.height, \
            f'Image size must be divisible by downscale factor'


class PlotDebugger:
    def __init__(self, enabled=False):
        self.enabled = enabled
        self.cache = []

    def __bool__(self):
        return self.enabled

    def __call__(self, *args, **kwargs):
        if not self.enabled:
            return

        from matplotlib import pyplot as plt
        plt.figure()
        plt.imshow(torch.concat((*self.cache, *args), dim=1).cpu())
        self.cache.clear()
        plt.show()


class DiffusionTrainer(PatchRegulariser):
    def __init__(self, opt, trainer):
        print('main.nerf running with options', opt)
        assert opt.patch_regulariser_path
        self.opt = opt
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        patch_diffusion_model = load_patch_diffusion_model(Path(opt.patch_regulariser_path))
        pose_generator = RandomCameraGenerator(cameras=[a for v in trainer.scene.train_cameras.values() for a in v])
        # pseudo_intrinsics = LLFF_DEFAULT_PSEUDO_INTRINSICS
        (pseudo_intrinsics, _), = groupby(Intrinsics(
            fx=fov2focal(a.FoVx, a.image_width), fy=fov2focal(a.FoVy, a.image_height),
            cx=a.image_width / 2, cy=a.image_height / 2, width=a.image_width, height=a.image_height,
        ) for v in trainer.scene.train_cameras.values() for a in v)

        print('Using patch full image pseudo intrinsics', pseudo_intrinsics)
        super().__init__(pose_generator=pose_generator,
                         patch_diffusion_model=patch_diffusion_model,
                         full_image_intrinsics=pseudo_intrinsics,
                         device=device,
                         planar_depths=True,
                         frustum_regulariser=None,
                         sample_downscale_factor=opt.patch_sample_downscale_factor,
                         uniform_in_depth_space=opt.normalise_diffusion_losses)
        self.model = trainer
        self.debug = PlotDebugger()

    def _get_random_patch_intrinsics(self, pose):
        intrinsics_random, intrinsics_downscaled = make_random_patch_intrinsics(
            patch_size=self._patch_size,
            full_image_intrinsics=self._full_image_intrinsics,
            downscale_factor=self._sample_downscale_factor,
        )
        return IntrinsicsCamera(pose, intrinsics_downscaled, **dataclasses.asdict(intrinsics_random))

    def _render_patch_with_intrinsics(self, intrinsics: IntrinsicsCamera, pose, model):
        pseudo_intrinsics = (intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy)
        bg = torch.tensor(self.model.bg_color, dtype=torch.float32, device="cuda")
        outputs = render(intrinsics, self.model, self.model.pipe, bg)

        pred_depth, pred_rgb, patch_rays = self._sample_patch_with_intrinsics(
            rearrange(outputs.image, 'C H W -> H W C'),
            rearrange(outputs.depth, 'C H W -> H W C'),
            pose, (intrinsics.fx, intrinsics.fy, intrinsics.image_width / 2, intrinsics.image_height / 2),
            pseudo_intrinsics, gt_context=nullcontext)

        if self.debug:
            self.debug(pred_rgb[0].detach())

        return pred_depth, pred_rgb, patch_rays, outputs

    def _sample_patch(self, image, image_intrinsics, pose, model):
        patch_intrinsics = self._get_random_patch_intrinsics(pose)
        pseudo_intrinsics = (patch_intrinsics.fx, patch_intrinsics.fy, patch_intrinsics.cx, patch_intrinsics.cy)
        rendered_depth, gt_rgb, patch_rays = self._sample_patch_with_intrinsics(
            rearrange(image.viewpoint_cam.original_image, 'C H W -> H W C'),
            rearrange(image.depth, 'C H W -> H W C'),
            pose, image_intrinsics, pseudo_intrinsics)

        if self.debug:
            pred_rgb = sample_patch_from_img(
                img_intrinsics=image_intrinsics, img=rearrange(image.image, 'C H W -> H W C'),
                rays_d=patch_rays['rays_d_cam'], patch_size=self._patch_size)
            self.debug(gt_rgb[0].detach(), pred_rgb[0].detach())

        return rendered_depth, gt_rgb, image

    def _sample_patch_with_intrinsics(self, image, depth, pose, image_intrinsics, pseudo_intrinsics,
                                      gt_context: Callable[[], AbstractContextManager] = torch.no_grad):
        patch_rays = get_rays(poses=pose.unsqueeze(0), intrinsics=pseudo_intrinsics,
                              H=self._patch_size, W=self._patch_size, N=-1)

        pred_depth = sample_patch_from_img(
            img_intrinsics=image_intrinsics, img=depth,
            patch_size=self._patch_size, rays_d=patch_rays['rays_d_cam'])

        if self._planar_depths:
            pred_depth *= patch_rays['rays_d_cam_z'].reshape(1, self._patch_size, self._patch_size, 1)

        with gt_context():
            gt_rgb = sample_patch_from_img(
                img_intrinsics=image_intrinsics, img=image,
                rays_d=patch_rays['rays_d_cam'], patch_size=self._patch_size)

        return pred_depth, gt_rgb, patch_rays

    def patch_regulariser(self, global_step, data):
        # t schedule
        lambda_t = np.clip((global_step - self.opt.patch_reg_start_step) /
                           (self.opt.patch_reg_finish_step - self.opt.patch_reg_start_step),
                           0., 1.)
        weight = self.opt.patch_weight_start + (self.opt.patch_weight_finish - self.opt.patch_weight_start) * lambda_t
        if global_step > self.opt.patch_reg_finish_step:
            time = 0.
        elif global_step > self.opt.patch_reg_start_step:
            time = self.opt.initial_diffusion_time * (1. - lambda_t)
        else:
            raise RuntimeError('Internal error')
        p_sample_patch = 0.25
        # self.debug.enabled = global_step % 500 == 0

        patch_outputs = self.get_diffusion_loss_with_rendered_patch(
            model=self.model, time=time
        ) if random.random() >= p_sample_patch else self.get_diffusion_loss_with_sampled_patch(
            model=self.model, time=time, image=data, image_intrinsics=(
                fov2focal(data.viewpoint_cam.FoVx, data.viewpoint_cam.image_width),
                fov2focal(data.viewpoint_cam.FoVy, data.viewpoint_cam.image_height),
                data.viewpoint_cam.image_width / 2, data.viewpoint_cam.image_height / 2),
            pose=data.viewpoint_cam.world_view_transform.T.inverse())

        return patch_outputs, weight, time

    # def get_linear_dynamic_reg_modifier(self, global_step):
    #     dynamic_reg_start_step = self.opt.reg_ramp_start_step
    #     dynamic_reg_max_strength_step = self.opt.reg_ramp_finish_step
    #     # Linear scheme
    #     if global_step > dynamic_reg_max_strength_step:
    #         dynamic_reg_modifier = 1.
    #     elif global_step > dynamic_reg_start_step:
    #         dynamic_reg_modifier = (global_step - dynamic_reg_start_step) / (
    #                 dynamic_reg_max_strength_step - dynamic_reg_start_step)
    #     else:
    #         dynamic_reg_modifier = 0.
    #
    #     # Geometric reg
    #     # if self.opt.apply_geom_reg_to_patches:
    #     #     spread_loss_weight = self.opt.spread_loss_strength * self.get_linear_dynamic_reg_modifier(global_step)
    #     #     loss += spread_loss_weight * patch_outputs.render_outputs['loss_dist']
    #
    #     return dynamic_reg_modifier


if __name__ == '__main__':
    run(DiffusionParams())

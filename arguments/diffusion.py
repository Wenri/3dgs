import random
from pathlib import Path

import numpy as np
import torch
from einops import rearrange

from arguments import ParamGroup
from gaussian_renderer import render
from scene.cameras import Camera
from submodules.diffusionerf.main_nerf import run
from submodules.diffusionerf.nerf.learned_regularisation.patch_pose_generator import PatchPoseGenerator, \
    unpack_4x4_transform
from submodules.diffusionerf.nerf.learned_regularisation.patch_regulariser import PatchRegulariser, \
    load_patch_diffusion_model, LLFF_DEFAULT_PSEUDO_INTRINSICS
from submodules.diffusionerf.nerf.utils import get_rays
from utils.graphics_utils import focal2fov


class DiffusionParams(ParamGroup):
    def __init__(self, parser=None):
        super().__init__(
            parser, "Diffusion Parameters",
            ckpt='latest',
            patch_regulariser_path='models/rgbd-patch-diffusion.pt',
            patch_sample_downscale_factor=4,
            patch_weight_start=0.2,
            patch_weight_finish=0.2,
            patch_reg_start_step=0,
            patch_reg_finish_step=2500,
            reg_ramp_start_step=3000,
            reg_ramp_finish_step=8000,
            initial_diffusion_time=0.1,
            normalise_diffusion_losses=True,
            apply_geom_reg_to_patches=False,
            spread_loss_strength=1.5e-05,
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


class DiffusionTrainer(PatchRegulariser):
    def __init__(self, opt, trainer):
        print('main.nerf running with options', opt)
        assert opt.patch_regulariser_path
        self.opt = opt
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        patch_diffusion_model = load_patch_diffusion_model(Path(opt.patch_regulariser_path))
        pose_generator = RandomCameraGenerator(cameras=[a for v in trainer.scene.train_cameras.values() for a in v])
        pseudo_intrinsics = LLFF_DEFAULT_PSEUDO_INTRINSICS
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

    def _render_patch_with_intrinsics(self, intrinsics, pose, model):
        pseudo_intrinsics = (intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy)

        patch_rays = get_rays(poses=pose.unsqueeze(0), intrinsics=pseudo_intrinsics,
                              H=self._patch_size, W=self._patch_size, N=-1)
        bg = torch.tensor(self.model.bg_color, dtype=torch.float32, device="cuda")
        R, t = unpack_4x4_transform(pose.cpu())
        viewpoint_cam = Camera(
            uid=None, colmap_id=None, image_name=None, gt_alpha_mask=None, R=R, T=t,
            image=torch.empty((0, intrinsics.width, intrinsics.height), dtype=torch.float32, device="cuda"),
            FoVx=focal2fov(intrinsics.fx, intrinsics.width), FoVy=focal2fov(intrinsics.fy, intrinsics.height),
        )
        outputs = render(viewpoint_cam, self.model, self.model.pipe, bg)
        B = 1

        pred_depth = rearrange(outputs.depth, 'C H W -> 1 H W C')
        if self._planar_depths:
            pred_depth = pred_depth * patch_rays['rays_d_cam_z'].reshape(B, intrinsics.height, intrinsics.width, 1)

        pred_rgb = rearrange(outputs.image, 'C H W -> 1 H W C')

        return pred_depth, pred_rgb, patch_rays, outputs

    def get_linear_dynamic_reg_modifier(self, global_step):
        dynamic_reg_start_step = self.opt.reg_ramp_start_step
        dynamic_reg_max_strength_step = self.opt.reg_ramp_finish_step
        # Linear scheme
        if global_step > dynamic_reg_max_strength_step:
            dynamic_reg_modifier = 1.
        elif global_step > dynamic_reg_start_step:
            dynamic_reg_modifier = (global_step - dynamic_reg_start_step) / (
                    dynamic_reg_max_strength_step - dynamic_reg_start_step)
        else:
            dynamic_reg_modifier = 0.
        return dynamic_reg_modifier

    def patch_regulariser(self, global_step, data):
        # t schedule
        initial_diffusion_time = self.opt.initial_diffusion_time
        patch_reg_start_step = self.opt.patch_reg_start_step
        patch_reg_finish_step = self.opt.patch_reg_finish_step
        weight_start = self.opt.patch_weight_start
        weight_finish = self.opt.patch_weight_finish

        lambda_t = (global_step - patch_reg_start_step) / (patch_reg_finish_step - patch_reg_start_step)
        lambda_t = np.clip(lambda_t, 0., 1.)
        weight = weight_start + (weight_finish - weight_start) * lambda_t

        if global_step > patch_reg_finish_step:
            time = 0.
        elif global_step > patch_reg_start_step:
            time = initial_diffusion_time * (1. - lambda_t)
        else:
            raise RuntimeError('Internal error')
        p_sample_patch = 0.25
        if random.random() >= p_sample_patch or True:
            patch_outputs = self.get_diffusion_loss_with_rendered_patch(model=self.model, time=time)
        else:
            patch_outputs = self.get_diffusion_loss_with_sampled_patch(
                model=self.model, time=time, image=data['images_full'][0], image_intrinsics=data['intrinsics'],
                pose=data['pose_c2w'][0]
            )
        loss = weight * patch_outputs.loss

        # Geometric reg
        if self.opt.apply_geom_reg_to_patches:
            spread_loss_weight = self.opt.spread_loss_strength * self.get_linear_dynamic_reg_modifier(global_step)
            loss += spread_loss_weight * patch_outputs.render_outputs['loss_dist']

        return loss, patch_outputs


if __name__ == '__main__':
    run(DiffusionParams())

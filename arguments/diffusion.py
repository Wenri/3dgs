import random
from pathlib import Path

import numpy as np
import torch

from arguments import ParamGroup
from submodules.diffusionerf.main_nerf import run
from submodules.diffusionerf.nerf.learned_regularisation.patch_pose_generator import PatchPoseGenerator
from submodules.diffusionerf.nerf.learned_regularisation.patch_regulariser import PatchRegulariser, \
    load_patch_diffusion_model, LLFF_DEFAULT_PSEUDO_INTRINSICS


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
            initial_diffusion_time=0.1,
            normalise_diffusion_losses=True,
            apply_geom_reg_to_patches=True,
        )


class DiffusionTrainer(PatchRegulariser):
    def __init__(self, opt, trainer):
        print('main.nerf running with options', opt)
        assert opt.patch_regulariser_path
        self.opt = opt
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        poses = [a.world_view_transform.T.cpu().numpy() for v in trainer.scene.train_cameras.values() for a in v]

        patch_diffusion_model = load_patch_diffusion_model(Path(opt.patch_regulariser_path))
        pose_generator = PatchPoseGenerator(poses=poses,
                                            spatial_perturbation_magnitude=0.2,
                                            angular_perturbation_magnitude_rads=0.2 * np.pi,
                                            no_perturb_prob=0.,
                                            frustum_checker=None)
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

    def patch_regulariser(self, data, global_step):
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
        if random.random() >= p_sample_patch:
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

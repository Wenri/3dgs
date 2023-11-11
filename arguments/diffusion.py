from pathlib import Path

from arguments import ParamGroup
from submodules.diffusionerf.main_nerf import run


class DiffusionParams(ParamGroup):
    def __init__(self, parser=None):
        super().__init__(
            parser, "Diffusion Parameters",
            path=Path('/home/wenri/pCloudDrive/ResearchProjects/3DGS/nerf_llff_data/room'),
            test=False,
            workspace=Path('runs/example/3_poses/room'),
            seed=0,
            iters=10000,
            lr=0.01,
            ckpt='latest',
            num_rays=1024,
            num_steps=512,
            upsample_steps=128,
            max_ray_batch=1024,
            downsample_val=8,
            downsample_train=8,
            test_mode=('test_eval',),
            eval_interval=5,
            max_val_imgs=None,
            num_train_poses=3,
            patch_regulariser_path='models/rgbd-patch-diffusion.pt',
            patch_sample_downscale_factor=4,
            patch_weight_start=0.2,
            patch_weight_finish=0.2,
            patch_reg_start_step=0,
            patch_reg_finish_step=2500,
            initial_diffusion_time=0.1,
            normalise_diffusion_losses=True,
            apply_geom_reg_to_patches=True,
            spread_loss_strength=1.5e-05,
            seg_loss_strength=1e-06,
            weights_sum_loss_strength=0.001,
            net_l2_loss_strength=1e-07,
            reg_ramp_start_step=3000,
            reg_ramp_finish_step=8000,
            use_frustum_regulariser=True,
            frustum_check_patches=False,
            frustum_regularise_patches=True,
            frustum_reg_initial_weight=1.0,
            frustum_reg_final_weight=0.01,
            mode='llff',
            color_space='srgb',
            preload=False,
            bound=7.5,
            scale=1.5,
            min_near=0.2,
            density_thresh=10,
            error_map=False,
            clip_text='',
            rand_pose=-1,
            only_run_on=('room',),
            num_train=(3,),
            force=False,
            normalise_length_scales_to=None,
            normalise_length_scales_to_at_least=7.5
        )


if __name__ == '__main__':
    run(DiffusionParams())

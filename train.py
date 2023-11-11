#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import uuid
from random import randint
from types import SimpleNamespace

import torch
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, OptimizationParams
from arguments.diffusion import DiffusionParams, DiffusionTrainer
from arguments.gaussian import parse_args
from gaussian_renderer import render, network_gui
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim


class GBCTrainer(GaussianModel):
    def __init__(self, dataset, opt, pipe, diffusion, checkpoint=None):
        super().__init__(dataset.sh_degree)
        self.tb_writer = self.prepare_output_and_logger(dataset)
        self.scene = Scene(dataset, self)
        self.training_setup(opt)
        self.opt = opt
        self.dataset = dataset
        self.pipe = pipe
        self.diffusion_trainer = DiffusionTrainer(diffusion, self)
        if checkpoint:
            (model_params, first_iter) = torch.load(checkpoint)
            self.restore(model_params, self.opt)

    def training(self, testing_iterations, saving_iterations, checkpoint_iterations, debug_from):
        first_iter = 0

        bg_color = [1, 1, 1] if self.dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        iter_start = torch.cuda.Event(enable_timing=True)
        iter_end = torch.cuda.Event(enable_timing=True)

        viewpoint_stack = None
        ema_loss_for_log = 0.0
        progress_bar = tqdm(range(first_iter, self.opt.iterations), desc="Training progress")
        first_iter += 1
        for iteration in range(first_iter, self.opt.iterations + 1):
            if network_gui.conn is None:
                network_gui.try_connect()
            while network_gui.conn is not None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, self.pipe.convert_SHs_python, self.pipe.compute_cov3D_python, keep_alive, \
                        scaling_modifer = network_gui.receive()
                    if custom_cam is not None:
                        net_image = render(custom_cam, self, self.pipe, background, scaling_modifer).image
                        net_image_bytes = (torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(
                            1, 2, 0).contiguous().cpu().numpy().data
                    network_gui.send(net_image_bytes, self.dataset.source_path)
                    if do_training and ((iteration < int(self.opt.iterations)) or not keep_alive):
                        break
                except Exception as e:
                    network_gui.conn = None

            iter_start.record()

            self.update_learning_rate(iteration)

            # Every 1000 its we increase the levels of SH up to a maximum degree
            if iteration % 1000 == 0:
                self.oneupSHdegree()

            # Pick a random Camera
            if not viewpoint_stack:
                viewpoint_stack = self.scene.getTrainCameras().copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

            # Render
            if (iteration - 1) == debug_from:
                self.pipe.debug = True

            bg = torch.rand(3, device="cuda") if self.opt.random_background else background

            render_pkg = render(viewpoint_cam, self, self.pipe, bg)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg.image, render_pkg.viewspace_points, \
                render_pkg.visibility_filter, render_pkg.radii

            # Loss
            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - self.opt.lambda_dssim) * Ll1 + self.opt.lambda_dssim * (1.0 - ssim(image, gt_image))
            loss.backward()

            iter_end.record()

            with torch.no_grad():
                # Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                if iteration % 10 == 0:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                    progress_bar.update(10)
                if iteration == self.opt.iterations:
                    progress_bar.close()

                # Log and save
                self.training_report(self.tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
                                     testing_iterations, self.scene, render, (self.pipe, background))
                if iteration in saving_iterations:
                    print("\n[ITER {}] Saving Gaussians".format(iteration))
                    self.scene.save(iteration)

                # Densification
                if iteration < self.opt.densify_until_iter:
                    # Keep track of max radii in image-space for pruning
                    self.max_radii2D[visibility_filter] = torch.max(self.max_radii2D[visibility_filter],
                                                                    radii[visibility_filter])
                    self.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    if iteration > self.opt.densify_from_iter and iteration % self.opt.densification_interval == 0:
                        size_threshold = 20 if iteration > self.opt.opacity_reset_interval else None
                        self.densify_and_prune(self.opt.densify_grad_threshold, 0.005, self.scene.cameras_extent,
                                               size_threshold)

                    if iteration % self.opt.opacity_reset_interval == 0 or (
                            self.dataset.white_background and iteration == self.opt.densify_from_iter):
                        self.reset_opacity()

                # Optimizer step
                if iteration < self.opt.iterations:
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

                if iteration in checkpoint_iterations:
                    print("\n[ITER {}] Saving Checkpoint".format(iteration))
                    torch.save((self.capture(), iteration), self.scene.model_path + "/chkpnt" + str(iteration) + ".pth")

    @staticmethod
    def prepare_output_and_logger(args):
        if not args.model_path:
            if os.getenv('OAR_JOB_ID'):
                unique_str = os.getenv('OAR_JOB_ID')
            else:
                unique_str = str(uuid.uuid4())
            args.model_path = os.path.join("./output/", unique_str[0:10])

        # Set up output folder
        print("Output folder: {}".format(args.model_path))
        os.makedirs(args.model_path, exist_ok=True)
        with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
            cfg_log_f.write(str(SimpleNamespace(**vars(args))))

        # Create Tensorboard writer
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_writer = SummaryWriter(args.model_path)
        except ImportError:
            print("Tensorboard not available: not logging progress")
            tb_writer = None

        return tb_writer

    @staticmethod
    def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene,
                        renderFunc, renderArgs):
        if tb_writer:
            tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
            tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
            tb_writer.add_scalar('iter_time', elapsed, iteration)

        # Report test and samples of training set
        if iteration in testing_iterations:
            torch.cuda.empty_cache()
            validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                                  {'name': 'train',
                                   'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in
                                               range(5, 30, 5)]})

            for config in validation_configs:
                if config['cameras'] and len(config['cameras']) > 0:
                    l1_test = 0.0
                    psnr_test = 0.0
                    for idx, viewpoint in enumerate(config['cameras']):
                        image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs).image, 0.0, 1.0)
                        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                        if tb_writer and (idx < 5):
                            tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name),
                                                 image[None], global_step=iteration)
                            if iteration == testing_iterations[0]:
                                tb_writer.add_images(
                                    config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                    gt_image[None], global_step=iteration)
                        l1_test += l1_loss(image, gt_image).mean().double()
                        psnr_test += psnr(image, gt_image).mean().double()
                    psnr_test /= len(config['cameras'])
                    l1_test /= len(config['cameras'])
                    print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test,
                                                                            psnr_test))
                    if tb_writer:
                        tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                        tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

            if tb_writer:
                tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
                tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
            torch.cuda.empty_cache()


def main(args, *extra_params):
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    GBCTrainer(*extra_params, checkpoint=args.start_checkpoint).training(
        args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.debug_from)

    # All done
    print("\nTraining complete.")


if __name__ == "__main__":
    main(*parse_args(ModelParams, OptimizationParams, PipelineParams, DiffusionParams))

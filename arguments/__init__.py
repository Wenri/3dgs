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
import sys
from argparse import ArgumentParser
from types import SimpleNamespace
from typing import Optional


class ParamGroup(SimpleNamespace):
    def __init__(self, parser: Optional[ArgumentParser] = None, name=None, fill_none=False, **kwargs):
        super().__init__(**kwargs)
        if parser:
            group = parser.add_argument_group(name or self.__class__.__name__)
            for key, value in vars(self).items():
                shorthand = False
                if key.startswith("_"):
                    shorthand = True
                    key = key[1:]
                t = type(value)
                value = value if not fill_none else None
                if shorthand:
                    if t == bool:
                        group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                    else:
                        group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
                else:
                    if t == bool:
                        group.add_argument("--" + key, default=value, action="store_true")
                    else:
                        group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = {k: v for k, v in vars(args).items() if k in vars(self) or f"_{k}" in vars(self)}
        return SimpleNamespace(**group)


class ModelParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        super().__init__(
            parser, "Loading Parameters", sentinel,
            sh_degree=3,
            _source_path="",
            _model_path="",
            _images="images",
            _resolution=-1,
            _white_background=False,
            data_device="cuda",
            eval=False
        )

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g


class PipelineParams(ParamGroup):
    def __init__(self, parser):
        super().__init__(
            parser, "Pipeline Parameters",
            convert_SHs_python=False,
            compute_cov3D_python=False,
            debug=False
        )


class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        super().__init__(
            parser, "Optimization Parameters",
            iterations=30_000,
            position_lr_init=0.00016,
            position_lr_final=0.0000016,
            position_lr_delay_mult=0.01,
            position_lr_max_steps=30_000,
            feature_lr=0.0025,
            opacity_lr=0.05,
            scaling_lr=0.005,
            rotation_lr=0.001,
            percent_dense=0.01,
            lambda_dssim=0.2,
            densification_interval=100,
            opacity_reset_interval=3000,
            densify_from_iter=500,
            densify_until_iter=15_000,
            densify_grad_threshold=0.0002,
            random_background=False,
        )


def get_combined_args(parser: ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k, v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return SimpleNamespace(**merged_dict)

import sys
from argparse import ArgumentParser
from operator import methodcaller

from . import get_combined_args


def parse_args(*params):
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    extracts = [p(parser) for p in params]
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    return args, *map(methodcaller("extract", args), extracts)


def parse_test_args(*params):
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    extracts = [p(parser) for p in params]
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    return args, *map(methodcaller("extract", args), extracts)

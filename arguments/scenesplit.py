import os
import random
from collections import UserDict
from difflib import get_close_matches
from pathlib import Path

from submodules.PatchmatchNet.colmap_input import colmap_input_parser, main as colmap_input_main
from submodules.PatchmatchNet.eval import eval_parser, main as eval_main


class SceneSplit:
    _SPLIT = {"train", "test"}

    def __init__(self, items, path):
        img_path = Path(path)
        self.path = img_path.parent

        if True or not all(self.path.joinpath(f'{name}.txt').exists() for name in self._SPLIT):
            self.gen_split(items, img_path, self.path.joinpath('3_views', 'images'))

        kwargs = {'input_folder': os.fspath(self.path.joinpath('3_views', 'dense'))}
        colmap_input_main(colmap_input_parser().parse_args(f'--{k}={v}' for k, v in kwargs.items()))

        kwargs.update({
            'output_folder': os.fspath(self.path.joinpath('3_views', 'patchmatch')),
            'checkpoint_path': os.path.join('submodules', 'PatchmatchNet', 'checkpoints', 'params_000007.ckpt'),
            'num_views': 7, 'image_max_dim': 2048, 'geo_mask_thres': 2, 'photo_thres': 0.8,
        })
        eval_main(eval_parser().parse_args(f'--{k}={v}' for k, v in kwargs.items()))

    def gen_split(self, items, img_path: Path, sub_path: Path):
        possibilities = [os.path.splitext(it)[0] for it in items]
        match = {img.stem: get_close_matches(img.stem, possibilities, n=1)[0] for img in img_path.iterdir()}

        with Path(self.path, 'train.txt').open('w') as f:
            for img in sub_path.iterdir():
                print(img.stem, match.pop(img.stem), file=f)

        with Path(self.path, 'test.txt').open('w') as f:
            for k, v in match.items():
                print(k, v, file=f)

    def get_mapping(self, name):
        return MappingInfo(Path(self.path, f"{name}.txt"), strict=name in self._SPLIT)


class MappingInfo(UserDict):
    def __init__(self, mapping: str | os.PathLike, strict: bool = False):
        self.mapping = mapping
        self.strict = strict
        try:
            with open(mapping) as f:
                super().__init__(map(str.split, filter(None, map(str.strip, f))))
        except FileNotFoundError:
            super().__init__()

    def map_name(self, name):
        return self.get(name, name) if not self.strict else self[name]

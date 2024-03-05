import os
import random
from collections import UserDict
from difflib import get_close_matches
from io import StringIO
from pathlib import Path


class SceneSplit:
    _SPLIT = {"train", "test"}

    def __init__(self, items, path):
        img_path = Path(path)
        self.path = img_path.parent

        if not all(self.path.joinpath(f'{name}.txt').exists() for name in self._SPLIT):
            self.gen_split(items, img_path)

    def gen_split(self, items, img_path: Path):
        possibilities = [os.path.splitext(it)[0] for it in items]

        with StringIO() as f:
            for img in img_path.iterdir():
                q = img.stem
                match, = get_close_matches(q, possibilities, n=1)
                print(match, q, file=f)
            mapping = f.getvalue().splitlines(keepends=True)

        random.shuffle(mapping)

        with Path(self.path, 'train.txt').open('w') as f:
            f.writelines(mapping[:3])

        with Path(self.path, 'test.txt').open('w') as f:
            f.writelines(mapping[3:])

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

import os.path
from pathlib import Path
from typing import Union

PathLike = Union[str, Path]


def root_dir() -> Path:
    current_path = Path.cwd()
    for path in [current_path, *current_path.parents]:
        if (path / "src").is_dir():
            return path
    raise FileNotFoundError("src not found. Are you sure this is a right project?")


def join(path: PathLike, *paths) -> Path:
    return os.path.join(path, paths)


def mkdirs(path: PathLike, exist_ok: bool=True):
    return os.makedirs(path, exist_ok=exist_ok)
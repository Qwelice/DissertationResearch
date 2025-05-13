from abc import ABC
from pathlib import Path
from typing import Union

PathLike = Union[str, Path]


def _find_project_root(current_path: PathLike = Path.cwd()) -> Path:
    if isinstance(current_path, str):
        current_path = Path(current_path)
    for path in [current_path, *current_path.parents]:
        if (path / "src").is_dir():
            return path
    raise FileNotFoundError("src not found. Are you sure this is a right project?")


class PathHandler(ABC):
    def __init__(self):
        pass
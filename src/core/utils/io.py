import os.path
from abc import ABC
from pathlib import Path
from typing import Union, Optional

PathLike = Union[str, Path]


def _find_project_root(current_path: PathLike = Path.cwd()) -> Path:
    if isinstance(current_path, str):
        current_path = Path(current_path)
    for path in [current_path, *current_path.parents]:
        if (path / "src").is_dir():
            return path
    raise FileNotFoundError("src not found. Are you sure this is a right project?")


class _PathManager:
    def __init__(self, home: Optional[PathLike]=None):
        if home is None:
            home = _find_project_root()
        if isinstance(home, str):
            home = Path(home)
            
        self._home: Path = home
        
    @property
    def home (self) -> Path:
        return self._home

    def join(self, path: PathLike, *paths) -> Path:
        return Path(os.path.join(path, *paths))

    def home_join(self, *paths) -> Path:
        return Path(self.join(self.home, *paths))

    def open(self, path: PathLike, mode: str, encoding: Optional[str]=None):
        return open(file=path, mode=mode, encoding=encoding)

    def openw(self, path: PathLike, encoding: Optional[str]=None):
        return self.open(path, 'w', encoding=encoding)

    def openr(self, path: PathLike, encoding: Optional[str]=None):
        return self.open(path, 'r', encoding=encoding)

    def mkdirs(self, path: PathLike, exist_ok: bool=True):
        return os.makedirs(path, exist_ok=exist_ok)


PathManager = _PathManager()
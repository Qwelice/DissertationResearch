import os.path
from abc import ABC
from pathlib import Path
from typing import Union, Optional, List


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

    def openw(self, path: PathLike):
        self._open(path, 'w')

    def openr(self, path: PathLike):
        self._open(path, 'r')

    def isfile(self, path: PathLike) -> bool:
        return self._isfile(self._resolve_path(path))

    def isdir(self, path: PathLike) -> bool:
        return self._isdir(self._resolve_path(path))

    def exists(self, path: PathLike) -> bool:
        return self._exists(self._resolve_path(path))

    def mkdirs(self, path: PathLike, exist_ok: bool=True):
        self._mkdirs(self._resolve_path(path), exist_ok=exist_ok)

    def listdir(self, path: PathLike) -> List[Path]:
        return self._listdir(self._resolve_path(path))

    def join(self, basepath: PathLike, paths: PathLike) -> PathLike:
        return self._join(basepath, paths)

    def get_local_path(self, cur_path: Optional[PathLike]=None) -> PathLike:
        return self._resolve_path(cur_path)

    def _open(self, path:PathLike, mode: Optional[str]=None):
        if mode is None:
            mode = 'r'
        return open(self._resolve_path(path), mode)

    def _isfile(self, path: PathLike) -> bool:
        raise NotImplementedError()

    def _isdir(self, path: PathLike) -> bool:
        raise NotImplementedError()

    def _exists(self, path: PathLike) -> bool:
        raise NotImplementedError()

    def _mkdirs(self, path: PathLike, exist_ok: bool=True):
        raise NotImplementedError()

    def _listdir(self, path: Path) -> List[Path]:
        raise NotImplementedError()

    def _join(self, basepath: PathLike, paths: PathLike) -> PathLike:
        raise NotImplementedError()

    def _resolve_path(self, path: Optional[PathLike]=None):
        raise NotImplementedError()


class LocalPathHandler(PathHandler):
    BASE_PREFIX = _find_project_root()

    def __init__(self):
        super().__init__()

    def _isfile(self, path: Union[str, Path]) -> bool:
        return os.path.isfile(path)

    def _isdir(self, path: Union[str, Path]) -> bool:
        return os.path.isdir(path)

    def _exists(self, path: Union[str, Path]) -> bool:
        return os.path.exists(path)

    def _mkdirs(self, path: Union[str, Path], exist_ok: bool=True):
        os.makedirs(path, exist_ok=exist_ok)

    def _listdir(self, path: Path) -> List[Path]:
        return list(path.iterdir())

    def _join(self, basepath: PathLike, paths: PathLike) -> PathLike:
        return os.path.join(basepath, paths)

    def _resolve_path(self, path: Optional[PathLike]=None) -> Path:
        if path is None:
            return self.BASE_PREFIX
        p = Path(path)
        if not p.is_absolute():
            return self.BASE_PREFIX / p
        return p


class BasePathManager:
    def __init__(self):
        self._handler = LocalPathHandler()

    def openw(self, path: PathLike):
        return self._handler.openw(path)

    def openr(self, path: PathLike):
        return self._handler.openr(path)

    def exists(self, path: PathLike) -> bool:
        return self._handler.exists(path)

    def listdir(self, path: PathLike) -> List[Path]:
        return self._handler.listdir(path)

    def mkdirs(self, path: PathLike, exist_ok: bool = True):
        self._handler.mkdirs(path, exist_ok=exist_ok)

    def isdir(self, path: PathLike) -> bool:
        return self._handler.isdir(path)

    def isfile(self, path: PathLike) -> bool:
        return self._handler.isfile(path)

    def join(self, basepath: PathLike, paths: PathLike) -> PathLike:
        return self._handler.join(basepath, paths)

    def get_local_path(self, cur_path: Optional[PathLike]=None) -> PathLike:
        return self._handler.get_local_path(cur_path)


PathManager = BasePathManager()
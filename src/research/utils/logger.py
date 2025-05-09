import logging
from typing import Optional

from .io import PathManager, PathLike


_default_dir_path = PathManager.join(PathManager.get_local_path(), "logs/internal")


def setup_logger(filename: str, filedir: Optional[PathLike]=None, min_log_level: Optional[int | str]=None):
    if min_log_level is None:
        min_log_level = logging.INFO

    if not filename.endswith('.log'):
        filename = f'{filename}.log'
    if filedir is None:
        filedir = _default_dir_path
    PathManager.mkdirs(filedir)
    filepath = PathManager.join(filedir, filename)

    logger = logging.getLogger()
    logger.setLevel(min_log_level)

    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(log_format)

    file_handler = logging.FileHandler(filepath, encoding='utf-8')
    file_handler.setFormatter(formatter)

    if logger.hasHandlers():
        logger.handlers.clear()

    logger.addHandler(file_handler)

    return logger
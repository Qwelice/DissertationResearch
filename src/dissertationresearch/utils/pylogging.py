import logging
import os
from typing import Optional


def setup_logger(filedir: str | os.PathLike, filename: str, min_log_level: Optional[int | str]=None):
    if min_log_level is None:
        min_log_level = logging.INFO

    if not filename.endswith('.log'):
        filename = f'{filename}.log'
    os.makedirs(filedir, exist_ok=True)
    filepath = os.path.join(filedir, filename)

    logger = logging.getLogger()
    logger.setLevel(min_log_level)

    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(log_format)

    file_handler = logging.FileHandler(filepath, encoding='utf-8')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger
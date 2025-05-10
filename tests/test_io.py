from typing import List

from research.utils import setup_logger, PathManager

logger = setup_logger(__name__,)


def test():
    logger.info(f"{PathManager.get_local_path()}")
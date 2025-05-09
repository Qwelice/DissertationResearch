from typing import List

from research.utils import setup_logger, PathManager

logger = setup_logger(__name__,)


def test():
    logger.info("starting io module tests.")
    dirs = PathManager.listdir("data")
    assert isinstance(dirs, List)
    logger.info(f"extracted dirs: {dirs}")
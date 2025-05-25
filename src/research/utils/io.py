import os.path
from pathlib import Path
import pandas as pd

PathLike = str


def root_dir() -> Path:
    current_path = Path.cwd()
    for path in [current_path, *current_path.parents]:
        if (path / "src").is_dir():
            return path
    raise FileNotFoundError("src not found. Are you sure this is a right project?")


def read_csv (path_to_df: PathLike) -> pd.DataFrame:
    if not path_to_df.endswith('.csv'):
        raise FileExistsError(f'`{path_to_df}` is not csv file')
    return pd.read_csv(path_to_df)


def read_anno_file (anno_dir: PathLike, anno_file: PathLike) -> pd.DataFrame:
    pth = os.path.join(anno_dir, anno_file)
    return read_csv(pth)
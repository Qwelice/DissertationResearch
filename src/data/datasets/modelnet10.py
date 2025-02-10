from os import PathLike
from typing import Union, List, Dict

import pandas as pd
from tqdm import tqdm


def _read_modelnet10_csv_dicts(metafile: Union[str, PathLike]) -> List[Dict]:
    objs = list()

    meta_df = pd.read_csv(metafile)
    pbar = tqdm(meta_df.iterrows(), total=len(meta_df), desc='dicts reading')

    for idx, row in pbar:
        name = row['object_id']
        mode = row['split']
        category = row['class']
        model = row['object_path']
        pbar.set_postfix({
            'category' : category
        })
        obj = {
            'name' : name,
            'mode' : mode,
            'category' : category,
            'model' : model
        }
        objs.append(obj)

    return objs


def _render_modelnet10_shapes(n_renders: int,
                              objs: List[Dict],
                              dataroot: Union[str, PathLike],
                              out_dir: Union[str, PathLike]):
    pass
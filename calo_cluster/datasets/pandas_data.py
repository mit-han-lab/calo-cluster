import logging
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd
from calo_cluster.datasets.base import BaseDataModule, BaseDataset
from tqdm.auto import tqdm


@dataclass
class PandasDataset(BaseDataset):
    "A generic pandas torch dataset."
    feats: List[str]
    coords: List[str]
    weight: str
    semantic_label: str
    instance_label: str

    def _get_df(self, index: int) -> pd.DataFrame:
        df = pd.read_pickle(self.files[index])
        return df

    def _get_numpy(self, index: int) -> Dict[str, Any]:
        df = self._get_df(index)

        #features = df[self.feats].to_numpy(dtype=np.half)
        features = df[self.feats].to_numpy(dtype=np.float32)
        #coordinates = df[self.coords].to_numpy(dtype=np.half)
        coordinates = df[self.coords].to_numpy(dtype=np.float32)

        return_dict = {'features': features, 'coordinates': coordinates}

        if self.semantic_label:
            return_dict['semantic_labels'] = df[self.semantic_label].to_numpy()
        if self.instance_label:
            return_dict['instance_labels'] = df[self.instance_label].to_numpy()
        if self.weight is not None:
            #return_dict['weights'] = df[self.weight].to_numpy(dtype=np.half)
            return_dict['weights'] = df[self.weight].to_numpy(dtype=np.float32)

        return return_dict


@dataclass
class PandasDataModuleMixin():
    feats: List[str]
    coords: List[str]
    weight: str
    semantic_label: str
    instance_label: str

    def make_dataset_kwargs(self) -> dict:
        kwargs = {
            'feats': self.feats,
            'coords': self.coords,
            'weight': self.weight,
            'semantic_label': self.semantic_label,
            'instance_label': self.instance_label
        }
        kwargs.update(super().make_dataset_kwargs())
        return kwargs

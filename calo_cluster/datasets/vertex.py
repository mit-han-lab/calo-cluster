import logging
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import uproot
from tqdm import tqdm

from .base import BaseDataModule, BaseDataset


@dataclass
class VertexDataset(BaseDataset):
    feats: list
    coords: list
    instance_label: str

    def __post_init__(self):
        if self.instance_label == 'reco':
            self.instance_label = 'reco_vtxID'
        elif self.instance_label == 'truth':
            raise NotImplementedError()
        else:
            raise NotImplementedError()
        return super().__post_init__()

    def _get_df(self, index: int) -> pd.DataFrame:
        df = pd.read_pickle(self.files[index])
        return df

    def _get_numpy(self, index: int) -> Tuple[np.array, np.array, Union[np.array, None], Union[np.array, None]]:
        df = self._get_df(index)
        #features = df[self.feats].to_numpy(dtype=np.half)
        features = df[self.feats].to_numpy(dtype=np.float32)
        if self.task == 'panoptic':
            raise NotImplementedError()
        elif self.task == 'semantic':
            raise NotImplementedError()
        elif self.task == 'instance':
            labels = df[self.instance_label].to_numpy()
        else:
            raise RuntimeError(f'Unknown task = "{self.task}"')
        #coordinates = df[self.coords].to_numpy(dtype=np.half)
        coordinates = df[self.coords].to_numpy(dtype=np.float32)

        weights = None
        return features, labels, weights, coordinates


@dataclass
class VertexDataModule(BaseDataModule):

    data_dir: str

    feats: List[str]
    coords: List[str]
    
    instance_label: str

    def __post_init__(self):
        super().__post_init__()

        self.data_dir = Path(self.data_dir)

        self._files = None

    @property
    def files(self) -> list:
        if self._files is None:
            self._files = []
            self._files.extend(
                sorted(self.data_dir.glob('*.pkl')))
        return self._files

    def make_dataset(self, files: List[Path], split: str) -> BaseDataset:
        kwargs = self.make_dataset_kwargs()
        return VertexDataset(files=files, **kwargs)

    def make_dataset_kwargs(self) -> dict:
        kwargs = {
            'feats': self.feats,
            'coords': self.coords,
            'instance_label': self.instance_label
        }
        kwargs.update(super().make_dataset_kwargs())
        return kwargs

    @staticmethod
    def root_to_pickle(root_data_path, raw_data_dir):
        ni = 0
        for f in sorted(root_data_path.glob('*.root')):
            root_dir = uproot.open(f)
            truth_tree = root_dir['Truth_Vertex_PV_Selected;6']
            reco_tree = root_dir['Reco_Vertex;4']
            jagged_dict = {}
            prefix = 'reco_vtx_fitted_trk_'
            for k, v in tqdm(reco_tree.items()):
                if not k.startswith(prefix):
                    continue
                jagged_dict[k[len(prefix):]] = v.array()
            jagged_dict['reco_vtxID'] = jagged_dict.pop('vtxID')
            coords = ['d0', 'z0', 'phi', 'theta', 'qp']
            scale = np.array([0.05, 500, 6, 2, 4])
            for n in tqdm(range(len(reco_tree[0].array()))):
                df_dict = {k: jagged_dict[k][n] for k in jagged_dict.keys()}
                flat_event = pd.DataFrame(df_dict)
                flat_event[coords] /= scale
                flat_event.to_pickle(raw_data_dir / f'event_{n+ni:05}.pkl')
            ni += n + 1
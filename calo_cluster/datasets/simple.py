import logging
import multiprocessing as mp
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from numpy.random import default_rng
from sklearn.datasets import make_blobs
from tqdm import tqdm

from .base import BaseDataModule
from .calo import CaloDataset


class SimpleDataset(CaloDataset):
    def __init__(self, **kwargs):
        super().__init__(semantic_label='semantic_label',
                         instance_label='instance_label', weight='E', **kwargs)


@dataclass
class SimpleDataModule(BaseDataModule):
    data_dir: str

    feats: List[str]
    coords: List[str]

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

    @staticmethod 
    def _generate_event(rng, l, n_noise, noise_scale, signal_scale):
        noise_x = rng.uniform(-l, l, size=n_noise)
        noise_y = rng.uniform(-l, l, size=n_noise)
        noise_z = rng.uniform(-l, l, size=n_noise)
        noise_E = rng.exponential(size=n_noise, scale=noise_scale)

        noise_df = pd.DataFrame(
            {'x': noise_x, 'y': noise_y, 'z': noise_z, 'E': noise_E})
        noise_df['semantic_label'] = 0
        noise_df['instance_label'] = -1

        s = 0
        while s == 0:
            n_clusters = rng.poisson(10, 1)
            n_samples = rng.poisson(5, n_clusters)
            s = n_samples.sum()
        X, y = make_blobs(n_samples=n_samples,
                            n_features=3, cluster_std=0.4)
        signal_E = rng.exponential(
            size=n_samples.sum(), scale=signal_scale)
        signal_df = pd.DataFrame(
            {'x': X[:, 0], 'y': X[:, 1], 'z': X[:, 2], 'E': signal_E, 'instance_label': y})
        signal_df['semantic_label'] = 1
        df = pd.concat((noise_df, signal_df))
        return df

    @staticmethod
    def generate(data_dir, l=10, n_noise=1000, noise_scale=0.5, signal_scale=10.0, n_events=10000) -> None:
        logging.info(f'Generating data at {data_dir}.')
        rng = np.random.default_rng()
        for i in tqdm(range(n_events)):
            df = SimpleDataModule._generate_event(rng, l, n_noise, noise_scale, signal_scale)
            event_path = data_dir / f'{i:05}.pkl'
            df.to_pickle(event_path)

    def make_dataset(self, files: List[Path], split: str) -> SimpleDataset:
        kwargs = self.make_dataset_kwargs()
        return SimpleDataset(files=files, **kwargs)

    def make_dataset_kwargs(self) -> dict:
        kwargs = {
            'feats': self.feats,
            'coords': self.coords
        }
        kwargs.update(super().make_dataset_kwargs())
        return kwargs
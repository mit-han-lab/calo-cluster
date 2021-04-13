import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from tqdm import tqdm

from .base import BaseDataModule, BaseDataset


class SimpleDataset(BaseDataset):
    def __init__(self, voxel_size, events, task):
        super().__init__(voxel_size, events, task, feats=[
            'x', 'y', 'z'], coords=['x', 'y', 'z'], instance_label='cluster')


@dataclass
class SimpleDataModule(BaseDataModule):
    @staticmethod
    def generate(data_dir, n_events=10000) -> None:
        logging.info(f'Generating data at {data_dir}.')
        rng = np.random.default_rng()
        for i in tqdm(range(n_events)):
            n_clusters = rng.poisson(20, 1)
            n_samples = rng.poisson(5, n_clusters)
            X, y = make_blobs(n_samples=n_samples,
                              n_features=3, cluster_std=0.3)
            event_path = data_dir / f'{i:05}.pkl'
            energy = rng.exponential(size=n_samples)
            df = pd.DataFrame(
                {'x': X[:, 0], 'y': X[:, 1], 'z': X[:, 2], 'energy': energy, 'cluster': y})
            df.to_pickle(event_path)

    def make_dataset(self, events) -> BaseDataset:
        return SimpleDataset(self.voxel_size, events, self.task)

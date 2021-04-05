import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.datasets import make_blobs
from torch.utils.data import DataLoader
from .base import BaseDataset
from tqdm import tqdm

from ..utils.comm import get_rank


class SimpleDataset(BaseDataset):
    def __init__(self, voxel_size, events, task):
        super().__init__(voxel_size, events, task, feats=['x', 'y', 'z'], coords=['x', 'y', 'z'], instance_label='cluster')


class SimpleDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int, num_epochs: int, num_workers: int, voxel_size: float, data_dir: str, seed: int = None, event_frac: float = 1.0, train_frac: float = 0.8, test_frac: float = 0.1, task: str = 'class', num_classes: int = 2, num_features: int = 5):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_epochs = num_epochs
        self.voxel_size = voxel_size
        self.seed = seed
        self.task = task

        self._validate_fracs(event_frac, train_frac, test_frac)
        self.event_frac = event_frac
        self.train_frac = train_frac
        self.test_frac = test_frac

        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _validate_fracs(self, event_frac, train_frac, test_frac):
        fracs = [event_frac, train_frac, test_frac]
        assert all(0.0 <= f <= 1.0 for f in fracs)
        assert train_frac + test_frac <= 1.0

    def train_val_test_split(self, events):
        num_events = int(self.event_frac * len(events))
        events = events[:num_events]
        num_train_events = int(self.train_frac * num_events)
        num_test_events = int(self.test_frac * num_events)
        num_val_events = num_events - num_train_events - num_test_events

        train_events = events[:num_train_events]
        val_events = events[num_train_events:-num_test_events]
        test_events = events[-num_test_events:]

        return train_events, val_events, test_events

    def data_exists(self) -> bool:
        return len(set(self.data_dir.glob('*'))) != 0

    def generate(self, n_events=10000) -> None:
        logging.info(f'Generating data at {self.data_dir}.')
        rng = np.random.default_rng()
        for i in tqdm(range(n_events)):
            n_clusters = rng.poisson(20, 1)
            n_samples = rng.poisson(5, n_clusters)
            X, y = make_blobs(n_samples=n_samples, n_features=3, cluster_std=0.1)
            event_path = data_dir / f'{i:05}.pkl'
            df = pd.DataFrame({'x': X[:, 0], 'y': X[:, 1], 'z': X[:, 2], 'cluster': y})
            df.to_pickle(event_path)

    def prepare_data(self) -> None:
        if not self.data_exists():
            logging.info(f'Data not found at {self.data_dir}.')
            self.generate()

    def setup(self, stage=None) -> None:
        if self.seed is None:
            self.seed = torch.initial_seed() % (2**32 - 1)
        self.seed = self.seed + get_rank() * self.num_workers * self.num_epochs
        logging.debug(f'setting seed={self.seed}')
        pl.seed_everything(self.seed)

        self.events = []
        for event in sorted(self.data_dir.glob('*.pkl')):
            self.events.append(event)
        train_events, val_events, test_events = self.train_val_test_split(
            self.events)

        if stage == 'fit' or stage is None:
            self.train_dataset = SimpleDataset(
                self.voxel_size, train_events, self.task)
            self.val_dataset = SimpleDataset(
                self.voxel_size, val_events, self.task)
        if stage == 'test' or stage is None:
            self.test_dataset = SimpleDataset(
                self.voxel_size, test_events, self.task)

    def dataloader(self, dataset: SimpleDataset) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=SimpleDataset.collate_fn,
            worker_init_fn=lambda worker_id: np.random.seed(self.seed + worker_id))

    def train_dataloader(self) -> DataLoader:
        return self.dataloader(self.train_dataset)

    def val_dataloader(self) -> DataLoader:
        return self.dataloader(self.val_dataset)

    def test_dataloader(self) -> DataLoader:
        return self.dataloader(self.test_dataset)

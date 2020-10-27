import logging
import os
import os.path as osp
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import requests
import torch
import uproot
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..modules.efficient_minkowski import sparse_collate, sparse_quantize
from ..utils.comm import get_rank
from .base import BaseDataset


class HCalDataset(BaseDataset):
    def __init__(self, voxel_size, events, task):
        feats = ['x', 'y', 'z', 'time', 'energy']
        coords = ['x', 'y', 'z']
        super().__init__(voxel_size, events, task, feats=feats, coords=coords,
                         class_label='hit', instance_label='RHClusterMatch')


class HCalDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int, num_epochs: int, num_workers: int, voxel_size: float, data_dir: str, data_url: str = 'https://cernbox.cern.ch/index.php/s/s19K02E9SAkxTeg/download', force_download: bool = False, seed: int = None, event_frac: float = 1.0, train_frac: float = 0.8, test_frac: float = 0.1, task: str = 'class', num_classes: int = 2):
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
        self.root_data_path = self.data_dir / 'data.root'
        self.data_url = data_url
        self.force_download = force_download
        self.raw_data_dir = self.data_dir / 'rawpointclouds'
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

    def extracted_data_exists(self) -> bool:
        return len(set(self.raw_data_dir.glob('*'))) != 0

    def download(self):
        try:
            logging.info(
                f'Downloading data to {self.data_dir} (this may take a few minutes).')
            with requests.get(self.data_url, allow_redirects=True, stream=True) as r:
                r.raise_for_status()
                with self.root_data_path.open(mode='wb') as f:
                    pbar = tqdm(total=int(r.headers['Content-Length']))
                    for chunk in r.iter_content(chunk_size=1024**2):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            logging.info('download complete.')
        except Exception:
            logging.error(
                f'Unable to download dataset; please download manually to {self.data_dir}')

    def extract(self) -> None:
        logging.info(f'Extracting data to {self.raw_data_dir}.')
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        root_dir = uproot.rootio.open(self.root_data_path)
        root_events = root_dir.get('Events;1')

        df = pd.DataFrame()
        for k, v in root_events[b'HcalRecHit'].items():
            df[k.decode('ascii').split('.')[1]] = v.array()

        for n in tqdm(range(df.shape[0])):
            jagged_event = df.loc[n]
            df_dict = {k: jagged_event[k] for k in jagged_event.keys()}
            flat_event = pd.DataFrame(df_dict)
            flat_event.astype({'hit': int})
            flat_event.to_pickle(self.raw_data_dir / f'event_{n:05}.pkl')

    def prepare_data(self) -> None:
        if self.force_download or not self.data_exists():
            if self.force_download:
                logging.info(f'force-download set!')
            else:
                logging.info(f'Data not found at {self.data_dir}.')
            self.download()
        if self.force_download or not self.extracted_data_exists():
            if not self.force_download:
                logging.info(f'Data not found at {self.raw_data_dir}.')
            self.extract()

    def setup(self, stage=None) -> None:
        if self.seed is None:
            self.seed = torch.initial_seed() % (2**32 - 1)
        self.seed = self.seed + get_rank() * self.num_workers * self.num_epochs
        logging.debug(f'setting seed={self.seed}')
        pl.seed_everything(self.seed)

        self.events = []
        for event in sorted(self.raw_data_dir.glob('*.pkl')):
            self.events.append(event)
        train_events, val_events, test_events = self.train_val_test_split(
            self.events)

        if stage == 'fit' or stage is None:
            self.train_dataset = HCalDataset(
                self.voxel_size, train_events, self.task)
            self.val_dataset = HCalDataset(
                self.voxel_size, val_events, self.task)
        if stage == 'test' or stage is None:
            self.test_dataset = HCalDataset(
                self.voxel_size, test_events, self.task)

    def dataloader(self, dataset: HCalDataset) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=HCalDataset.collate_fn,
            worker_init_fn=lambda worker_id: np.random.seed(self.seed + worker_id))

    def train_dataloader(self) -> DataLoader:
        return self.dataloader(self.train_dataset)

    def val_dataloader(self) -> DataLoader:
        return self.dataloader(self.val_dataset)

    def test_dataloader(self) -> DataLoader:
        return self.dataloader(self.test_dataset)

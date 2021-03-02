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

from ..utils.comm import get_rank
from .base import BaseDataset


class VertexDataset(BaseDataset):
    def __init__(self, voxel_size, events, task):
        feats = ['Z', 'E', 'Px', 'Py', 'Pz', 'Eta', 'Phi']
        coords = ['Eta', 'Phi']
        super().__init__(voxel_size, events, task, feats=feats, coords=coords,
                         class_label='IsPU', instance_label='vertex_id')


class VertexDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int, num_epochs: int, num_workers: int, voxel_size: float, data_dir: str, data_url: str = 'https://cernbox.cern.ch/index.php/s/BkPgj9OjWaNBYqz/download', seed: int = None, event_frac: float = 1.0, train_frac: float = 0.8, test_frac: float = 0.1, task: str = 'class', num_classes: int = 2, num_features: int = 7):
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
        self.pkl_data_dir = self.data_dir / 'data'

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.pkl_data_dir.mkdir(parents=True, exist_ok=True)

        self._events = None

    @property
    def events(self) -> list:
        if self._events is None:
            self._events = []
            self._events.extend(sorted(self.pkl_data_dir.glob('*.pkl')))
        return self._events

    def _validate_fracs(self, event_frac, train_frac, test_frac):
        fracs = [event_frac, train_frac, test_frac]
        assert all(0.0 <= f <= 1.0 for f in fracs)
        assert train_frac + test_frac <= 1.0

    def train_val_test_split(self, events):
        num_events = int(self.event_frac * len(events))
        events = events[:num_events]
        num_train_events = int(self.train_frac * num_events)
        num_test_events = int(self.test_frac * num_events)

        train_events = events[:num_train_events]
        val_events = events[num_train_events:-num_test_events]
        test_events = events[-num_test_events:]

        return train_events, val_events, test_events

    def data_exists(self) -> bool:
        return len(set(self.data_dir.glob('*.root'))) != 0

    def extracted_data_exists(self) -> bool:
        return len(set(self.pkl_data_dir.glob('*'))) != 0

    def download(self):
        try:
            logging.info(
                f'Downloading data to {self.root_data_path} (this may take a few minutes).')
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
                f'Unable to download dataset; please download manually to {self.root_data_path}')

    def extract(self) -> None:
        logging.info(f'Extracting data to {self.pkl_data_dir}.')
        self.pkl_data_dir.mkdir(parents=True, exist_ok=True)
        root_dir = uproot.rootio.open(self.root_data_path)
        tree = root_dir.get('Delphes;1')
        particles = tree['Particle']
        vertices = tree['GenVertex']

        particle_df = pd.DataFrame()
        for k, v in tqdm(particles.items()):
            name = k.decode('ascii').split('.')[1]
            if name == 'fBits':
                continue
            particle_df[name] = v.array()

        vertex_df = pd.DataFrame()
        for k, v in tqdm(vertices.items()):
            name = k.decode('ascii').split('.')[1]
            if name == 'fBits':
                continue
            vertex_df[name] = v.array()

        for n in tqdm(range(particle_df.shape[0])):
            jagged_particles = particle_df.loc[n]
            particle_dict = {k: jagged_particles[k]
                             for k in jagged_particles.keys()}
            flat_p = pd.DataFrame(particle_dict)
            jagged_vertices = vertex_df.loc[n]
            vertex_dict = {k: jagged_vertices[k]
                           for k in jagged_vertices.keys()}
            flat_v = pd.DataFrame(vertex_dict)
            flat_p['vertex_id'] = -1
            vertex_id = 0
            for i in range(flat_v.shape[0]):
                mask = np.isin(flat_p['fUniqueID'],
                               flat_v.loc[i, 'Constituents'])
                flat_p.loc[mask, 'vertex_id'] = vertex_id
                vertex_id += 1
            flat_p = flat_p[flat_p['vertex_id'] != -1]
            flat_p.to_pickle(self.pkl_data_dir / f'event_{n:05}.pkl')

    def prepare_data(self) -> None:
        if not self.extracted_data_exists():
            logging.info(f'Raw dataset not found at {self.pkl_data_dir}.')
            if not self.data_exists():
                self.download()
            self.extract()

    def setup(self, stage=None) -> None:
        if self.seed is None:
            self.seed = torch.initial_seed() % (2**32 - 1)
        self.seed = self.seed + get_rank() * self.num_workers * self.num_epochs
        logging.debug(f'setting seed={self.seed}')
        pl.seed_everything(self.seed)

        train_events, val_events, test_events = self.train_val_test_split(
            self.events)

        if stage == 'fit' or stage is None:
            self.train_dataset = VertexDataset(
                self.voxel_size, train_events, self.task)
            self.val_dataset = VertexDataset(
                self.voxel_size, val_events, self.task)
        if stage == 'test' or stage is None:
            self.test_dataset = VertexDataset(
                self.voxel_size, test_events, self.task)

    def dataloader(self, dataset: VertexDataset) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=VertexDataset.collate_fn,
            worker_init_fn=lambda worker_id: np.random.seed(self.seed + worker_id))

    def train_dataloader(self) -> DataLoader:
        return self.dataloader(self.train_dataset)

    def val_dataloader(self) -> DataLoader:
        return self.dataloader(self.val_dataset)

    def test_dataloader(self) -> DataLoader:
        return self.dataloader(self.test_dataset)

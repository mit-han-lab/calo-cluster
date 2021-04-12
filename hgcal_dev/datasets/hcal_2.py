import logging
import multiprocessing as mp
import os
import os.path as osp
import random
from functools import partial
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
from sklearn.utils import shuffle


class Hcal2Dataset(BaseDataset):
    def __init__(self, voxel_size, events, task, instance_label):
        feats = ['x', 'y', 'z', 'time', 'energy']
        coords = ['x', 'y', 'z']

        if instance_label == 'truth':
            instance_label = 'trackId'
        elif instance_label == 'antikt':
            instance_label = 'RHAntiKtCluster'
        else:
            raise RuntimeError()
        super().__init__(voxel_size, events, task, feats=feats, coords=coords,
                         class_label='hit', instance_label=instance_label, weight='energy')

    def _get_pc_feat_labels(self, index):
        event = pd.read_pickle(self.events[index])
        if self.task == 'panoptic':
            block, labels_ = event[self.feats], event[[
                self.class_label, self.instance_label]].to_numpy()
        elif self.task == 'semantic':
            block, labels_ = event[self.feats], event[self.class_label].to_numpy(
            )
        elif self.task == 'instance':
            block, labels_ = event[self.feats], event[self.instance_label].to_numpy(
            )
        else:
            raise RuntimeError(f'Unknown task = "{self.task}"')

        pc_ = np.round(block[self.coords].to_numpy() / self.voxel_size)
        pc_ -= pc_.min(0, keepdims=1)

        feat_ = block.to_numpy()
        if self.weight is not None:
            weights_ = event[self.weight].to_numpy()
        else:
            weights_ = None
        return pc_, feat_, labels_, weights_


class HCal2DataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int, num_epochs: int, num_workers: int, voxel_size: float, data_dir: str, seed: int = None, event_frac: float = 1.0, train_frac: float = 0.8, test_frac: float = 0.1, task: str = 'class', num_classes: int = 2, min_cluster_energy: float = 6.0, min_hits_per_cluster: int = 3, num_features: int = 5, instance_label: str = 'truth'):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_epochs = num_epochs
        self.voxel_size = voxel_size
        self.seed = seed
        self.task = task
        self.instance_label = instance_label

        self._validate_fracs(event_frac, train_frac, test_frac)
        self.event_frac = event_frac
        self.train_frac = train_frac
        self.test_frac = test_frac

        self.data_dir = Path(data_dir)
        self.raw_data_dir = self.data_dir / 'min_energy_0.0_min_hits_0'
        self.selected_data_dir = self.data_dir / \
            f'min_energy_{min_cluster_energy}_min_hits_{min_hits_per_cluster}'

        self.min_hits_per_cluster = min_hits_per_cluster
        self.min_cluster_energy = min_cluster_energy

        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.selected_data_dir.mkdir(parents=True, exist_ok=True)

        self._events = None

    @property
    def events(self) -> list:
        if self._events is None:
            self._events = []
            self._events.extend(sorted(self.selected_data_dir.glob('*.pkl')))
        return self._events

    @property
    def raw_events(self) -> list:
        if self._events is None:
            self._events = []
            self._events.extend(sorted(self.raw_data_dir.glob('*.pkl')))
        return self._events

    def _validate_fracs(self, event_frac, train_frac, test_frac):
        fracs = [event_frac, train_frac, test_frac]
        assert all(0.0 <= f <= 1.0 for f in fracs)
        assert train_frac + test_frac <= 1.0

    def train_val_test_split(self, events):
        events = shuffle(events, random_state=42)
        num_events = int(self.event_frac * len(events))
        events = events[:num_events]
        num_train_events = int(self.train_frac * num_events)
        num_test_events = int(self.test_frac * num_events)
        num_val_events = num_events - num_train_events - num_test_events

        train_events = events[:num_train_events]
        val_events = events[num_train_events:-num_test_events]
        test_events = events[-num_test_events:]

        return train_events, val_events, test_events

    @classmethod
    def apply_selection(cls, event_path, data_dir, min_cluster_energy, min_hits_per_cluster, ignore_id=-99):
        event = pd.read_pickle(event_path)
        tracks = cls.get_clusters(event, truth=True)

        tracks = tracks[(tracks['energy'] >= min_cluster_energy) | (tracks['clusterId'] == ignore_id)]
        tracks = tracks[(tracks['nconstituents'] >= min_hits_per_cluster) | (tracks['clusterId'] == ignore_id)]
        event = event[event['trackId'].isin(tracks['clusterId'])]
        event = event.reset_index(drop=True)

        out_path = data_dir / event_path.name
        if event.shape[0] > 0:
            event.to_pickle(out_path)

    @classmethod
    def _make_selected_data(cls, selected_data_dir, event_paths, min_cluster_energy, min_hits_per_cluster, ncpus=32):
        logging.info(f'Making selected data at {selected_data_dir}')
        with mp.Pool(ncpus) as p:
            with tqdm(total=len(event_paths)) as pbar:
                for _ in p.imap_unordered(partial(cls.apply_selection, data_dir=selected_data_dir, min_cluster_energy=min_cluster_energy, min_hits_per_cluster=min_hits_per_cluster), event_paths):
                    pbar.update()

    def make_selected_data(self):
        self._make_selected_data(self.selected_data_dir, self.raw_events, self.min_cluster_energy, self.min_hits_per_cluster)

    def raw_data_exists(self) -> bool:
        return len(set(self.raw_data_dir.glob('*'))) != 0

    def selected_data_exists(self) -> bool:
        return len(set(self.selected_data_dir.glob('*'))) != 0

    def prepare_data(self) -> None:
        if not self.selected_data_exists():
            logging.info(
                f'selected dataset not found at {self.selected_data_dir}.')
            if not self.raw_data_exists():
                logging.error(f'Raw dataset not found at {self.raw_data_dir}.')
                raise RuntimeError()
            self.make_selected_data()

    def setup(self, stage=None) -> None:
        train_events, val_events, test_events = self.train_val_test_split(self.events)

        if self.seed is None:
            self.seed = torch.initial_seed() % (2**32 - 1)
            self.seed = self.seed + get_rank() * self.num_workers * self.num_epochs
        logging.debug(f'setting seed={self.seed}')
        pl.seed_everything(self.seed)


        if stage == 'fit' or stage is None:
            self.train_dataset = Hcal2Dataset(
                self.voxel_size, train_events, self.task, self.instance_label)
            self.val_dataset = Hcal2Dataset(
                self.voxel_size, val_events, self.task, self.instance_label)
        if stage == 'test' or stage is None:
            self.test_dataset = Hcal2Dataset(
                self.voxel_size, test_events, self.task, self.instance_label)

    def dataloader(self, dataset: Hcal2Dataset) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=Hcal2Dataset.collate_fn,
            worker_init_fn=lambda worker_id: np.random.seed(self.seed + worker_id))

    def train_dataloader(self) -> DataLoader:
        return self.dataloader(self.train_dataset)

    def val_dataloader(self) -> DataLoader:
        return self.dataloader(self.val_dataset)

    def test_dataloader(self) -> DataLoader:
        return self.dataloader(self.test_dataset)

    @classmethod
    def get_clusters(cls, event, truth=True):
        if truth:
            cluster_col = 'trackId'
        else:
            cluster_col = 'RHAntiKtCluster'
        event['weta'] = event['eta'] * event['energy']
        event['wphi'] = event['phi'] * event['energy']
        result = event.groupby(['trackId'])[['energy', 'weta', 'wphi']].agg(['mean', 'count'])
        energy = result[('energy', 'mean')]
        energy.name = 'energy'
        nconstituents = result[('energy', 'count')]
        nconstituents.name = 'nconstituents'
        eta = result[('weta', 'mean')] / energy
        eta.name = 'eta'
        phi = result[('wphi', 'mean')] / energy
        phi.name = 'phi'
        return pd.concat([energy, eta, phi, nconstituents], axis=1).reset_index().rename(columns={cluster_col: 'clusterId'})

    @classmethod
    def merge_event(cls, event_path, data_dir, granularity=0.15):
        event = pd.read_pickle(event_path)
        noise = event[event['trackId'] == -99].reset_index(drop=True)
        true_hits = event[event['trackId'] != -99].reset_index(drop=True)
        
        while True:
            # Find delta R between each cluster, identify pairs that are mergeable and sort according to energy.
            tracks = cls.get_clusters(true_hits)
            eta = tracks['eta'].values
            phi = tracks['phi'].values
            dR2 = (np.expand_dims(eta, axis=1) - eta)**2 + (np.expand_dims(phi, axis=1) - phi)**2
            np.fill_diagonal(dR2, 1.0)
            mergeable = dR2 < granularity**2
            sorted_indices = np.argsort(tracks['energy'].values)[::-1]

            # Keep the highest energy pairs.
            X, Y = np.where(mergeable[sorted_indices])
            new_ids = {}
            track_ids = tracks['clusterId'].values
            for x, y in zip(X, Y):
                x = track_ids[sorted_indices[x]]
                y = track_ids[y]
                if y not in new_ids and x not in new_ids:
                    new_ids[y] = x
            if len(new_ids) == 0:
                break

            # Assign the new ids
            new_hits = true_hits.copy()
            for k, v in new_ids.items():
                new_hits.loc[true_hits['trackId']==k, 'trackId'] = v
            true_hits = new_hits

        # Fix ids.
        track_ids = true_hits['trackId'].unique()
        for i, track_id in enumerate(track_ids):
            true_hits.loc[true_hits['trackId']==track_id, 'trackId'] = i
        merged_event = pd.concat([noise, true_hits]).sample(frac=1).reset_index(drop=True)
        merged_event_path = data_dir / event_path.name
        merged_event.to_pickle(merged_event_path)

    @classmethod
    def merge_events(cls, raw_data_dir, data_dir, granularity=0.15, ncpus=32):
        with mp.Pool(ncpus) as p:
            raw_event_paths = [f for f in raw_data_dir.glob('*.pkl')]
            with tqdm(total=len(raw_event_paths)) as pbar:
                for _ in p.imap_unordered(partial(cls.merge_event, data_dir=data_dir, granularity=granularity), raw_event_paths):
                    pbar.update()

    @staticmethod
    def root_to_pickle(root_data_path, raw_data_dir):
        ni = 0
        for f in sorted(root_data_path.glob('*.root')):
            root_dir = uproot.rootio.open(f)
            root_events = root_dir.get('Events;1')
            df = pd.DataFrame()
            for k, v in root_events[b'RecHit'].items():
                df[k.decode('ascii').split('.')[1]] = v.array()

            for n in tqdm(range(df.shape[0])):
                jagged_event = df.loc[n]
                df_dict = {k: jagged_event[k] for k in jagged_event.keys()}
                flat_event = pd.DataFrame(df_dict)
                flat_event.astype({'hit': int})
                flat_event.to_pickle(raw_data_dir / f'event_{n+ni:05}.pkl')
            ni = n + 1
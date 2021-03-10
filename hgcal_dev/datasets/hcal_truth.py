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


class HCalTruthDataset(BaseDataset):
    def __init__(self, voxel_size, events, task):
        feats = ['x', 'y', 'z', 'time', 'energy']
        coords = ['x', 'y', 'z']

        super().__init__(voxel_size, events, task, feats=feats, coords=coords,
                         class_label='hit', instance_label='trackId')

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
        return pc_, feat_, labels_


class HCalTruthDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int, num_epochs: int, num_workers: int, voxel_size: float, data_dir: str, seed: int = None, event_frac: float = 1.0, train_frac: float = 0.8, test_frac: float = 0.1, task: str = 'class', num_classes: int = 2, min_cluster_energy: float = 6.0, min_hits_per_cluster: int = 3, num_features: int = 5):
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
        pl.seed_everything(42)
        events = shuffle(events)
        num_events = int(self.event_frac * len(events))
        events = events[:num_events]
        num_train_events = int(self.train_frac * num_events)
        num_test_events = int(self.test_frac * num_events)
        num_val_events = num_events - num_train_events - num_test_events

        train_events = events[:num_train_events]
        val_events = events[num_train_events:-num_test_events]
        test_events = events[-num_test_events:]

        return train_events, val_events, test_events

    def get_tracks(self, event):
        tracks = pd.DataFrame(columns=['trackId', 'energy', 'nhits'])
        tracks['trackId'] = event['trackId'].unique()
        for i, id in enumerate(tracks['trackId']):
            mask = (event['trackId'] == id)
            tracks.loc[i, 'energy'] = event.loc[mask, 'energy'].sum()
            tracks.loc[i, 'eta'] = (
                (event.loc[mask, 'eta'] * event.loc[mask, 'energy']) / tracks.loc[i, 'energy']).sum()
            tracks.loc[i, 'phi'] = (
                (event.loc[mask, 'phi'] * event.loc[mask, 'energy']) / tracks.loc[i, 'energy']).sum()
            tracks.loc[i, 'nhits'] = mask.sum()
        return tracks

    def apply_selection(self, event, tracks):
        tracks = tracks[tracks['energy'] >= self.min_cluster_energy]
        tracks = tracks[tracks['nhits'] >= self.min_hits_per_cluster]
        event = event[event['trackId'].isin(tracks['trackId'])]
        return event.reset_index(drop=True), tracks.reset_index(drop=True)

    def make_selected_data(self):
        logging.info(f'Making selected data at {self.selected_data_dir}')
        for event_path in tqdm(self.raw_events):
            event = pd.read_pickle(event_path)
            tracks = self.get_tracks(event)
            event, tracks = self.apply_selection(event, tracks)
            out_path = self.selected_data_dir / event_path.name
            if event.shape[0] > 0:
                event.to_pickle(out_path)

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
            self.train_dataset = HCalTruthDataset(
                self.voxel_size, train_events, self.task)
            self.val_dataset = HCalTruthDataset(
                self.voxel_size, val_events, self.task)
        if stage == 'test' or stage is None:
            self.test_dataset = HCalTruthDataset(
                self.voxel_size, test_events, self.task)

    def dataloader(self, dataset: HCalTruthDataset) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=HCalTruthDataset.collate_fn,
            worker_init_fn=lambda worker_id: np.random.seed(self.seed + worker_id))

    def train_dataloader(self) -> DataLoader:
        return self.dataloader(self.train_dataset)

    def val_dataloader(self) -> DataLoader:
        return self.dataloader(self.val_dataset)

    def test_dataloader(self) -> DataLoader:
        return self.dataloader(self.test_dataset)

    @classmethod
    def get_track_id_map(cls, tracks, granularity=0.01):
        def deltaR_sq(t1, t2):
            return (t2.eta - t1.eta)**2 + (t2.phi - t1.phi)**2
        granularity_sq = granularity**2
        merged = np.full_like(tracks['trackId'].values, False, dtype=bool)
        new_track_ids = {}
        for t1 in tracks.sort_values('energy', ascending=False).itertuples():
            if merged[t1.Index]:
                continue
            for t2 in tracks[~merged].itertuples():
                if t2.Index == t1.Index:
                    continue
                if deltaR_sq(t1, t2) < granularity_sq:
                    merged[t2.Index] = True
                    new_track_ids[t2.trackId] = t1.trackId
        return new_track_ids

    @classmethod
    def merge_track(cls, raw_event_path, data_dir, granularity):
        event = pd.read_pickle(raw_event_path)
        tracks = cls.get_tracks(event)
        id_map = cls.get_track_id_map(tracks, granularity)
        for old_id, new_id in id_map.items():
            mask = (event['trackId'] == old_id)
            event.loc[mask, 'trackId'] = new_id
        event_path = data_dir / raw_event_path.name
        event.to_pickle(event_path)

    @classmethod
    def merge_tracks(cls, raw_data_dir, data_dir, granularity=0.01):
        with mp.Pool(4) as p:
            raw_event_paths = [f for f in raw_data_dir.glob('*.pkl')]
            with tqdm(total=len(raw_event_paths)) as pbar:
                for _ in p.imap_unordered(partial(cls.merge_track, data_dir=data_dir, granularity=granularity), raw_event_paths):
                    pbar.update()

    @classmethod
    def root_to_pickle(root_data_path, raw_data_dir):
        root_dir = uproot.rootio.open(root_data_path)
        root_events = root_dir.get('Events;1')
        
        df = pd.DataFrame()
        for k, v in root_events[b'RecHit'].items():
            df[k.decode('ascii').split('.')[1]] = v.array()

        for n in tqdm(range(df.shape[0])):
            jagged_event = df.loc[n]
            df_dict = {k: jagged_event[k] for k in jagged_event.keys()}
            flat_event = pd.DataFrame(df_dict)
            flat_event.astype({'hit': int})
            flat_event.to_pickle(raw_data_dir / f'event_{n:05}.pkl')
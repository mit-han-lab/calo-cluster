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
from sklearn.utils import shuffle
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..utils.comm import get_rank
from .base import BaseDataset


class HCalDataset(BaseDataset):
    def __init__(self, voxel_size, events, task, use_2d, scale):
        self.use_2d = use_2d
        self.scale = scale
        if use_2d:
            feats = ['eta', 'phi', 'time', 'energy']
            coords = ['eta', 'phi']
        else:
            feats = ['x', 'y', 'z', 'time', 'energy']
            coords = ['x', 'y', 'z']

        super().__init__(voxel_size, events, task, feats=feats, coords=coords,
                         class_label='hit', instance_label='RHClusterMatch')

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
        if self.scale:
            if self.use_2d:
                raise NotImplementedError()
            else:
                stds = np.array([119.310562, 117.978790, 286.380585, 8.158741, 2.077959])
                means = np.array([-1.991564, -1.404704, 0.642713, 0.35379654, 0.447480])
                t_mask = (feat_[:, 3] == -9999.0)
                feat_ = (feat_ - means) / stds
                feat_[t_mask, 3] = -2
        return pc_, feat_, labels_


class HCalDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int, num_epochs: int, num_workers: int, voxel_size: float, data_dir: str, data_url: str = 'https://cernbox.cern.ch/index.php/s/s19K02E9SAkxTeg/download', seed: int = None, event_frac: float = 1.0, train_frac: float = 0.8, test_frac: float = 0.1, task: str = 'class', num_classes: int = 2, num_features: int = 5, noise_level: float = 1.0, noise_seed: int = 31, use_2d: bool = False, split_clusters: bool = False, scale: bool = False):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_epochs = num_epochs
        self.voxel_size = voxel_size
        self.seed = seed
        self.task = task
        self.scale = scale

        self._validate_fracs(event_frac, train_frac, test_frac)
        self.event_frac = event_frac
        self.train_frac = train_frac
        self.test_frac = test_frac

        self.data_dir = Path(data_dir)
        self.root_data_path = self.data_dir / 'data.root'
        self.data_url = data_url
        self.raw_data_dir = self.data_dir / '1.0_noise'
        self.raw_data_dir2 = self.data_dir / \
            '1.0_noise_split'  # for split_clusters = True

        assert 0.0 <= noise_level <= 1.0
        self.noise_seed = noise_seed
        self.noise_level = noise_level
        self.noisy_data_dir = self.data_dir / f'{noise_level}_noise'
        self.noisy_data_dir2 = self.data_dir / \
            f'{noise_level}_noise_split'  # for split_clusters = True

        self.use_2d = use_2d
        # if true, split instance clusters based on semantic label, to enforce each cluster possessing a unique semantic label.
        self.split_clusters = split_clusters

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.noisy_data_dir.mkdir(parents=True, exist_ok=True)
        self.noisy_data_dir2.mkdir(parents=True, exist_ok=True)

        self._raw_events = None
        self._events = None

    @property
    def raw_events(self) -> list:
        if self._raw_events is None:
            self._raw_events = []
            self._raw_events.extend(sorted(self.raw_data_dir.glob('*.pkl')))
        return self._raw_events

    @property
    def events(self) -> list:
        if self._events is None:
            self._events = []
            self._events.extend(sorted(self.noisy_data_dir.glob('*.pkl')))
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

    def data_exists(self) -> bool:
        return len(set(self.data_dir.glob('*.root'))) != 0

    def extracted_data_exists(self) -> bool:
        return len(set(self.raw_data_dir.glob('*'))) != 0

    def noisy_data_exists(self) -> bool:
        return len(set(self.noisy_data_dir.glob('*'))) != 0

    def noisy_split_exists(self) -> bool:
        return len(set(self.noisy_data_dir2.glob('*'))) != 0

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

    def make_noisy_data(self) -> None:
        if self.noise_level == 1.0:
            return
        logging.info(f'Saving noisy data to {self.noisy_data_dir}.')
        for event_path in tqdm(self.raw_events):
            event = pd.read_pickle(event_path)
            non_noise_indices = np.where(event['hit'].values != 0)[0]
            noise_indices = ~non_noise_indices
            np.random.seed(self.noise_seed)
            selected_noise_indices = np.random.choice(
                noise_indices, size=int(noise_indices.shape[0]*self.noise_level))
            selected_indices = np.concatenate(
                (non_noise_indices, selected_noise_indices))
            if selected_indices.shape[0] > 0:
                selected = event.iloc[selected_indices]
                noisy_event_path = self.noisy_data_dir / \
                    f'{event_path.stem}.pkl'
                selected.to_pickle(noisy_event_path)

    def make_split_data(self) -> None:
        logging.info(
            f'Splitting dataset and saving to {self.noisy_data_dir2}.')
        for event_path in tqdm(self.events):
            event = pd.read_pickle(event_path)

            instance_labels = event['RHClusterMatch']
            unique_instance_labels = instance_labels.unique()
            new_instance_labels = instance_labels.copy()
            semantic_labels = event['hit']

            hit_mask = semantic_labels == 1
            noise_mask = ~hit_mask

            current_label = 0
            for instance_label in unique_instance_labels:
                cluster_mask = instance_labels == instance_label
                new_instance_labels[cluster_mask & hit_mask] = current_label
                new_instance_labels[cluster_mask &
                                    noise_mask] = current_label + 1
                current_label += 2

            event['RHClusterMatch'] = new_instance_labels
            split_event_path = self.noisy_data_dir2 / f'{event_path.stem}.pkl'
            event.to_pickle(split_event_path)

    def prepare_data(self) -> None:
        if not self.noisy_data_exists():
            logging.info(f'Noisy dataset not found at {self.noisy_data_dir}.')
            if not self.extracted_data_exists():
                logging.info(f'Raw dataset not found at {self.raw_data_dir}.')
                if not self.data_exists():
                    self.download()
                self.extract()
            self.make_noisy_data()
        if self.split_clusters:
            if not self.noisy_split_exists():
                logging.info(
                    f'Split dataset not found at {self.noisy_data_dir2}.')
                self.make_split_data()
            self.noisy_data_dir = self.noisy_data_dir2

    def setup(self, stage=None) -> None:
        train_events, val_events, test_events = self.train_val_test_split(
            self.events)
        if self.seed is None:
            self.seed = torch.initial_seed() % (2**32 - 1)
        self.seed = self.seed + get_rank() * self.num_workers * self.num_epochs
        logging.debug(f'setting seed={self.seed}')
        pl.seed_everything(self.seed)

        if stage == 'fit' or stage is None:
            self.train_dataset = HCalDataset(
                self.voxel_size, train_events, self.task, self.use_2d, self.scale)
            self.val_dataset = HCalDataset(
                self.voxel_size, val_events, self.task, self.use_2d, self.scale)
        if stage == 'test' or stage is None:
            self.test_dataset = HCalDataset(
                self.voxel_size, test_events, self.task, self.use_2d, self.scale)

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

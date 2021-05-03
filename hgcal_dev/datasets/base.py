import logging
import multiprocessing as mp
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import uproot
from hgcal_dev.utils.sparse_collate import sparse_collate_fn
from sklearn.utils import shuffle
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchsparse import SparseTensor
from torchsparse.utils import sparse_quantize
from tqdm import tqdm

from ..utils.comm import get_rank


@dataclass
class BaseDataset(Dataset):
    "Base torch dataset."
    voxel_size: float
    events: list
    task: str
    feats: list = None
    coords: list = None
    weight: str = None
    class_label: str = 'class'
    instance_label: str = 'instance'
    ignore_label: int = -1
    scale: bool = False
    std: list = None
    mean: list = None

    def __len__(self):
        return len(self.events)

    def _get_pc_feat_labels(self, index):
        event = pd.read_pickle(self.events[index])
        feat_ = event[self.feats].to_numpy()
        if self.task == 'panoptic':
            labels_ = event[[self.class_label, self.instance_label]].to_numpy()
        elif self.task == 'semantic':
            labels_ = event[self.class_label].to_numpy()
        elif self.task == 'instance':
            labels_ = event[self.instance_label].to_numpy(
            )
        else:
            raise RuntimeError(f'Unknown task = "{self.task}"')
        pc_ = np.round(event[self.coords].to_numpy() / self.voxel_size)
        pc_ -= pc_.min(0, keepdims=1)

        if self.scale:
            feat_ = (feat_ - np.array(self.mean)) / np.array(self.std)

        if self.weight is not None:
            weights_ = event[self.weight].to_numpy()
        else:
            weights_ = None
        return pc_, feat_, labels_, weights_

    def __getitem__(self, index):
        pc_, feat_, labels_, weights_ = self._get_pc_feat_labels(index)
        inds, labels, inverse_map = sparse_quantize(pc_,
                                                    feat_,
                                                    labels_,
                                                    return_index=True,
                                                    return_invs=True,
                                                    ignore_label=self.ignore_label)
        pc = pc_[inds]
        feat = feat_[inds]

        features = SparseTensor(feat, pc)

        labels = SparseTensor(labels, pc)
        labels_ = SparseTensor(labels_, pc_)
        inverse_map = SparseTensor(inverse_map, pc_)
        return_dict = {'features': features, 'labels': labels,
                       'labels_mapped': labels_, 'inverse_map': inverse_map}
        if weights_ is not None:
            weights = weights_[inds]
            weights = SparseTensor(weights, pc)
            return_dict['weights'] = weights
        return return_dict

    def get_inds_labels(self, index):
        pc_, feat_, labels_, _ = self._get_pc_feat_labels(index)
        inds, labels, inverse_map = sparse_quantize(pc_,
                                                    feat_,
                                                    labels_,
                                                    return_index=True,
                                                    return_invs=True,
                                                    ignore_label=self.ignore_label)
        return inds, labels

    @staticmethod
    def collate_fn(inputs):
        return sparse_collate_fn(inputs)


@dataclass
class BaseDataModule(pl.LightningDataModule):
    """The base pytorch-lightning data module that handles common data loading tasks.

    If you set self.transformed_data_dir != self.raw_data_dir in a subclass, and self._transformed_data_dir is empty
    or does not exist, then the function returned by self.get_transform_function will be applied to each raw event
    and the result will be saved to self.transformed_data_dir when prepare_data is called. 

    This allows subclasses to define arbitrary transformations that should be performed before serving data,
    e.g., merging, applying selections, reducing noise levels, etc.
    
    When creating a base class, make sure to override make_dataset appropriately."""

    num_features: int
    batch_size: int
    num_epochs: int
    num_workers: int
    voxel_size: float
    data_dir: str
    raw_data_dir: str
    seed: int
    event_frac: float
    train_frac: float
    test_frac: float
    task: str
    num_classes: int

    def __post_init__(self):
        super().__init__()

        self._validate_fracs()

        self.data_dir = Path(self.data_dir)
        self.raw_data_dir = Path(self.raw_data_dir)
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.transformed_data_dir = self.raw_data_dir

        self._events = None
        self._raw_events = None

    @property
    def events(self) -> list:
        if self._events is None:
            self._events = []
            self._events.extend(
                sorted(self.transformed_data_dir.glob('*.pkl')))
        return self._events

    @property
    def raw_events(self) -> list:
        if self._raw_events is None:
            self._raw_events = []
            self._raw_events.extend(sorted(self.raw_data_dir.glob('*.pkl')))
        return self._raw_events

    def _validate_fracs(self):
        fracs = [self.event_frac, self.train_frac, self.test_frac]
        assert all(0.0 <= f <= 1.0 for f in fracs)
        assert self.train_frac + self.test_frac <= 1.0

    def train_val_test_split(self, events):
        events = shuffle(events, random_state=42)
        num_events = int(self.event_frac * len(events))
        events = events[:num_events]
        num_train_events = int(self.train_frac * num_events)
        num_test_events = int(self.test_frac * num_events)

        train_events = events[:num_train_events]
        val_events = events[num_train_events:-num_test_events]
        test_events = events[-num_test_events:]

        return train_events, val_events, test_events

    def make_transformed_data(self, ncpus=32):
        transform = self.get_transform_function()
        logging.info(f'Making transformed data at {self.transformed_data_dir}')
        with mp.Pool(ncpus) as p:
            with tqdm(total=len(self.raw_events)) as pbar:
                for _ in p.imap_unordered(transform, self.raw_events):
                    pbar.update()

    def raw_data_exists(self) -> bool:
        return len(set(self.raw_data_dir.glob('*'))) != 0

    def transformed_data_exists(self) -> bool:
        return len(set(self.transformed_data_dir.glob('*'))) != 0

    def prepare_data(self) -> None:
        if not self.transformed_data_exists():
            logging.info(
                f'transformed dataset not found at {self.transformed_data_dir}.')
            if not self.raw_data_exists():
                logging.error(f'Raw dataset not found at {self.raw_data_dir}.')
                raise RuntimeError()
            self.make_transformed_data()

    def setup(self, stage=None) -> None:
        train_events, val_events, test_events = self.train_val_test_split(
            self.events)

        logging.debug(f'setting seed={self.seed}')
        pl.seed_everything(self.seed)

        if stage == 'fit' or stage is None:
            self.train_dataset = self.make_dataset(train_events)
            self.val_dataset = self.make_dataset(val_events)
        if stage == 'test' or stage is None:
            self.test_dataset = self.make_dataset(test_events)

    def dataloader(self, dataset: BaseDataset) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=dataset.collate_fn,
            worker_init_fn=lambda worker_id: np.random.seed(self.seed + worker_id))

    def train_dataloader(self) -> DataLoader:
        return self.dataloader(self.train_dataset)

    def val_dataloader(self) -> DataLoader:
        return self.dataloader(self.val_dataset)

    def test_dataloader(self) -> DataLoader:
        return self.dataloader(self.test_dataset)

    def get_transform_function(self):
        """In subclasses, should return a function that accepts an event path as its sole argument."""
        raise NotImplementedError()

    def make_dataset(self, events) -> BaseDataset:
        raise NotImplementedError()
        
    def voxel_occupancy(self):
        self.batch_size = 1
        dataloader = self.train_dataloader()
        voxel_occupancies = np.zeros(len(dataloader.dataset))
        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader.dataset)):
            voxel_occupancies[i] = len(batch['inverse_map'].C) / len(batch['features'].C) 

        return voxel_occupancies
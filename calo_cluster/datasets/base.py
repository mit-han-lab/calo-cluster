import logging
import multiprocessing as mp
from dataclasses import dataclass
from functools import partial
from itertools import repeat
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Union

import hydra
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import uproot
from hydra import compose, initialize_config_dir
from numpy.core.arrayprint import str_format
from omegaconf import OmegaConf
from sklearn.utils import shuffle
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchsparse import SparseTensor
from torchsparse.utils.collate import sparse_collate_fn
from tqdm import tqdm

from ..utils.comm import get_rank
from ..utils.quantize import sparse_quantize


@dataclass
class BaseDataset(Dataset):
    """Base torch dataset.

    A subclass of this dataset needs to:
    1. override _get_numpy().
    2. override collate_fn if a different collate method is required for the dataset."""
    files: List[Path]
    voxel_size: float

    task: str

    transform_features: bool
    features_loc: List[float]
    features_scale: List[float]

    transform_coords: bool
    coords_loc: List[float]
    coords_scale: List[float]

    sparse: bool

    def __post_init__(self):
        super().__init__()

    def __len__(self):
        return len(self.files)

    def _get_numpy(self, index: int) -> Tuple[np.array, np.array, Union[np.array, None], Union[np.array, None]]:
        """Returns (features, labels, weights, coordinates) for a given file index.

        Override this function."""
        raise NotImplementedError()

    def _get_numpy_scaled(self, index: int) -> Tuple[np.array, np.array, Union[np.array, None], Union[np.array, None]]:
        """Simple wrapper for _get_numpy that scales the features/coords."""
        features, labels, weights, coordinates = self._get_numpy(index)

        if self.transform_features:
            features = (features - np.array(self.features_loc)) / \
                np.array(self.features_scale)

        if self.transform_coords:
            coordinates = (coordinates - np.array(self.coords_loc)
                           ) / np.array(self.coords_scale)

        return features, labels, weights, coordinates

    def _get_sparse_tensors(self, index: int) -> Dict[str, SparseTensor]:
        features_, labels_, weights_, coordinates_ = self._get_numpy_scaled(
            index)
        coordinates_ = np.round(coordinates_ / self.voxel_size)
        coordinates_ -= coordinates_.min(0, keepdims=1)

        _, inds, inverse_map = sparse_quantize(coordinates_,
                                               return_index=True,
                                               return_inverse=True)
        coordinates = coordinates_[inds]
        features = features_[inds]
        labels = labels_[inds]
        features = SparseTensor(features, coordinates)
        labels = SparseTensor(labels, coordinates)
        inverse_map = SparseTensor(inverse_map, coordinates_)

        if weights_ is not None:
            weights = weights_[inds]
            weights = SparseTensor(weights, coordinates)
        else:
            weights = None

        return_dict = {'features': features, 'labels': labels,
                       'inverse_map': inverse_map, 'weights': weights}
        return return_dict

    def get_numpy(self, index: int) -> Dict[str, np.array]:
        features, labels, weights, _ = self._get_numpy_scaled(index)
        return_dict = {'features': features,
                       'labels': labels, 'weights': weights}

        return return_dict

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if self.sparse:
            return self._get_sparse_tensors(index)
        else:
            return self.get_numpy(index)

    @property
    def collate_fn(self) -> Callable[[List[Any]], Any]:
        """Returns a function that collates data into batches for the dataloader."""
        if self.sparse:
            return sparse_collate_fn
        else:
            return None


@dataclass
class BaseDataModule(pl.LightningDataModule):
    """The base pytorch-lightning data module that handles common data loading tasks.


    This module assumes that the data is organized into a set of files, with one event per file.
    When creating a base class, make sure to override make_dataset appropriately.

    Parameters:
    seed -- a seed used by the RNGs
    task -- the type of ML task that will be performed on this dataset (semantic, instance, panoptic)
    num_epochs -- the number of epochs
    batch_size -- the batch size
    sparse -- whether the data should be provided as SparseTensors (for spvcnn), or not. 

    num_workers -- the number of CPU processes to use for data workers.

    event_frac -- the fraction of total data to use
    train_frac -- the fraction of train data to use
    test_frac -- the fraction of test data to use

    transform_features -- if true, use scaling on the features (x = (x - features_loc) / features_scale)
    transform_coords -- same as transform_features, but for coords

    cluster_ignore_labels -- the semantic labels that should be ignored when clustering (needs to be supported by clusterer) and in embed criterion (needs to be supported by embed criterion)
    semantic_ignore_label -- the semantic label that should be ignored in semantic segmentation criterion (needs to be supported by semantic criterion)

    batch_dim -- the dimension that contains batch information, if sparse=False. If sparse=True, the batch should be stored in the last dimension of the coordinates.

    num_classes -- the number of semantic classes
    num_features -- the number of features used as input to the ML model
    voxel_size -- the length of a voxel along one coordinate dimension 
    """

    seed: int
    task: str
    num_epochs: int
    batch_size: int
    sparse: bool

    num_workers: int

    event_frac: float
    train_frac: float
    test_frac: float

    transform_features: bool
    features_loc: Union[List[float], None]
    features_scale: Union[List[float], None]

    transform_coords: bool
    coords_loc: Union[List[float], None]
    coords_scale: Union[List[float], None]

    cluster_ignore_labels: List[int]
    semantic_ignore_label: Union[int, None]

    batch_dim: int

    num_classes: int
    num_features: int
    voxel_size: float

    @property
    def files(self) -> List[Path]:
        raise NotImplementedError()

    def __post_init__(self):
        super().__init__()

        self._validate_fracs()

    def _validate_fracs(self):
        fracs = [self.event_frac, self.train_frac, self.test_frac]
        assert all(0.0 <= f <= 1.0 for f in fracs)
        assert self.train_frac + self.test_frac <= 1.0

    def train_val_test_split(self) -> Tuple[Union[List[Path], None], Union[List[Path], None], Union[List[Path], None]]:
        """Returns train, val, and test file lists

        Assumes that self.files is defined and there is no preset split in the dataset.
        If the dataset already has train/val/test files defined, override this function
        and return them."""
        files = shuffle(self.files, random_state=42)
        num_files = int(self.event_frac * len(files))
        files = files[:num_files]
        num_train_files = int(self.train_frac * num_files)
        num_test_files = int(self.test_frac * num_files)

        train_files = files[:num_train_files]
        val_files = files[num_train_files:-num_test_files]
        test_files = files[-num_test_files:]

        return train_files, val_files, test_files

    def setup(self, stage: str = None) -> None:
        train_files, val_files, test_files = self.train_val_test_split()

        logging.debug(f'setting seed={self.seed}')
        pl.seed_everything(self.seed)

        if stage == 'fit' or stage is None:
            self.train_dataset = self.make_dataset(train_files, split='train')
            self.val_dataset = self.make_dataset(val_files, split='val')
        if stage == 'test' or stage is None:
            self.test_dataset = self.make_dataset(test_files, split='test')

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

    def _voxel_occupancy(self) -> np.array:
        """Returns the average voxel occupancy for each batch in the train dataloader."""
        if not self.sparse:
            raise RuntimeError(
                'voxel_occupancy called, but dataset is not sparse!')

        self.batch_size = 1
        dataloader = self.train_dataloader()
        voxel_occupancies = np.zeros(len(dataloader.dataset))
        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader.dataset)):
            voxel_occupancies[i] = len(
                batch['inverse_map'].C) / len(batch['features'].C)

        return voxel_occupancies

    @staticmethod
    def voxel_occupancy(voxel_size, dataset):
        with initialize_config_dir(config_dir='/home/alexj/hgcal-dev/configs'):
            cfg = compose(config_name='config', overrides=[
                          f'dataset={dataset}', f'dataset.voxel_size={voxel_size}', 'dataset.sparse=True', 'train=single_gpu', 'dataset.num_workers=0'])
            dm = hydra.utils.instantiate(cfg.dataset, task='instance')
        dm.prepare_data()
        dm.setup('fit')
        return dm._voxel_occupancy()

    def make_dataset(self, files: List[Path], split: str) -> BaseDataset:
        raise NotImplementedError()

    def make_dataset_kwargs(self) -> dict:
        kwargs = {
            'voxel_size': self.voxel_size,
            'task': self.task,
            'sparse': self.sparse,
            'transform_features': self.transform_features,
            'features_loc': self.features_loc,
            'features_scale': self.features_scale,
            'transform_coords': self.transform_coords,
            'coords_loc': self.coords_loc,
            'coords_scale': self.coords_scale
        }
        return kwargs

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
from .base import BaseDataset

@dataclass
class BaseOffsetDataset(Dataset):
    """Base offset torch dataset.

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

    def _get_numpy(self, index: int) -> Tuple[np.array, np.array, Union[np.array, None], Union[np.array, None], Union[np.array, None]]:
        """Returns (features, labels, weights, coordinates, offsets) for a given file index.

        Override this function."""
        raise NotImplementedError()

    def _get_numpy_scaled(self, index: int) -> Tuple[np.array, np.array, Union[np.array, None], Union[np.array, None], Union[np.array, None]]:
        """Simple wrapper for _get_numpy that scales the features/coords."""
        features, labels, weights, coordinates, offsets = self._get_numpy(index)

        if self.transform_features:
            features = (features - np.array(self.features_loc)) / \
                np.array(self.features_scale)

        if self.transform_coords:
            coordinates = (coordinates - np.array(self.coords_loc)
                           ) / np.array(self.coords_scale)

        return features, labels, weights, coordinates, offsets

    def _get_sparse_tensors(self, index: int) -> Dict[str, SparseTensor]:
        features_, labels_, weights_, coordinates_, offsets_ = self._get_numpy_scaled(
            index)
        pc = coordinates_
        coordinates_ = np.round(coordinates_ / self.voxel_size)
        coordinates_ -= coordinates_.min(0, keepdims=1)

        _, inds, inverse_map = sparse_quantize(coordinates_,
                                               return_index=True,
                                               return_inverse=True)
        pc = pc[inds]
        coordinates = coordinates_[inds]
        features = features_[inds]
        labels = labels_[inds]
        offsets = offsets_[inds]
        pc = SparseTensor(pc, coordinates)
        features = SparseTensor(features, coordinates)
        labels = SparseTensor(labels, coordinates)
        offsets = SparseTensor(offsets, coordinates)
        inverse_map = SparseTensor(inverse_map, coordinates_)

        if weights_ is not None:
            weights = weights_[inds]
            weights = SparseTensor(weights, coordinates)
        else:
            weights = None

        return_dict = {'features': features, 'labels': labels,
                       'inverse_map': inverse_map, 'weights': weights, 'offsets': offsets, 'coordinates': pc}
        return return_dict

    def get_numpy(self, index: int) -> Dict[str, np.array]:
        features, labels, weights, _, offsets = self._get_numpy_scaled(index)
        return_dict = {'features': features,
                       'labels': labels, 'weights': weights, 'offsets': offsets}

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
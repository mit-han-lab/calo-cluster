from dataclasses import dataclass
from typing import Any, Callable, Dict, List

import numpy as np
from calo_cluster.utils.quantize import sparse_quantize
from torchsparse import SparseTensor
from torchsparse.utils.collate import sparse_collate_fn

from .base import AbstractBaseDataset


@dataclass
class SparseDatasetMixin(AbstractBaseDataset):
    sparse: bool

    def _get_sparse_tensors(self, index: int) -> Dict[str, SparseTensor]:
        dense_dict = self._get_numpy(index)
        coordinates_ = dense_dict['coordinates']
        coordinates_ = np.round(coordinates_ / self.voxel_size)
        coordinates_ -= coordinates_.min(0, keepdims=1)

        _, inds, inverse_map = sparse_quantize(coordinates_,
                                               return_index=True,
                                               return_inverse=True)

        coordinates = coordinates_[inds]
        sparse_dict = {k: SparseTensor(v[inds], coordinates)
                       for k, v in dense_dict.items() if k != 'coordinates'}
        inverse_map = SparseTensor(inverse_map, coordinates_)
        sparse_dict['inverse_map'] = inverse_map

        return sparse_dict

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if self.sparse:
            return self._get_sparse_tensors(index)
        else:
            return super().__getitem__(index)

    @property
    def collate_fn(self) -> Callable[[List[Any]], Any]:
        """Returns a function that collates data into batches for the dataloader."""
        if self.sparse:
            return sparse_collate_fn
        else:
            return super().collate_fn

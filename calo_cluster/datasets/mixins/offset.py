from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from .base import AbstractBaseDataModule, AbstractBaseDataset


@dataclass
class OffsetDatasetMixin(AbstractBaseDataset):
    "A dataset that supports cluster offsets."

    def __post_init__(self):
        super().__post_init__()

    def _get_offset(self, coordinates: np.array, semantic_labels: np.array, instance_labels: np.array):
        labels = (instance_labels.astype(np.int64) << 32) + semantic_labels.astype(np.int64)
        offsets = np.zeros_like(coordinates, dtype=np.float32)
        for i in np.unique(labels):
            if (i & 0xFFFFFFFF) not in self.valid_semantic_labels_for_clustering:
                continue
            i_indices = (labels == i).reshape(-1)
            coordinates_i = coordinates[i_indices]
            if coordinates_i.shape[0] <= 0:
                continue
            mean_coordinates_i = (np.max(coordinates_i, axis=0) + np.min(coordinates_i, axis=0)) / 2.0
            offsets[i_indices] = mean_coordinates_i - coordinates_i
        return offsets

    def _get_numpy(self, index: int) -> Dict[str, Any]:
        return_dict = super()._get_numpy(index)
        return_dict['offsets'] = self._get_offset(
            return_dict['coordinates'], return_dict['semantic_labels'], return_dict['instance_labels'])
        return return_dict

@dataclass
class OffsetDataModuleMixin(AbstractBaseDataModule):
    pass

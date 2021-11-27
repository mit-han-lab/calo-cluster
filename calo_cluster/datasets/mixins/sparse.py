from dataclasses import dataclass
from typing import Any, Callable, Dict, List

import numpy as np
import plotly.express as px
from torchsparse import SparseTensor
from torchsparse.utils.collate import sparse_collate_fn
from tqdm.auto import tqdm

from calo_cluster.utils.quantize import sparse_quantize

from .base import AbstractBaseDataModule, AbstractBaseDataset


@dataclass
class SparseDatasetMixin(AbstractBaseDataset):
    """

    Parameters:
    sparse -- whether the data should be provided as SparseTensors (for spvcnn), or not.
    voxel_size -- the length of a voxel along one coordinate dimension. """
    sparse: bool
    voxel_size: float

    def _get_sparse_tensors(self, index: int) -> Dict[str, SparseTensor]:
        """Keys with 'raw' indicate that the data is unmodified from the dense dict. All other data is the result of voxelization."""
        dense_dict = self._get_numpy(index)
        coordinates_ = dense_dict['coordinates']
        coordinates_ = np.round(coordinates_ / self.voxel_size)
        coordinates_ -= coordinates_.min(0, keepdims=1)

        _, inds, inverse_map = sparse_quantize(coordinates_,
                                               return_index=True,
                                               return_inverse=True)

        coordinates = coordinates_[inds]
        sparse_dict = {k: SparseTensor(v[inds], coordinates)
                       for k, v in dense_dict.items()}
        inverse_map = SparseTensor(inverse_map, coordinates_)
        sparse_dict['inverse_map'] = inverse_map
        for k, v in dense_dict.items():
            sparse_dict[f'{k}_raw'] = SparseTensor(v, coordinates_)
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


@dataclass
class SparseDataModuleMixin(AbstractBaseDataModule):
    sparse: bool
    voxel_size: float

    def make_dataset_kwargs(self) -> Dict[str, Any]:
        kwargs = {
            'sparse': self.sparse,
            'voxel_size': self.voxel_size
        }
        kwargs.update(super().make_dataset_kwargs())
        return kwargs

    def voxel_occupancy(self, only_different_labels: bool = False, label_type: str = None) -> np.array:
        """Returns the average number of points in each occupied voxel for each file in the train dataset.

        Parameters:
        only_different_labels: if true, only count the number of points with different labels.
        label_type: one of [None, instance, semantic] -- the type of label to use if only_different_labels is true.
        """
        if not self.sparse:
            raise RuntimeError(
                'voxel_occupancy called, but dataset is not sparse!')
        self.batch_size = 1
        dataloader = self.train_dataloader()
        dataset = dataloader.dataset
        if only_different_labels:
            if label_type == 'instance':
                label = dataset.instance_label
                batch_label = 'instance_labels'
            elif label_type == 'semantic':
                label = dataset.semantic_label
                batch_label = 'semantic_labels'
            else:
                raise NotImplementedError()

        voxel_occupancies = np.zeros(len(dataset))
        for i, batch in tqdm(enumerate(dataloader), total=len(dataset)):
            if only_different_labels:
                original = dataset._get_df(i)[label].values
                sampled = batch[batch_label].F[batch['inverse_map'].F].numpy()
                mask = (original == sampled)
                kept = mask.sum()
            else:
                kept = len(batch['features'].C)
            total = voxel_occupancies[i] = len(batch['inverse_map'].C)
            voxel_occupancies[i] = total / kept

        return voxel_occupancies

    @classmethod
    def occupancy_test(cls, voxel_sizes):
        occupancies = {}
        for vx in tqdm(voxel_sizes):
            dm = cls.from_config(
                ['dataset.event_frac=0.01', f'dataset.voxel_size={vx}'])
            occupancies[vx] = dm.voxel_occupancy(
                only_different_labels=True, label_type='instance')
        # %%
        for vx in voxel_sizes:
            print(f'mean occupancy (vx = {vx}): {occupancies[vx].mean()}')

from dataclasses import dataclass
from typing import Any, Callable, Dict, List
import hydra

import numpy as np
from calo_cluster.utils.quantize import sparse_quantize
from hydra import compose, initialize_config_dir
from torchsparse import SparseTensor
from torchsparse.utils.collate import sparse_collate_fn
from tqdm import tqdm

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
        # NOTE: fix this for other users.
        with initialize_config_dir(config_dir='/home/alexj/hgcal-dev/configs'):
            cfg = compose(config_name='config', overrides=[
                          f'dataset={dataset}', f'dataset.voxel_size={voxel_size}', 'dataset.sparse=True', 'train=single_gpu', 'dataset.num_workers=0'])
            dm = hydra.utils.instantiate(cfg.dataset, task='instance')
        dm.prepare_data()
        dm.setup('fit')
        return dm._voxel_occupancy()

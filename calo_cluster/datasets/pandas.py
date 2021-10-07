import logging
import multiprocessing as mp
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from itertools import repeat
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Union

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
import uproot
from calo_cluster.datasets.base import BaseDataModule, BaseDataset
from calo_cluster.datasets.base_offset import BaseOffsetDataset
from hydra import compose, initialize_config_dir
from numpy.core.arrayprint import str_format
from omegaconf import OmegaConf
from sklearn.utils import shuffle
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchsparse import SparseTensor
from torchsparse.utils.collate import sparse_collate_fn
from torchsparse.utils.quantize import sparse_quantize
from tqdm import tqdm

import pandas as pd

from ..utils.comm import get_rank
from ..utils.quantize import sparse_quantize

@dataclass
class PandasDataset(BaseDataset):
    "A generic pandas torch dataset."
    feats: list
    coords: list
    weight: str
    semantic_label: str
    instance_label: str

    def _get_df(self, index: int) -> pd.DataFrame:
        df = pd.read_pickle(self.files[index])
        return df

    def _get_numpy(self, index: int) -> Dict[str, Any]:
        df = self._get_df(index)

        #features = df[self.feats].to_numpy(dtype=np.half)
        features = df[self.feats].to_numpy(dtype=np.float32)
        #coordinates = df[self.coords].to_numpy(dtype=np.half)
        coordinates = df[self.coords].to_numpy(dtype=np.float32)

        return_dict = {'features': features, 'coordinates': coordinates}

        if self.semantic_label:
            return_dict['semantic_labels'] = df[self.semantic_label].to_numpy()
        if self.instance_label:
            return_dict['instance_labels'] = df[self.instance_label].to_numpy()
        if self.weight is not None:
            #return_dict['weights'] = df[self.weight].to_numpy(dtype=np.half)
            return_dict['weights'] = df[self.weight].to_numpy(dtype=np.float32)

        return return_dict


@dataclass
class OffsetDatasetMixin(AbstractBaseDataset):
    "A dataset that supports cluster offsets."
    valid_labels: list

    @abstractmethod
    def _get_offset(self, coordinates, semantic_labels, instance_labels):
        pass

    def _get_numpy(self, index: int) -> Dict[str, Any]:
        return_dict = super()._get_numpy()
        return_dict['offsets'] = self._get_offset(return_dict['coordinates'], return_dict['semantic_labels'], return_dict['instance_labels'])
        return return_dict

@dataclass
class CombineLabelsMixin(AbstractBaseDataset):
    "Combines labels into single 'label' field."
    task: str

    def _get_numpy(self, index: int) -> Dict[str, Any]:
        return_dict = super()._get_numpy()

        semantic_labels = return_dict.pop('semantic_labels')
        instance_labels = return_dict.pop('instance_labels')

        if self.task == 'panoptic':
            return_dict['labels'] = np.stack((semantic_labels, instance_labels), axis=-1)
        elif self.task == 'semantic':
            return_dict['labels'] = semantic_labels
        elif self.task == 'instance':
            return_dict['labels'] = instance_labels
        else:
            raise RuntimeError(f'Unknown task = "{self.task}"')

        return return_dict

class HCalZllJetsMixin(PandasDataset):
    def __init__(self, instance_target, **kwargs):
        if instance_target == 'truth':
            semantic_label = 'hit'
            instance_label = 'trackId'
        elif instance_target == 'antikt':
            raise NotImplementedError()
            #instance_label = 'RHAntiKtCluster_reco'
        elif instance_target == 'pf':
            semantic_label = 'pf_hit'
            instance_label = 'PFcluster0Id'
        else:
            raise RuntimeError()
        super().__init__(semantic_label=semantic_label, instance_label=instance_label, weight='energy', **kwargs)

class HCalZllJetsDataset(CombineLabelsMixin, SparseDatasetMixin, ScaledDatasetMixin, HCalZllJetsMixin):
    pass

class HCalZllJetsOffsetDataset(CombineLabelsMixin, SparseDatasetMixin, OffsetDatasetMixin, ScaledDatasetMixin, HCalZllJetsMixin):
    pass
import logging
import multiprocessing as mp
import os
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import uproot
from torch.utils.data import DataLoader
from torchsparse import SparseTensor
from torchsparse.utils.collate import sparse_collate_fn
from torchsparse.utils.quantize import sparse_quantize
from tqdm.auto import tqdm

from calo_cluster.datasets.mixins.sparse import SparseDataModuleMixin, SparseDatasetMixin
from calo_cluster.datasets.mixins.offset import OffsetDataModuleMixin, OffsetDatasetMixin
from calo_cluster.datasets.mixins.scaled import ScaledDataModuleMixin, ScaledDatasetMixin
from calo_cluster.datasets.mixins.semantic_kitti import SemanticKITTIDataModuleMixin, SemanticKITTIDatasetMixin


@dataclass
class SemanticKITTIDataset(SparseDatasetMixin, ScaledDatasetMixin, SemanticKITTIDatasetMixin):
    pass

@dataclass
class SemanticKITTIOffsetDataset(SparseDatasetMixin, OffsetDatasetMixin, ScaledDatasetMixin, SemanticKITTIDatasetMixin):
    pass

@dataclass
class SemanticKITTIDataModule(SparseDataModuleMixin, ScaledDataModuleMixin, SemanticKITTIDataModuleMixin):
    def make_dataset(self, files: List[Path], split: str) -> SemanticKITTIDataset:
        kwargs = self.make_dataset_kwargs()
        return SemanticKITTIDataset(files=files, **kwargs)

@dataclass
class SemanticKITTIOffsetDataModule(SparseDataModuleMixin, OffsetDataModuleMixin, ScaledDataModuleMixin, SemanticKITTIDataModuleMixin):
    def make_dataset(self, files: List[Path], split: str) -> SemanticKITTIOffsetDataset:
        kwargs = self.make_dataset_kwargs()
        return SemanticKITTIOffsetDataset(files=files, split=split, **kwargs)
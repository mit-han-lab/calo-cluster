import logging
import multiprocessing as mp
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from calo_cluster.datasets.base import BaseDataModule
from calo_cluster.datasets.mixins.combine_labels import CombineLabelsDataModuleMixin, CombineLabelsDatasetMixin
from calo_cluster.datasets.mixins.offset import OffsetDataModuleMixin, OffsetDatasetMixin
from calo_cluster.datasets.mixins.scaled import ScaledDataModuleMixin, ScaledDatasetMixin
from calo_cluster.datasets.mixins.simple import SimpleDataModuleMixin
from calo_cluster.datasets.mixins.sparse import SparseDataModuleMixin, SparseDatasetMixin
from numpy.random import default_rng
from sklearn.datasets import make_blobs
from tqdm.auto import tqdm

from calo_cluster.datasets.pandas_data import PandasDataset

@dataclass
class SimpleDataset(SparseDatasetMixin, CombineLabelsDatasetMixin, ScaledDatasetMixin, PandasDataset):
    pass


@dataclass
class SimpleOffsetDataset(SparseDatasetMixin, CombineLabelsDatasetMixin, OffsetDatasetMixin, ScaledDatasetMixin, PandasDataset):
    pass


@dataclass
class SimpleDataModule(SparseDataModuleMixin, CombineLabelsDataModuleMixin, ScaledDataModuleMixin, SimpleDataModuleMixin):
    def make_dataset(self, files: List[Path], split: str) -> SimpleDataset:
        kwargs = self.make_dataset_kwargs()
        return SimpleDataset(files=files, **kwargs)

@dataclass
class SimpleOffsetDataModule(SparseDataModuleMixin, CombineLabelsDataModuleMixin, OffsetDataModuleMixin, ScaledDataModuleMixin, SimpleDataModuleMixin):
    def make_dataset(self, files: List[Path], split: str) -> SimpleOffsetDataset:
        kwargs = self.make_dataset_kwargs()
        return SimpleOffsetDataset(files=files, **kwargs)
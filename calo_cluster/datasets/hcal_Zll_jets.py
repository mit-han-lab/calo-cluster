from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import uproot
from tqdm.auto import tqdm

from calo_cluster.datasets.mixins.combine_labels import CombineLabelsDataModuleMixin, CombineLabelsDatasetMixin
from calo_cluster.datasets.mixins.sparse import SparseDataModuleMixin, SparseDatasetMixin
from calo_cluster.datasets.mixins.offset import OffsetDataModuleMixin, OffsetDatasetMixin
from calo_cluster.datasets.mixins.scaled import ScaledDataModuleMixin, ScaledDatasetMixin
from calo_cluster.datasets.mixins.hcal_Zll_jets import HCalZllJetsDataModuleMixin, HCalZllJetsMixin


@dataclass
class HCalZllJetsDataset(SparseDatasetMixin, CombineLabelsDatasetMixin, ScaledDatasetMixin, HCalZllJetsMixin):
    pass

@dataclass
class HCalZllJetsOffsetDataset(SparseDatasetMixin, CombineLabelsDatasetMixin, OffsetDatasetMixin, ScaledDatasetMixin, HCalZllJetsMixin):
    pass

@dataclass
class HCalZllJetsDataModule(SparseDataModuleMixin, CombineLabelsDataModuleMixin, ScaledDataModuleMixin, HCalZllJetsDataModuleMixin):
    def make_dataset(self, files: List[Path], split: str) -> HCalZllJetsDataset:
        kwargs = self.make_dataset_kwargs()
        return HCalZllJetsDataset(files=files, **kwargs)

    @staticmethod
    def fix_overrides(overrides: List[str]):
        overrides.append('dataset=hcal_Zll_jets')
        return overrides

@dataclass
class HCalZllJetsOffsetDataModule(SparseDataModuleMixin, CombineLabelsDataModuleMixin, OffsetDataModuleMixin, ScaledDataModuleMixin, HCalZllJetsDataModuleMixin):
    def make_dataset(self, files: List[Path], split: str) -> HCalZllJetsOffsetDataset:
        kwargs = self.make_dataset_kwargs()
        return HCalZllJetsOffsetDataset(files=files, **kwargs)

    @staticmethod
    def fix_overrides(overrides: List[str]):
        overrides.append('dataset=hcal_Zll_jets_offset')
        return overrides
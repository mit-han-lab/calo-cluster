from dataclasses import dataclass

import pandas as pd
from tqdm.auto import tqdm

from calo_cluster.datasets.mixins.calo import CaloDataModule
from calo_cluster.datasets.pandas_data import PandasDataset

# see https://github.com/cms-pepr/pytorch_cmspepr/blob/68ef83386c6656a388815031f4679c19a3ca62db/torch_cmspepr/dataset.py#L104

@dataclass
class HGCalTausMixin(PandasDataset):
    pass


@dataclass
class HGCalTausDataModuleMixin(CaloDataModule):
    pass


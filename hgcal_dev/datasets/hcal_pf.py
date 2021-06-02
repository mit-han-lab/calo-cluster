import multiprocessing as mp
from dataclasses import dataclass
from functools import partial

import numpy as np
import pandas as pd
import uproot
from tqdm import tqdm

from .hcal import HCalDataModule
from .base import BaseDataset


class HCalPFDataset(BaseDataset):
    def __init__(self, voxel_size, events, task, instance_label, feats, coords):
        if instance_label == 'truth':
            instance_label = 'trackId'
        elif instance_label == 'antikt':
            instance_label = 'RHAntiKtCluster_reco'
        elif instance_label == 'pf':
            instance_label = 'PFcluster0Id'
        else:
            raise RuntimeError()
        super().__init__(voxel_size, events, task, feats=feats, coords=coords,
                         class_label='hit', instance_label=instance_label, weight='energy')



@dataclass
class HCalPFDataModule(HCalDataModule):
    def make_dataset(self, events) -> HCalPFDataset:
        return HCalPFDataset(self.voxel_size, events, self.task, self.instance_label, self.feats, self.coords)
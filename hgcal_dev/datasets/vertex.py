import logging
import os
import os.path as osp
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import requests
import torch
import uproot
from sklearn.utils import shuffle
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..utils.comm import get_rank
from .base import BaseDataset, BaseDataModule

from dataclasses import dataclass

class VertexDataset(BaseDataset):
    def __init__(self, voxel_size, events, task, scale):
        feats = ['Z']
        coords = ['Eta', 'Phi']
        super().__init__(voxel_size, events, task, feats=feats, coords=coords,
                         semantic_label='IsPU', instance_label='vertex_id', scale=scale, mean=[-55.67877], std=[10541.643])

@dataclass
class VertexDataModule(BaseDataModule):
    scale: bool

    def make_dataset(self, events) -> BaseDataset:
        return VertexDataset(self.voxel_size, events, self.task, self.scale)
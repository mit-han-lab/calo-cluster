from dataclasses import dataclass
import logging

import numpy as np
from calo_cluster.datasets.base import BaseDataModule
from calo_cluster.datasets.pandas_data import PandasDataModuleMixin, PandasDataset
from tqdm.auto import tqdm
import pandas as pd
from sklearn.datasets import make_blobs

X, y = make_blobs(centers=10, cluster_std=0.5)

from calo_cluster.clustering.meanshift import MeanShift
cpu_cluster = MeanShift(bandwidth=1.0, use_gpu=True)
pred_labels = cpu_cluster.cluster(X)
print(pred_labels)
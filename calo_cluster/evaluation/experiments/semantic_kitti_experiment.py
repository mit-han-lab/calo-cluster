import logging
import math
from pathlib import Path
from re import L
from typing import Union

import cycler
import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import torch
import wandb
import yaml
from calo_cluster.clustering.meanshift import MeanShift
from calo_cluster.evaluation.experiments.base_experiment import (BaseEvent,
                                                              BaseExperiment)
from calo_cluster.models.spvcnn import SPVCNN
from omegaconf import OmegaConf
from plotly.subplots import make_subplots
from sklearn.metrics import auc, confusion_matrix, roc_curve
from tqdm import tqdm


class SemanticKITTIEvent(BaseEvent):
    def __init__(self, input_path, label_map, pred_path=None, task='panoptic'):
        self.label_map = label_map
        super().__init__(input_path, pred_path=pred_path, semantic_label='label_id', instance_label='instance_id', task=task, clusterer=MeanShift(bandwidth=0.01), weight_name='energy')

    def _load_input(self):
        with self.input_path.open('rb') as b:
            block = np.fromfile(b, dtype=np.float32).reshape(-1, 4)

        data = {'x': block[:, 0],
                'y': block[:, 1],
                'z': block[:, 2],
                'reflectivity': block[:, 3]}

        label_file = Path(str(self.input_path).replace('velodyne', 'labels').replace(
            '.bin', '.label'))
        if label_file.exists():
            with label_file.open('rb') as a:
                all_labels = np.fromfile(a, dtype=np.int32).reshape(-1)
        else:
            all_labels = np.zeros(block.shape[0]).astype(np.int32)

        if self.task == 'semantic':
            labels_ = self.label_map[all_labels & 0xFFFF].astype(np.int64)
            data['label_id'] = labels_ 
        elif self.task == 'instance':
            labels_ = ((all_labels >> 4) & 0xFFFF).astype(np.int64)
            data['instance_id'] = labels_
        elif self.task == 'panoptic':
            semantic_labels = self.label_map[all_labels & 0xFFFF].astype(
                np.int64)
            instance_labels = ((all_labels >> 4) & 0xFFFF).astype(np.int64)
            data['label_id'] = semantic_labels
            data['instance_id'] = instance_labels
        df = pd.DataFrame(data)
        return df
        
        

class SemanticKITTIExperiment(BaseExperiment):
    def __init__(self, wandb_version, ckpt_name=None):
        super().__init__(wandb_version, ckpt_name=ckpt_name)

    def make_event(self, input_path, pred_path):
        return SemanticKITTIEvent(input_path=input_path, pred_path=pred_path, task=self.task)
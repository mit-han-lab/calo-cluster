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
from tqdm.auto import tqdm


label_name_mapping = {
    0: 'unlabeled',
    1: 'outlier',
    10: 'car',
    11: 'bicycle',
    13: 'bus',
    15: 'motorcycle',
    16: 'on-rails',
    18: 'truck',
    20: 'other-vehicle',
    30: 'person',
    31: 'bicyclist',
    32: 'motorcyclist',
    40: 'road',
    44: 'parking',
    48: 'sidewalk',
    49: 'other-ground',
    50: 'building',
    51: 'fence',
    52: 'other-structure',
    60: 'lane-marking',
    70: 'vegetation',
    71: 'trunk',
    72: 'terrain',
    80: 'pole',
    81: 'traffic-sign',
    99: 'other-object',
    252: 'moving-car',
    253: 'moving-bicyclist',
    254: 'moving-person',
    255: 'moving-motorcyclist',
    256: 'moving-on-rails',
    257: 'moving-bus',
    258: 'moving-truck',
    259: 'moving-other-vehicle'
}

kept_labels = [
    'road', 'sidewalk', 'parking', 'other-ground', 'building', 'car', 'truck',
    'bicycle', 'motorcycle', 'other-vehicle', 'vegetation', 'trunk', 'terrain',
    'person', 'bicyclist', 'motorcyclist', 'fence', 'pole', 'traffic-sign'
]

class SemanticKITTIEvent(BaseEvent):
    def __init__(self, input_path, pred_path, task, label_map):
        self.label_map = label_map
        super().__init__(input_path, pred_path=pred_path, task=task)

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
        reverse_label_name_mapping = {}
        self.label_map = np.zeros(260)
        cnt = 0
        for label_id in label_name_mapping:
            if label_id > 250:
                if label_name_mapping[label_id].replace('moving-',
                                                        '') in kept_labels:
                    self.label_map[label_id] = reverse_label_name_mapping[
                        label_name_mapping[label_id].replace('moving-', '')]
                else:
                    self.label_map[label_id] = 255
            elif label_id == 0:
                self.label_map[label_id] = 255
            else:
                if label_name_mapping[label_id] in kept_labels:
                    self.label_map[label_id] = cnt
                    reverse_label_name_mapping[
                        label_name_mapping[label_id]] = cnt
                    cnt += 1
                else:
                    self.label_map[label_id] = 255

        self.reverse_label_name_mapping = reverse_label_name_mapping
        super().__init__(wandb_version, ckpt_name=ckpt_name)

    def make_event(self, input_path, pred_path):
        return SemanticKITTIEvent(input_path=input_path, pred_path=pred_path, task=self.task, label_map=self.label_map)
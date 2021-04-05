import cycler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from hgcal_dev.clustering.meanshift import MeanShift
from hgcal_dev.metrics.instance import PanopticQuality
from plotly.subplots import make_subplots
from sklearn.metrics import auc, confusion_matrix, roc_curve
from tqdm import tqdm


palette = 'rgb(120,62,62), rgb(94,79,49), rgb(53,69,53), rgb(56,90,107), rgb(80,56,107), rgb(82,42,42), rgb(69,64,53), rgb(92,120,98), rgb(82,97,107), rgb(63,42,82), rgb(120,92,92), rgb(56,49,29), rgb(62,120,81), rgb(63,74,82), rgb(91,73,94), rgb(69,53,53), rgb(94,89,73), rgb(49,94,67), rgb(56,83,107), rgb(120,62,120), rgb(94,55,49), rgb(120,114,92), rgb(73,94,81), rgb(42,63,82), rgb(69,36,69), rgb(56,33,29), rgb(107,100,56), rgb(36,69,51), rgb(29,44,56), rgb(94,49,91), rgb(94,76,73), rgb(82,76,42), rgb(43,56,50), rgb(49,67,94), rgb(56,29,54), rgb(120,78,62), rgb(120,120,62), rgb(92,120,109), rgb(36,49,69), rgb(120,92,116), rgb(69,45,36), rgb(68,69,53), rgb(56,107,90), rgb(92,103,120), rgb(56,43,54), rgb(120,100,92), rgb(91,94,73), rgb(29,56,49), rgb(62,81,120), rgb(82,63,75), rgb(94,64,49), rgb(85,94,49), rgb(62,120,112), rgb(49,55,94), rgb(120,62,97), rgb(94,80,73), rgb(60,69,36), rgb(49,94,88), rgb(29,33,56), rgb(82,42,63), rgb(120,85,62), rgb(112,120,92), rgb(53,69,67), rgb(92,96,120), rgb(56,29,42), rgb(56,40,29), rgb(45,56,29), rgb(36,69,67), rgb(73,76,94), rgb(107,56,73), rgb(94,70,49), rgb(51,56,43), rgb(73,94,93), rgb(53,54,69), rgb(69,36,45), rgb(120,105,92), rgb(81,120,62), rgb(56,104,107), rgb(36,36,69), rgb(120,92,100), rgb(56,49,43), rgb(45,69,36), rgb(92,118,120), rgb(59,56,107), rgb(94,73,78), rgb(120,93,62), rgb(78,94,73), rgb(43,54,56), rgb(91,82,107), rgb(94,49,55), rgb(69,53,36), rgb(55,94,49), rgb(49,85,94), rgb(40,29,56), rgb(56,29,31), rgb(120,101,62), rgb(29,56,29), rgb(36,62,69), rgb(70,63,82)'.split(', ')

class BaseEvent():
    """Aggregates the input/output/clustering for a single event."""

    def __init__(self, input_path, class_label: str = None, instance_label: str = None, pred_path=None, task='panoptic', clusterer=MeanShift(), num_classes=2):
        self.input_path = input_path
        self.pred_path = pred_path
        self.task = task
        self.class_label = class_label
        self.instance_label = instance_label
        self._pred_instance_labels = None
        self.num_classes = num_classes
        self._load()

    def _load(self):
        input_event = pd.read_pickle(self.input_path)
        event_prediction = np.load(self.pred_path)
        if self.task == 'panoptic':
            self.embedding = event_prediction['embedding']
            self.pred_class_labels = event_prediction['labels']
        elif self.task == 'instance':
            self.embedding = event_prediction['embedding']
        elif self.task == 'semantic':
            self.pred_class_labels = event_prediction['labels']
        self.input_event = input_event

    @property
    def pred_instance_labels(self):
        if self._pred_instance_labels is None:
            self._pred_instance_labels = self.clusterer.cluster(self.embedding)
        return self._pred_instance_labels

    def pq(self):
        assert self.task == 'panoptic' or self.task == 'instance'
        if self.task == 'panoptic':
            pq_metric = PanopticQuality(num_classes=self.num_classes)

            outputs = (self.pred_class_labels, self.pred_instance_labels)
            targets = (self.input_event[self.class_label].values, self.input_event[self.instance_label].values)
        elif self.task == 'instance':
            pq_metric = PanopticQuality(semantic=False)

            outputs = self.pred_instance_labels
            targets = self.input_event[self.instance_label].values
        pq_metric.add(outputs, targets)
        
        return pq_metric.compute()


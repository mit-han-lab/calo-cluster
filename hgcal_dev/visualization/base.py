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
from sklearn import cluster
from sklearn.metrics import auc, confusion_matrix, roc_curve
from tqdm import tqdm


palette = 'rgb(120,62,62), rgb(94,79,49), rgb(53,69,53), rgb(56,90,107), rgb(80,56,107), rgb(82,42,42), rgb(69,64,53), rgb(92,120,98), rgb(82,97,107), rgb(63,42,82), rgb(120,92,92), rgb(56,49,29), rgb(62,120,81), rgb(63,74,82), rgb(91,73,94), rgb(69,53,53), rgb(94,89,73), rgb(49,94,67), rgb(56,83,107), rgb(120,62,120), rgb(94,55,49), rgb(120,114,92), rgb(73,94,81), rgb(42,63,82), rgb(69,36,69), rgb(56,33,29), rgb(107,100,56), rgb(36,69,51), rgb(29,44,56), rgb(94,49,91), rgb(94,76,73), rgb(82,76,42), rgb(43,56,50), rgb(49,67,94), rgb(56,29,54), rgb(120,78,62), rgb(120,120,62), rgb(92,120,109), rgb(36,49,69), rgb(120,92,116), rgb(69,45,36), rgb(68,69,53), rgb(56,107,90), rgb(92,103,120), rgb(56,43,54), rgb(120,100,92), rgb(91,94,73), rgb(29,56,49), rgb(62,81,120), rgb(82,63,75), rgb(94,64,49), rgb(85,94,49), rgb(62,120,112), rgb(49,55,94), rgb(120,62,97), rgb(94,80,73), rgb(60,69,36), rgb(49,94,88), rgb(29,33,56), rgb(82,42,63), rgb(120,85,62), rgb(112,120,92), rgb(53,69,67), rgb(92,96,120), rgb(56,29,42), rgb(56,40,29), rgb(45,56,29), rgb(36,69,67), rgb(73,76,94), rgb(107,56,73), rgb(94,70,49), rgb(51,56,43), rgb(73,94,93), rgb(53,54,69), rgb(69,36,45), rgb(120,105,92), rgb(81,120,62), rgb(56,104,107), rgb(36,36,69), rgb(120,92,100), rgb(56,49,43), rgb(45,69,36), rgb(92,118,120), rgb(59,56,107), rgb(94,73,78), rgb(120,93,62), rgb(78,94,73), rgb(43,54,56), rgb(91,82,107), rgb(94,49,55), rgb(69,53,36), rgb(55,94,49), rgb(49,85,94), rgb(40,29,56), rgb(56,29,31), rgb(120,101,62), rgb(29,56,29), rgb(36,62,69), rgb(70,63,82)'.split(', ')

class BaseEvent():

    def __init__(self, input_path, pred_path=None, task='panoptic'):
        self.palette = palette
        self.input_path = input_path
        self.pred_path = pred_path
        self.task = task
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
        input_event.rename(columns={'vertex_id': 'labels_i', 'IsPU': 'labels_s'}, inplace=True)
        self.input_event = input_event

    def top_n_clusters(self, n):
        for cluster in self.input_event['labels_i'].unique():
            cluster_mask = self.input_event['labels_i'] == cluster
            cluster_energy = self.input_event.loc[cluster_mask, 'E'].sum()
            self.input_event.loc[cluster_mask, 'cluster_energy'] = cluster_energy
        cluster_energies = np.sort(self.input_event['cluster_energy'].unique())[::-1][:n]
        mask = self.input_event['cluster_energy'].isin(cluster_energies)
        out = HCalEvent(self.input_path, self.pred_path, self.task)
        out.input_event = self.input_event[mask]
        if hasattr(out, 'embedding'):
            out.embedding = self.embedding[mask]
        if hasattr(out, 'pred_class_labels'):
            out.pred_class_labels = self.pred_class_labels[mask]
        return out


    def plot(self, truth=True, semantic=True, pred_instance_labels=None, scale_size=True):
        hits = self.input_event
        if scale_size:
            size='E'
        else:
            size=None
        if truth:
            if semantic:
                labels = hits['labels_s']
            else:
                labels = hits['labels_i']
        else:
            if semantic:
                labels = self.pred_class_labels
            else:
                labels = pred_instance_labels
        if semantic:
            hits['type'] = 'Primary'
            hits.loc[labels == 1, 'type'] = 'PU'
            fig = px.scatter(hits, x='Eta', y='Phi',
                                color='type', size=size)
        else:
            hits['instance'] = labels.astype(str)
            fig = px.scatter(hits, x='Eta', y='Phi',
                    color='instance', size=size, color_discrete_sequence=palette)
        return fig

    def pq(self, pred_instance_labels):
        assert self.task == 'panoptic'
        pq_metric = PanopticQuality(num_classes=2)

        outputs = (self.pred_class_labels, pred_instance_labels)
        targets = (self.input_event['labels_s'].values, self.input_event['labels_i'].values)
        pq_metric.add(outputs, targets)
        
        return pq_metric.compute()


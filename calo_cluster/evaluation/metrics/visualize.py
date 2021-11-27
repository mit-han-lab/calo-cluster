from dataclasses import dataclass
import torch
from torchmetrics import IoU
from tqdm.auto import tqdm

from calo_cluster.evaluation.metrics.metric import Metric
import pandas as pd
import plotly.express as px

from calo_cluster.evaluation.utils import get_palette

@dataclass
class Visualize(Metric):

    def __post_init__(self):
        self.reset()

    def reset(self):
        self.truth_figs = []
        self.pred_figs = []

    def add(self, coords, pred_labels, labels, weights):
        """Add data."""
        d = {'x': coords[:, 0], 'y': coords[:, 1], 'z': coords[:, 2], 'pred_label': pred_labels, 'label': labels, 'weight': weights}
        plot_df = pd.DataFrame(d)
        plot_df['label'] = plot_df['label'].astype(str)
        plot_df['pred_label'] = plot_df['pred_label'].astype(str)
        self.truth_figs.append(px.scatter_3d(plot_df, x='x', y='y', z='z', color='label', size='weight', color_discrete_sequence=get_palette(plot_df['label'])))
        self.pred_figs.append(px.scatter_3d(plot_df, x='x', y='y', z='z', color='pred_label', size='weight', color_discrete_sequence=get_palette(plot_df['pred_label'])))

    def add_from_dict(self, subbatch):
        """Add data from subbatch dict."""
        self.add(coords=subbatch['coordinates'].cpu().numpy(), pred_labels=subbatch['pred_semantic_labels'].cpu().numpy(), labels=subbatch['semantic_labels_raw'].cpu().numpy(), weights=subbatch['weights_raw'].cpu().numpy())

    def _save(self, path):
        for i in tqdm(range(len(self.truth_figs))):
            name = path / f'truth_{i}.png'
            self.truth_figs[i].write_image(name)
            name = path / f'pred_{i}.png'
            self.pred_figs[i].write_image(name)

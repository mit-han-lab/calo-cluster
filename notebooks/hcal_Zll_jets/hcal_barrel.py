# %%
from calo_cluster.datasets.hcal_Zll_jets import HCalZllJetsOffsetDataModule
from calo_cluster.evaluation.experiments.base_experiment import BaseExperiment
from calo_cluster.evaluation.experiments.base_offset_experiment import BaseOffsetExperiment
from pathlib import Path 
from calo_cluster.clustering.meanshift import MeanShift
from tqdm.auto import tqdm
from calo_cluster.evaluation.metrics.classification import mIoU
from calo_cluster.evaluation.metrics.instance import PanopticQuality
import numpy as np
import plotly
import plotly.express as px
from calo_cluster.evaluation.utils import get_palette
import pandas as pd
# %%
exp = BaseOffsetExperiment('2d01uh1b')
# %%
events = exp.get_events(split='val', n=100)

# %%
def save():
    clusterer = MeanShift(bandwidth=0.001, use_semantic=True, ignore_semantic_labels=(0,))
    for i, evt in tqdm(enumerate(events)):
        semantic_truth = evt.input_event['pf_hit'].to_numpy()
        evt.pred_instance_labels = clusterer.cluster(embedding=evt.embedding, semantic_labels=semantic_truth)
        np.save(f'tmp/pred_instance_labels{i}.npy', evt.pred_instance_labels)

# %%
def load(i):
    events[i].pred_instance_labels = np.load(f'../../tmp/pred_instance_labels{i}.npy')

# %%
from tqdm.auto import tqdm
bws = [0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
rets = {}
for bw in bws:
    print(f'bw = {bw}')
    clusterer = MeanShift(bandwidth=bw, use_semantic=True, ignore_semantic_labels=(0,), use_gpu=False)
    pq = PanopticQuality(num_classes=2, ignore_index=None, ignore_semantic_labels=(0,))
    for evt in tqdm(events):
        semantic_truth = evt.input_event['pf_hit'].to_numpy()
        evt.pred_instance_labels = clusterer.cluster(embedding=evt.embedding, semantic_labels=semantic_truth)
        pq.add((semantic_truth, evt.pred_instance_labels), (semantic_truth, evt.input_event['PFcluster0Id'].astype(int).to_numpy()))
    rets[bw] = pq.compute()
    print(f'(bw = {bw}) pq = {rets[bw]}')
# %%
n = 6
evt = events[n]
d = {'eta': evt.input_event['eta'], 'phi': evt.input_event['phi'], 'energy': evt.input_event['energy'], 'hit': evt.input_event['pf_hit'], 'id': evt.input_event['PFcluster0Id'], 'eta_pred': evt.embedding[:,0], 'phi_pred': evt.embedding[:,1]}
plot_df = pd.DataFrame(d)
plot_df['id'] = plot_df['id'].astype(str)
plot_df = plot_df[plot_df['hit'] == 1]
# %%
px.scatter(plot_df, x='eta', y='phi', size='energy', color='id', color_discrete_sequence=get_palette(plot_df['id']))
# %%
px.scatter(plot_df, x='eta_pred', y='phi_pred', size='energy', color='id', color_discrete_sequence=get_palette(plot_df['id']))
# %%
# true semantic labels, pred instance labels from coordinates
clusterer = MeanShift(bandwidth=bw, use_semantic=True, ignore_semantic_labels=(0,))
# %%
for i, evt in tqdm(enumerate(events)):
    semantic_truth = evt.input_event['pf_hit'].to_numpy()
    load(i)
# %%
n = 6
evt = events[n]
d = {'eta': evt.input_event['eta'], 'phi': evt.input_event['phi'], 'energy': evt.input_event['energy'], 'hit': evt.input_event['pf_hit'], 'id': evt.pred_instance_labels, 'eta_pred': evt.embedding[:,0], 'phi_pred': evt.embedding[:,1]}
plot_df = pd.DataFrame(d)
plot_df['id'] = plot_df['id'].astype(str)
plot_df = plot_df[plot_df['hit'] == 1]
# %%
px.scatter(plot_df, x='eta_pred', y='phi_pred', size='energy', color='id', color_discrete_sequence=get_palette(plot_df['id']))
# %%

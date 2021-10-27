# %%
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
# %%
exp = BaseOffsetExperiment('1blpfu5z')
# %%
exp.save_predictions(batch_size=16, num_workers=32)
# %%
evts = exp.get_events('val', n=100)
# %%
def fix_instance_labels(pred_instance_labels, semantic_labels, labels_to_fix):
    fixed_labels = pred_instance_labels.copy()
    unique_labels = np.unique(semantic_labels)
    for l in unique_labels:
        if l not in labels_to_fix:
            continue
        mask = semantic_labels == l
        fixed_labels[mask] = 0
    return fixed_labels

# %%
evts[0].input_event.columns

# %%
# pred semantic labels, pred instance labels from coordinates
from tqdm.auto import tqdm
bws = [1e-1, 1.0, 1e1, 1e2]
#bws = [0.2]
rets = {}
for bw in bws:
    print(f'bw = {bw}')
    clusterer = MeanShift(use_gpu=False, bandwidth=bw, use_semantic=True, ignore_semantic_labels=(0,))
    pq = PanopticQuality(num_classes=2, ignore_index=None, ignore_semantic_labels=(0,))
    for evt in tqdm(evts):
        evt.pred_instance_labels = clusterer.cluster(embedding=evt.embedding, semantic_labels=evt.pred_semantic_labels)
        evt.pred_instance_labels = fix_instance_labels(evt.pred_instance_labels, evt.pred_semantic_labels, labels_to_fix=np.array([0]))
        pq.add((evt.pred_semantic_labels, evt.pred_instance_labels), (evt.input_event['pf_hit'].to_numpy(), evt.input_event['PFcluster0Id'].astype(int).to_numpy()))
    rets[bw] = pq.compute()
    print(f'(bw = {bw}) pq = {rets[bw]}')
    
# %%
# pred semantic labels, pred instance labels from coordinates
from tqdm.auto import tqdm
bws = [1e1]
#bws = [0.2]
rets = {}
for bw in bws:
    print(f'bw = {bw}')
    clusterer = MeanShift(use_gpu=False, bandwidth=bw, use_semantic=True, ignore_semantic_labels=(0,))
    pq = PanopticQuality(num_classes=2, ignore_index=None, ignore_semantic_labels=(0,))
    for evt in tqdm(evts):
        evt.pred_instance_labels = clusterer.cluster(embedding=evt.embedding, semantic_labels=evt.pred_semantic_labels)
        evt.pred_instance_labels = fix_instance_labels(evt.pred_instance_labels, evt.pred_semantic_labels, labels_to_fix=np.array([0]))
        pq.add((evt.pred_semantic_labels, evt.pred_instance_labels), (evt.input_event['pf_hit'].to_numpy(), evt.input_event['PFcluster0Id'].astype(int).to_numpy()))
    rets[bw] = pq.compute()
    print(f'(bw = {bw}) pq = {rets[bw]}')

# %%
def plot_event(evt, use_offsets=False, use_pred_labels=False, instance_label='PFcluster0Id', min_eta=None, max_eta=None):
    #mask = evt.pred_semantic_labels == 1
    plot_df = evt.input_event
    if use_offsets:
        plot_df['xp'] = evt.embedding[:, 0]
        plot_df['yp'] = evt.embedding[:, 1]
        plot_df['zp'] = evt.embedding[:, 2]
        x='xp'
        y='yp'
        z='zp'
    else:
        x='x'
        y='y'
        z='z'
    if use_pred_labels:
        plot_df['pred_instance_labels'] = evt.pred_instance_labels
        color='s_pred_instance_labels'
        plot_df['s_pred_instance_labels'] = plot_df['pred_instance_labels'].astype(str)
    else:
        color='s_instance_label'
    #plot_df = plot_df[mask]
    plot_df['s_instance_label'] = plot_df[instance_label].astype(str)
    if min_eta is not None:
        plot_df = plot_df[(plot_df['eta'] > min_eta)]
    if max_eta is not None:
        plot_df = plot_df[(plot_df['eta'] < max_eta)]
    fig = px.scatter_3d(plot_df, x=x, y=y, z=z, color=color)
    fig.update_traces(marker=dict(size=3),
                    selector=dict(mode='markers'))
    return fig
# %%
n=3
min_eta=None
max_eta=None
fig = plot_event(evts[n], use_offsets=False, use_pred_labels=False, min_eta=min_eta, max_eta=max_eta)
fig.show()
# %%
fig = plot_event(evts[n], use_offsets=True, use_pred_labels=False, min_eta=min_eta, max_eta=max_eta)
#fig.update_layout(scene_zaxis_range=[400, 520])
fig.show()
# %%
iou = mIoU(evts, num_classes=2, ignore_index=0, semantic_label='pf_hit', reduction='none')
print(iou.mean())
# %%
n=1
min_eta=200
fig = plot_event(evts[n], use_offsets=True, use_pred_labels=False, min_eta=min_eta, max_eta=max_eta)
fig.show()
# %%
fig = plot_event(evts[n], use_offsets=True, use_pred_labels=True, min_eta=min_eta, max_eta=max_eta)
fig.show()
# %%

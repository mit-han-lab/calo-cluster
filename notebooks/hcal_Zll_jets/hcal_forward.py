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
from calo_cluster.evaluation.utils import get_palette
import pandas as pd
# %%
sem_wandb_version = '1d19n1c3'
inst_wandb_version = '1ho42pfq'
sem_exp = BaseExperiment(sem_wandb_version)
inst_exp = BaseOffsetExperiment(inst_wandb_version)
# %%
sem_exp.save_predictions(batch_size=128, num_workers=8)
inst_exp.save_predictions(batch_size=128, num_workers=8)
# %%
sem_evts = sem_exp.get_events('val', n=100)
inst_evts = inst_exp.get_events('val', n=100)

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
sem_evts[0].input_event.columns

# %%
# true semantic labels, pred instance labels from coordinates
from tqdm.auto import tqdm
bws = [0.0001, 0.00025, 0.00050, 0.00075, 0.001]
rets = {}
for bw in bws:
    print(f'bw = {bw}')
    clusterer = MeanShift(bandwidth=bw, use_semantic=True, ignore_semantic_labels=(0,))
    pq = PanopticQuality(num_classes=2, ignore_index=None, ignore_semantic_labels=(0,))
    for s_evt, i_evt in tqdm(zip(sem_evts, inst_evts)):
        s_evt.pred_instance_labels = clusterer.cluster(embedding=i_evt.embedding, semantic_labels=s_evt.pred_semantic_labels)
        s_evt.pred_instance_labels = fix_instance_labels(s_evt.pred_instance_labels, s_evt.pred_semantic_labels, labels_to_fix=np.array([0]))
        pq.add((s_evt.pred_semantic_labels, s_evt.pred_instance_labels), (s_evt.input_event['pf_hit'].to_numpy(), s_evt.input_event['PFcluster0Id'].astype(int).to_numpy()))
    rets[bw] = pq.compute()
    print(f'(bw = {bw}) pq = {rets[bw]}')

# %%
figs_2d = []
figs_3d = []
figs_3d_xyz = []
for n in range(10):
    evt = sem_evts[n].input_event
    evt['instance_id'] = evt['PFcluster0Id'].astype(str)
    evt['r'] = (evt['x']**2 + evt['y']**2)**0.5
    figs_2d.append(px.scatter(evt, x='eta', y='phi', color='instance_id', color_discrete_sequence=get_palette(evt['instance_id']), size='energy'))
    figs_3d.append(px.scatter_3d(evt, x='eta', y='phi', z='r', color='instance_id', color_discrete_sequence=get_palette(evt['instance_id']), size='energy'))
    figs_3d_xyz.append(px.scatter_3d(evt, x='x', y='y', z='z', color='instance_id', color_discrete_sequence=get_palette(evt['instance_id']), size='energy'))
# %%
for f2d, f3d in zip(figs_2d, figs_3d):
    f2d.show()
    f3d.show()
# %%
figs_2d[6].show()
figs_3d[6].show()
figs_3d_xyz[6].show()

# %%
figs_2d[0].show()
figs_2d[1].show()
figs_2d[2].show()
figs_2d[3].show()
# %%
def test_r():
    r = []
    for evt in evts:
        evt = evt.input_event
        mask = (evt['eta'] < 1.5) & (evt['eta'] > -1.5)
        evt = evt[mask]
        evt['instance_id'] = evt['PFcluster0Id'].astype(str)
        evt['r'] = (evt['x']**2 + evt['y']**2)**0.5
        r.append(evt['r'])
    flat_r = pd.concat(r)
    print(f'min(r) = {flat_r.min()}, max(r) = {flat_r.max()}')

# %%
test_r()
# %%
offsets = []
for batch in inst_exp.datamodule.val_dataloader():
    offsets.append(batch['offsets'].F)
# %%
offsets = np.concatenate(offsets)
# %%
np.mean(np.abs(offsets), axis=0)

# %%
pred_offsets = []
for evt in inst_evts:
    pred_offset = evt.embedding - evt.input_event[['x', 'y', 'z']]
    pred_offsets.append(pred_offset.to_numpy())
# %%
pred_offsets = np.concatenate(pred_offsets)
# %%
np.mean(np.abs(pred_offsets), axis=0)
# %%
pred_offsets.shape
# %%
offsets.shape
# %%
np.mean(pred_offsets[pred_offsets != 0])
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
    fig = px.scatter_3d(plot_df, x=x, y=y, z=z, color=color, size='energy')
    return fig
# %%
n=3
min_eta=-1.2
max_eta=1.2
fig = plot_event(sem_evts[n], use_offsets=False, use_pred_labels=False, min_eta=min_eta, max_eta=max_eta)
fig.show()
# %%
fig = plot_event(sem_evts[n], use_offsets=True, use_pred_labels=False, min_eta=min_eta, max_eta=max_eta)
#fig.update_layout(scene_zaxis_range=[400, 520])
fig.show()
# %%
for s_evt, i_evt in zip(sem_evts, inst_evts):
    s_evt.embedding = i_evt.embedding
# %%

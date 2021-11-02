# %%
from calo_cluster.evaluation.experiments.base_offset_experiment import BaseOffsetExperiment
from pathlib import Path 
from calo_cluster.clustering.meanshift import MeanShift
from tqdm.auto import tqdm
from calo_cluster.evaluation.utils import get_palette
import plotly.express as px
# %%
wandb_version = '1b8lkbs8'
exp = BaseOffsetExperiment(wandb_version)

# %%
evts = exp.get_events('val', n=100)

# %%
clusterer = MeanShift(bandwidth=0.02, use_semantic=True, ignore_semantic_labels=(0,))
for evt in tqdm(evts):
    evt.pred_instance_labels = clusterer.cluster(embedding=evt.embedding, semantic_labels=evt.input_event['pf_hit'].to_numpy())
    
# %%
from calo_cluster.evaluation.metrics.instance import PanopticQuality
# %%
evts[0].input_event
# %%
plot_df = evts[0].input_event
plot_df['pred_label'] = evts[0].pred_instance_labels
plot_df['pred_label'] = plot_df['pred_label'].astype(str)
plot_df['PFcluster0Id'] = plot_df['PFcluster0Id'].astype(str)
import plotly.express as px
mask = plot_df['pf_hit'] != 0
px.scatter_3d(plot_df[mask], x='x', y='y', z='z', size='energy', color='pred_label')
# %%
px.scatter_3d(plot_df[mask], x='x', y='y', z='z', size='energy', color='PFcluster0Id')
# %%
pq = PanopticQuality(num_classes=2, semantic=False)
for evt in tqdm(evts):
    mask = evt.input_event['pf_hit'] != 0
    pq.add(evt.pred_instance_labels[mask], evt.input_event['PFcluster0Id'].astype(int).to_numpy()[mask])
print(f'pq = {pq.compute()}')
# %%


# %%
evts[0].input_event.columns
# %%
n=3
import pandas as pd
for i, batch in enumerate(exp.datamodule.val_dataloader().dataset):
    if i == n:
        c = batch['coordinates'].F
        d = {'nx': c[:, 0], 'ny': c[:, 1], 'nz': c[:, 2], 'id': batch['labels'].F[:, 1]}
        plot_df2 = pd.DataFrame(d)
        plot_df2['id'] = plot_df2['id'].astype(str)
        break
    
plot_df = evts[n].input_event
pc = evts[n].embedding
plot_df['cx'] = pc[:,0]
plot_df['cy'] = pc[:,1]
plot_df['cz'] = pc[:,2]
plot_df['id'] = plot_df['instance_id'].astype(str)
px.scatter_3d(plot_df2, x='nx', y='ny', z='nz', color='id', color_discrete_sequence=get_palette(plot_df2['id'])).show()
px.scatter_3d(plot_df, x='cx', y='cy', z='cz', color='id', color_discrete_sequence=get_palette(plot_df['id'])).show()
# %%
px.scatter_3d(plot_df, x='x', y='y', z='z', color='id', color_discrete_sequence=get_palette(plot_df['id']))
# %%

# %%
from calo_cluster.training.criterion import offset_loss

# %%
import torch
import numpy as np
losses = []
for i, batch in enumerate(exp.datamodule.val_dataloader()):
    gt_offsets = batch['offsets'].F
    pred_offsets = torch.zeros_like(gt_offsets)
    valid_labels = torch.tensor([1,])
    semantic_labels = batch['labels'].F[:, 0]
    if valid_labels.device != semantic_labels.device:
        valid_labels = valid_labels.to(semantic_labels.device)
    valid = (semantic_labels[..., None] == valid_labels).any(-1)
    loss = offset_loss(pred_offsets, gt_offsets, valid)
    losses.append(loss)
    if i > 100:
        break
losses = np.array(losses)
# %%
print(losses.mean())
# %%

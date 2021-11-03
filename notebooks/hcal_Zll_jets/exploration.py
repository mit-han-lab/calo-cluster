# %%
from calo_cluster.datasets.hcal_Zll_jets import HCalZllJetsOffsetDataModule
import pandas as pd
import plotly.express as px
from calo_cluster.evaluation.utils import get_palette
import scipy
import numpy as np
import math
from tqdm.auto import tqdm
# %%
dm = HCalZllJetsOffsetDataModule.from_config(['dataset.instance_target=truth'])
# %%
dataset = dm.train_dataloader().dataset
# %%
coords_list = []
n = 100
for i, batch in tqdm(enumerate(dm.train_dataloader())):
    coords = batch['offsets'].F
    if i > n:
        break
    coords_list.append(coords)
flat_coords = np.concatenate(coords_list)
# %%
mean = np.median(np.abs(flat_coords[np.any(flat_coords!=0, axis=1)]), axis=0)
std = flat_coords.std(axis=0)
print(f'mean = {mean}')
print(f'std = {std}')
# %%
hcal_p_list = []
hcal_c_list = []
for i in range(100):
    particles = dataset._get_df(i)
    mask = particles['pf_hit'] != 1
    particles = particles[mask]
    clusters = dm.get_clusters(particles, cluster_col='PFcluster0Id')
    hcal_p_list.append(particles)
    hcal_c_list.append(clusters)
hcal_flat_c = pd.concat(hcal_c_list)
hcal_flat_p = pd.concat(hcal_p_list)
# %%
d = dataset[1]
coords = d['coordinates'].F
feats = d['features'].F
labels = d['labels'].F
# %%
offsets = np.abs(d['offsets'].F)
np.mean(offsets[offsets != 0])
# %%
df_dict = {'eta': coords[:,0], 'phi': coords[:,1], 'energy': feats[:,5], 'id': labels[:,1]}
plot_df = pd.DataFrame(df_dict)
plot_df['id'] = plot_df['id'].astype(str)
# %%
px.scatter(plot_df, x='eta', y='phi', color='id', size='energy', color_discrete_sequence=get_palette(plot_df['id']))
# %%
feats
# %%
dm.voxel_occupancy(n=1000)
# %%
# %%
import torch
import numpy as np
from calo_cluster.training.criterion import offset_loss
losses = []
for i, batch in enumerate(dm.val_dataloader()):
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

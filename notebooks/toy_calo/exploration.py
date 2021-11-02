# %%
from calo_cluster.datasets.toy_calo import ToyCaloOffsetDataModule
import plotly.express as px
import pandas as pd
import numpy as np
from pathlib import Path
from calo_cluster.evaluation.utils import get_palette
from calo_cluster.training.criterion import offset_loss
# %%
dm = ToyCaloOffsetDataModule.from_config(['dataset.np_mu=0.55'])
dset = dm.train_dataset

# %%
n=7
c = dset[n]['features'].F
plot_df = pd.DataFrame({'x': c[:, 0], 'y': c[:, 1], 'energy': c[:,2], 'id': dset[n]['labels'].F[:,1]})
plot_df['id'] = plot_df['id'].astype(str)
px.scatter(plot_df, x='x', y='y', color='id', color_discrete_sequence=get_palette(plot_df['id']), size='energy')
# %%
c.shape
# %%
from tqdm.auto import tqdm
voxel_sizes = [0.01, 0.1, 1.0]
occupancies = {}
for vx in tqdm(voxel_sizes):
    dm = ToyCaloOffsetDataModule.from_config([f'dataset.voxel_size={vx}', 'dataset.event_frac=0.1'])
    occupancies[vx] = dm.voxel_occupancy(only_different_labels=True, label_type='instance')
# %%
for vx in voxel_sizes:
    print(f'mean occupancy (vx = {vx}): {occupancies[vx].mean()}')
# %%
px.histogram(occupancies)
# %%
for fig in figs:
    fig.show()
# %%
(occupancies<1.1).sum() / occupancies.shape[0]
# %%
coords_list = []
for i, evt in enumerate(dm.train_dataloader().dataset):
    coords = evt['coordinates'].F
    coords_list.append(coords)

# %%
flat_coords = np.concatenate(coords_list)
# %%
flat_coords.mean(axis=0)
# %%
flat_coords.std(axis=0)
# %%
import torch
import numpy as np
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

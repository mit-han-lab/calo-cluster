# %%
from calo_cluster.datasets.vertex import VertexOffsetDataModule
import pandas as pd
import plotly.express as px
from calo_cluster.evaluation.utils import get_palette
import scipy
import numpy as np
import math
from tqdm.auto import tqdm

# %%
# voxel_sizes = [0.001, 0.01, 0.1, 1.0, 10.0]
voxel_sizes = [ 0.01 * i for i in range(1,20)]
# voxel_sizes = [] 
occupancies = {}
for vx in tqdm(voxel_sizes):
    dm = VertexOffsetDataModule.from_config(['dataset.event_frac=0.01', f'dataset.voxel_size={vx}'])
    # dm = VertexOffsetDataModule.from_config(['dataset.event_frac=0.01', f'dataset.voxel_size={vx}', 'dataset.num_classes=2', 'dataset.num_features=5', 'dataset.data_dir=/global/cscratch1/sd/hrzhao/calo_cluster/data'])
    occupancies[vx] = dm.voxel_occupancy(only_different_labels=True, label_type='instance')
# %%
for vx in voxel_sizes:
    print(f'mean occupancy (vx = {vx}): {occupancies[vx].mean()}')
# %%
px.histogram(occupancies)
# %%
dataloader = dm.train_dataloader()
for i, batch in tqdm(enumerate(dataloader), total=len(dataloader.dataset)):
    if i == 0:
        break
# %%
batch
# %%
batch['labels']
# %%
dataloader.dataset._get_df(0)['PFcluster0Id'].shape
# %%
original = dataloader.dataset._get_df(0)['PFcluster0Id'].values
sampled = batch['labels'].F[:,1][batch['inverse_map'].F].numpy()
# %%
mask = sampled != original
# %%
mask.sum()
# %%

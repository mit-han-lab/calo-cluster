# %%
from calo_cluster.datasets.hcal_Zll_jets import HCalZllJetsOffsetDataModule
import pandas as pd
import plotly.express as px
from calo_cluster.evaluation.utils import get_palette
import scipy
import numpy as np
import math
from tqdm import tqdm

# %%
voxel_sizes = [0.001, 0.01, 0.1, 1.0, 10.0]
occupancies = {}
for vx in tqdm(voxel_sizes):
    dm = HCalZllJetsOffsetDataModule.from_config(['dataset.event_frac=0.01', f'dataset.voxel_size={vx}'])
    occupancies[vx] = dm.voxel_occupancy(only_different_labels=True, label='instance_label')
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

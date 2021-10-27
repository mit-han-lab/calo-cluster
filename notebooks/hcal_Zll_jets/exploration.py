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
dm = HCalZllJetsOffsetDataModule.from_config()
# %%
dataset = dm.train_dataloader().dataset
# %%
coords_list = []
n = 100
for i, batch in tqdm(enumerate(dm.train_dataloader())):
    coords = batch['coordinates'].F
    if i > n:
        break
    coords_list.append(coords)
flat_coords = np.concatenate(coords_list)
# %%
mean = flat_coords.mean(axis=0)
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
df_dict = {'x': coords[:,0], 'y': coords[:,1], 'z': coords[:,2], 'energy': feats[:,4], 'id': labels[:,1]}
plot_df = pd.DataFrame(df_dict)
plot_df['id'] = plot_df['id'].astype(str)
# %%
px.scatter_3d(plot_df, x='x', y='y', z='z', color='id', size='energy', color_discrete_sequence=get_palette(plot_df['id']))
# %%
feats
# %%
dm.voxel_occupancy(n=1000)
# %%

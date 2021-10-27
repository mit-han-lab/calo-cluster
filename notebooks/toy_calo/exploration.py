# %%
from calo_cluster.datasets.toy_calo import ToyCaloOffsetDataModule
import plotly.express as px
import pandas as pd
import numpy as np
from pathlib import Path
from calo_cluster.evaluation.utils import get_palette
# %%
p_s = 5
dm = ToyCaloOffsetDataModule.from_config([f'dataset.p_s={p_s}', 'dataset.include_noise=True', 'dataset.voxel_size=0.2'])
dset = dm.train_dataset

# %%
n=5
c = dset[n]['features'].F
plot_df = pd.DataFrame({'x': c[:, 0], 'y': c[:, 1], 'z': c[:, 2], 'energy': c[:,3], 'id': dset[n]['labels'].F[:,1]})
plot_df['id'] = plot_df['id'].astype(str)
px.scatter_3d(plot_df, x='x', y='y', z='z', color='id', color_discrete_sequence=get_palette(plot_df['id']), size='energy')
# %%
c.shape
# %%
import tqdm
p_s = 5
figs = []
for voxel_size in tqdm.tqdm([0.1, 1.0, 10, 100]):
    dm = ToyCaloOffsetDataModule.from_config([f'dataset.p_s={p_s}', 'dataset.include_noise=False', f'dataset.voxel_size={voxel_size}', 'dataset.event_frac=0.1'])
    occupancies = dm.voxel_occupancy()
    figs.append(px.histogram(occupancies, title=f'{voxel_size}'))
# %%
for fig in figs:
    fig.show()
# %%
(occupancies<1.1).sum() / occupancies.shape[0]
# %%
dset[]
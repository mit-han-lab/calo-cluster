# %%
import h5py
# %%
f = h5py.File('/global/cscratch1/sd/schuya/calo_cluster/data/garnet_hgcal/events_0.h5')

# %%
f.keys()
# %%
cluster = f['cluster']
# %%
cluster.keys()
# %%
cluster[0]
# %%
cluster[0].shape
# %%
cluster = cluster.value
# %%
import numpy as np

# %%
dset = np.array(cluster)
# %%
dset[0]

# %%
dset[0][1]
# %%
import plotly.express as px

# %%
import pandas as pd
n = 2
plot_df = pd.DataFrame({'x': dset[n, :, 0], 'y': dset[n, :, 1], 'z': dset[n, :, 2], 'energy': dset[n, :, 3]})
px.scatter_3d(plot_df, x='x', y='y', z='z', size='energy')
# %%
f.keys()
# %%
f['raw']
# %%

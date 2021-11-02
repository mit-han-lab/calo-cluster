# %%
from sklearn.datasets import make_blobs
from numpy.random import default_rng
import numpy as np
import pandas as pd
import plotly.express as px

rng = default_rng()
l = 10
from pathlib import Path

# %%
def generate_event(l, n_noise, noise_scale, signal_scale):
    # noise
    noise_x = rng.uniform(-l, l, size=n_noise)
    noise_y = rng.uniform(-l, l, size=n_noise)   
    noise_z = rng.uniform(-l, l, size=n_noise)
    noise_E = rng.exponential(size=n_noise, scale=noise_scale)

    noise_df = pd.DataFrame({'x': noise_x, 'y': noise_y, 'z': noise_z, 'E': noise_E})
    noise_df['semantic_label'] = 0
    noise_df['instance_label'] = -1

    n_clusters = rng.poisson(10, 1)
    n_samples = rng.poisson(5, n_clusters)
    X, y = make_blobs(n_samples=n_samples,
                        n_features=3, cluster_std=0.4)
    signal_E = rng.exponential(size=n_samples.sum(), scale=signal_scale)
    signal_df = pd.DataFrame({'x': X[:, 0], 'y': X[:, 1], 'z': X[:, 2], 'E': signal_E, 'instance_label': y}) 
    signal_df['semantic_label'] = 1
    df = pd.concat((noise_df, signal_df))
    return df
# %%
df = generate_event(l=l, n_noise=1000, noise_scale=0.5, signal_scale=10.0)
df['s_instance_label'] = df['instance_label'].astype(str)
# %%
px.scatter_3d(df, x='x', y='y', z='z', size='E', color='s_instance_label')

# %%
signal_scale = 10.0
n_clusters = rng.poisson(10, 1)
n_samples = rng.poisson(5, n_clusters)
X, y = make_blobs(n_samples=n_samples,
                    n_features=3, cluster_std=0.8)
signal_E = rng.exponential(size=n_samples.sum(), scale=signal_scale)

# %%
X.shape

# %%
y.shape
# %%
n_samples.sum()

# %%
from calo_cluster.datasets.simple import SimpleDataModule
from pathlib import Path
# %%
SimpleDataModule.generate(Path('/global/cscratch1/sd/schuya/calo_cluster/data/simple'), n_events=10000)
# %%
# %%
from calo_cluster.datasets.simple import SimpleOffsetDataModule
import plotly.express as px
import pandas as pd
import numpy as np
from pathlib import Path
from calo_cluster.evaluation.utils import get_palette
from calo_cluster.training.criterion import offset_loss
# %%
dm = SimpleOffsetDataModule.from_config(['dataset.voxel_size=0.05'])
dset = dm.train_dataset

# %%
n=9
c = dset[n]['features'].F
plot_df = pd.DataFrame({'x': c[:, 0], 'y': c[:, 1], 'z': c[:, 2], 'energy': c[:,3], 'id': dset[n]['labels'].F[:,1]})
plot_df['id'] = plot_df['id'].astype(str)
px.scatter_3d(plot_df, x='x', y='y', z='z', color='id', color_discrete_sequence=get_palette(plot_df['id']))
# %%
import numpy as np
from tqdm.auto import tqdm
def make_dist_plots(dm, ignore_label, n=500):
    dm.batch_size = 1
    ds = dm.train_dataloader().dataset
    n_clusters = np.zeros(n)
    
    #n clusters
    for i, event in tqdm(enumerate(ds)):
        if i >= n:
            break
        labels = event['labels'].F[:, 1]
        valid = labels != ignore_label
        labels = labels[valid]
        unique_labels = set(np.unique(labels))
        nc = len(unique_labels)
        n_clusters[i] = nc
    px.histogram(n_clusters, title="number of clusters").show()
    print(f'mean number of clusters: {n_clusters.mean()}')
    n_clusters = None

    #n points per cluster
    n_points = []
    for i, event in tqdm(enumerate(ds)):
        if i >= n:
            break
        c = event['coordinates'].F
        labels = event['labels'].F[:, 1]
        valid = labels != ignore_label
        c = c[valid]
        labels = labels[valid]
        unique_labels = set(np.unique(labels))
        nc = len(unique_labels)
        npts = np.zeros(nc)
        for i, cid in enumerate(unique_labels):
            mask = (labels == cid)
            npts[i] = mask.sum()
            n_points.append(npts)
    n_points = np.concatenate(n_points)
    px.histogram(n_points, title="points per cluster").show()
    print(f'mean points per cluster: {n_points.mean()}')
    n_points = None
    
    #cluster offset
    offsets = []
    for i, event in tqdm(enumerate(ds)):
        if i >= n:
            break
        c = event['coordinates'].F
        labels = event['labels'].F[:, 1]
        valid = labels != ignore_label
        c = c[valid]
        offsets.append(event['offsets'].F[valid]) # offsets
    offsets = np.concatenate(offsets)
    for i in range(offsets.shape[1]):
        px.histogram(offsets[:,i], title=f"offset from cluster center (coordinate {i})").show()
    print(f'mean cluster offsets: {offsets.mean(axis=0)}')
    print(f'std cluster offsets: {offsets.std(axis=0)}')
    offsets = None
    
    # cluster center in coordinates
    centers = []
    for i, event in tqdm(enumerate(ds)):
        if i >= n:
            break
        # n clusters, n points per cluster, cluster offset, 
        c = event['coordinates'].F
        labels = event['labels'].F[:, 1]
        valid = labels != ignore_label
        c = c[valid]
        labels = labels[valid]
        unique_labels = set(np.unique(labels))
        nc = len(unique_labels)
        for i, cid in enumerate(unique_labels):
            mask = (labels == cid)
            centers.append(c[mask]) # cluster center in coords
    centers = np.concatenate(centers)
    for ci in range(centers.shape[1]):
        px.histogram(centers[:,ci], title=f"cluster center (coordinate {ci})").show()
    
# %%
n_points, centers, offsets, n_clusters = make_dist_plots(dm, ignore_label=-1)
# %%
n_points.mean()
# %%
n_clusters.mean()
# %%
from calo_cluster.datasets.hcal_Zll_jets import HCalZllJetsOffsetDataModule
import plotly.express as px
import pandas as pd
import numpy as np
from pathlib import Path
from calo_cluster.evaluation.utils import get_palette
from calo_cluster.training.criterion import offset_loss
from tqdm.auto import tqdm
# %%
dm = HCalZllJetsOffsetDataModule.from_config(['dataset.min_eta=-1.5', 'dataset.max_eta=1.5', 'dataset.coords=[eta,phi]', 'dataset.feats=[eta,phi,energy]', 'dataset.transform_coords=False', 'dataset.event_frac=1.0'])
dset = dm.train_dataset
# %%
n=10
c = dset[n]['features'].F
plot_df = pd.DataFrame({'eta': c[:, 0], 'phi': c[:, 1], 'energy': c[:,2], 'id': dset[n]['labels'].F[:,1]})
plot_df['id'] = plot_df['id'].astype(str)
px.scatter(plot_df, x='eta', y='phi', color='id', color_discrete_sequence=get_palette(plot_df['id']), size='energy')
# %%
make_dist_plots(dm, ignore_label=-1, n=1000)
# %%
dm.batch_size = 1
ds = dm.train_dataloader().dataset
n_clusters = np.zeros(len(n))
ignore_label = -1
#n clusters
n = 1000
for i, event in tqdm(enumerate(ds)):
    if i >= n:
        break
    labels = event['labels'].F[:, 1]
    valid = labels != ignore_label
    labels = labels[valid]
    unique_labels = set(np.unique(labels))
    nc = len(unique_labels)
    n_clusters[i] = nc
px.histogram(n_clusters, title="number of clusters").show()
print(f'mean number of clusters: {n_clusters.mean()}')
n_clusters = None
# %%
len(ds)
# %%

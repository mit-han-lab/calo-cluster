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
from hgcal_dev.datasets.simple import SimpleDataModule
from pathlib import Path
# %%
SimpleDataModule.generate(Path('/home/alexj/data/hgcal-dev/simple'), n_events=10000)
# %%
